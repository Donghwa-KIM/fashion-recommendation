from serve_seg import *
from cgd.pooler import ROIpool
import detectron2.data.transforms as T

import torch
import os, argparse, json, pickle
import numpy as np
import random
import math
from itertools import compress
import pytz
from datetime import datetime
import yaml
from varname import nameof




import warnings
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)



class FailException:
    def __init__(self, json_path, func2failnum):
        self.json_path = json_path
        self.func2failnum = func2failnum
        
    def __call__(self, func):
        def inner_function(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                e.args = (f'{self.func2failnum[func.__name__]} in {func.__name__} => ' + e.args[0] ,)
                save_json(self.json_path, self.func2failnum[func.__name__],{})
                raise 

        return inner_function  
    
def load_model_configs(args):
    with open(args.config_path) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    return configs



parser = argparse.ArgumentParser()

parser.add_argument("--save_path", type=str, default="./dataset/rec_images",
                    help='path to save final json output')
parser.add_argument("--image_path", type=str, default="./dataset/samples/245993.jpg",
                    help='input image')
parser.add_argument("--model_path", type=str, default="Misc/cascade_mask_rcnn_R_101_FPN_3x.yaml", 
                    help='--pretrained COCO dataset for semgentation task')
parser.add_argument("--model_weights", type=str, default="./model/kfashion_cascade_mask_rcnn",
                    help='model checkpoints')
parser.add_argument("--cgd_path", type=str, default="./model/",
                    help='cgd root path')
parser.add_argument("--config_path", type=str, default="./src/configs.yaml", 
                    help='-- convenient configs for models')
parser.add_argument("--seg_path", type = str, default = './dataset/segDB')
parser.add_argument("--abs_seg_path", type = str, default = '/home/korea/fashion-recommendation/dataset/segDB')
parser.add_argument("--extractor_type", type = str, default = 'cgd_pca_pairItem')
parser.add_argument("--extractor_path", type = str, default = './dataset/feature_extraction')
parser.add_argument("--target_category", type = str, default = 'lower',
                    help='category option, selected from (upper, lower, outer)')
parser.add_argument("--target_color", type = str, default = None)
parser.add_argument("--target_style", type = str, default = "섹시")
parser.add_argument("--top_k", type = int, default = 5,
                    help = "How many items to recommend?")

args = parser.parse_args()


# Exception
class FailException:
    def __init__(self, json_path, func2failnum):
        self.json_path = json_path
        self.func2failnum = func2failnum
        
    def __call__(self, func):
        def inner_function(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                e.args = (f'{self.func2failnum[func.__name__]} in {func.__name__} => ' + e.args[0] ,)
                save_json(self.json_path, self.func2failnum[func.__name__],{})
                raise 

        return inner_function  


def save_json(path, code, body):
    tz_kor = pytz.timezone('Asia/Seoul') 
    json_dict = {}

    json_dict["resultCode"] = code
    json_dict["resultMessage"] = "SUCCESS" 
    json_dict["body"] = body
    json_dict["responseDate"] = datetime.now(tz_kor).strftime('%Y-%m-%d %H:%M:%S')

    with open(path, "w") as f:
        json.dump(json_dict, f)
        logger.info("Saved json in {}".format(path))

        

def base_extract(args, cate_master_dict):

    # model configs
    configs = load_model_configs(args)    

    # categoies info
    fashion_metadata = build_categories(configs)

    # model index
    args.model_idx = get_checkpoint(args)

    # model for semgmentation
    predictor= get_predictor(args, configs)
    logger.info(f"Extracting for {args.image_path}")
    im = get_image(args)
    check_image(im)

    # prediction
    outputs = predictor(im)
    labels  = get_labels(configs, outputs)
    logger.info(f"Extracted {len(labels)} item(s): {labels}")

    # cgd
    model = torch.load( os.path.join(args.cgd_path, 'cgd_model.pt'), map_location=torch.device('cpu'))
    logger.info(f"load cgd model from {os.path.join(args.cgd_path, 'cgd_model.pt')}")

    ####################
    ### Error code 1 ###
    ####################
    if not labels:
        return None, None, None

    else:
        
        # test resize func
        resizefunc = T.ResizeShortestEdge(
                [predictor.cfg.INPUT.MIN_SIZE_TEST, predictor.cfg.INPUT.MIN_SIZE_TEST], predictor.cfg.INPUT.MAX_SIZE_TEST
            )
        # test size
        im = resizefunc.get_transform(im).apply_image(im)

        image_batch = [{'image':torch.Tensor(im.transpose(2,0,1))}]
        roi_pooler  = ROIpool(predictor.model)

        with torch.no_grad():
            features, pred_ins, is_empty = roi_pooler.batches(image_batch)
            cgd, _ = model(features)

        classes = [configs['Detectron2']['LABEL_LIST']['kfashion'][cls_.item()] 
                for ins in pred_ins for cls_ in ins.pred_classes]

        # bigger categories:
        hlv_classes = [dict_[0] for detected_ in classes for dict_ in cate_master_dict.items() if detected_ in dict_[1]]

        with open(f'./model/pca_model.pkl', 'rb') as f:
            pca = pickle.load(f)

        X = cgd.detach().cpu().numpy()
        cgd_pca = pca.transform(X)
        
        return classes, hlv_classes, cgd_pca


# upper/lower 등 higher category 반환
def check_what_cate(master_dict, given_label):
    cate_ = [k for k, v in master_dict.items() if given_label in v]
    hlv_cate = cate_[0] if cate_ else None
    return hlv_cate


# 여러 카테고리 검출 등 경우 하나의 클래스만 선정
def set_one_class(target_category, *args):
    classes = args[0]
    hlv_classes = args[1]


def apply_option(original, *args):
    
    target_category, target_color, target_style = args[0], args[1], args[2]
    
    A = list(set([cgd['id'].split('.')[0] for cgd in original if check_what_cate(cate_option_dict, cgd['label']) == target_category]))
    B = list(set([cgd['id'].split('.')[0] for cgd in original if cgd['category_color'] == target_color] if target_color is not None else [None]))
    C = list(set([cgd['id'].split('.')[0] for cgd in original if cgd['style'] == target_style] if target_style is not None else [None]))
    
    valid = [np.array(k) for k in [A, B, C] if k != [None]]
    result = np.array(list(set([cgd['id'].split('.')[0] for cgd in original])))
    
    for K in valid:
        result = np.intersect1d(result, K)

    result = list(result)
    
    return result
    


def load_features(pooling_dir, selected_method, cate_option_dict, detected_class, *args):

    pooling_dir  = pooling_dir

    pooling_method     = ''.join((selected_method, '.pkl'))
    pooling_method_dir = os.path.join(pooling_dir, pooling_method)

    with open(pooling_method_dir, 'rb') as tmp_file:
        current_method = pickle.load(tmp_file)
    
    # Filter1: color/style 적용
    filter1 = apply_option(current_method, *args)
    
    # Filter 1: 색상/스타일/카테고리 옵션
#     filter1 = [cgd['id'].split('.')[0] for cgd in current_method if cgd['category_color']==target_color and
#                                                                     cgd['style']==target_style and
#                                                                     check_what_cate(cate_option_dict, cgd['label'])==target_category]
    
    # Filter 2: d
    filter2 = list(set([cgd['id'].split('.')[0] for cgd in current_method if cgd['label']==detected_class]))
    
    # 교집합
    filtered = list(np.intersect1d(np.array(filter1), np.array(filter2)))
    
    logger.info(f"Items left after filtering: {len(filtered)}")
    
    #selected_feature = [cgd for cgd in current_method if cgd['id'].split('.')[0] in filtered]
    tmp_list_method = np.array([cgd['id'].split('.')[0] for cgd in current_method])
    
    tf_filter = list(np.isin(tmp_list_method, filtered))
    selected_features = list(compress(current_method, tf_filter))
    
    return selected_features


def retrieve_other_image(base_dir, input_id, wanted_type):

    to_retrieve = []
    for id_ in input_id:
        # get name only
        cand_images = list(map(lambda r: r.split('.')[0], os.listdir(os.path.join(base_dir, id_))))

        tmp_retrieve = [img_ for img_ in cand_images if img_ in cate_master_dict[wanted_type]]
        tmp_retrieve = list(map(lambda r: ''.join((r, '.png')), tmp_retrieve))[0]

        to_retrieve.append(tmp_retrieve)

    return to_retrieve


def recommend_other_cate(query,
                         DB,
                         label = None,
                         wanted_type = None,
                         k = 5,
                         seg_dir = None,
                         abs_seg_dir = None):

    assert (seg_dir is not None), "Segmented dataset's directory should be provided."

    k = k
    crit = label
    
    # 원하는 추천 카테고리 선택 (하의, 아우터.. 현재는 하의만)
    wanted_type = wanted_type
    
    # Query and DB
    query = query
    db = DB
    #db = [item_ for item_ in DB if item_['label'] == crit]
    db_same_label_num = len(db)

    if db_same_label_num == 0:
        print(f"///Warning/// Query item type: {crit}, but no matching item found in DB.")
        return None, None

    else:
        # Calculate cosine similarity
        query_feat = torch.from_numpy(query)
        query_feat_norm = query_feat / query_feat.norm()

        db_feats_list = [item_['feature'] for item_ in db]

        db_feats = torch.from_numpy(np.array(db_feats_list))
        db_feats_norm = db_feats.t() / db_feats.norm(dim = 1)

        feats_sim = torch.mm(query_feat_norm.unsqueeze(0), db_feats_norm).squeeze(0)
        
        feats_sim_index = feats_sim.sort(descending = True)[1].tolist()
        feats_sim_scores = feats_sim.sort(descending = True)[0].tolist()

        # sorted db
        sorted_db = [db[a] for a in feats_sim_index]

        # top-k 추천
        topk = min(db_same_label_num, k)

        # 하나씩 추천하되, 동일한 pair_id는 넣지 않도록
        recom_done = False
        recom_db, recom_score, tmp_pair_ids = [], [], []

        k_index = 0
        while not recom_done:
            # 이미 추가한 pair_id이면 넣지 않음
            if sorted_db[k_index]['pair_id'] not in tmp_pair_ids:
                recom_db.append(sorted_db[k_index])
                recom_score.append(feats_sim_scores[k_index])
                tmp_pair_ids.append(sorted_db[k_index]['pair_id'])
            else:
                pass

            if (len(recom_db) == topk) or (len(sorted_db) == k_index + 1):
                recom_done = True
            else:
                k_index += 1

        recom_ids = [a['id'].split('.')[0] for a in recom_db]
        other_items = retrieve_other_image(seg_dir, input_id = recom_ids, wanted_type = wanted_type)
        other_item_path = [os.path.join(abs_seg_dir, id_, item_) for id_, item_ in zip(recom_ids, other_items)]

        print(f"Query item type: {crit}, {len(recom_ids)} items recommended.")

        return recom_ids, recom_score, other_item_path


    

# def save_valid_seg(cate_master_dict, cgd_feature, base_dir, target_category, target_color, target_style):
#     '''
#     CGD feature vector 기준으로 필터링 수행
#     Filter1: option(arg)에서 스타일, 컬러 일치
#     Filter2: (상의/하의/아우터) 중 2개 카테고리 이상 포함
#     '''
    
#     # 2개 이상(상의/하의 등) 있는 것들만 저장
#     save_folders = []
    
#     cgd_crit1 = [cgd for cgd in cgd_feature if check_what_cate(cgd['label']) == target_category and
#                                                cgd['category_color'] == target_color and
#                                                cgd['style'] == target_style]
    
    
    
#     for afolder in os.listdir(base_dir):

#         suffix_ = ('.png', '.jpg', 'PNG', 'JPG')
#         tmp_items = [file_ for file_ in os.listdir(os.path.join(base_dir, afolder)) if file_.endswith(suffix_)]
#         items = list(map(lambda r: r.split('.')[0], tmp_items))

#         check = list(set([check_what_cate(cate_master_dict, item_) for item_ in items]))
#         #check = [cate for cate in check if cate != 'block']

#         if len(check) >= 2:
#             save_folders.append(afolder)

#     return save_folders


    
    
if __name__ == "__main__":
    
    # (0) Master division
    configs = load_model_configs(args)
    cate_master_dict = configs['rec']['CATE_MASTER_DICT']
    cate_option_dict = configs['rec']['CATE_OPTION_DICT']
    
    condition_hlv = list(cate_master_dict.keys())
    condition_hlv.remove(args.target_category)

    #--------------------------
    #     Define exception
    #--------------------------
    json_path = os.path.join(args.save_path, 'jsons', f"{os.path.basename(args.image_path).split('.')[0]}.json")
    fail_exception = FailException(json_path, configs['exception']['rec'])

    
    @fail_exception
    def check_option_input(arguments):
        if not (arguments.target_category):
            raise Exception('Category option not provided')    

    @fail_exception
    def check_image(im):
        if im is None:
            raise Exception('Can not load the image, check the image path!')
    
    @fail_exception
    def check_detected(classes):
        ####################
        ### Error code 1 ###
        ####################
        if classes is None:
            raise Exception("No Item detected in the input image.")

    @fail_exception
    def check_category_redncy(classes, hlv_classes, features, condition_hlv, target_category):
        ####################
        ### Error code 2 ###
        ####################
        if list(set(hlv_classes)) == [target_category]:
            raise Exception(f"Input category and target category are the same:{list(set(hlv_classes))}={[target_category]}")
        
        else:
            to_use = [(classes[i], features[i]) for i, a in enumerate(hlv_classes) if a in condition_hlv]
            return to_use[0]
        
        
    @fail_exception
    def check_DB(DB):
        ####################
        ### Error code 2 ###
        ####################
        if not DB:
            raise Exception(f"No item to recommend, DB is empty which fits the given conditions")
    
    
    # option input check
    check_option_input(args)
    
    # (0) make file directory
    os.makedirs(os.path.join(args.save_path, 'jsons'), exist_ok = True)

    # (1) Extract
    classes, hlv_classes, pooled_feature = base_extract(args, cate_master_dict)
    check_detected(classes)
    
    target_label, target_feature = check_category_redncy(classes, hlv_classes, pooled_feature, condition_hlv, args.target_category)
    logger.info(f"Item to use for recommendation: {target_label}")
    
    # *** check not block
    #check_not_block(hlv_classes)

    # 니트, 바지 검출 -> upper, lower
    # (1) target이 lower라면 upper인 knit가 타겟 클래스
    # (2) target이 outer라면 앞선 upper를 선택

#     # (2) Save valid folders (including 2 categories)
#     filter_crit = save_valid_seg(base_dir = args.seg_path)


    # (3) Set DB
    DB = load_features(args.extractor_path,
                       args.extractor_type,
                       cate_option_dict,
                       target_label,
                       args.target_category,
                       args.target_color,
                       args.target_style)
    
    check_DB(DB)
    
    # (4) Recommend items of another category
    seg_result_json = []
    tmp_seg_result_json = []
    
    query = target_feature
    #wanted_type = [hlv for hlv in hlv_master if hlv != hlv_classes[i]][0]

    recom_ids, recom_scores, recom_items = recommend_other_cate(query = query,
                                                                DB = DB,
                                                                label = target_label,
                                                                wanted_type = args.target_category,
                                                                k = args.top_k,
                                                                seg_dir = args.seg_path,
                                                                abs_seg_dir = args.abs_seg_path)

    if recom_ids is not None:
        for zips in zip(recom_ids, recom_items, recom_scores):
            tmp_json = dict(workId    = zips[0],
                            file_path = zips[1],
                            score     = zips[2])
            tmp_seg_result_json.append(tmp_json)            
    else:
        tmp_seg_result_json.append({})

    seg_result_json = tmp_seg_result_json
    
    # When recommendation in successful, save json
    result_code = "SUCCESS"
    save_json(json_path, result_code, seg_result_json)