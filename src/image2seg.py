from serve_seg import *
from cgd.pooler import ROIpool

import torch
import os, argparse, json, pickle
import numpy as np
import random
import math
from itertools import compress
import pytz
from datetime import datetime
import yaml



parser = argparse.ArgumentParser()

parser.add_argument("--save_path", type=str, default="/home/appuser/fashion_repo/src",
                    help='path to save final json output')
parser.add_argument("--image_path", type=str, default="../dataset/samples/100401.jpg",
                    help='input image')
parser.add_argument("--model_weights", type=str, default="../model",
                    help='model checkpoints')
parser.add_argument("--cgd_path", type=str, default="../model/",
                    help='cgd root path')
parser.add_argument("--model_path", type=str, default="Misc/cascade_mask_rcnn_R_101_FPN_3x.yaml", 
                    help='--pretrained COCO dataset for semgentation task')
parser.add_argument("--config_path", type=str, default="../src/configs.yaml", 
                    help='-- convenient configs for models')
parser.add_argument("--seg_path", type = str, default = '../dataset/segDB')
parser.add_argument("--abs_seg_path", type = str, default = '/home/appuser/fashion_repo/src/dataset/segDB')
parser.add_argument("--extractor_type", type = str, default = 'cgd_pca')
parser.add_argument("--extractor_path", type = str, default = '../model')
parser.add_argument("--top_k", type = int, default = 5,
                    help = "How many items to recommend?")

args = parser.parse_args(args=[])


# Exception
func2failnum = {
    'load_model_configs': 'FAIL-001',
    'build_categories': 'FAIL-004',
    'plot': 'FAIL-003',
    'get_labels': 'FAIL-005',
    'get_image': 'FAIL-006',
    'base_extract': 'FAIL_007',
    'check_not_block': 'FAIL_008'
}



class FailException:
    def __init__(self, json_path, func2failnum):
        self.json_path = json_path
        self.func2failnum = func2failnum

    def handler(self, func):
        def inner_function(*args, **kwargs):
            try:
                func(*args, **kwargs)
            except Exception as e:
                e.args = (f'{self.func2failnum[func.__name__]} in {func.__name__} => ' + e.args[0] ,)
                save_json(self.json_path, self.func2failnum[func.__name__],{})
                raise 

        return inner_function

    
json_path = os.path.join(args.save_path, 'jsons', f"{os.path.basename(args.image_path).split('.')[0]}.json")
fail_exception = FailException(json_path, func2failnum)



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

        
        
#@fail_exception.handler  
def base_extract(args):

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

    if im is None:
        raise NotImplementedError('Can not load the image, check the image path!')

    # prediction
    outputs = predictor(im)
    labels  = get_labels(configs, outputs)
    logger.info(f"Extracted {len(labels)} item(s): {labels}")

    # cgd
    model = torch.load( os.path.join(args.cgd_path, 'cgd_model.pt')).to('cpu')
    logger.info(f"load cgd model from {os.path.join(args.cgd_path, 'cgd_model.pt')}")

    ####################
    ### Error code 1 ###
    ####################
    if not labels:
        raise Exception("No item detected.")

    image_batch = [{'image':torch.Tensor(im.transpose(2,0,1))}]
    roi_pooler  = ROIpool(predictor.model)

    with torch.no_grad():
        features, pred_ins, is_empty = roi_pooler.batches(image_batch)
        cgd, _ = model(features)

    classes = [configs['Detectron2']['LABEL_LIST']['kfashion'][cls_.item()] 
               for ins in pred_ins for cls_ in ins.pred_classes]

    # bigger categories:
    hlv_classes = [dict_[0] for detected_ in classes for dict_ in cate_master_dict.items() if detected_ in dict_[1]]

    with open(f'../model/pca_model.pkl', 'rb') as f:
        pca = pickle.load(f)

    X = cgd.detach().cpu().numpy()
    cgd_pca = pca.transform(X)

    return classes, hlv_classes, cgd_pca



@fail_exception.handler
def check_not_block(hlv_classes):
    ####################
    ### Error code 2 ###
    ####################
    if 'block' in hlv_classes:
        raise Exception("Single dress detected, no item to recommend")

        

# upper/lower 등 higher category 반환
def check_what_cate(master_dict, given_label):
    cate_ = [k for k, v in master_dict.items() if given_label in v][0]
    assert cate_ is not None, "No matching category in master dictionary."
    return cate_



def load_features(pooling_dir, selected_method, save_folders):

    pooling_dir  = pooling_dir
    save_folders = save_folders

    pooling_method     = ''.join((selected_method, '.pkl'))
    pooling_method_dir = os.path.join(pooling_dir, pooling_method)

    with open(pooling_method_dir, 'rb') as tmp_file:
        current_method = pickle.load(tmp_file)

    tmp_list_method = np.array([list(current_method[i].values())[1].split('.')[0] for i in range(len(current_method))])
    tf_filter = list(np.isin(tmp_list_method, save_folders))
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
    '''
    label: knitwear, t-shirts, ...
    wanted_type: upper/lower
    '''

    assert (seg_dir is not None), "Segmented dataset's directory should be provided."

    k = k
    crit = label
    
    # 원하는 추천 카테고리 선택 (하의, 아우터.. 현재는 하의만)
    wanted_type = wanted_type
    
    # Query and DB
    query = query
    db = [item_ for item_ in DB if item_['label'] == crit]
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


def save_valid_seg(base_dir):

    # 2개 이상(상의/하의 등) 있는 것들만 저장
    save_folders = []

    for afolder in os.listdir(base_dir):

        suffix_ = ('.png', '.jpg', 'PNG', 'JPG')
        tmp_items = [file_ for file_ in os.listdir(os.path.join(base_dir, afolder)) if file_.endswith(suffix_)]
        items = list(map(lambda r: r.split('.')[0], tmp_items))

        check = list(set([check_what_cate(cate_master_dict, item_) for item_ in items]))
        check = [cate for cate in check if cate != 'block']

        if len(check) >= 2:
            save_folders.append(afolder)

    return save_folders


    
    
if __name__ == "__main__":
    
    # (0) Master division
    with open(args.config_path) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    
    cate_master_dict = configs['reg']['CATE_MASTER_DICT']

    hlv_master = list(cate_master_dict.keys())
    hlv_master.remove('block')


    # (0) make file directory
    os.makedirs(os.path.join(args.save_path, 'jsons'), exist_ok = True)
    os.makedirs(os.path.join(args.save_path, 'images'), exist_ok = True)


    # (1) Extract
    classes, hlv_classes, pooled_feature = base_extract(args)


    # *** check not block
    check_not_block(hlv_classes)


    # (2) Save valid folders (including 2 categories)
    save_folders = save_valid_seg(base_dir = args.seg_path)


    # (3) Set DB
    DB = load_features(pooling_dir     = args.extractor_path,
                       selected_method = args.extractor_type,
                       save_folders    = save_folders)


    # (4) Recommend items of another category
    seg_result_json = []
    for i, target_label in enumerate(classes):

        tmp_seg_result_json = []

        query = pooled_feature[i]
        wanted_type = [hlv for hlv in hlv_master if hlv != hlv_classes[i]][0]

        recom_ids, recom_scores, recom_items = recommend_other_cate(query = query,
                                                                    DB = DB,
                                                                    label = target_label,
                                                                    wanted_type = wanted_type,
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
            
        seg_result_json.append(tmp_seg_result_json)
    

    # When recommendation in successful, save json
    result_code = "SUCCESS"
    save_json(json_path, result_code, seg_result_json)