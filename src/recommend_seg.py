import os
import json
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import math
import argparse
import pickle
import cv2
from time import time
from itertools import compress

# upper/lower 등 higher category 반환
def check_what_cate(given_label):
    cate_ = [k for k, v in hlv_cate_dict.items() if given_label in v][0]
    assert cate_ is not None
    return cate_


def load_features(pooling_dir, selected_method, save_folders):

    pooling_dir = pooling_dir
    save_folders = save_folders

    pooling_method = ''.join((selected_method, '.pkl'))
    pooling_method_dir = os.path.join(pooling_dir, pooling_method)

    with open(pooling_method_dir, 'rb') as tmp_file:
        current_method = pickle.load(tmp_file)

    tmp_list_method = np.array([list(current_method[i].values())[1].split('.')[0] for i in range(len(current_method))])
    tf_filter = list(np.isin(tmp_list_method, save_folders))
    selected_features = list(compress(current_method, tf_filter))
    #selected_features = [a for a in current_method if a['id'].split('.')[0] in save_folders]
    
    return selected_features


####### plot #######
def plot_retrievals(query_img, db_rec_img, retrieval_num, show_length = 5):
    '''
    query_img, db_rec_img should be in numpy.array (w, h, c) form
    '''
    #plt.show()
    plt.figure(figsize = (retrieval_num * 4, retrieval_num * 4 / 2.5))
    
    max_row = math.ceil(retrieval_num/show_length) + 1
    
    ax = plt.subplot(max_row, show_length, 1)
    ax.imshow(query_img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title("Query",  fontsize=14)
    
    for i, db_imb in enumerate(db_rec_img):
        
        ax = plt.subplot(max_row, show_length, show_length + 1 + i)
        ax.imshow(db_imb)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title(f"Recommend {i+1}")

    # save figure
    plt.savefig('../dataset/reg_images/seg.png', dpi = 300)

def recommend_other_cate(label,
                        wanted_type,
                        k = 5,
                        seg_dir = None,
                        tot_dir = None):
    '''
    label: knitwear, t-shirts, ...
    wanted_type: upper/lower
    idx: random한 번호
    '''
    assert (seg_dir is not None) and (tot_dir is not None), "Segment and total_image directories should be provided."

    k = k
    
    # 원하는 카테고리 선택(knitwear 등)
    crit = label
    crit_type = check_what_cate(crit)
    assert crit_type != wanted_type, f"Target Label = {crit_type}, wanted type = {wanted_type}!"

    # label이 동일한 것들(니트, 티셔츠 등)만 가져옴
    crit_features = [a for a in selected_features if a['label'] == crit]
    crit_hlv_cate = [list(hlv_cate_dict.keys())[i] for i, a in enumerate(list(hlv_cate_dict.values())) if crit in a][0]

    # 원하는 추천 카테고리 선택 (하의, 아우터.. 현재는 하의만)
    # 수정
    wanted_type = wanted_type

    idx = random.randint(0, len(crit_features))
    query = crit_features[idx]
    
    # db는 query와 다른 것들 중 query와 pair_id가 다른 것만 가져옴
    # 추가: 서로 달라야...
    db = [a for i, a in enumerate(crit_features) if (i != idx) and (a['pair_id'] != query['pair_id'])]

    db_same_label_num = len(db)
    if db_same_label_num == 0:
        print("Error: No matching item in DB!")

    else:

        query_feat = torch.from_numpy(query['feature'])
        query_feat_norm = query_feat / query_feat.norm()

        db_feats_list = [a['feature'] for a in db]
        
        db_feats = torch.from_numpy(np.array(db_feats_list))
        db_feats_norm = db_feats.t() / db_feats.norm(dim = 1)

        # to_device
        # cuda.is_available()

        # feature similarity
        feats_sim = torch.mm(query_feat_norm.unsqueeze(0), db_feats_norm).squeeze(0)

        # similarity index(descending order)
        feats_sim_index = feats_sim.sort(descending = True)[1].tolist()

        # sorted db
        sorted_db = [db[a] for a in feats_sim_index]

        # top-k 추천
        topk = min(db_same_label_num, k)
        
        # 하나씩 추천하되, 동일한 pair_id는 넣지 않도록
        recom_done = False
        recom_db = []
        tmp_pair_ids = []
        
        k_index = 0
        while not recom_done:
            # 이미 추가한 pair_id이면 넣지 않음
            if sorted_db[k_index]['pair_id'] not in tmp_pair_ids:
                recom_db.append(sorted_db[k_index])
                tmp_pair_ids.append(sorted_db[k_index]['pair_id'])
            else:
                pass
            
            if (len(recom_db) == topk) or (len(sorted_db) == k_index + 1):
                recom_done = True
            else:
                k_index += 1

        recom_ids = [a['id'].split('.')[0] for a in recom_db]

    query_id = query['id'].split('.')[0]

    #print(f"Query id: {query_id}, Recom_ids: {recom_ids}")

    
    ######################
    ### PLOT segmented ###
    ######################
    query_img_file = os.path.join(seg_dir, query_id, ''.join((crit, '.png')))

    # DB 이미지
    db_img_files = []

    db_img_dirs = [os.path.join(base_dir, a) for a in recom_ids]

    for recom_id in recom_ids:

        # ['pants', 'skirts'] 등 원하는 추천 타겟
        base_targets = [a.split('.')[0] for a in os.listdir(os.path.join(seg_dir, recom_id))if a.split('.')[0] != crit]

        wanted_target = [a for a in base_targets if check_what_cate(a) == wanted_type]

        # 하나만 random하게 선택
        db_img_files.append(os.path.join(seg_dir, recom_id, ''.join((random.choice(wanted_target), '.png'))))

    # read images   
    query_img = cv2.cvtColor(cv2.imread(query_img_file), cv2.COLOR_BGR2RGB)
    db_img = [cv2.cvtColor(cv2.imread(a), cv2.COLOR_BGR2RGB) for a in db_img_files]
    
    ######################
    ### PLOT original  ###
    ######################
    # org_query_img_file = os.path.join(tot_dir, ''.join((query_id, '.jpg')))

    # # DB 이미지
    # org_db_img_files = [os.path.join(tot_dir, a) for a in [''.join((b, '.jpg')) for b in recom_ids]]

    # # read images    
    # org_query_img = cv2.cvtColor(cv2.imread(org_query_img_file), cv2.COLOR_BGR2RGB)
    # org_db_img = [cv2.cvtColor(cv2.imread(a), cv2.COLOR_BGR2RGB) for a in org_db_img_files]

    
    # plot
    plot_retrievals(query_img, db_img, retrieval_num = topk)
    #plot_retrievals(org_query_img, org_db_img, retrieval_num = topk)

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--seg_path", type = str, default = './home/sks/kfashion/kfashion_dataset/seg')
    parser.add_argument("--model_path", type = str, default = './results/feature_extraction')
    parser.add_argument("--img_path", type = str, default = './datasets/for/images')
    parser.add_argument("--extractor_type", type = str, default = 'cgd_pca')
    parser.add_argument("--target_label", default = None, help = "What label to evaluate?")
    parser.add_argument("--wanted_type", default = None, help = "What other category do you want to retrieve?")
    parser.add_argument("--top_k", type = int, default = 5, help = "How many items to recommend?")

    args = parser.parse_args()

    # segmented image가 있는 디렉토리
    base_dir = args.seg_path
    tot_dir = args.img_path

    # hierarchical division
    hlv_cate_dict = {'upper': ['jacket', 'padded jacket', 'coat', 'jumper', 'dress', 'cardigan',
                            't-shirts', 'top', 'jumpsuit', 'hoody', 'blouse', 'knitwear', 'shirts', 'vest'],
                    'lower': ['pants', 'jean', 'leggings', 'skirt']}

    # 2개 이상(상의/하의 등) 있는 것들만 저장
    save_folders = []

    for afolder in os.listdir(base_dir):
        
        suffix_ = ('.png', '.jpg', 'PNG', 'JPG')
        tmp_items = [file_ for file_ in os.listdir(os.path.join(base_dir, afolder)) if file_.endswith(suffix_)]
        
        items = [a.split('.')[0] for a in tmp_items]

        check = set([check_what_cate(a) for a in items])

        if len(check) >= 2:
            save_folders.append(afolder)


    # 2개 이상인 feature 값만 load
    selected_features = load_features(pooling_dir = args.model_path, selected_method = args.extractor_type, save_folders = save_folders)

    recommend_other_cate(label = args.target_label,
                         wanted_type = args.wanted_type,
                         k = args.top_k,
                         seg_dir = base_dir,
                         tot_dir = tot_dir)
    
    print("Recommendation Done!")

