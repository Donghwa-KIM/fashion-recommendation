import logging
import argparse
import torch
import numpy as np
import pandas as pd
import math
import os, fnmatch, shutil
import matplotlib.pyplot as plt
import cv2
import itertools
import pickle
import yaml
from time import time


LOG_PATH = './dataset/results/TTA/recommendation/'
os.makedirs(LOG_PATH, exist_ok=True)
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", 
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler(os.path.join(LOG_PATH,"recomm_log.txt")),
                        logging.StreamHandler()
                    ])


class FailException:
    def __init__(self, func2failnum):
        self.func2failnum = func2failnum
        
    def __call__(self, func):
        def inner_function(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                e.args = (f'{self.func2failnum[func.__name__]} in {func.__name__} => ' + e.args[0] ,)
                raise 

        return inner_function


def load_model_configs(args):
    with open(args.config_path,  encoding="utf-8") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    return configs


def split_pooled(train_dataset, test_dataset, unique_labels):

    train_db = {}
    for _label in unique_labels:
        tmp_db = [a for a in train_dataset if (a['label'] == _label) and (a['split'] in ('train', 'val'))]
        train_db[_label] = tmp_db

    test_db = {}
    for _label in unique_labels:
        tmp_db = [a for a in test_dataset if (a['label'] == _label) and (a['split'] == 'test')]
        test_db[_label] = tmp_db
        
    return train_db, test_db


def retrieve_from_db(query, db, k, print_rate = False):
    '''
    query: dictionary form {id, feature, pair_id}
    '''
    db_same_label_num = len(db)
    if db_same_label_num == 0:
        return None, None, None, None
    
    else:
        query_feat = torch.from_numpy(query['feature'])
        query_feat_norm = query_feat / query_feat.norm()

        db_feats_list = [a['feature'] for a in db]

        db_feats = torch.from_numpy(np.array(db_feats_list))
        db_feats_norm = db_feats.t() / db_feats.norm(dim = 1)

        # feature similarity
        feats_sim = torch.mm(query_feat_norm.unsqueeze(0), db_feats_norm).squeeze(0)

        # similarity index(descending order)
        feats_sim_index = feats_sim.sort(descending = True)[1].tolist()

        # sorted db
        sorted_db = [db[a] for a in feats_sim_index]
        
        test_single_result = []
        for target_k in k:

            k = min(db_same_label_num, target_k)

            recom_db = sorted_db[:k]
            recom_ids = [os.path.splitext(a['id'])[0] for a in recom_db]
            recom_pair_ids = [a['pair_id'] for a in recom_db]
            
            recom_set = list(zip(recom_ids, recom_pair_ids))
            logger.info(f"k: {k} | Ground Truth Pair_ID: {query['pair_id']}, Predicted: {recom_set}")
            
            test_single_result.append([os.path.splitext(query['id'])[0], query['pair_id'], recom_set])

        return test_single_result

    
def calc_metrics(unique_labels, set_k, *DB):

    train_db, test_db = DB[0], DB[1]
    
    result_pd = pd.DataFrame(columns = ['image_id', 'hit_k@1', 'hit_k@10'])
    
    real_fin_result = []
    for target_label in unique_labels:
        logger.info(f'================ Test on single item ==================')
        tot_num_per_target_label = len(test_db[target_label])

        M_recall_rate_list, M_hit_rate_list, M_rr_list = [], [], []

        for idx in range(tot_num_per_target_label):

            query = test_db[target_label][idx]
            db    = train_db[target_label]

            test_single_result = retrieve_from_db(query, db, k = set_k, print_rate = False)

    return test_single_result
    
    
if __name__ == "__main__":
    
    t0 = time()

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--extractor_path", type = str, default = './dataset/feature_extraction/TTA')
    parser.add_argument("--extractor_train", type = str, default = 'cgd')
    parser.add_argument("--extractor_test", type = str, default = 'cgd_test_sample')
    parser.add_argument("--target_id", type = str, default = '188293')
    parser.add_argument("--target_item", type = str, default = 'pants')
    parser.add_argument("--image_from_path", type = str, default = './dataset/kfashion_dataset_new/train_tot_images')
    parser.add_argument("--image_save_path", type = str, default = './dataset/results/TTA/recommendation')
    parser.add_argument("--set_ks", type = str, default = "1,10", help = "Set of k's used for testing")
    parser.add_argument("--config_path", type = str, default = './src/configs.yaml')
    
    args = parser.parse_args()
    
    args.image_save_path = os.path.join(args.image_save_path,f'{args.target_id}_{args.target_item}')

    configs = load_model_configs(args)
    
    # ---------------------
    #      EXCEPTION
    # ---------------------
    fail_exception = FailException(configs['exception']['rec_test'])
    # ---------------------
    #       FAIL-001
    # ---------------------
    @fail_exception
    def check_cgd_load(cgd_dir):
        if not os.path.exists(cgd_dir):
            raise Exception('Check directory of the selected extractor')
    
    
    # TRAIN DATASET(.pkl)
    train_method_dir = os.path.join(args.extractor_path, ''.join((args.extractor_train, '.pkl')))
    check_cgd_load(train_method_dir)
    
    with open(train_method_dir, 'rb') as tmp_file:
        train_dataset = pickle.load(tmp_file)
    
    
    # TEST DATASET(.pkl)
    test_method_dir = os.path.join(args.extractor_path, ''.join((args.extractor_test, '.pkl')))
    check_cgd_load(test_method_dir)
    
    with open(test_method_dir, 'rb') as tmp_file:
        test_dataset = pickle.load(tmp_file)

    target_test_dataset = [a for a in test_dataset if os.path.splitext(a['id'])[0] == args.target_id and a['label'] == args.target_item]
    
    # label split
    # unique_label은 테스트 기준(250개)
    unique_labels = list(set([a['label'] for a in target_test_dataset]))
    train_db, test_db = split_pooled(train_dataset, target_test_dataset, unique_labels)

    set_ks = list(map(int, args.set_ks.split(',')))

    test_single_result = calc_metrics(unique_labels, set_ks, train_db, test_db)
    
    # Save Images
    # (0) make file directory

    
    for k_ in set_ks:
        tmp_route = ''.join(('k@', str(k_)))
        tmp_dir = os.path.join(args.image_save_path, tmp_route)
        logger.info(f"Make directory to save images at: {tmp_dir}")

        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok = True)
        
    save_folder_ks = [''.join(('k@', str(k_))) for k_ in set_ks]
    
    for i, kset in enumerate(test_single_result):
        
        query_image_name = ''.join((kset[0], '.jpg'))
        db_image_name = [''.join((a[0], '.jpg')) for a in kset[-1]]
        
        query_image_path = os.path.join(args.image_from_path, query_image_name)
        db_image_path = [os.path.join(args.image_from_path, bb) for bb in db_image_name]
        
        to_query_image_path = os.path.join(args.image_save_path, save_folder_ks[i], ''.join(('Query_', query_image_name)))
        to_db_image_path = [os.path.join(args.image_save_path, save_folder_ks[i], bb) for bb in db_image_name]
        
        # copy image files
        logger.info(f"Copy image for test {save_folder_ks[i]}")

        # (1) Query image
        shutil.copyfile(query_image_path, to_query_image_path)
        
        # (2) db images
        for i, db_image in enumerate(db_image_path):
            shutil.copyfile(db_image, to_db_image_path[i])

        logger.info(f"Copy Done.")

    print("[DONE] Total time spent : {:.4f} seconds.".format(time()-t0))
