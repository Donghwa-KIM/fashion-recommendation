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
        best_score = feats_sim.sort(descending = True)[0].tolist()[0]
        
        # sorted db
        sorted_db = [db[a] for a in feats_sim_index]
        
        recall_rate_list, hit_rate_list, rr_list = [], [], []
        for target_k in k:

            k = min(db_same_label_num, target_k)

            recom_db = sorted_db[:k]
            recom_ids = [a['id'] for a in recom_db]

            # db에 있는 동일한 pair_id 개수
            query_pair_id = query['pair_id']
            same_id_dbs = [a for a in db if a['pair_id'] == query_pair_id]
            query_id_len = len(same_id_dbs)

            if not same_id_dbs:
#                 tmp_id = query['id']
#                 tmp_label = query['label']
#                 logger.info(f'{tmp_id}, {tmp_label} has no pair id in DB, so passing calculation')
                continue

            else:
                # -----------------
                # RECALL
                # -----------------
                max_recall = query_id_len
                recall_success = len([rec for rec in recom_db if rec['pair_id'] == query_pair_id])
                recall_rate = recall_success/max_recall

                # -----------------
                # HIT: 관련 항목(pair_id) 중 하나라도 추천(k) 내에 속하면 1, 아니면 0
                # -----------------
                hit_success = len([rec for rec in recom_db if rec['pair_id'] == query_pair_id])
                hit_rate = min(1, hit_success)

                # -----------------
                # MRR: k 개중 관련 항목(pair_id)가 처음으로 등장하는 rank를 r이라고 하고, 이것의 역수 = RR, 전체 평균 = MRR
                # -----------------
                same_query_id_index = [i for i, a in enumerate(recom_db) if a['pair_id'] == query_pair_id]
                if same_query_id_index:
                    _rr = min([i for i, a in enumerate(recom_db) if a['pair_id'] == query_pair_id]) + 1
                    rr = 1/_rr
                else:
                    rr = 0
            
            recall_rate_list.append(recall_rate)
            hit_rate_list.append(hit_rate)
            rr_list.append(rr)
            

        # return
        return recom_ids, recall_rate_list, hit_rate_list, rr_list, best_score

    
def calc_metrics(unique_labels, set_k, *DB):

    train_db, test_db = DB[0], DB[1]
    
    result_pd = pd.DataFrame(columns = ['image_id', 'label', 'hit_k@1', 'hit_k@10'])
    
    real_fin_result = []
    for target_label in unique_labels:
        logger.info(f'================ Test on [{target_label}] ==================')
        tot_num_per_target_label = len(test_db[target_label])

        M_recall_rate_list, M_hit_rate_list, M_rr_list = [], [], []

        for idx in range(tot_num_per_target_label):

            query = test_db[target_label][idx]
            db    = train_db[target_label]

            _, recall_rate_list, hit_rate_list, rr_list, best_score = retrieve_from_db(query, db, k = set_k, print_rate = False)
            
            if recall_rate_list and hit_rate_list and rr_list:
                M_recall_rate_list.append(recall_rate_list)
                M_hit_rate_list.append(hit_rate_list)
                M_rr_list.append(rr_list)
                
                result_pd = pd.concat([result_pd, pd.DataFrame({'image_id': [os.path.splitext(query['id'])[0]],
                                                               'label'    : target_label,
                                                               'hit_k@1'  : [hit_rate_list[0]],
                                                               'hit_k@10' : [hit_rate_list[1]],
                                                               'best_sim': best_score})], axis = 0)

        # If none was calcuated, pass the label
        if M_hit_rate_list:
            
            fin_result = []
            for i, k in enumerate(set_k):

                RC = np.mean(np.array([a[i] for a in M_recall_rate_list]))
                HT = np.mean(np.array([b[i] for b in M_hit_rate_list]))
                RR = np.mean(np.array([c[i] for c in M_rr_list]))

                logger.info(f'Tested on: {target_label} with k: {k} || Hit: {HT:.4f}')

                fin_result.append((RC, HT, RR))

            real_fin_result.append(fin_result)

    # ---------------------------
    #     Calculate(Average)
    # ---------------------------
    RESULT = np.mean(np.array(real_fin_result), axis = 0)
    for i, k in enumerate(set_k):
        logger.info(f"For k = {k}  HIT: {RESULT[i][1]:.4f}  MRR: {RESULT[i][2]:.4f}")

    return RESULT, result_pd
    
    
if __name__ == "__main__":
    
    t0 = time()

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--extractor_path", type = str, default = './dataset/feature_extraction/TTA')
    parser.add_argument("--extractor_train", type = str, default = 'cgd')
    parser.add_argument("--extractor_test", type = str, default = 'cgd_test_sample')
    parser.add_argument("--csv_save_path", type = str, default = './dataset/results/TTA/recommendation')
    parser.add_argument("--set_ks", type = str, default = "1,10", help = "Set of k's used for testing")
    parser.add_argument("--config_path", type = str, default = './src/configs.yaml')
    
    args = parser.parse_args()
    
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

    # label split
    # unique_label은 테스트 기준(250개)
    unique_labels = list(set([a['label'] for a in test_dataset]))
    train_db, test_db = split_pooled(train_dataset, test_dataset, unique_labels)

    set_ks = list(map(int, args.set_ks.split(',')))

    result_metrics, result_pd = calc_metrics(unique_labels, set_ks, train_db, test_db)
    
    # Save result into csv file
    csv_save_dir = os.path.join(args.csv_save_path, 'recomm_result.csv')
    result_pd_save = result_pd.sort_values(by = 'best_sim', ascending = False)

    logger.info(f'Writing result file into csv file: {csv_save_dir}')
    result_pd_save.to_csv(csv_save_dir, index = False, float_format='%.4f')

    print("[DONE] Total time spent : {:.4f} seconds.".format(time()-t0))
