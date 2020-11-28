import argparse
import yaml
import os
import logging
import torch
import pickle
from tqdm import tqdm
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import FashionTrainer
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data import (build_detection_train_loader,
                             build_detection_test_loader,
                             build_detection_val_loader)
import detectron2.data.transforms as T

from torch import nn
from torch.nn import functional as F
from utils import *
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from cgd.trainer import Trainer
from cgd.losses import *
from cgd.model import CGD,GeM,L2Norm
from cgd.pooler import ROIpool
from sklearn.decomposition import PCA



logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)



def train(args):
    set_seed(args.seed)
    os.makedirs(args.output_path, exist_ok=True)

    with open(args.config_path) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    
    logger.info(configs)

    dataset = Dataset(args.input_path, args.data_name)

    for d in ["train", "val"]:
        DatasetCatalog.register(f"{args.data_name}_" + d, lambda d=d: dataset.get_fashion_dicts(d))
        MetadataCatalog.get(f"{args.data_name}_" + d).set(thing_classes = configs['Detectron2']['LABEL_LIST'][args.data_name])
        
    experiment_folder = os.path.join(args.output_path,f"{args.data_name}_{args.model_name}")
    model_idx = get_best_checkpoint(experiment_folder)
    
    cfg = get_cfg()
    cfg.OUTPUT_DIR = os.path.join(args.output_path,f"{args.data_name}_{args.model_name}")
    cfg.merge_from_file(model_zoo.get_config_file(args.model_path))
    cfg.DATASETS.TRAIN = (f'{args.data_name}_train',)
    cfg.DATASETS.TEST = (f'{args.data_name}_val',)
    cfg.DATALOADER.NUM_WORKERS = configs['Detectron2']['DATALOADER_NUM_WORKERS'] # cpu
    cfg.SOLVER.IMS_PER_BATCH = configs['cgd']['SOLVER_IMS_PER_BATCH'] 
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = configs['Detectron2']['MODEL_ROI_HEADS_BATCH_SIZE_PER_IMAGE']  # number of items in batch update
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(configs['Detectron2']['LABEL_LIST'][args.data_name])  # num classes
    cfg.MODEL.WEIGHTS = os.path.join(experiment_folder,f"model_{model_idx.zfill(7)}.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
 
    predictor = DefaultPredictor(cfg)
    
    train_loader = build_detection_train_loader(cfg)
    val_loader = build_detection_val_loader(cfg ,
                               cfg.DATASETS.TEST[0],
                               DatasetMapper(cfg,True)
                               )
    
    device=predictor.model.device
    
    selector = BatchHardTripletSelector().to(device)
    triple_loss = TripletLoss(margin=configs['cgd']['TRIPLE_MARGIN']).to(device)
    ce_loss = LabelSmoothingCrossEntropy(smoothing=configs['cgd']['SMOOTHING'], temperature_scale= configs['cgd']['TEMP_SCALE']).to(device)
    
    ###
    
    #cgd= torch.load("./model/middle_cgd_model.pt")
    cgd = CGD(configs['cgd']['GD_CONFIG'], configs['cgd']['FEATURE_DIM'], configs['cgd']['BASE_DIM'], configs['cgd']['NUM_CLASS']).to(device)
    
    
    optimizer = Adam(cgd.parameters(), lr= configs['cgd']['LR'])
    lr_scheduler = MultiStepLR(optimizer, milestones=[int(0.6 * configs['cgd']['MAX_ITERS']), int(0.8 * configs['cgd']['MAX_ITERS'])], gamma=0.1)

    detectron = predictor.model
    assert not detectron.training, "Current detectron is training mode"
    
    roi_pooler = ROIpool(detectron)
    
    trainer= Trainer(detectron, roi_pooler, selector, ce_loss, triple_loss, optimizer=optimizer, 
                     scheduler= lr_scheduler,
                     save_path = os.path.join(args.output_path, 'cgd_model.pt'))
    

    
    
    trainer.train(train_loader, val_loader, cgd,  configs['cgd']['MAX_ITERS'], eval_period= configs['cgd']['EVAL_PERIOD'])    

    
    
    
def evaluate(args):
    set_seed(args.seed)
    os.makedirs(args.output_path, exist_ok=True)

    with open(args.config_path) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    
    logger.info(configs)

    dataset = Dataset(args.input_path, args.data_name)
    
    for d in ["train", "val", 'test']:
        DatasetCatalog.register(f"{args.data_name}_" + d, lambda d=d: dataset.get_fashion_dicts(d))
        MetadataCatalog.get(f"{args.data_name}_" + d).set(thing_classes = configs['Detectron2']['LABEL_LIST'][args.data_name])
        
    experiment_folder = os.path.join(args.output_path,f"{args.data_name}_{args.model_name}")
    model_idx = get_best_checkpoint(experiment_folder)
    
    cfg = get_cfg()
    cfg.OUTPUT_DIR = os.path.join(args.output_path,f"{args.data_name}_{args.model_name}")
    cfg.merge_from_file(model_zoo.get_config_file(args.model_path))
    cfg.DATASETS.TRAIN = ()
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = configs['Detectron2']['DATALOADER_NUM_WORKERS'] # cpu
    cfg.SOLVER.IMS_PER_BATCH = configs['cgd']['SOLVER_IMS_PER_BATCH'] 
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = configs['Detectron2']['MODEL_ROI_HEADS_BATCH_SIZE_PER_IMAGE']  # number of items in batch update
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(configs['Detectron2']['LABEL_LIST'][args.data_name])  # num classes
    cfg.MODEL.WEIGHTS = os.path.join(experiment_folder,f"model_{model_idx.zfill(7)}.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold

    predictor = DefaultPredictor(cfg)
    
    train_dataset_dicts = dataset.get_fashion_dicts('train')
    val_dataset_dicts = dataset.get_fashion_dicts('val')
    test_dataset_dicts = dataset.get_fashion_dicts('test')
    
    # cgd
    model = torch.load( os.path.join(args.output_path, 'cgd_model.pt'))
    
    # detectron
    detectron = predictor.model
    # test resize func
    resizefunc = T.ResizeShortestEdge(
            [predictor.cfg.INPUT.MIN_SIZE_TEST, predictor.cfg.INPUT.MIN_SIZE_TEST], predictor.cfg.INPUT.MAX_SIZE_TEST
        )
    
    assert not detectron.training, "Current detectron is training mode"

    roi_pooler = ROIpool(detectron)
    
    total_dict = []
    total_dict = get_features(train_dataset_dicts, roi_pooler, model, configs, 'train', total_dict, resizefunc)
    total_dict = get_features(val_dataset_dicts, roi_pooler, model, configs, 'val', total_dict, resizefunc)
    total_dict = get_features(test_dataset_dicts, roi_pooler, model, configs, 'test', total_dict, resizefunc)
    
    os.makedirs('./dataset/feature_extraction', exist_ok=True)

    with open(f'./dataset/feature_extraction/cgd.pkl', 'wb') as f:
        pickle.dump( total_dict,f)

    pca_dict, pca = get_pca(total_dict)
    
    with open(f'./dataset/feature_extraction/cgd_pca.pkl', 'wb') as f:
        pickle.dump(pca_dict,f )
        
    with open(f'./model/pca_model.pkl', 'wb') as f:
        pickle.dump(pca,f )        
        
        
def get_pca(total_dict):
    X = np.array([dict_['feature'] for dict_ in total_dict])
    pca = PCA(whiten=True, n_components=256)
    whitened = pca.fit_transform(X)

    pca_dict = []
    for v, dict_ in zip(whitened, total_dict):
        dict_['feature'] = v
        pca_dict.append(dict_)
    return pca_dict, pca



def get_features(dataset_dicts, roi_pooler, model, configs, split, total_dict,
                 resizefunc):

    model.eval()
    for d in tqdm(dataset_dicts):
        with torch.no_grad():
            im = cv2.imread(d["file_name"])
            # test size
            im = resizefunc.get_transform(im).apply_image(im)
           
            image_batch = [{'image':torch.Tensor(im.transpose( 2,0,1))}]
            features, pred_ins, is_empty = roi_pooler.batches(image_batch)
            if is_empty:
                continue
            cgd, _ = model(features)

        N = len(features)
        classes = [configs['Detectron2']['LABEL_LIST']['kfashion'][cls_.item()] 
                   for ins in pred_ins for cls_ in ins.pred_classes]

        meta_lists = [(d['pair_id'], os.path.basename(d['file_name']))]
        total_meta_lists = meta_lists*N

        mapping_dict = {c: meta for c, meta in zip(classes , total_meta_lists)}
        
        # test
        #set_ = [i.split('.')[0] for i in os.listdir(f"./dataset/segDB/{os.path.basename(d['file_name']).split('.')[0]}")]
        #print(os.path.basename(d['file_name']),':',sorted([key for key in mapping_dict]),'=>',sorted(set_))
        #assert sorted([key for key in mapping_dict]) == sorted(set_)

        for cls_ in mapping_dict:
            meta_lists = mapping_dict[cls_]
            total_dict.extend([{'split':split,
                                 'id': meta_lists[1],
                                 'pair_id': meta_lists[0],
                                 'feature':cgd.detach().cpu().numpy()[0],
                                 'label':cls_}])
    return total_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_eval', action='store_true', help='do evaluation' )
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument("--data_name", type=str, default="kfashion", help='dataset name')
    parser.add_argument("--model_name", type=str, default="cascade_mask_rcnn", help='model_name')
    parser.add_argument("--input_path", type=str, default="../dataset/kfashion_dataset", help='input root path')
    parser.add_argument("--output_path", type=str, default="./model/", help='output root path')
    parser.add_argument("--model_path", type=str, default="Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml", 
                        help='--pretrained COCO dataset for semgentation task')
    parser.add_argument("--config_path", type=str, default="./configs.yaml", 
                        help='-- convenient configs for models')
    
    
    args = parser.parse_args()

    logger.info({ arg: vars(args)[arg] for arg in vars(args)})
    
        
    if args.do_eval: 
        evaluate(args)
    else:
        train(args)
    