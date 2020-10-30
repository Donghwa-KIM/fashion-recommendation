import argparse
import yaml
import os
import logging

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import FashionTrainer

from utils import *


logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument("--data_name", type=str, default="Deepfashion", help='dataset name')
    parser.add_argument("--model_name", type=str, default="mask_rcnn", help='model_name')
    parser.add_argument("--input_path", type=str, default="../data/Deepfashion_dataset", help='input root path')
    parser.add_argument("--output_path", type=str, default="../model/", help='output root path')
    parser.add_argument("--model_path", type=str, default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", 
                        help='--pretrained COCO dataset for semgentation task')
    parser.add_argument("--config_path", type=str, default="./configs.yaml", 
                        help='-- convenient configs for models')
    parser.add_argument("--pretrained_path", type=str, default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", 
                        help='--pretrained weights for semgentation task of COCO dataset')

    args = parser.parse_args()


    logger.info({ arg: vars(args)[arg] for arg in vars(args)})

    set_seed(args.seed)
    os.makedirs(args.output_path, exist_ok=True)

    with open(args.config_path) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    dataset = Dataset(args.input_path, args.data_name)

    for d in ["train", "val"]:
        DatasetCatalog.register(f"{args.data_name}_" + d, lambda d=d: dataset.get_fashion_dicts(d))
        MetadataCatalog.get(f"{args.data_name}_" + d).set(thing_classes = configs['Detectron2']['LABEL_LIST'][args.data_name])
    
    cfg = get_cfg()
    cfg.OUTPUT_DIR = f"{args.output_path}/{args.data_name}_{args.model_name}"
    cfg.merge_from_file(model_zoo.get_config_file(args.model_path))
    cfg.DATASETS.TRAIN = (f"{args.data_name}_train",)
    cfg.DATASETS.TEST = (f"{args.data_name}_val",) # we modfiy in hooks.py, defaults.py, so it works
    cfg.TEST.EVAL_PERIOD = configs['Detectron2']['EVAL_PERIOD'] # compute validation loss
    cfg.DATALOADER.NUM_WORKERS = configs['Detectron2']['DATALOADER_NUM_WORKERS'] # cpu
    cfg.SOLVER.IMS_PER_BATCH = configs['Detectron2']['SOLVER_IMS_PER_BATCH'] # allocation to 9000m
    cfg.SOLVER.BASE_LR = configs['Detectron2']['SOLVER_BASE_LR'] # 0.00025  
    cfg.SOLVER.CHECKPOINT_PERIOD = configs['Detectron2']['SOLVER_CHECKPOINT_PERIOD'] # saved model
    cfg.SOLVER.MAX_ITER = configs['Detectron2']['SOLVER_MAX_ITER']   #20000  
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = configs['Detectron2']['MODEL_ROI_HEADS_BATCH_SIZE_PER_IMAGE']  # number of items in batch update
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.pretrained_path) 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(configs['Detectron2']['LABEL_LIST'][args.data_name])  # num classes

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = FashionTrainer(cfg) 
    trainer.resume_or_load(resume=False) # init
    
    trainer.train()
    
if __name__ == "__main__":
    train()