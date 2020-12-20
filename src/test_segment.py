import argparse
import yaml
import os
import logging

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from utils import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


LOG_PATH = './dataset/results/TTA/segmentation/'
os.makedirs(LOG_PATH, exist_ok=True)
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", 
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler(os.path.join(LOG_PATH,"segment_log.txt")),
                        logging.StreamHandler()
                    ])


    
def evaluate(args):
    
    logger.info("Start segmentation task!")

    with open(args.config_path,  encoding="utf-8") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    
    #logger.info(configs)
    
    dataset = Dataset(args.input_path, args.data_name)

    d = args.test_folder_name
    
    DatasetCatalog.register(f"{args.data_name}_" + d, lambda d=d: dataset.get_fashion_dicts(d))
    MetadataCatalog.get(f"{args.data_name}_" + d).set(thing_classes = configs['Detectron2']['LABEL_LIST'][args.data_name])
    
    experiment_folder = os.path.join(args.output_path,f"{args.data_name}_{args.model_name}")
    model_idx = get_best_checkpoint(experiment_folder)
    
    logger.info("Build model ...")

    cfg = get_cfg()
    cfg.OUTPUT_DIR = os.path.join(args.output_path,f"{args.data_name}_{args.model_name}")
    cfg.merge_from_file(model_zoo.get_config_file(args.model_path))
    cfg.DATASETS.TRAIN = ()
    cfg.DATASETS.TEST = () 
    cfg.DATALOADER.NUM_WORKERS = configs['Detectron2']['DATALOADER_NUM_WORKERS'] # cpu
    cfg.SOLVER.IMS_PER_BATCH = configs['Detectron2']['SOLVER_IMS_PER_BATCH'] # allocation to 9000m
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = configs['Detectron2']['MODEL_ROI_HEADS_BATCH_SIZE_PER_IMAGE']  # number of items in batch update
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(configs['Detectron2']['LABEL_LIST'][args.data_name])  # num classes
    cfg.MODEL.WEIGHTS = os.path.join(args.output_path,f"{args.data_name}_{args.model_name}",f"model_{model_idx.zfill(7)}.pth")
    predictor = DefaultPredictor(cfg)
    
    logger.info("Completed model ...")
    
    #os.makedirs('./results/TTA',exist_ok=True)

    evaluator = COCOEvaluator(f"{args.data_name}_{d}", cfg, False, 
                          output_dir=LOG_PATH)
    val_loader = build_detection_test_loader(cfg, f"{args.data_name}_{d}" )
    # another equivalent way to evaluate the model is to use `trainer.test`
    output_dict = inference_on_dataset(predictor.model, val_loader, evaluator)
    
    with open(os.path.join(LOG_PATH,'eval_dict.json'), 'w') as outfile:
        json.dump(output_dict, outfile)
    
    logging.info('Attach Test label & Path Loading')
    gt_data = dataset.get_fashion_dicts(d)
    id2path = {d['image_id']: d['file_name'] for d in gt_data}
    cat2name = {i:name for i, name in enumerate(configs['Detectron2']['LABEL_LIST'][args.data_name])}
    df = evaluator._score_df['segm']
    df['path'] = df['image_id'].map(id2path)
    df['name'] = df['category_id'].map(cat2name)
    df.to_csv(os.path.join(LOG_PATH,'/prediction.csv'))
    
    logger.info("Saved the predictied results.")

if __name__ == "__main__":
    logging.info(f'Runing {os.path.basename(__file__)}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--do_eval', action='store_false', help='do evaluation' )
    parser.add_argument("--data_name", type=str, default="kfashion", help='dataset name')
    parser.add_argument("--model_name", type=str, default="cascade_mask_rcnn", help='model_name')
    parser.add_argument("--input_path", type=str, default="./dataset/kfashion_dataset_new", help='input root path')
    parser.add_argument("--test_folder_name", type=str, default="test_sample", help='test_folder_name')
    parser.add_argument("--output_path", type=str, default="./model", help='output root path')
    parser.add_argument("--model_path", type=str, default="Misc/cascade_mask_rcnn_R_101_FPN_3x.yaml", 
                        help='--pretrained COCO dataset for semgentation task')
    parser.add_argument("--config_path", type=str, default="./src/configs.yaml", 
                        help='-- convenient configs for models')
    parser.add_argument("--pretrained_path", type=str, default="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml", 
                        help='--pretrained weights for semgentation task of COCO dataset')

    args = parser.parse_args()

    logger.info("Get Argparser ...")
    
    
    if args.do_eval: 
        evaluate(args)
