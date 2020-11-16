import argparse
import yaml
import os
import logging
import matplotlib.pyplot as plt

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

from utils import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def cv2_imshow(a, **kwargs):
    a = a.clip(0, 255).astype('uint8')
    # cv2 stores colors as BGR; convert to RGB
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)

    return plt.imshow(a, **kwargs)
    
def get_checkpoint(args):    
    # best model search
    experiment_folder = os.path.join(args.model_weights)
    model_idx = get_best_checkpoint(experiment_folder)
    return model_idx

def load_model_configs(args):
    with open(args.config_path) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    return configs

def build_categories(configs):
    MetadataCatalog.get('inference').set(thing_classes = configs['Detectron2']['LABEL_LIST']['kfashion'])
    fashion_metadata = MetadataCatalog.get('inference')
    return fashion_metadata

def get_labels(configs, outputs):
    return np.array(configs['Detectron2']['LABEL_LIST']['kfashion'])[
        outputs['instances'].pred_classes.detach().cpu().numpy()].tolist()

def get_image(args):
    return cv2.imread(args.image_path)

def get_predictor(args, configs):
    # model build and load
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.model_path))
    cfg.DATASETS.TRAIN = ()
    cfg.DATASETS.TEST = () 
    cfg.DATALOADER.NUM_WORKERS = configs['Detectron2']['DATALOADER_NUM_WORKERS'] # cpu
    cfg.SOLVER.IMS_PER_BATCH = configs['Detectron2']['SOLVER_IMS_PER_BATCH'] # allocation t
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = configs['Detectron2']['MODEL_ROI_HEADS_BATCH_SIZE_PER_IMAGE']  # number of items in batch update
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(configs['Detectron2']['LABEL_LIST']['kfashion'])  # num classes
    cfg.MODEL.WEIGHTS = os.path.join(args.model_weights,f"model_{model_idx.zfill(7)}.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold

    predictor = DefaultPredictor(cfg)
    
    return predictor
def plot(args, fashion_metadata, im, outputs, labels):
    plt.figure(figsize=(7,7))
    v = Visualizer(im[:, :, ::-1],
                   metadata=fashion_metadata, 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    cv2_imshow(out.get_image()[:, :, ::-1])
    plt.axis('off')
    logger.info("Saved in {}".format(os.path.join(args.save_path, f"{'_'.join(labels)}_{os.path.basename(args.image_path)}")))
    
    plt.savefig(os.path.join(args.save_path, f"{'_'.join(labels)}_{os.path.basename(args.image_path)}"),bbox_inches='tight')
    plt.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="./dataset/samples/056665.jpg", help='input image')
    parser.add_argument("--save_path", type=str, default="./dataset/seg_images", help='save root')
    parser.add_argument("--model_weights", type=str, default="./model/kfashion_cascade_mask_rcnn", help='model checkpoints')
    parser.add_argument("--model_path", type=str, default="Misc/cascade_mask_rcnn_R_101_FPN_3x.yaml", 
                        help='--pretrained COCO dataset for semgentation task')
    parser.add_argument("--config_path", type=str, default="./src/configs.yaml", 
                        help='-- convenient configs for models')


    args = parser.parse_args()
    
    # model configs
    configs = load_model_configs(args)    
    # categoies info
    fashion_metadata = build_categories(configs)
    # model index
    model_idx = get_checkpoint(args)
    # save path
    os.makedirs(args.save_path, exist_ok =True)

    # model for semgmentation
    predictor= get_predictor(args, configs)
    logger.info(f"Extracting for {args.image_path}")
    im = get_image(args)
    
    if im is None:
        raise NotImplementedError('Can not load the image, check the image path!')
    
    # prediction
    outputs = predictor(im)
    labels = get_labels(configs, outputs)
    logger.info(f"Extracted {len(labels)} items")

    # save the segmented image
    plot(args, fashion_metadata, im, outputs, labels)
