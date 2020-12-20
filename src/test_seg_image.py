import argparse
import yaml
import os
import logging
import matplotlib.pyplot as plt
import pytz

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from time import gmtime, strftime
from datetime import datetime
from time import time
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


def segment2points(anno):
    
    """
    format ((x1,y1), (x2,y2), ...) -> (x1, x2, ...) , (y1, y2, ...)
    """
    x_points=[]
    y_points=[]
    for i, v in enumerate(anno['segmentation'][0]):
        if i % 2 ==0:
            x_points.append(v)
        else:
            y_points.append(v)
    return x_points, y_points

def filter_zero_point(px,py):
    '''
    remove zero point mislabeled
    '''
    new_px=[]
    new_py=[]
    for x,y in zip(px,py):
        if x!=0 or y!=0:
            new_px.append(x)
            new_py.append(y)
    return new_px, new_py


def add_filename(json_):
    """
    Args:
        string: json path

    Returns:
        dict: annotion label
    """
    with open(json_) as f:
        imgs_anns = json.load(f)
        img_extension = json_.split('/')[-1].split('.')[0]+'.jpg'
        imgs_anns['filename'] = img_extension
    return imgs_anns    

def get_label_dict(args):
    #args.json_name = '../dataset/kfashion_dataset_new/test_sample/annos/123957.json'
    img_root = os.path.dirname(args.image_path)

    v = add_filename(args.json_path)

    record = {}
    filename = os.path.join(img_root, v["filename"])
    height, width = cv2.imread(filename).shape[:2]
    record["file_name"] = filename
    record["height"] = height
    record["width"] = width
    record["pair_id"] = v["pair_id"]
    items = sorted([item for item in v.keys() if 'item' in item])
    objs = []
    for key in items:
        anno = v[key]
        px, py  = segment2points(anno)
        px, py  = filter_zero_point(px,py)
        poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py) ]
        poly = [p for x in poly for p in x]

        # no label
        if len(poly) % 2 == 0 and len(poly) >= 6:                        

            obj = {
            "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": [poly],
            "category_id": anno['category_id'],
            "category_name": anno['category_name']
            }                        
            objs.append(obj)
    record["annotations"] = objs        
    return record

def load_model_configs(args):
    with open(args.config_path, encoding="utf-8") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    return configs

    
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
    cfg.MODEL.WEIGHTS = os.path.join(args.model_weights,f"model_{args.model_idx.zfill(7)}.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold

    predictor = DefaultPredictor(cfg)
    
    return predictor





def plot(args, fashion_metadata, im, outputs, labels, gt):
    #-----------true---------#    
    im = cv2.imread(gt["file_name"])

    plt.figure(figsize=(20,20))
    plt.subplot(1,2,1)
    plt.title('ground truths')
    v = Visualizer(im[:, :, ::-1],
                   metadata=fashion_metadata, 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    
    # ground truths
    out = v.draw_dataset_dict(gt)
    cv2_imshow(out.get_image()[:, :, ::-1])
    plt.axis('off')
   
    
    #-----------pred----------#
    plt.subplot(1,2,2)
    plt.title('pred truths')


    v = Visualizer(im[:, :, ::-1],
                   metadata=fashion_metadata, 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2_imshow(out.get_image()[:, :, ::-1])
    plt.axis('off')

    plt.savefig(os.path.join(args.save_path,'images', f"{os.path.basename(args.image_path)}"),bbox_inches='tight')
    logger.info("Saved image in {}".format(os.path.join(args.save_path, 'images', f"{os.path.basename(args.image_path)}")))
    plt.close()
 
    
    



if __name__ == "__main__":
    logging.info(f'Runing {os.path.basename(__file__)}')
    try:
        print("[1/4] parser_args.")
        parser = argparse.ArgumentParser()

        parser.add_argument("--image_path", type=str, default="./dataset/kfashion_dataset_new/test_sample/image/123957.jpg", help='input image')
        parser.add_argument("--json_path", type=str, default="./dataset/kfashion_dataset_new/test_sample/annos/123957.json", help='json label')

        parser.add_argument("--save_path", type=str, default="./dataset/results/TTA/segmentation/", help='save root')
        parser.add_argument("--model_weights", type=str, default="./model/kfashion_cascade_mask_rcnn", help='model checkpoints')
        parser.add_argument("--model_path", type=str, default="Misc/cascade_mask_rcnn_R_101_FPN_3x.yaml", help='--pretrained COCO dataset for semgentation task')
        parser.add_argument("--config_path", type=str, default="./src/configs.yaml", help='-- convenient configs for models')
        args = parser.parse_args()
    except Exception as ex:
        print("ERROR CODE :", 101)
        print(ex)
        exit()

    # model configs
    try:
        print("[2/4] load_model_configs.")
        configs = load_model_configs(args)
    except Exception as ex:
        print("ERROR CODE :", 102)
        print(ex)
        exit()



    try:
        print("[3/4] build_categories.")
        # categoies info
        fashion_metadata = build_categories(configs)
        # model index
        args.model_idx = get_checkpoint(args)
        # save path
        os.makedirs(os.path.join(args.save_path,'images'), exist_ok =True)
    except Exception as ex:
        print("ERROR CODE :", 104)
        print(ex)
        exit()

    try:
        print("[4/4] prediction.")
        # model for semgmentation
        predictor = get_predictor(args, configs)
        logger.info(f"Extracting for {args.image_path}")
        # get image
        im = get_image(args)
        # prediction
        outputs = predictor(im)
        labels = get_labels(configs, outputs)
        logger.info(f"Extracted {len(labels)} items")
        # save the segmented image
        gt = get_label_dict(args)
        plot(args, fashion_metadata, im, outputs, labels, gt)
    except Exception as ex:
        print("ERROR CODE :", 105)
        print(ex)
        exit()
