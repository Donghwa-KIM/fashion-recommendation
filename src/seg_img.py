import argparse
import yaml
import os
import logging
from tqdm import tqdm
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import FashionTrainer
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from utils import *


# vis 
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode



def cv2_imshow(a, **kwargs):
    a = a.clip(0, 255).astype('uint8')
    # cv2 stores colors as BGR; convert to RGB
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)

    return a


class VisFashion:
    def __init__(self, mask, img):
        self.w_s, self.w_e, self.h_s, self.h_e = self.get_forground_index(mask) 
        self.img_ = self.white_background(mask, img)
        self.out_image = self.resize_foreground()
    
    def white_background(self, mask, img):
        img[~ mask,:] = [255,255,255]
        return img
    
    def get_forground_index(self, mask):
        for w_s, v in enumerate(mask.sum(axis=0)):
            if v!=0:
                break
        for w_e, v in enumerate(reversed(mask.sum(axis=0))):
            if v!=0:
                break
        for h_s, v in enumerate(mask.sum(axis=1)):
            if v!=0:
                break
        for h_e, v in enumerate(reversed(mask.sum(axis=1))):
            if v!=0:
                break
        return w_s, w_e, h_s, h_e
        
    def resize_foreground(self):
        return self.img_[self.h_s:-self.h_e, self.w_s:-self.w_e, :]
        
        
        

        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--do_eval', action='store_true', help='do evaluation' )
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument("--data_name", type=str, default="kfashion", help='dataset name')
    parser.add_argument("--model_name", type=str, default="cascade_mask_rcnn", help='model_name')
    parser.add_argument("--input_path", type=str, default="../dataset/kfashion_dataset_new", help='input root path')
    parser.add_argument("--output_path", type=str, default="../model/", help='output root path')
    parser.add_argument("--model_path", type=str, default="Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml", 
                        help='--pretrained COCO dataset for semgentation task')
    parser.add_argument("--config_path", type=str, default="./configs.yaml", 
                        help='-- convenient configs for models')


    args = parser.parse_args()

    logger.info({ arg: vars(args)[arg] for arg in vars(args)})
    
    
    
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
    cfg.DATASETS.TRAIN = (f'{args.data_name}_train',)
    cfg.DATASETS.TEST = (f'{args.data_name}_val',)
    cfg.DATALOADER.NUM_WORKERS = configs['Detectron2']['DATALOADER_NUM_WORKERS'] # cpu
    cfg.SOLVER.IMS_PER_BATCH = configs['cgd']['SOLVER_IMS_PER_BATCH'] # allocation to 9000m
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = configs['Detectron2']['MODEL_ROI_HEADS_BATCH_SIZE_PER_IMAGE']  # number of items in batch update
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(configs['Detectron2']['LABEL_LIST'][args.data_name])  # num classes
    cfg.MODEL.WEIGHTS = os.path.join(experiment_folder,f"model_{model_idx.zfill(7)}.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)    
    
    
    train_dataset_dicts = dataset.get_fashion_dicts('train')
    val_dataset_dicts = dataset.get_fashion_dicts('val')
    test_dataset_dicts = dataset.get_fashion_dicts('test')
    
    
    for loader in [train_dataset_dicts, val_dataset_dicts, test_dataset_dicts]:
        for d in tqdm(loader):

            im = cv2.imread(d["file_name"])
            outputs = predictor(im)
            masks = outputs['instances'].pred_masks.detach().cpu().numpy()
            labels = np.array(configs['Detectron2']['LABEL_LIST']['kfashion'])[
                outputs['instances'].pred_classes.detach().cpu().numpy()].tolist()

            for mask, label in zip(masks, labels):

                im = cv2.imread(d["file_name"])
                save_path = os.path.join('dataset', 
                                         'segDB',
                                         os.path.basename(d["file_name"]).split('.')[0])

                os.makedirs(save_path, exist_ok =True)

                vis = VisFashion(mask, im)
                if vis.out_image.shape[0] ==0 or vis.out_image.shape[1]==0:
                    continue
                plt.figure(figsize=(10,10))
                a = cv2_imshow(vis.out_image)
                plt.imshow(a)
                plt.axis('off')
                plt.savefig(os.path.join(save_path, f'{label}.png'),bbox_inches='tight')
                plt.close()