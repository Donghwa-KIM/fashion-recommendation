#
import os, json, cv2
from tqdm import tqdm

# seed
import numpy as np
import random
import torch
import logging

#
from detectron2.structures import BoxMode

logger = logging.getLogger(__name__)


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



def cv2_imshow(a, **kwargs):
    """
    Args:
        array

    Returns:
        figure: a image
    """
    a = a.clip(0, 255).astype('uint8')
    # cv2 stores colors as BGR; convert to RGB
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)

    return plt.imshow(a, **kwargs)



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

def set_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(SEED)
    
    
    
class Dataset:
    '''
    reformat to train detectron2    
    '''
    def __init__(self, path, data_name):
        self.path = path
        self.name = data_name

            
    def get_fashion_dicts(self, d):
                
        img_root = f"{self.path}/{d}/image"
        ann_root = f"{self.path}/{d}/annos"
        
        json_files = [os.path.join(ann_root, f) for f in os.listdir(ann_root)]
        
        logger.info(f'Preprocessing for {img_root}')
        
        dataset_dicts = []
        for idx, j in enumerate(tqdm(json_files)):
            v = add_filename(j)
            
            # consider only images from shop
            if v['source'] == 'shop':

                record = {}
                filename = os.path.join(img_root, v["filename"])
                height, width = cv2.imread(filename).shape[:2]
                record["file_name"] = filename
                record["image_id"] = idx
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

                    if self.name == 'Deepfashion':
                        # label started from 1
                        anno['category_id']= anno['category_id']-1
                        
                    obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": anno['category_id'],
                    "category_name": anno['category_name']
                    }                        
                    objs.append(obj)
                record["annotations"] = objs        

                dataset_dicts.append(record)
#                 if idx % 10000==0: 
#                     print(d,idx,'//',len(json_files)) 
#                     break

        return dataset_dicts