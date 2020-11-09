# Environment

- Version
    - `torch 1.6.0`, `cu101` 


# Build Detectron2 Using Docker

- refered from [Facebook github](https://github.com/facebookresearch/detectron2)
    
```shell
# build image
docker build --build-arg USER_ID=$UID -t fashion ./docker

```


```shell
# build container from the image
docker run --gpus all -it -p 8282:8282 -p 6006:6006 --shm-size=8gb --env="DISPLAY" -v ~/Documents/dataset/fashion/:/home/appuser/fashion_repo/dataset --name=fashion fashion:latest 
```

- `~/Documents/dataset/fashion/`: your data path in host
- `/home/appuser/fashion_repo/dataset`: your data path in docker

```shell
# Access to container
docker exec -it fashion bash
```



# Segmentation Task

- Train/Test Cascade mask rcnn on *DeepfashionV2*

```shell
sh script/run_seg_deepfahsion_cascade.sh 
```

- Train/Test Cascade mask rcnn on *Kfashion*

```shell
sh script/run_seg_deepfahsion_cascade.sh 
```

# Recommendation Task


- Train/Test combined global descriptors on *Kfashion*

```shell
sh script/run_cgd_extraction.sh 
```



---

# Extract Segmented Image for DB


- segmented images for *Kfashion*, extracted by *Cascade mask rcnn*

```shell
sh script/run_seg_image.sh 
```



