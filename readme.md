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

# Dataset

- Deepfashion-v2 (*landmark task*)
    - https://github.com/switchablenorms/DeepFashion2 
- Kfashion
    - https://drive.google.com/file/d/1Dz4k_MesgG2Uqno-LzrS7ebPqKG8p0XZ/view?usp=sharing

![](http://drive.google.com/uc?export=view&id=1D4YgJw9-IsN2VTDKUdeBn0FMIeBtmYJ8)

# Segmentation Task
- Cascade mask rcnn

<img src="http://drive.google.com/uc?export=view&id=1gDc3edWBcKZr2tsMhuJTu0GoVPuI9W2s" width="700">


- Train/Test Cascade mask rcnn on *DeepfashionV2*
    - completed model: [Deepfashion_cascade_mask_rcnn](http://drive.google.com/uc?export=view&id=1D4YgJw9-IsN2VTDKUdeBn0FMIeBtmYJ8)
    - its results: [Deepfashion_cascade_mask_rcnn](https://drive.google.com/file/d/1FSnYl10_I2A-dhpu75YSOJVG6lAYFXLJ/view?usp=sharing)

```shell
sh script/run_seg_deepfahsion_cascade.sh 
```

- Train/Test Cascade mask rcnn on *Kfashion*
    - completed model: [kfashion_cascade_mask_rcnn](https://drive.google.com/file/d/1h_BIcdZUl98zghhcmtRntdGHISp_GH5g/view?usp=sharing)
    - its results: [kfashion_cascade_mask_rcnn](https://drive.google.com/file/d/1kqwq6PFUT3cvV7wUfqNhVbwLK-w9y-NW/view?usp=sharing)

```shell
sh script/run_seg_deepfahsion_cascade.sh 
```

- Output

<img src="http://drive.google.com/uc?export=view&id=1BhHko1U4xcTCg-yLQmR-mrm2-4QFDJzQ" width="700">

# Performance

- Distribution of instances among all 21 categories:

|   category    | #instances | category | #instances | category | #instances |
| :-----------: | :--------- | :------: | :--------- | :------: | :--------- |
|   cardigan    | 133        | knitwear | 662        |  dress   | 2042       |
|   leggings    | 43         |   vest   | 99         |  bratop  | 14         |
|    blouse     | 1281       |  shirts  | 344        |  skirt   | 957        |
|    jacket     | 267        |  jumper  | 149        | jumpsuit | 81         |
| jogger pants  | 68         |  zipup   | 28         |   jean   | 698        |
|     coat      | 234        |   top    | 453        | t-shirts | 1514       |
| padded jacket | 66         |  pants   | 1592       |  hoody   | 156        |
|               |            |          |            |          |            |
|     total     | 10881      |          |            |          |            |

- Evaluation results for bbox:

|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
| :----: | :----: | :----: | :----: | :----: | :----: |
| 48.194 | 60.508 | 53.806 | 45.000 | 57.415 | 48.119 |

- Per-category bbox AP:

| category      | AP     | category | AP     | category | AP     |
| :------------ | :----- | :------- | :----- | :------- | :----- |
| cardigan      | 37.397 | knitwear | 66.892 | dress    | 83.845 |
| leggings      | 19.579 | vest     | 48.953 | bratop   | 0.000  |
| blouse        | 68.244 | shirts   | 43.211 | skirt    | 72.168 |
| jacket        | 43.539 | jumper   | 45.737 | jumpsuit | 41.825 |
| jogger pants  | 20.207 | zipup    | 5.803  | jean     | 71.190 |
| coat          | 67.621 | top      | 29.423 | t-shirts | 65.624 |
| padded jacket | 73.085 | pants    | 63.442 | hoody    | 44.280 |



- Evaluation results for segm:

|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
| :----: | :----: | :----: | :----: | :----: | :----: |
| 42.243 | 59.154 | 48.035 | 30.000 | 39.219 | 42.198 |

- Per-category segm AP:

| category      | AP     | category | AP     | category | AP     |
| :------------ | :----- | :------- | :----- | :------- | :----- |
| cardigan      | 25.927 | knitwear | 63.572 | dress    | 78.047 |
| leggings      | 9.894  | vest     | 36.384 | bratop   | 0.000  |
| blouse        | 62.206 | shirts   | 40.120 | skirt    | 71.696 |
| jacket        | 33.228 | jumper   | 36.688 | jumpsuit | 34.431 |
| jogger pants  | 17.901 | zipup    | 4.554  | jean     | 61.952 |
| coat          | 61.531 | top      | 26.688 | t-shirts | 59.336 |
| padded jacket | 68.655 | pants    | 54.951 | hoody    | 39.343 |

# Recommendation Task

<img src="http://drive.google.com/uc?export=view&id=1AlMMbejlJKZM0L1ynpIUo_zL9DfeV6mp" width="700">


- Train/Test combined global descriptors on *Kfashion*
    - completed model: [cgd_model.pt](https://drive.google.com/file/d/1h_BIcdZUl98zghhcmtRntdGHISp_GH5g/view?usp=sharing)
    - its results: [feature_extraction](https://drive.google.com/file/d/1OOjOxvDOVQa8mAQnLN0D6wRkcf-yU2Fi/view?usp=sharing)

```shell
sh script/run_cgd_extraction.sh 
```



- Get items by using cosine similarities, and evaluation  



---

# Extract Segmented Image for DB


- [segmented images](https://drive.google.com/file/d/1IKSz-7P3ToUg76kS6QDwr-vmdxkJ6Ukw/view?usp=sharing) for *Kfashion*, extracted by *Cascade mask rcnn*

```shell
sh script/run_seg_image.sh 
```

- Output

<img src="http://drive.google.com/uc?export=view&id=1fFy6E5G9zFRj48RtF5CjZQ4-3iW567v4" width="700">


