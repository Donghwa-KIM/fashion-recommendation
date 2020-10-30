# Environment

- Version
    - `torch 1.6.0`, `cu101` 
- Using Docker
    ```docker
    docker build --build-arg USER_ID=$UID -t detectron2 .
    ```
    docker run -t -d --name fashion -v 로컬패스:도커패스 detectron2     docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest
    
    ```
- Access to container
    ```docker
    docker exec -it fashion
    ```
    
# Segmentation Task

- Training maskrcnn for DeepfashionV2

```shell
CUDA_VISIBLE_DEVICES=1 sh script/run_seg_deepfahsion_maskrcnn.sh 
```
