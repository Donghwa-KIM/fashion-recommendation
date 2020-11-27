SAVE_PATH="./dataset/rec_images"
IMAGE_PATH="./dataset/samples/055981.jpg"
TOP_K=5


CUDA_VISIBLE_DEVICES=${GPU_NUM} python ./src/image2seg.py \
    --save_path ${SAVE_PATH} \
    --image_path ${IMAGE_PATH} \
    --top_k ${TOP_K}
