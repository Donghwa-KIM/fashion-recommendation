SAVE_PATH="./dataset/rec_images"
IMAGE_PATH="./dataset/samples/055981.jpg"
TARGET_CATEGORY="lower"
TARGET_COLOR="블랙"
TARGET_STYLE="섹시"
TOP_K=5


CUDA_VISIBLE_DEVICES=${GPU_NUM} python ./src/image2seg.py \
    --save_path ${SAVE_PATH} \
    --image_path ${IMAGE_PATH} \
    --target_category ${TARGET_CATEGORY} \
    --target_color ${TARGET_COLOR} \
    --target_style ${TARGET_STYLE} \
    --top_k ${TOP_K}
