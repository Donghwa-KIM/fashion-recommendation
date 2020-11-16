GPU_NUM=0  
SEG_PATH="../dataset/seg_DB"
MODEL_PATH="../dataset/feature_extraction"
IMAGE_PATH="../dataset/tot_dataset"
EXTRACTOR_TYPE="cgd_pca"
TARGET_LABEL="knitwear"
WANTED_TYPE="lower"
TOP_K=10

#CONFIG_PATH="./src/configs.yaml"

CUDA_VISIBLE_DEVICES=${GPU_NUM} python ../src/recommend_seg.py \
    --seg_path ${SEG_PATH} \
    --model_path ${MODEL_PATH} \
    --img_path ${IMAGE_PATH} \
    --extractor_type ${EXTRACTOR_TYPE} \
    --target_label ${TARGET_LABEL} \
    --wanted_type ${WANTED_TYPE} \
    --top_k ${TOP_K}
