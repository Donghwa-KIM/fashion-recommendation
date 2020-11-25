GPU_NUM=0
SAVE_PATH="/home/appuser/fashion_repo/src"
IMAGE_PATH="../dataset/samples/000237.jpg"
MODEL_WEIGHTS="../model"
CGD_PATH="../model"
MODEL_PATH="Misc/cascade_mask_rcnn_R_101_FPN_3x.yaml"
CONFIG_PATH="./configs.yaml"
SEG_PATH="../dataset/seg_DB"
ABS_SEG_PATH="/home/appuser/fashion_repo/src/dataset/segDB"
EXTRACTOR_TYPE="cgd_pca"
EXTRACTOR_PATH="../model"
TOP_K=5

#CONFIG_PATH="./src/configs.yaml"

CUDA_VISIBLE_DEVICES=${GPU_NUM} python ../src/image2seg.py \
    --save_path ${SAVE_PATH} \
    --image_path ${IMAGE_PATH} \
    --model_weights ${MODEL_WEIGHTS} \
    --cgd_path ${CGD_PATH} \
    --model_path ${MODEL_PATH} \
    --config_path ${CONFIG_PATH} \
    --seg_path ${SEG_PATH} \
    --abs_seg_path ${ABS_SEG_PATH} \
    --extractor_type ${EXTRACTOR_TYPE} \
    --extractor_path ${EXTRACTOR_PATH} \
    --top_k ${TOP_K}
