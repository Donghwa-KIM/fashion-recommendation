GPU_NUM=0
SAVE_PATH="../dataset/rec_images"
IMAGE_PATH="../dataset/samples/055981.jpg"
MODEL_PATH="Misc/cascade_mask_rcnn_R_101_FPN_3x.yaml"
MODEL_WEIGHTS="../model/kfashion_cascade_mask_rcnn"
CGD_PATH="../model"
CONFIG_PATH="../src/configs.yaml"
SEG_PATH="../dataset/segDB"
ABS_SEG_PATH="/home/korea/fashion-recommendation/dataset/segDB"
EXTRACTOR_TYPE="cgd_pca"
EXTRACTOR_PATH="../dataset/feature_extraction"
TOP_K=5

#CONFIG_PATH="./src/configs.yaml"

CUDA_VISIBLE_DEVICES=${GPU_NUM} python ../src/image2seg.py \
    --save_path ${SAVE_PATH} \
    --image_path ${IMAGE_PATH} \
    --model_path ${MODEL_PATH} \
    --model_weights ${MODEL_WEIGHTS} \
    --cgd_path ${CGD_PATH} \
    --config_path ${CONFIG_PATH} \
    --seg_path ${SEG_PATH} \
    --abs_seg_path ${ABS_SEG_PATH} \
    --extractor_type ${EXTRACTOR_TYPE} \
    --extractor_path ${EXTRACTOR_PATH} \
    --top_k ${TOP_K}
