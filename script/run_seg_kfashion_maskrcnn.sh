
GPU_NUM=1  
DATA_NAME="kfashion"
MODEL_NAME="mask_rcnn"
INPUT_PATH="./dataset/kfashion_dataset_new"
OUTPUT_PATH="./model"
MODEL_PATH="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
CONFIG_PATH="./src/configs.yaml"

    

# # mask r-cnn
# CUDA_VISIBLE_DEVICES=${GPU_NUM} python src/segment.py \
#     --data_name ${DATA_NAME} \
#     --model_name ${MODEL_NAME} \
#     --input_path ${INPUT_PATH} \
#     --output_path ${OUTPUT_PATH} \
#     --model_path ${MODEL_PATH} \
#     --config_path ${CONFIG_PATH} \

# evaluate
CUDA_VISIBLE_DEVICES=${GPU_NUM} python src/segment.py \
    --do_eval \
    --data_name ${DATA_NAME} \
    --model_name ${MODEL_NAME} \
    --input_path ${INPUT_PATH} \
    --output_path ${OUTPUT_PATH} \
    --model_path ${MODEL_PATH} \
    --config_path ${CONFIG_PATH}