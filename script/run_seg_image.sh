
GPU_NUM=0  
DATA_NAME="kfashion"
MODEL_NAME="cascade_mask_rcnn"
INPUT_PATH="./dataset/kfashion_dataset_new"
OUTPUT_PATH="./model"
MODEL_PATH="Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"
CONFIG_PATH="./src/configs.yaml"


# evaluate
CUDA_VISIBLE_DEVICES=${GPU_NUM} python src/seg_img.py \
    --do_eval \
    --data_name ${DATA_NAME} \
    --model_name ${MODEL_NAME} \
    --input_path ${INPUT_PATH} \
    --output_path ${OUTPUT_PATH} \
    --model_path ${MODEL_PATH} \
    --config_path ${CONFIG_PATH}