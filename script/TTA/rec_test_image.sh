# TRAGET_ID='251648' # 078000
# TARGET_ITEM='coat'
TRAGET_ID='107393' #
TARGET_ITEM='skirt'

IMAGE_FROM_PATH='./dataset/kfashion_dataset_new/train_tot_images'
IMAGE_SAVE_PATH='./dataset/results/TTA/recommendation'

# evaluate
sudo python src/test_recomm_image.py \
    --target_id ${TRAGET_ID} \
    --target_item ${TARGET_ITEM} \
    --image_from_path ${IMAGE_FROM_PATH} \
    --image_save_path ${IMAGE_SAVE_PATH} 

