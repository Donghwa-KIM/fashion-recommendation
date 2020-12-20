# good example
IMAGE_PATH="./dataset/kfashion_dataset_new/test_sample/image/098727.jpg"
JSON_PATH="./dataset/kfashion_dataset_new/test_sample/annos/098727.json"

## bad example
# IMAGE_PATH="./dataset/kfashion_dataset_new/test_sample/image/265848.jpg"
# JSON_PATH="./dataset/kfashion_dataset_new/test_sample/annos/265848.json"
# IMAGE_PATH="./dataset/kfashion_dataset_new/test_sample/image/230191.jpg"
# JSON_PATH="./dataset/kfashion_dataset_new/test_sample/annos/230191.json"

sudo python src/test_seg_image.py --image_path ${IMAGE_PATH} --json_path ${JSON_PATH}

