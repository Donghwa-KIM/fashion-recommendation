# dataset root path
INPUT_PATH="./dataset/kfashion_dataset_new"
# test folder name
TEST_FOLDER_NAME="test_sample"

# evaluate
sudo python3 src/test_segment.py --input_path ${INPUT_PATH} --test_folder_name ${TEST_FOLDER_NAME}