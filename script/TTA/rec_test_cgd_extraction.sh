INPUT_PATH="./dataset/kfashion_dataset_new"
TEST_FOLDER_NAME="test_sample"

    

# evaluate
python src/test_extractor.py \
    --do_eval \
    --test_folder_name ${TEST_FOLDER_NAME} \
    --input_path ${INPUT_PATH} 
