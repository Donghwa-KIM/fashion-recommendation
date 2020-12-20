# cgd extraction
INPUT_PATH="./dataset/kfashion_dataset_new"
TEST_FOLDER_NAME="test_sample"
SAVE_FOLDER="./dataset/feature_extraction/TTA"
# recommendation
EXTRACTOR_PATH='./dataset/feature_extraction/TTA'
EXTRACTOR_TRAIN='cgd'
EXTRACTOR_TEST='cgd_test_sample'
CSV_SAVE_PATH='./dataset/results/TTA/recommendation'

# (1) cgd extraction
sudo python src/test_recomm_extractor.py \
    --do_eval \
    --save_folder ${SAVE_FOLDER} \
    --test_folder_name ${TEST_FOLDER_NAME} \
    --input_path ${INPUT_PATH} 

# (2) recommendation
sudo python src/test_recomm.py \
    --extractor_path ${EXTRACTOR_PATH} \
    --extractor_train ${EXTRACTOR_TRAIN} \
    --extractor_test ${EXTRACTOR_TEST} \
    --csv_save_path ${CSV_SAVE_PATH}\
