EXTRACTOR_PATH="./dataset/feature_extraction"
EXTRACTOR_TYPE="cgd_pca"
SET_KS="1,2,5,10,20"
CONFIG_PATH="./src/configs.yaml"


# evaluate
python src/recomm_test.py \
    --extractor_path ${EXTRACTOR_PATH} \
    --extractor_type ${EXTRACTOR_TYPE} \
    --set_ks ${SET_KS} \
    --config_path ${CONFIG_PATH}
