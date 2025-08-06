TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
LOGFILE="./save/log/ConResnet50_MPNCOV_NC_Maha_Set_01__${TIMESTAMP}.log"

python main_mpncovresnet_nc_feat.py \
--batch_size 256 \
--learning_rate 0.05 \
--model mpncovresnet50 \
--epochs 100 \
--cosine \
--warm \
--data_folder ./data/set_01/train_01/preprocessed/six_features/TFRecord_w64_s32/2 \
--close_set_test_data ./data/set_01/test_01_known_vehicle_known_attack/preprocessed/six_features/TFRecord_w64_s32/2 \
--open_set_test_data ./data/set_01/test_03_known_vehicle_unknown_attack/preprocessed/six_features/TFRecord_w64_s32/2 \
--n_classes 5 \
--loss p \
--test_contains_unknown \
--representation MPNCOV \
--attention Cov \
--trial F-DEF_4_Maha_Set_01 \
> "$LOGFILE" 2>&1

echo "✅ Log saved to $LOGFILE"
