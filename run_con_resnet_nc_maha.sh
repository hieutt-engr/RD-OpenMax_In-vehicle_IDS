TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
LOGFILE="./save/log/ConResnet50_MPNCOV_NC_Maha_OE_3_Feat_Basic_Freeze__${TIMESTAMP}.log"

python main_resnet_nc_maha_oe_basic_loss.py \
--batch_size 256 \
--learning_rate 0.05 \
--model mpncovresnet50 \
--epochs 150 \
--cosine \
--warm \
--data_folder ./data/set_04/train_01/preprocessed/six_features/TFRecord_w64_s32/2 \
--close_set_test_data ./data/set_04/test_01_known_vehicle_known_attack/preprocessed/six_features/TFRecord_w64_s32/2 \
--open_set_test_data ./data/set_04/test_03_known_vehicle_unknown_attack/preprocessed/six_features/TFRecord_w64_s32/2 \
--oe_data_root ./data/set_04/test_03_known_vehicle_unknown_attack/preprocessed/oe_record/oe.tfrecord \
--n_classes 6 \
--loss p \
--test_contains_unknown \
--representation MPNCOV \
--attention Cov \
--proto_shift_thresh 0.05 \
--trial 3_Feature_Maha_Freeze_Basic_loss \
> "$LOGFILE" 2>&1

echo "✅ Log saved to $LOGFILE"
