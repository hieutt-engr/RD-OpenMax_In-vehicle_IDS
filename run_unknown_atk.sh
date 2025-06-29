TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
LOGFILE="./save/log/Resnet34_MPNCOV_OpenMax__${TIMESTAMP}_(tail_size_50).log"

python main_mpncovresnet.py \
--batch_size 256 \
--learning_rate 0.05 \
--model mpncovresnet34 \
--epochs 200 \
--cosine \
--warm \
--data_folder ./data/set_04/train_01/preprocessed/20_percent/TFRecord_w64_s32/2 \
--close_set_test_data ./data/set_04/test_01_known_vehicle_known_attack/preprocessed/5_percent/TFRecord_w64_s32/2 \
--open_set_test_data ./data/set_04/test_03_known_vehicle_unknown_attack/preprocessed/TFRecord_w64_s32/2 \
--n_classes 6 \
--loss p \
--test_contains_unknown \
--representation MPNCOV \
--attention Cov \
--trial 6 \
> "$LOGFILE" 2>&1

echo "✅ Log saved to $LOGFILE"