TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
LOGFILE="./save/log/ConResnet50_GAN_MPNCOV_OpenMax__${TIMESTAMP}.log"

python main_con_mpncovresnet_gan.py \
--batch_size 256 \
--learning_rate 0.05 \
--model mpncovresnet50 \
--epochs 100 \
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
--trial 1 \
> "$LOGFILE" 2>&1

echo "✅ Log saved to $LOGFILE"