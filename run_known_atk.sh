
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
LOGFILE="./save/log/EfficientNet_OpenMax__${TIMESTAMP}.log"

python main.py --batch_size 64 --learning_rate 0.05 --temp 0.7 --cosine --warm --trial 1 \
--data_folder ./data/set_04/train_01/preprocessed/TFRecord_w64_s32/2 \
--test_data_folder ./data/set_04/test_01_known_vehicle_known_attack/preprocessed/TFRecord_w64_s32/2 \
--test_contains_unknown \
--n_classes 6 \
> "$LOGFILE" 2>&1

echo "✅ Log saved to $LOGFILE"