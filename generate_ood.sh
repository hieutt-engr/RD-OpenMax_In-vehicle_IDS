TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
LOGFILE="./save/log/Generate_OOD__${TIMESTAMP}.log"

python train_shrinkage_ae_generate_ood.py \
--data_folder ./data/set_04/train_01/preprocessed/six_features/TFRecord_w64_s32/2 \
--ood_save_path ./data/set_04/test_03_known_vehicle_unknown_attack/preprocessed/ood/using_atk/generated_ood.tfrecord \
--support_1_folder ./data/set_04/test_03_known_vehicle_unknown_attack/preprocessed/one_type/dos/TFRecord_w64_s32/2 \
--support_2_folder ./data/set_04/test_03_known_vehicle_unknown_attack/preprocessed/one_type/force-neutral/TFRecord_w64_s32/2 \
--num_workers 8 \
--batch_size 128 \
> "$LOGFILE" 2>&1

echo "✅ Log saved to $LOGFILE"
