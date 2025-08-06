TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
LOGFILE="./save/log/Generate_OOD_ACGAN__${TIMESTAMP}.log"

python train_acgan_generate_ood.py \
--support_1_folder ./data/set_04/test_03_known_vehicle_unknown_attack/preprocessed/one_type/dos/TFRecord_w64_s32/2 \
--support_2_folder ./data/set_04/test_03_known_vehicle_unknown_attack/preprocessed/one_type/force-neutral/TFRecord_w64_s32/2 \
--output_path ./data/set_04/test_03_known_vehicle_unknown_attack/preprocessed/ood/acgan/generated_ood.tfrecord \
--epochs 100 \
--batch_size 256 \
--num_ood 1000 \
> "$LOGFILE" 2>&1

echo "✅ Log saved to $LOGFILE"
