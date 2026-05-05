MODEL_NAME=/mnt/disk1/backup_user/dang.nh4/VLM_Embed/meta_train/rkd_meta_vqa/checkpoint-epoch-0
OUTPUT_DIR="./eval-res"
DATASET_NAME="TIGER-Lab/MMEB-eval"
IMAGE_DIR="/mnt/disk1/backup_user/dang.nh4/eval-data"
BATCH_SIZE=36

# export CUDA_VISIBLE_DEVICES=2
datasets=("OK-VQA" "A-OKVQA" "DocVQA" "InfographicsVQA" "ChartQA" "Visual7W")
# datasets=("OK-VQA")
# datasets=(Visual7W)

echo "Starting comprehensive model evaluation for $MODEL_NAME"
echo "Targeting ${#datasets[@]} MMEB subsets."

for SUBSET_NAME in "${datasets[@]}"; do
    echo "=========================================================================="
    echo "   [STARTING] Evaluation for subset: $SUBSET_NAME"
    echo "=========================================================================="

    python eval_mmeb.py \
        --model_name "$MODEL_NAME" \
        --encode_output_path "$OUTPUT_DIR" \
        --pooling eos \
        --normalize True \
        --lora True \
        --lora_r 64 \
        --gpu_id 2 \
        --bf16 \
        --dataset_name "$DATASET_NAME" \
        --subset_name "$SUBSET_NAME" \
        --dataset_split test \
        --per_device_eval_batch_size "$BATCH_SIZE" \
        --image_dir "$IMAGE_DIR" \
        --tgt_prefix_mod

    echo "--------------------------------------------------------------------------"
    echo "   [FINISHED] Evaluation for subset: $SUBSET_NAME completed successfully."
    echo "--------------------------------------------------------------------------"
    echo ""
done

echo "--------------------------------------------------------------------------"
echo "All $MODEL_NAME evaluations on all specified subsets are complete."
echo "--------------------------------------------------------------------------"