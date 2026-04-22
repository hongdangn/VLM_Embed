MODEL_NAME=/mnt/disk1/backup_user/dang.nh4/VLM_Embed/training/gvendi_phase2/checkpoint-epoch-0
OUTPUT_DIR="./eval-res"
DATASET_NAME="TIGER-Lab/MMEB-eval"
IMAGE_DIR="/mnt/disk1/backup_user/dang.nh4/eval-data"
BATCH_SIZE=48

# datasets=(ImageNet-1K HatefulMemes SUN397 N24News VOC2007) 
datasets=(Place365 ImageNet-A ImageNet-R ObjectNet Country211)
# export CUDA_VISIBLE_DEVICES=2
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
        --gpu_id 5 \
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