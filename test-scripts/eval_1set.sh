#sft cls e1, sft vqa e7, sft ret e3, sft ground e1
#rkd vqa e0, rkd cls e0
#uld cls e0, uld vqa e1, uld ret e0
#ours cls e0, ours vqa e0, ours ret e0
set -e

# --- Configuration Variables ---
MODEL_NAME="/home/mcn/VLM_Embed/training/ours_ground/checkpoint-epoch-0"
OUTPUT_DIR="./eval-res"
DATASET_NAME="TIGER-Lab/MMEB-eval"
IMAGE_DIR="/home/mcn/VLM_Embed/eval-data"
BATCH_SIZE=5

# List of all dataset subsets to evaluate
# datasets=(ImageNet-1K HatefulMemes SUN397 N24News VOC2007 Place365 ImageNet-A ImageNet-R ObjectNet Country211) 
# datasets=(MSCOCO)
# datasets=(VOC2007)
# datasets=(CIRR)
datasets=(OK-VQA)

# --- End Configuration ---
# --- End Configuration ---

echo "Starting comprehensive model evaluation for $MODEL_NAME"
echo "Targeting ${#datasets[@]} MMEB subsets."

# Loop through each dataset in the array
for SUBSET_NAME in "${datasets[@]}"; do
    echo "=========================================================================="
    echo "   [STARTING] Evaluation for subset: $SUBSET_NAME"
    echo "=========================================================================="

    # Execute the Python evaluation script
    python eval_mmeb.py \
        --model_name "$MODEL_NAME" \
        --encode_output_path "$OUTPUT_DIR" \
        --pooling eos \
        --normalize True \
        --lora True \
        --lora_r 64 \
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