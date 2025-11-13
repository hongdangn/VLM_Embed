#!/bin/bash
#
# Script to run the model evaluation on multiple MMEB subsets sequentially.
#
# Usage:
# 1. Save this content as 'run_all_evals.sh'.
# 2. Make it executable: chmod +x run_all_evals.sh
# 3. Run the script: ./run_all_evals.sh

# Exit immediately if any command fails
set -e

# --- Configuration Variables ---
MODEL_NAME="/home/user2/dangnh/VLM_Embed/training/no_deepspeed_sft/checkpoint-epoch1"
OUTPUT_DIR="./eval-res"
DATASET_NAME="TIGER-Lab/MMEB-eval"
IMAGE_DIR="/home/user2/dangnh/VLMEmbed/eval-data"
BATCH_SIZE=8

# List of all dataset subsets to evaluate
# datasets=(ImageNet-1K HatefulMemes SUN397 N24News VOC2007 Place365 ImageNet-A ImageNet-R ObjectNet Country211) 
datasets=(CIRR)

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