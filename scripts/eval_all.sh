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
MODEL_NAME="nganng0510/sft_dang_1108_0.0.1"
OUTPUT_DIR="./eval-res"
DATASET_NAME="TIGER-Lab/MMEB-eval"
IMAGE_DIR="/home/user2/dangnh/VLMEmbed/eval-data"
BATCH_SIZE=32

# List of all dataset subsets to evaluate
datasets=(Wiki-SS-NQ VisDial CIRR VisualNews_t2i VisualNews_i2t MSCOCO_t2i MSCOCO_i2t NIGHTS WebQA OVEN FashionIQ EDIS OK-VQA A-OKVQA DocVQA InfographicsVQA ChartQA Visual7W ScienceQA GQA TextVQA VizWiz ImageNet-1K HatefulMemes SUN397 N24News VOC2007 Place365 ImageNet-A ImageNet-R ObjectNet Country211 MSCOCO RefCOCO RefCOCO-Matching Visual7W-Pointing)

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
        --lora \
        --lora_r 8 \
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