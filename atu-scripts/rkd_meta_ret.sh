NUM_GPUS_PER_NODE=1
TRAIN_SCRIPT="train_distill_ddp.py"

torchrun --nproc_per_node=$NUM_GPUS_PER_NODE --master_port 30000 $TRAIN_SCRIPT \
    --model_name "apple/FastVLM-0.5B" \
    --teacher_model_name "raghavlite/B3_Qwen2_2B" \
    --lora True \
    --lora_r 64 \
    --gpu_id 2 \
    --teacher_lora True \
    --teacher_lora_r 8 \
    --teacher_pooling "eos" \
    --teacher_backbone "qwen2_vl" \
    --pooling "eos" \
    --dataset_name "TIGER-Lab/MMEB-train" \
    --subset_name VisDial CIRR VisualNews_t2i VisualNews_i2t MSCOCO_t2i MSCOCO_i2t NIGHTS WebQA \
    --dataset_split "original" \
    --model_backbone "llava_qwen2" \
    --image_dir "./vlm2vec_train/MMEB-train/" \
    --output_dir "training/rkd_meta_ret" \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --bf16 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --save_strategy "epoch" \
    --seed 42 \
    --weight_decay 0.01 \
    --normalize True \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --image_resolution mid \
    --kd_loss_type "rkd" \
    --projector_config_path "./config/projector_config.json" \
    --projector_lr 5e-4

MODEL_NAME="./training/rkd_meta_ret/checkpoint-epoch-0"
OUTPUT_DIR="./eval-res"
DATASET_NAME="TIGER-Lab/MMEB-eval"
IMAGE_DIR="./eval-data"
BATCH_SIZE=32

## eval ret
datasets=(VisDial CIRR VisualNews_t2i VisualNews_i2t MSCOCO_t2i MSCOCO_i2t NIGHTS WebQA FashionIQ Wiki-SS-NQ OVEN EDIS) 

echo "Starting comprehensive model evaluation for $MODEL_NAME"
echo "Targeting ${#datasets[@]} MMEB subsets."

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