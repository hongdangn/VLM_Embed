#!/bin/bash

# Số lượng GPU trên mỗi node (máy)
NUM_GPUS_PER_NODE=1

# Đường dẫn tới file script training của bạn
TRAIN_SCRIPT="train_one_model_no_deepspeed.py"
MODEL_NAME="nganng0510/propose_ddp_V_0911"


# =========================================================================
# Dùng torchrun để khởi chạy
# =========================================================================
torchrun --nproc_per_node=$NUM_GPUS_PER_NODE --rdzv-endpoint=localhost:29501 $TRAIN_SCRIPT \
    --model_name $MODEL_NAME \
    --lora True \
    --lora_r 64 \
    --load_pretrained_lora True \
    --pooling "eos" \
    --dataset_name "TIGER-Lab/MMEB-train" \
    --subset_name "MSCOCO"  \
    --dataset_split "original" \
    --model_backbone "llava_qwen2" \
    --image_dir "./vlm2vec_train/MMEB-train/" \
    --output_dir "training/sft_ground" \
    --per_device_train_batch_size 14 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --num_train_epochs 5 \
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
    --projector_config_path "./config/projector_config.json" \
    --projector_lr 5e-4 \
