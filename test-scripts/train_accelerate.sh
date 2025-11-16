#!/bin/bash

# Số lượng GPU trên mỗi node (máy)
NUM_GPUS_PER_NODE=1

# Đường dẫn tới file script training của bạn
TRAIN_SCRIPT="train_one_model_accelerate.py"
MODEL_NAME="nganng0510/dang_no_deepspeed_sft_cp2_1411"


# =========================================================================
# Dùng torchrun để khởi chạy
# =========================================================================
accelerate launch $TRAIN_SCRIPT \
    --model_name $MODEL_NAME \
    --lora True \
    --lora_r 64 \
    --pooling "eos" \
    --dataset_name "TIGER-Lab/MMEB-train" \
    --subset_name "VOC2007" "OK-VQA"  \
    --dataset_split "original" \
    --model_backbone "llava_qwen2" \
    --image_dir "./vlm2vec_train/MMEB-train/" \
    --output_dir "training/sft_cls_vqa" \
    --per_device_train_batch_size 12 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --num_train_epochs 8 \
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
    --projector_lr 5e-5 \
