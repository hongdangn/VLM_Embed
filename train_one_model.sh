#!/bin/bash

# Số lượng GPU trên mỗi node (máy)
NUM_GPUS_PER_NODE=1

# Đường dẫn tới file script training của bạn
TRAIN_SCRIPT="train_one_model_no_deepspeed.py"

# =========================================================================
# Dùng torchrun để khởi chạy
# =========================================================================
# torchrun --nproc_per_node=$NUM_GPUS_PER_NODE $TRAIN_SCRIPT \
    # --subset_name "ImageNet_1K" "N24News" "HatefulMemes" "VOC2007" "SUN397" "OK-VQA" "A-OKVQA" "DocVQA" "InfographicsVQA" "ChartQA" "Visual7W" "VisDial" "CIRR" "VisualNews_t2i" "VisualNews_i2t" "MSCOCO_i2t" "MSCOCO_t2i" "NIGHTS" "WebQA" "MSCOCO" \


python $TRAIN_SCRIPT \
    --model_name "apple/FastVLM-0.5B" \
    --lora True \
    --lora_r 64 \
    --pooling "eos" \
    --dataset_name "TIGER-Lab/MMEB-train" \
    --subset_name "HatefulMemes" \
    --dataset_split "original" \
    --model_backbone "llava_qwen2" \
    --image_dir "./vlm2vec_train/MMEB-train/" \
    --output_dir "training/no_deepspeed_sft" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
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
    --image_resolution low \
    --projector_config_path "./config/projector_config.json" \
    --projector_lr 5e-5 \