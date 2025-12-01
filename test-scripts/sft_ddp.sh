#!/bin/bash

# Số lượng GPU trên mỗi node (máy)
NUM_GPUS_PER_NODE=1

# Đường dẫn tới file script training của bạn
TRAIN_SCRIPT="train_one_model_ddp.py"

# =========================================================================
# Dùng torchrun để khởi chạy
# =========================================================================

## classify
torchrun --nproc_per_node=$NUM_GPUS_PER_NODE --master_port 30000 $TRAIN_SCRIPT \
    --model_name "apple/FastVLM-0.5B" \
    --lora True \
    --lora_r 64 \
    --pooling "eos" \
    --dataset_name "TIGER-Lab/MMEB-train" \
    --subset_name "ImageNet_1K" "N24News" "HatefulMemes" "VOC2007" "SUN397" \
    --dataset_split "original" \
    --model_backbone "llava_qwen2" \
    --image_dir "./vlm2vec_train/MMEB-train/" \
    --output_dir "training/sft_meta_cls" \
    --per_device_train_batch_size 12 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --num_train_epochs 2 \
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
    --projector_lr 5e-4 \

# # torchrun --nproc_per_node=$NUM_GPUS_PER_NODE 
# python $TRAIN_SCRIPT \
#     --model_name "apple/FastVLM-0.5B" \
#     --lora True \
#     --lora_r 64 \
#     --pooling "eos" \
#     --dataset_name "TIGER-Lab/MMEB-train" \
#     --subset_name OK-VQA A-OKVQA DocVQA InfographicsVQA ChartQA Visual7W \
#     --dataset_split "original" \
#     --model_backbone "llava_qwen2" \
#     --image_dir "./vlm2vec_train/MMEB-train/" \
#     --output_dir "training/sft_meta_vqa" \
#     --per_device_train_batch_size 8 \
#     --gradient_accumulation_steps 2 \
#     --learning_rate 1e-5 \
#     --num_train_epochs 2 \
#     --bf16 \
#     --save_total_limit 2 \
#     --logging_steps 1 \
#     --save_strategy "epoch" \
#     --seed 42 \
#     --weight_decay 0.01 \
#     --normalize True \
#     --lr_scheduler_type "cosine" \
#     --warmup_ratio 0.03 \
#     --image_resolution low \
#     --projector_config_path "./config/projector_config.json" \
#     --projector_lr 5e-5 \

# # torchrun --nproc_per_node=$NUM_GPUS_PER_NODE 
# python $TRAIN_SCRIPT \
#     --model_name "apple/FastVLM-0.5B" \
#     --lora True \
#     --lora_r 64 \
#     --pooling "eos" \
#     --dataset_name "TIGER-Lab/MMEB-train" \
#     --subset_name VisDial	CIRR	VisualNews_t2i	VisualNews_i2t	MSCOCO_t2i	MSCOCO_i2t	NIGHTS	WebQA \
#     --dataset_split "original" \
#     --model_backbone "llava_qwen2" \
#     --image_dir "./vlm2vec_train/MMEB-train/" \
#     --output_dir "training/sft_meta_ret" \
#     --per_device_train_batch_size 8 \
#     --gradient_accumulation_steps 2 \
#     --learning_rate 1e-5 \
#     --num_train_epochs 2 \
#     --bf16 \
#     --save_total_limit 2 \
#     --logging_steps 1 \
#     --save_strategy "epoch" \
#     --seed 42 \
#     --weight_decay 0.01 \
#     --normalize True \
#     --lr_scheduler_type "cosine" \
#     --warmup_ratio 0.03 \
#     --image_resolution low \
#     --projector_config_path "./config/projector_config.json" \
#     --projector_lr 5e-5 \

# torchrun --nproc_per_node=$NUM_GPUS_PER_NODE --master_port 30000 $TRAIN_SCRIPT \
#     --model_name "apple/FastVLM-0.5B" \
#     --lora True \
#     --lora_r 64 \
#     --pooling "eos" \
#     --dataset_name "TIGER-Lab/MMEB-train" \
#     --subset_name MSCOCO \
#     --dataset_split "original" \
#     --model_backbone "llava_qwen2" \
#     --image_dir "./vlm2vec_train/MMEB-train/" \
#     --output_dir "training/sft_meta_ground_mm_projector" \
#     --per_device_train_batch_size 10 \
#     --gradient_accumulation_steps 1 \
#     --learning_rate 1e-4 \
#     --num_train_epochs 2 \
#     --bf16 \
#     --save_total_limit 2 \
#     --logging_steps 1 \
#     --save_strategy "epoch" \
#     --seed 42 \
#     --weight_decay 0.01 \
#     --normalize True \
#     --lr_scheduler_type "cosine" \
#     --warmup_ratio 0.03 \
#     --image_resolution low \
#     --projector_config_path "./config/projector_config.json" \
#     --projector_lr 5e-4 \

## 
# torchrun --nproc_per_node=$NUM_GPUS_PER_NODE $TRAIN_SCRIPT \
#     --model_name "apple/FastVLM-0.5B" \
#     --lora True \
#     --lora_r 64 \
#     --pooling "eos" \
#     --dataset_name "TIGER-Lab/MMEB-train" \
#     --subset_name "MSCOCO" \
#     --dataset_split "original" \
#     --model_backbone "llava_qwen2" \
#     --image_dir "./vlm2vec_train/MMEB-train/" \
#     --output_dir "training/no_deepspeed_sft" \
#     --per_device_train_batch_size 32 \
#     --gradient_accumulation_steps 2 \
#     --learning_rate 1e-5 \
#     --num_train_epochs 10 \
#     --bf16 \
#     --save_total_limit 2 \
#     --logging_steps 1 \
#     --save_strategy "epoch" \
#     --seed 42 \
#     --weight_decay 0.01 \
#     --normalize True \
#     --lr_scheduler_type "cosine" \
#     --warmup_ratio 0.03 \
#     --image_resolution low \
#     --projector_config_path "./config/projector_config.json" \
#     --projector_lr 5e-5 \