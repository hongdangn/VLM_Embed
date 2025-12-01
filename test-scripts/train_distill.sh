#!/bin/bash

# Số lượng GPU trên mỗi node (máy)
NUM_GPUS_PER_NODE=1

# Đường dẫn tới file script training của bạn
TRAIN_SCRIPT="train_distill_ddp.py"

# =========================================================================
# Dùng torchrun để khởi chạy
# =========================================================================
torchrun --standalone \
    --nproc_per_node=$NUM_GPUS_PER_NODE $TRAIN_SCRIPT \
    --model_name "nganng0510/propose_ddp_V_0911" \
    --teacher_model_name "raghavlite/B3_Qwen2_2B" \
    --lora True \
    --load_pretrained_lora True \
    --teacher_lora True \
    --lora_r 64 \
    --teacher_lora_r 8 \
    --teacher_pooling "eos" \
    --teacher_backbone "qwen2_vl" \
    --model_backbone "llava_qwen2" \
    --pooling "eos" \
    --dataset_name "TIGER-Lab/MMEB-train" \
    --subset_name "CIRR" \
    --dataset_split "original" \
    --image_dir "vlm2vec_train/MMEB-train" \
    --output_dir "training/ours_ret" \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --num_train_epochs 2 \
    --bf16 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --save_strategy "epoch" \
    --seed 42 \
    --weight_decay 0.01 \
    --normalize True \
    --teacher_normalize True \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --kd_weight 0.3 \
    --kd_loss_type "ours" \
    --image_resolution "low" \
    --projector_config_path "./config/projector_config.json" \
    --projector_lr 5e-4 \

# torchrun --standalone \
#     --nproc_per_node=$NUM_GPUS_PER_NODE $TRAIN_SCRIPT \
#     --model_name "nganng0510/propose_ddp_V_0911" \
#     --teacher_model_name "raghavlite/B3_Qwen2_2B" \
#     --lora True \
#     --load_pretrained_lora True \
#     --teacher_lora True \
#     --lora_r 64 \
#     --teacher_lora_r 8 \
#     --teacher_pooling "eos" \
#     --teacher_backbone "qwen2_vl" \
#     --model_backbone "llava_qwen2" \
#     --pooling "eos" \
#     --dataset_name "TIGER-Lab/MMEB-train" \
#     --subset_name "CIRR" \
#     --dataset_split "original" \
#     --image_dir "vlm2vec_train/MMEB-train" \
#     --output_dir "training/rkd_ret" \
#     --per_device_train_batch_size 8 \
#     --gradient_accumulation_steps 2 \
#     --learning_rate 1e-4 \
#     --num_train_epochs 2 \
#     --bf16 \
#     --save_total_limit 2 \
#     --logging_steps 1 \
#     --save_strategy "epoch" \
#     --seed 42 \
#     --weight_decay 0.01 \
#     --normalize True \
#     --teacher_normalize True \
#     --lr_scheduler_type "cosine" \
#     --warmup_ratio 0.03 \
#     --kd_weight 0.3 \
#     --kd_loss_type "ours" \
#     --image_resolution "low" \
#     --projector_config_path "./config/projector_config.json" \
#     --projector_lr 5e-4 \

# torchrun --standalone \
#     --nproc_per_node=$NUM_GPUS_PER_NODE $TRAIN_SCRIPT \
#     --model_name "dangnguyens1/sft-fastvlm-ground" \
#     --teacher_model_name "raghavlite/B3_Qwen2_2B" \
#     --lora True \
#     --load_pretrained_lora True \
#     --teacher_lora True \
#     --lora_r 64 \
#     --teacher_lora_r 8 \
#     --teacher_pooling "eos" \
#     --teacher_backbone "qwen2_vl" \
#     --model_backbone "llava_qwen2" \
#     --pooling "eos" \
#     --dataset_name "TIGER-Lab/MMEB-train" \
#     --subset_name "MSCOCO" \
#     --dataset_split "original" \
#     --image_dir "vlm2vec_train/MMEB-train" \
#     --output_dir "training/rkd_ground" \
#     --per_device_train_batch_size 6 \
#     --gradient_accumulation_steps 2 \
#     --learning_rate 1e-4 \
#     --num_train_epochs 1 \
#     --bf16 \
#     --save_total_limit 2 \
#     --logging_steps 1 \
#     --save_strategy "epoch" \
#     --seed 42 \
#     --weight_decay 0.01 \
#     --normalize True \
#     --teacher_normalize True \
#     --lr_scheduler_type "cosine" \
#     --warmup_ratio 0.03 \
#     --kd_weight 0.3 \
#     --kd_loss_type "rkd" \
#     --image_resolution "low" \
#     --projector_config_path "./config/projector_config.json" \
#     --projector_lr 5e-4 \

