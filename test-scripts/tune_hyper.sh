#!/bin/bash

# Số lượng GPU trên mỗi node (máy)
NUM_GPUS_PER_NODE=1

# Đường dẫn tới file script training của bạn
TRAIN_SCRIPT="train_distill_ddp.py"
INTRA_RKD=(0.6 0.9)
# IMG_ALIGN=(0) # (If you decide to uncomment it)
CRS_MODAL=(0.002 0.004 0.006)
OT=(0.03 0.06 0.09)
# =========================================================================
# Dùng torchrun để khởi chạy
# =========================================================================
for intra in "${INTRA_RKD[@]}"; do
    for modal in "${CRS_MODAL[@]}"; do
        for ot in "${OT[@]}"; do
            SUBSET_NAME="CIRR_intra${intra}_cm${modal}_ot${ot}"
            outdir="training/dang_propose/${SUBSET_NAME}"
            echo "=========================================================================="
            echo "   [STARTING] Evaluation for subset: $SUBSET_NAME"
            echo "=========================================================================="

            torchrun --standalone \
                --nproc_per_node=$NUM_GPUS_PER_NODE $TRAIN_SCRIPT \
                --model_name "apple/FastVLM-0.5B" \
                --teacher_model_name "raghavlite/B3_Qwen2_2B" \
                --lora True \
                --teacher_lora True \
                --lora_r 8 \
                --teacher_lora_r 8 \
                --teacher_pooling "eos" \
                --teacher_backbone "qwen2_vl" \
                --model_backbone "llava_qwen2" \
                --pooling "eos" \
                --dataset_name "TIGER-Lab/MMEB-train" \
                --subset_name "CIRR" \
                --dataset_split "original" \
                --image_dir "vlm2vec_train/MMEB-train" \
                --output_dir $outdir \
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
                --teacher_normalize True \
                --lr_scheduler_type "cosine" \
                --warmup_ratio 0.03 \
                --kd_weight 0.3 \
                --kd_loss_type "dang_propose" \
                --image_resolution "low" \
                --projector_config_path "./config/projector_config.json" \
                --projector_lr 5e-5 \
                --rkd_loss_weight 0 \
                --simple_kd_weight 0.3 \
                --intra_rkd_weight $intra \
                --img_align_loss_weight 0 \
                --cross_modal_kd_weight $modal \
                --ot_loss_weight $ot >> log.txt
        done
    done
done