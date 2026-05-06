NUM_GPUS_PER_NODE=1

TRAIN_SCRIPT="gvendi_phase1.py"
# teacher_cache_dir="./teacher_gradients/sft_fastvlm_SUN397_phase1"
teacher_cache_dir="./teacher_gradients/gvendi_fastvlm_SUN397_phase1"
GVENDI_CODEBOOK_METHOD="${GVENDI_CODEBOOK_METHOD:-sinkhorn}"

export TORCH_DISTRIBUTED_DEBUG=DETAIL

export CUDA_VISIBLE_DEVICES=0

#phase 1 training
torchrun --standalone \
    --nproc_per_node=$NUM_GPUS_PER_NODE $TRAIN_SCRIPT \
    --model_name apple/FastVLM-0.5B \
    --teacher_model_name "training/gvendi_SUN397_fastvlm/checkpoint-epoch-0" \
    --lora True \
    --teacher_lora True \
    --lora_r 64 \
    --lora_alpha 64 \
    --teacher_lora_r 8 \
    --teacher_pooling "eos" \
    --teacher_backbone "llava_qwen2" \
    --model_backbone "llava_qwen2" \
    --pooling "eos" \
    --dataset_name "TIGER-Lab/MMEB-train" \
    --subset_name "SUN397" \
    --dataset_split "original" \
    --image_dir "./vlm2vec_train/MMEB-train" \
    --percent_data 1.0 \
    --output_dir "training/sft_fastvlm_vqa_phase1" \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --bf16 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --save_strategy "epoch" \
    --seed 42 \
    --weight_decay 0.01 \
    --normalize True \
    --teacher_normalize True \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --kd_weight 2.5 \
    --w_cross_modal_loss 2.5 \
    --kd_loss_type "gvendi_phase1" \
    --image_resolution "low" \
    --projector_lr 5e-4 \
    --need_hash True \
    --teacher_cache_dir $teacher_cache_dir \
    --gvendi_codebook_method "$GVENDI_CODEBOOK_METHOD" \
    --phase_1 True \
