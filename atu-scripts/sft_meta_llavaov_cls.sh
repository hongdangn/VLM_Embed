NUM_GPUS_PER_NODE=1
TRAIN_SCRIPT="train_one_model_ddp.py"

torchrun --nproc_per_node=$NUM_GPUS_PER_NODE --master_port 30000 $TRAIN_SCRIPT \
    --model_name "llava-hf/llava-onevision-qwen2-0.5b-ov-hf" \
    --gpu_id 0 \
    --lora True \
    --lora_r 64 \
    --pooling "eos" \
    --dataset_name "TIGER-Lab/MMEB-train" \
    --subset_name "ImageNet_1K" "N24News" "HatefulMemes" "VOC2007" "SUN397" \
    --dataset_split "original" \
    --model_backbone "llava_onevision" \
    --image_dir "./vlm2vec_train/MMEB-train/" \
    --output_dir "training/sft_meta_llavaov_cls" \
    --per_device_train_batch_size 2 \
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
    --kd_loss_type "contrastive" \
    --image_resolution low \
    --projector_config_path "./config/projector_config.json" \
    --projector_lr 5e-4
