NUM_GPUS_PER_NODE=1
TRAIN_SCRIPT="train_distill_ddp.py"

torchrun --nproc_per_node=$NUM_GPUS_PER_NODE --master_port 30000 $TRAIN_SCRIPT \
    --model_name "llava-hf/llava-onevision-qwen2-0.5b-ov-hf" \
    --teacher_model_name "raghavlite/B3_Qwen2_2B" \
    --lora True \
    --lora_r 64 \
    --gpu_id 3 \
    --teacher_lora True \
    --teacher_lora_r 8 \
    --teacher_pooling "eos" \
    --teacher_backbone "qwen2_vl" \
    --pooling "eos" \
    --dataset_name "TIGER-Lab/MMEB-train" \
    --subset_name "OK-VQA" "A-OKVQA" "DocVQA" "InfographicsVQA" "ChartQA" "Visual7W" \
    --dataset_split "original" \
    --model_backbone "llava_onevision" \
    --image_dir "./vlm2vec_train/MMEB-train/" \
    --output_dir "training/emo_meta_llavaov_vqa" \
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
    --image_resolution low \
    --kd_loss_type "vqa" \
    --projector_config_path "./config/projector_config.json" \
    --projector_lr 5e-4