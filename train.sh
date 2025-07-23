#!/bin/bash

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=6,7 accelerate launch --config_file=deepspeed_zero3.yaml sft_vlm.py \
    --dataset_name HuggingFaceH4/llava-instruct-mix-vsft \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --output_dir test \
    --use_peft \
    --torch_dtype bfloat16 \
    --logging_strategy steps \
    --eval_strategy steps \
    --save_strategy steps \
    --logging_steps 1 \
    --eval_steps 0.0415 \
    --save_steps 0.0415 \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --report_to wandb \
    # --gradient_checkpointing \