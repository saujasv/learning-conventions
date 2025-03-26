#!/bin/bash

#SBATCH --mem=256G
#SBATCH --constraint="L40|L40S|6000Ada"
#SBATCH --gres=gpu:4
#SBATCH --time 48:00:00
#SBATCH -o slurm-out/pixtral-ft-%j.out
#SBATCH -e slurm-out/pixtral-ft-%j.err

echo $HOST

source /usr/share/Modules/init/bash
module load cuda-12.4

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tangrams

export NCCL_P2P_DISABLE=1

export WANDB_PROJECT="tangrams"

PORT=$((RANDOM % 101 + 29500))

echo $CONFIG

accelerate launch --main_process_port $PORT --config_file /data/tir/projects/tir3/users/svadugur/huggingface/accelerate/default_config.yaml run_trainer.py \
    --dataset_name $PROJECTS_PATH/tangrams/base_model_training_games/$TASK \
    --dataset_train_split train \
    --dataset_test_split validation \
    --output_dir $PROJECTS_PATH/tangrams/models/$SAVE_PATH \
    --agent_type $AGENT \
    --images_base_path square-black-imgs \
    --context_presentation last_shuffle \
    --feedback_label $FEEDBACK \
    --use_length_token $LENGTH \
    --learning_rate 0.0001 \
    --warmup_steps 25 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_checkpointing \
    --use_peft \
    --lora_target_modules $LORA_MODULES \
    --lora_r 128 \
    --lora_alpha 32 \
    --lora_dropout 0 \
    --max_seq_length 8192 \
    --model_name_or_path saujasv/pixtral-12b \
    --bf16 \
    --gradient_accumulation_steps 4 \
    --torch_dtype bfloat16 \
    --eval_on_start \
    --eval_strategy epoch \
    --eval_steps 1 \
    --save_strategy epoch \
    --save_steps 1 \
    --logging_strategy steps \
    --logging_steps 1 \
    --num_train_epochs 5 \
    --run_name $SAVE_PATH 