#!/bin/bash

#SBATCH --partition=general
#SBATCH --mem=64G
#SBATCH --constraint="A6000|6000Ada|L40"
#SBATCH --gres=gpu:2
#SBATCH --time 48:00:00
#SBATCH -o slurm-out/tangrams-base-model-%j.out
#SBATCH -e slurm-out/tangrams-base-model-%j.err

echo $HOST

source /usr/share/Modules/init/bash
module load cuda-12.1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tangrams

python ~/requeue_failed_gpu.py

export NCCL_DISABLE_P2P=1

accelerate launch run_trainer.py \
    --dataset_name $PROJECTS_PATH/tangrams/base_model_training_games/$EXPT \
    --dataset_train_split train \
    --dataset_test_split validation \
    --output_dir $PROJECTS_PATH/tangrams/models/idefics3_base_adapter-$EXPT \
    --agent_type $AGENT \
    --images_base_path square-black-imgs \
    --context_presentation last_shuffle \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --use_peft \
    --lora_target_modules all-linear \
    --gradient_checkpointing \
    --model_name_or_path saujasv/Idefics3-8B-Llama3 \
    --bf16 \
    --gradient_accumulation_steps 8 \
    --torch_dtype bfloat16 \
    --eval_strategy steps \
    --eval_steps 10 \
    --save_strategy steps \
    --save_steps 10 \
    --logging_strategy steps \
    --logging_steps 1 \
    --num_train_epochs 5 \

