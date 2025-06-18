#!/bin/bash

#SBATCH --mem=384G
#SBATCH --constraint="6000Ada|L40|L40S"
#SBATCH --gres=gpu:4
#SBATCH -t 24:00:00
#SBATCH -c 32
#SBATCH -o slurm-out/dpo-%j.out
#SBATCH -e slurm-out/dpo-%j.err

echo $HOST 
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tangrams

PORT=$((RANDOM % 500 + 29500))

export NCCL_P2P_DISABLE=1

export WANDB_PROJECT="tangrams"

# sbatch --partition=general --export="DATASET_BASE_PATH=results/preference_pairs_gemma,RUN_NAME=correctness_and_cost-speaker=gemma_pt-listener=gemma_ft,ALL" slurm-scripts/dpo_gemma.sbatch
# sbatch --partition=general --export="DATASET_BASE_PATH=results/preference_pairs_gemma,RUN_NAME=informativity-speaker=gemma_pt-listener=gemma_ft,ALL" slurm-scripts/dpo_gemma.sbatch
# sbatch --partition=preempt --export="DATASET_BASE_PATH=results/preference_pairs_gemma,RUN_NAME=cost-speaker=gemma_pt-listener=gemma_ft,ALL" slurm-scripts/dpo_gemma.sbatch
# sbatch --partition=preempt --export="DATASET_BASE_PATH=results/preference_pairs_gemma,RUN_NAME=correctness-speaker=gemma_pt-listener=gemma_ft,ALL" slurm-scripts/dpo_gemma.sbatch
# sbatch --partition=preempt --export="DATASET_BASE_PATH=results/preference_pairs_gemma,RUN_NAME=hard_correctness-speaker=gemma_pt-listener=gemma_ft,ALL" slurm-scripts/dpo_gemma.sbatch

echo "Unzipping COCO2014..."
mkdir -p /scratch/svadugur/${SLURM_JOB_ID}/COCO2014; unzip -q /data/tir/projects/tir3/users/svadugur/learning-conventions/COCO2014/train2014.zip -d /scratch/svadugur/${SLURM_JOB_ID}/COCO2014; unzip -q /data/tir/projects/tir3/users/svadugur/learning-conventions/COCO2014/val2014.zip -d /scratch/svadugur/${SLURM_JOB_ID}/COCO2014
echo "Unzipped COCO2014."

export IMAGE_BASE_PATH=/scratch/svadugur/${SLURM_JOB_ID}/COCO2014

export HUB_NAME=gemma-$(echo $RUN_NAME | cut -d- -f1)

accelerate launch --main_process_port $PORT --config_file experiment_configs/deepspeed_config_gemma_4.yaml run_dpo_trainer.py \
    --dataset_name ${DATASET_BASE_PATH}/$RUN_NAME \
    --output_dir /data/tir/projects/tir3/users/svadugur/learning-conventions/models/${RUN_NAME}-${SLURM_JOB_ID} \
    --run_name ${RUN_NAME}-${SLURM_JOB_ID} \
    --dataloader_num_workers 32 \
    --dataloader_pin_memory \
    --max_length 16384 \
    --loss_type apo_zero \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --use_peft \
    --lora_target_modules all-linear \
    --lora_r 32 \
    --lora_alpha 32 \
    --gradient_checkpointing \
    --model_name_or_path google/gemma-3-12b-pt \
    --bf16 \
    --learning_rate 5e-6 \
    --gradient_accumulation_steps 8 \
    --torch_dtype bfloat16 \
    --eval_on_start \
    --eval_strategy steps \
    --eval_steps 0.02 \
    --save_strategy steps \
    --save_steps 0.02 \
    --logging_strategy steps \
    --logging_steps 1 \
    --num_train_epochs 10 \
    --load_best_model_at_end \
    --metric_for_best_model 'rewards/margins' \
    --chat_template_file agents/gemma-3-pt-chat-template.jinja \
    --model_type base \
    --context_presentation once \
    --feedback_label true \
    --demonstration_game data/coco-4-image-contexts/demo.json

rm -rf /scratch/svadugur/${SLURM_JOB_ID}/
