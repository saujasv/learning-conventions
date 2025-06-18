#!/bin/bash

#SBATCH --mem=64G
#SBATCH --constraint="6000Ada|L40|L40S"
#SBATCH --gres=gpu:2
#SBATCH --time 6:00:00
#SBATCH -o slurm-out/sampling-%j.out
#SBATCH -e slurm-out/sampling-%j.err

export NCCL_P2P_DISABLE=1

echo "Unzipping COCO2014..."
mkdir -p /scratch/svadugur/${SLURM_JOB_ID}/COCO2014
unzip -q /data/tir/projects/tir3/users/svadugur/learning-conventions/COCO2014/train2014.zip -d /scratch/svadugur/${SLURM_JOB_ID}/COCO2014
unzip -q /data/tir/projects/tir3/users/svadugur/learning-conventions/COCO2014/val2014.zip -d /scratch/svadugur/${SLURM_JOB_ID}/COCO2014
echo "Unzipped COCO2014."

export IMAGE_BASE_PATH=/scratch/svadugur/${SLURM_JOB_ID}/COCO2014

python -m fire training/preference_pairs.py run_sampling experiment_configs/sampling_gemma/contexts-${CTX}-next-${NEXT}.json

rm -rf /scratch/svadugur/${SLURM_JOB_ID}/
