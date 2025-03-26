#!/bin/bash

#SBATCH --mem=128G
#SBATCH --constraint="A6000|6000Ada|L40|L40S"
#SBATCH --gres=gpu:2
#SBATCH --time 18:00:00
#SBATCH -o slurm-out/tangrams-gpu-prompt-%j.out
#SBATCH -e slurm-out/tangrams-gpu-prompt-%j.err

# python -m fire evaluate_finetuned_models.py evaluate saujasv/pixtral-12b saujasv/pixtral-12b lexgram/exp1_data-icca-no_control.json square-black-imgs pixtral-eval-results/tangrams-samples=model-listener=base.json --tangrams --listener_feedback_label

# python -m fire evaluate_finetuned_models.py evaluate saujasv/pixtral-12b saujasv/pixtral-12b lexgram/exp1_data-icca-no_control.json square-black-imgs pixtral-eval-results/tangrams-samples=model-listener=ft.json --tangrams --listener_feedback_label --listener_adapter_path /data/tir/projects/tir3/users/svadugur/tangrams/models/pixtral-l-vision_only-listener_parts/checkpoint-1950

# python -m fire evaluate_finetuned_models.py evaluate_dropout saujasv/pixtral-12b saujasv/pixtral-12b lexgram/exp1_data-icca-no_control.json square-black-imgs pixtral-eval-results/tangrams-samples=dropout-listener=base.json --tangrams --listener_feedback_label

# python -m fire evaluate_finetuned_models.py evaluate_dropout saujasv/pixtral-12b saujasv/pixtral-12b lexgram/exp1_data-icca-no_control.json square-black-imgs pixtral-eval-results/tangrams-samples=dropout-listener=ft.json --tangrams --listener_feedback_label --listener_adapter_path /data/tir/projects/tir3/users/svadugur/tangrams/models/pixtral-l-vision_only-listener_parts/checkpoint-1950

# python -m fire evaluate_finetuned_models.py evaluate saujasv/pixtral-12b saujasv/pixtral-12b data/coco_games.json /data/tir/projects/tir1/corpora/MSCOCO/images/ pixtral-eval-results/coco-samples=base-listener=base.json --listener_feedback_label

# python -m fire evaluate_finetuned_models.py evaluate_dropout saujasv/pixtral-12b saujasv/pixtral-12b data/coco_games.json /data/tir/projects/tir1/corpora/MSCOCO/images/ pixtral-eval-results/coco-samples=dropout-listener=base.json--listener_feedback_label