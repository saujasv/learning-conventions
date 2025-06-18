#!/bin/bash

#SBATCH --mem=16G
#SBATCH -t 30:00
#SBATCH -c 4
#SBATCH -o slurm-out/preferences-%j.out
#SBATCH -e slurm-out/preferences-%j.err

mkdir -p results/preference_pairs_gemma

export fn=`echo $samples | grep -o 'next=[^.]*' | cut -d'=' -f2`
export ctx=`echo $samples | grep -o 'contexts=[^-]*' | cut -d'=' -f2`

echo $fn
echo $ctx

python -m fire training/preference_pairs.py run_preference_pairs_copeland results/sampling_gemma/$samples ${fn}_preference results/preference_pairs_gemma/${fn}-speaker=gemma_pt-listener=gemma_ft-contexts=${ctx}-next=${fn}.jsonl