#!/bin/bash
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 08:00:00
#SBATCH -J cramming_eval_test2h
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
mkdir -p "$HOME/cramming/logs"

module load lang/Miniconda3/23.9.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cram

cd "$HOME/cramming"
nvidia-smi -L

python eval.py eval=GLUE_sane name=test2h \
  eval.checkpoint=latest \
  impl.shuffle_in_dataloader=True \
  impl.compile_torch=False \
  impl.microbatch_size=8
