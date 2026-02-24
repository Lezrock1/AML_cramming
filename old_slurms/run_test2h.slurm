#!/bin/bash
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=40G
#SBATCH -t 02:05:00
#SBATCH -J cramming_test2h
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# --- setup folders ---
mkdir -p "$HOME/cramming/logs"
mkdir -p "$HOME/cramming/outputs/wandb"

# --- conda env ---
module load lang/Miniconda3/23.9.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cram

# --- W&B (offline atm) ---
export WANDB_MODE=offline
export WANDB_DIR="$HOME/cramming/outputs/wandb"

# WITH internet - with online logging, comment the 2 lines above and uncomment:
# export WANDB_API_KEY="DEIN_KEY"
# export WANDB_MODE=online

cd "$HOME/cramming"

echo "=== Host: $(hostname) ==="
nvidia-smi -L
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
python -c "import wandb; print('wandb', wandb.__version__)"

python pretrain.py \
  name=test2h \
  arch=crammed-bert \
  train=bert-o4 \
  data=pile-readymade \
  budget=2 \
  impl.threads=16 \
  impl.microbatch_size=8 \
  impl.compile_torch=False \
  impl.fullgraph=false \
  wandb.enabled=True \
  wandb.project=cramming

