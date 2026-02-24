#!/bin/bash
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 08:30:00
#SBATCH -J LLaMAbase_8h_H100
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

module load lang/Miniconda3/23.9.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cram

nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cd "$HOME/cramming"

python pretrain_LLaMA_base.py