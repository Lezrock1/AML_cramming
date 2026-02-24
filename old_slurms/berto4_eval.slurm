#!/bin/bash
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 6:20:00
#SBATCH -J cramming_eval
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

module load lang/Miniconda3/23.9.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cram

nvidia-smi -L
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cd "$HOME/cramming"

python eval.py eval=GLUE_sane name=berto4_8h_H100 eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True impl.compile_torch=False

python eval.py eval=GLUE_sane name=berto4_24h_H100 eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True impl.compile_torch=False
