#!/bin/bash
#SBATCH -J pretrain_sonata
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=512G
#SBATCH --time=5-00:00:00
#SBATCH -o slurm/%x-%j.out
#SBATCH -e slurm/%x-%j.err

set -euo pipefail

mkdir -p slurm

# Load conda in non-interactive shell.
if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi
conda activate pointcept-torch2.5.0-cu12.4

cd /public/home/xinwuye/UniDock-Pointcept

NUM_GPU=${NUM_GPU:-8}
DATASET=${DATASET:-molecule}
CONFIG=${CONFIG:-pretrain_sonata}
EXP_NAME=${EXP_NAME:-pretrain_sonata_afdb10000}
BATCH_SIZE=${BATCH_SIZE:-256}

# Extra overrides can be appended to sbatch command, e.g.:
# sbatch scripts/train_pretrain_sonata.sbatch data_root=data/rna_go_5class data.train.split=train
OVERRIDES=("$@")

sh scripts/train.sh \
  -g "${NUM_GPU}" \
  -d "${DATASET}" \
  -c "${CONFIG}" \
  -n "${EXP_NAME}" \
  -- \
  batch_size="${BATCH_SIZE}" \
  "${OVERRIDES[@]}"
