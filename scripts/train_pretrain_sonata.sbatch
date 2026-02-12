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
# NOTE: conda activation scripts may read unset vars (e.g. MKL_INTERFACE_LAYER),
# which conflicts with `set -u`. Temporarily disable nounset for activation.
set +u
if [ -f "/public/home/xinwuye/miniconda/etc/profile.d/conda.sh" ]; then
  source "/public/home/xinwuye/miniconda/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE=$(conda info --base)
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "ERROR: conda not found. Please load conda module or update this script." >&2
  exit 1
fi

conda activate pointcept-torch2.5.0-cu12.4
set -u

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
