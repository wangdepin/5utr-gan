#!/bin/bash
#SBATCH --job-name=5utr-gan
#SBATCH --output=/wrk-kappa/users/depinwan/tries/2026-09-29-5utr-gan/logs/%x_%j.out
#SBATCH --error=/wrk-kappa/users/depinwan/tries/2026-09-29-5utr-gan/logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --account=cs_ukko2

# Train one configuration. The objective is passed in so the corrected and the
# original pretraining losses can be run as a matched pair under one budget.
#
#   sbatch run_turso.sbatch <nll|bce_gumbel> <run_name>

set -euo pipefail

WORK=/wrk-kappa/users/depinwan/tries/2026-09-29-5utr-gan
OBJECTIVE=${1:?usage: sbatch run_turso.sbatch <nll|bce_gumbel> <run_name>}
RUN_NAME=${2:?usage: sbatch run_turso.sbatch <nll|bce_gumbel> <run_name>}

export PATH=/wrk-kappa/users/depinwan/pixi/bin:$PATH
export PIXI_CACHE_DIR=$WORK/.pixicache

cd "$WORK/code"

echo "[$(date -u +%FT%TZ)] host=$(hostname) objective=$OBJECTIVE run=$RUN_NAME"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# The environment is resolved on the login node; compute nodes have no network.
pixi run --frozen python train.py \
    --objective "$OBJECTIVE" \
    --max_len 50 \
    --pre_epochs 100 \
    --total_batch 500 \
    --d_steps 5 \
    --d_pretrain_steps 30 \
    --sample_every 50 \
    --out_dir "$WORK/runs/$RUN_NAME" \
    --log_dir "$WORK/runs/$RUN_NAME/log"

echo "[$(date -u +%FT%TZ)] DONE $RUN_NAME"
