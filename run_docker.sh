#!/usr/bin/env bash
# Run training in the official PyTorch image with the repository mounted.
set -euo pipefail

docker run --gpus all -it --rm --ipc=host \
    -v "$(pwd)":/workspace -w /workspace \
    --name pytorch-5utr-gan \
    pytorch/pytorch:latest \
    bash -c "pip install -r requirements.txt && python train.py $*"
