#!/usr/bin/env bash
set -euo pipefail

project_root="${1:-/efs/yangkaixing/FlowerDance_T2M_Long}"
gpu_ids="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
cd "$project_root"

export CUDA_VISIBLE_DEVICES="$gpu_ids"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

exec .venv/bin/accelerate launch \
  --multi_gpu \
  --num_machines 1 \
  --num_processes 4 \
  --main_process_port 29609 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  train_t2m_long.py \
  --data-root third_party/FloodDiffusion/raw_data \
  --embedding-root data/umt5_xxl_sentence_features \
  --output runs/t2m_long_tr_t0_full \
  --batch-size 64 \
  --workers 8 \
  --max-length 200 \
  --max-steps 300000 \
  --save-every 10000 \
  --log-every 20 \
  --learning-rate 2e-4 \
  --weight-decay 0.02 \
  --max-grad-norm 1.0 \
  --min-t 0.05 \
  --min-interval 0.05 \
  --fm-time-weight-power 2.0 \
  --fm-weight 1.0 \
  --rec-weight 0.636 \
  --vel-weight 0.323 \
  --acc-weight 0.2 \
  --fk-weight 0.646 \
  --mixed-precision bf16
