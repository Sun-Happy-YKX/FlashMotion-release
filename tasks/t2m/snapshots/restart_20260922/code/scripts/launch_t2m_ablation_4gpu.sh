#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 VARIANT OUTPUT_DIR GPU_IDS PORT" >&2
  exit 2
fi

variant="$1"
output="$2"
gpu_ids="$3"
port="$4"
if [[ "$variant" == "tr_t0_full_fm_only" ]]; then
  echo "[$(date -Is)] skip retired variant=$variant; use tr_t0_full_rec_only"
  exit 0
fi
repo="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo"

mkdir -p "$output" logs/t2m_ablation
exec 9>"$output/train.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] skip $variant: another host owns $output"
  exit 0
fi
if [[ -f "$output/train.complete" ]]; then
  echo "[$(date -Is)] skip $variant: complete"
  exit 0
fi

unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK
unset MASTER_ADDR MASTER_PORT TORCHELASTIC_RUN_ID
export CUDA_VISIBLE_DEVICES="$gpu_ids"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

resume=()
if [[ -f "$output/checkpoints/latest.pt" ]]; then
  resume=(--resume "$output/checkpoints/latest.pt")
fi

echo "[$(date -Is)] start variant=$variant output=$output cuda=$gpu_ids resume=${resume[*]:-none}"
.venv/bin/accelerate launch \
  --multi_gpu \
  --num_machines 1 \
  --num_processes 4 \
  --main_process_port "$port" \
  --mixed_precision bf16 \
  --dynamo_backend no \
  train_t2m_long.py \
  --data-root third_party/FloodDiffusion/raw_data \
  --embedding-root data/umt5_xxl_sentence_features \
  --output "$output" \
  --x0-variant "$variant" \
  --batch-size 64 \
  --workers 8 \
  --max-length 200 \
  --max-steps 300000 \
  --save-every-epochs 50 \
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
  --cfg-dropout-prob 0.1 \
  --cfg-omega 2.0 \
  --mixed-precision bf16 \
  "${resume[@]}"

touch "$output/train.complete"
echo "[$(date -Is)] complete variant=$variant output=$output"
