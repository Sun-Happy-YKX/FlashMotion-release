#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 OUTPUT_DIR GPU_IDS PORT CROSS_MODAL_FUSION CONDITION_SELECTION DIRECTION_FUSION FLOOR_WEIGHT" >&2
  exit 2
fi

output="$1"
gpu_ids="$2"
port="$3"
cross_modal_fusion="$4"
condition_selection="$5"
direction_fusion="$6"
floor_weight="$7"

[[ "$cross_modal_fusion" == "add" || "$cross_modal_fusion" == "adain" ]] || exit 2
[[ "$condition_selection" == "0" || "$condition_selection" == "1" ]] || exit 2
[[ "$direction_fusion" == "sum" || "$direction_fusion" == "cond_gate" ]] || exit 2

repo="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo"
mkdir -p "$output" logs/t2m_factorial
exec 9>"$output/train.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] skip: another host owns $output"
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
selection=()
if [[ "$condition_selection" == "1" ]]; then
  selection=(--condition-selection)
fi

echo "[$(date -Is)] start variant=tr_only fusion=$cross_modal_fusion condition_selection=$condition_selection direction=$direction_fusion floor=$floor_weight output=$output cuda=$gpu_ids resume=${resume[*]:-none}"
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
  --x0-variant tr_only \
  --cross-modal-fusion "$cross_modal_fusion" \
  "${selection[@]}" \
  --direction-fusion "$direction_fusion" \
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
  --floor-weight "$floor_weight" \
  --cfg-dropout-prob 0.1 \
  --cfg-omega 2.0 \
  --mixed-precision bf16 \
  "${resume[@]}"

touch "$output/train.complete"
echo "[$(date -Is)] complete output=$output"
