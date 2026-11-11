#!/usr/bin/env bash
set -u

if [[ $# -lt 4 ]]; then
  echo "usage: $0 VARIANT OUTPUT_DIR GPU_ID MIN_STEP" >&2
  exit 2
fi

variant="$1"
train_output="$2"
gpu_id="$3"
min_step="$4"
if [[ "$variant" == "tr_t0_full_fm_only" ]]; then
  echo "[$(date -Is)] skip retired variant=$variant; use tr_t0_full_rec_only"
  exit 0
fi
repo="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo"
mkdir -p "$train_output" logs/t2m_ablation "eval/t2m_ablation_periodic/$variant"
log="logs/t2m_ablation/${variant}_periodic_eval.log"
exec 8>"$train_output/eval.lock"
if ! flock -n 8; then
  echo "[$(date -Is)] skip duplicate evaluator for $variant" >>"$log"
  exit 0
fi
exec 7>"/tmp/t2m_ablation_eval_gpu_${gpu_id}.lock"

while true; do
  did_work=0
  while IFS= read -r checkpoint; do
    [[ -n "$checkpoint" ]] || continue
    name="$(basename "$checkpoint" .pt)"
    step="${name#step-}"
    step=$((10#$step))
    (( step > min_step )) || continue
    eval_output="eval/t2m_ablation_periodic/$variant/$name"
    [[ -f "$eval_output/metrics.json" ]] && continue
    did_work=1
    # Use exactly the frozen 311-episode BABEL-long-v2 cohort (>200 frames,
    # multi-caption, no upper cap). Its 621 scored segments keep the 512-D
    # covariance full-rank while matching the comparison table protocol.
    cohort_dir="eval/babel_long_v2/motions/flooddiffusion"
    echo "[$(date -Is)] evaluate variant=$variant step=$step cohort=babel_long_v2" | tee -a "$log"
    # Multiple training runs may hit the same 50-epoch boundary together.
    # Serialize evaluators assigned to one GPU so all 1/2/4-step artifacts are
    # completed instead of racing into an avoidable OOM.
    if flock 7 env CUDA_VISIBLE_DEVICES="$gpu_id" .venv/bin/python evaluate_t2m_ablation.py \
      --checkpoint "$checkpoint" \
      --output "$eval_output" \
      --steps 1 2 4 \
      --cohort-dir "$cohort_dir" \
      --limit 0 \
      >>"$log" 2>&1; then
      echo "[$(date -Is)] evaluation-complete variant=$variant step=$step" | tee -a "$log"
    else
      echo "[$(date -Is)] evaluation-failed variant=$variant step=$step; retrying later" | tee -a "$log"
      sleep 60
      break
    fi
  done < <(find "$train_output/checkpoints" -maxdepth 1 -type f -name 'step-*.pt' 2>/dev/null | sort)

  if [[ -f "$train_output/train.complete" ]]; then
    pending=0
    while IFS= read -r checkpoint; do
      [[ -n "$checkpoint" ]] || continue
      name="$(basename "$checkpoint" .pt)"
      step="${name#step-}"
      step=$((10#$step))
      if (( step > min_step )) && [[ ! -f "eval/t2m_ablation_periodic/$variant/$name/metrics.json" ]]; then
        pending=1
        break
      fi
    done < <(find "$train_output/checkpoints" -maxdepth 1 -type f -name 'step-*.pt' 2>/dev/null | sort)
    if (( pending == 0 )); then
      echo "[$(date -Is)] all evaluations complete variant=$variant" | tee -a "$log"
      exit 0
    fi
  fi
  (( did_work == 0 )) && sleep 30
done
