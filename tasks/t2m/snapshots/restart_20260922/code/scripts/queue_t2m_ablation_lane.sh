#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 GPU_IDS BASE_PORT" >&2
  exit 2
fi

gpu_ids="$1"
base_port="$2"
repo="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo"
mkdir -p logs/t2m_ablation

run_variant() {
  local variant="$1" output="$2" port="$3" min_step="$4"
  local eval_gpu="${gpu_ids%%,*}"
  nohup setsid bash scripts/watch_t2m_ablation_eval.sh \
    "$variant" "$output" "$eval_gpu" "$min_step" \
    >>"logs/t2m_ablation/${variant}_watcher_launcher.log" 2>&1 < /dev/null &
  bash scripts/launch_t2m_ablation_4gpu.sh \
    "$variant" "$output" "$gpu_ids" "$port" \
    2>&1 | tee -a "logs/t2m_ablation/${variant}_train.log"
}

# CFG remains last, as agreed; the existing tr_t0_full run is resumed rather
# than restarted so its first 260k optimization steps remain useful.
run_variant t0_only runs/t2m_ablation_t0_only_bs64 "$((base_port + 1))" 0
run_variant tr_only runs/t2m_ablation_tr_only_bs64 "$((base_port + 2))" 0
run_variant tr_t0_aux runs/t2m_ablation_tr_t0_aux_bs64 "$((base_port + 3))" 0
run_variant tr_t0_full_rec_only runs/t2m_ablation_tr_t0_full_rec_only_bs64 "$((base_port + 4))" 0
run_variant tr_t0_full runs/t2m_long_tr_t0_full "$((base_port + 5))" 260000
run_variant tr_cfg_baked runs/t2m_ablation_tr_cfg_baked_bs64 "$((base_port + 6))" 0

touch logs/t2m_ablation/all_six_train.complete
echo "[$(date -Is)] all six T2M ablation training lanes complete"
