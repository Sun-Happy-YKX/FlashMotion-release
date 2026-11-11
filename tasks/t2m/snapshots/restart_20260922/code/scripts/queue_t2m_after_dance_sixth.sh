#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo"
mkdir -p logs/t2m_ablation
echo "[$(date -Is)] waiting for Dance FM-only lane 0,4,6,7"
while pgrep -f "train.py --x0_variant tr_t0_full .*x0_ablation_tr_t0_full_fm_only_bs64" >/dev/null; do
  sleep 60
done
echo "[$(date -Is)] Dance FM-only ended; opening second T2M lane"
exec bash scripts/queue_t2m_ablation_lane.sh 0,4,6,7 29730
