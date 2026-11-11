#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 CHAIN_NAME" >&2
  exit 2
fi

repo="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo"
mkdir -p logs/t2m_factorial

wait_for_complete() {
  local output="$1"
  while [[ ! -f "$output/train.complete" ]]; do
    sleep 60
  done
}

run_one() {
  local tag="$1" output="$2" gpus="$3" port="$4" fusion="$5"
  local selection="$6" direction="$7" floor="$8"
  bash scripts/launch_t2m_tr_only_factorial_4gpu.sh \
    "$output" "$gpus" "$port" "$fusion" "$selection" "$direction" "$floor" \
    >"logs/t2m_factorial/${tag}.backlog.log" 2>&1
}

case "$1" in
  a3_g1)
    wait_for_complete runs/t2m_factorial_tr_only_adain_nosel_sum_floor1
    run_one adain_nosel_condgate_floor1 \
      runs/t2m_factorial_tr_only_adain_nosel_condgate_floor1 \
      0,4,6,7 29811 adain 0 cond_gate 1.0
    ;;
  a3_g2)
    wait_for_complete runs/t2m_factorial_tr_only_adain_sel_sum_floor1
    run_one add_sel_sum_nofloor \
      runs/t2m_factorial_tr_only_add_sel_sum_nofloor \
      0,2,3,5 29812 add 1 sum 0.0
    run_one add_sel_sum_floor1 \
      runs/t2m_factorial_tr_only_add_sel_sum_floor1 \
      0,2,3,5 29812 add 1 sum 1.0
    ;;
  a3_g3)
    wait_for_complete runs/t2m_factorial_tr_only_adain_sel_condgate_floor1
    run_one add_sel_condgate_nofloor \
      runs/t2m_factorial_tr_only_add_sel_condgate_nofloor \
      1,4,6,7 29813 add 1 cond_gate 0.0
    run_one add_sel_condgate_floor1 \
      runs/t2m_factorial_tr_only_add_sel_condgate_floor1 \
      1,4,6,7 29813 add 1 cond_gate 1.0
    ;;
  a2_g4)
    wait_for_complete runs/t2m_factorial_tr_only_add_nosel_sum_floor1
    run_one add_nosel_condgate_floor1 \
      runs/t2m_factorial_tr_only_add_nosel_condgate_floor1 \
      1,4,5,7 29821 add 0 cond_gate 1.0
    ;;
  a2_g5)
    wait_for_complete runs/t2m_factorial_tr_only_add_nosel_condgate_nofloor
    run_one adain_nosel_condgate_nofloor \
      runs/t2m_factorial_tr_only_adain_nosel_condgate_nofloor \
      0,2,5,6 29822 adain 0 cond_gate 0.0
    ;;
  *)
    echo "unknown chain: $1" >&2
    exit 2
    ;;
esac
