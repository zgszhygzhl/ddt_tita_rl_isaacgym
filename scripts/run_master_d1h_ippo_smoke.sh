#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SEED="${SEED:-0}"

${PYTHON} train.py \
  --task=master_d1h_climb \
  --headless \
  --max_iterations 20 \
  --seed "${SEED}" \
  --run_name "master_d1h_ppo_smoke_seed${SEED}"

${PYTHON} train.py \
  --task=master_d1h_climb_ippo \
  --headless \
  --max_iterations 20 \
  --seed "${SEED}" \
  --ippo_select_mode topk \
  --ippo_retain_ratio 0.5 \
  --run_name "master_d1h_ippo_smoke_topk50_seed${SEED}"

