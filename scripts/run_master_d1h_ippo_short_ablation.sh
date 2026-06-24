#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
ITERATIONS="${ITERATIONS:-1000}"
SEED="${SEED:-0}"

COMMON_ARGS="--headless --max_iterations ${ITERATIONS} --seed ${SEED}"

${PYTHON} train.py \
  --task=master_d1h_climb \
  ${COMMON_ARGS} \
  --run_name "master_d1h_ppo_seed${SEED}"

${PYTHON} train.py \
  --task=master_d1h_climb_ippo \
  ${COMMON_ARGS} \
  --ippo_select_mode random_same_ratio \
  --ippo_retain_ratio 0.5 \
  --run_name "master_d1h_random_filter50_seed${SEED}"

${PYTHON} train.py \
  --task=master_d1h_climb_ippo \
  ${COMMON_ARGS} \
  --ippo_select_mode weight_only \
  --ippo_weight_clip 3.0 \
  --run_name "master_d1h_ippo_weight_only_seed${SEED}"

${PYTHON} train.py \
  --task=master_d1h_climb_ippo \
  ${COMMON_ARGS} \
  --ippo_select_mode topk \
  --ippo_retain_ratio 0.5 \
  --ippo_weight_clip 3.0 \
  --run_name "master_d1h_ippo_topk50_seed${SEED}"

${PYTHON} train.py \
  --task=master_d1h_climb_ippo \
  ${COMMON_ARGS} \
  --ippo_select_mode topk \
  --ippo_retain_ratio 0.3 \
  --ippo_weight_clip 3.0 \
  --run_name "master_d1h_ippo_topk30_seed${SEED}"

