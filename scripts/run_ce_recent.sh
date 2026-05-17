#!/usr/bin/env bash
set -u

cd /home/ubuntu/quant_project || exit 1
mkdir -p logs results checkpoints

ts=$(date +%Y%m%d_%H%M%S)
log="logs/ce_recent_${ts}.log"

echo "START=$(date)" > "$log"
echo "IDEA_6=recent_train_window_for_nonstationarity" >> "$log"

run_variant() {
  local name="$1"
  shift
  echo "RUN variant=${name} START=$(date)" >> "$log"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u src/train_ce_recent.py \
    --data-dir binance_btcusdt_1m_multifeature_selected_T60_H10 \
    --variant-name "$name" \
    --epochs 25 \
    --memory-size 50000 \
    --batch-size 1024 \
    --eval-batch-size 4096 \
    --k 200 \
    --device cuda \
    --num-workers 4 \
    "$@" >> "$log" 2>&1
  echo "RUN variant=${name} EXIT=$? END=$(date)" >> "$log"
}

run_variant recent200k_ce \
  --train-tail-size 200000 \
  --label-smoothing 0 \
  --dropout 0.1 \
  --weight-decay 1e-4 \
  --class-weight none \
  --focal-gamma 0

run_variant recent300k_ce \
  --train-tail-size 300000 \
  --label-smoothing 0 \
  --dropout 0.1 \
  --weight-decay 1e-4 \
  --class-weight none \
  --focal-gamma 0

run_variant recent200k_smooth002 \
  --train-tail-size 200000 \
  --label-smoothing 0.02 \
  --dropout 0.1 \
  --weight-decay 1e-4 \
  --class-weight none \
  --focal-gamma 0

run_variant recent300k_smooth002 \
  --train-tail-size 300000 \
  --label-smoothing 0.02 \
  --dropout 0.1 \
  --weight-decay 1e-4 \
  --class-weight none \
  --focal-gamma 0

echo "END=$(date)" >> "$log"
