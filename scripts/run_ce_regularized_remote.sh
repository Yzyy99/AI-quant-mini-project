#!/usr/bin/env bash
set -u

cd /home/ubuntu/quant_project || exit 1
mkdir -p logs results checkpoints

ts=$(date +%Y%m%d_%H%M%S)
log="logs/ce_regularized_${ts}.log"

echo "START=$(date)" > "$log"
echo "IDEA_5=ce_regularization_label_smoothing_class_weight_sweep" >> "$log"

run_variant() {
  local name="$1"
  shift
  echo "RUN variant=${name} START=$(date)" >> "$log"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u src/train_ce_regularized.py \
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

run_variant smooth005 \
  --label-smoothing 0.05 \
  --dropout 0.1 \
  --weight-decay 1e-4 \
  --class-weight none \
  --focal-gamma 0

run_variant smooth005_dropout02_wd5e4 \
  --label-smoothing 0.05 \
  --dropout 0.2 \
  --weight-decay 5e-4 \
  --class-weight none \
  --focal-gamma 0

run_variant smooth002_invsqrt \
  --label-smoothing 0.02 \
  --dropout 0.1 \
  --weight-decay 1e-4 \
  --class-weight inverse_sqrt \
  --focal-gamma 0

run_variant focal1_smooth002 \
  --label-smoothing 0.02 \
  --dropout 0.1 \
  --weight-decay 1e-4 \
  --class-weight none \
  --focal-gamma 1.0

echo "END=$(date)" >> "$log"
