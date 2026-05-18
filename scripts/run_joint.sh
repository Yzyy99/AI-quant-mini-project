#!/usr/bin/env bash
set -u

cd /home/ubuntu/quant_project || exit 1
mkdir -p logs results checkpoints

ts=$(date +%Y%m%d_%H%M%S)
log="logs/joint_ce_supcon_${ts}.log"

echo "START=$(date)" > "$log"
echo "FINAL=ce_plus_supcon_joint_loss_lambda_0p02" >> "$log"

lam=0.02
echo "RUN lambda=$lam START=$(date)" >> "$log"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u src/train_joint.py \
  --data-dir binance_btcusdt_1m_multifeature_selected_T60_H10 \
  --epochs 30 \
  --supcon-weight "$lam" \
  --memory-size 50000 \
  --batch-size 1024 \
  --eval-batch-size 4096 \
  --k 200 \
  --device cuda \
  --num-workers 4 >> "$log" 2>&1
echo "RUN lambda=$lam EXIT=$? END=$(date)" >> "$log"

echo "END=$(date)" >> "$log"
