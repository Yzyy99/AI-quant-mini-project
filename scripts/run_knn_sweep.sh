#!/usr/bin/env bash
set -u

cd /home/ubuntu/quant_project || exit 1
mkdir -p logs results

ts=$(date +%Y%m%d_%H%M%S)
log="logs/knn_sweep_${ts}.log"

{
  echo "START=$(date)"
  echo "IDEA_1=weighted_knn_k_memory_sweep_supcon_cls"
} > "$log"

COMMON_DATA="--data-dir binance_btcusdt_1m_multifeature_selected_T60_H10"
COMMON_SWEEP="--memory-sizes 10000,50000,100000,200000 --ks 5,10,20,50,100,200 --votes majority,sim,exp0.05,exp0.1,exp0.2 --eval-batch-size 2048 --device cuda"

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u src/knn_sweep.py \
  $COMMON_DATA \
  --checkpoint checkpoints/supcon_transformer_F23.pt \
  --checkpoint-type supcon \
  --embedding cls \
  $COMMON_SWEEP \
  --out results/knn_sweep_supcon_cls_F23.json >> "$log" 2>&1
echo "IDEA_1_EXIT=$? AT=$(date)" >> "$log"

echo "IDEA_2=supcon_projection_embedding_knn" >> "$log"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u src/knn_sweep.py \
  $COMMON_DATA \
  --checkpoint checkpoints/supcon_transformer_F23.pt \
  --checkpoint-type supcon \
  --embedding projection \
  $COMMON_SWEEP \
  --out results/knn_sweep_supcon_projection_F23.json >> "$log" 2>&1
echo "IDEA_2_EXIT=$? AT=$(date)" >> "$log"

echo "IDEA_3=ce_embedding_weighted_knn_sweep" >> "$log"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u src/knn_sweep.py \
  $COMMON_DATA \
  --checkpoint checkpoints/ce_transformer_F23.pt \
  --checkpoint-type ce \
  --embedding cls \
  $COMMON_SWEEP \
  --out results/knn_sweep_ce_cls_F23.json >> "$log" 2>&1
echo "IDEA_3_EXIT=$? AT=$(date)" >> "$log"

echo "END=$(date)" >> "$log"
