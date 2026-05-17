# AI Quant Mini Project

Transformer-based BTCUSDT minute-level classification experiments.

This repository contains the training/evaluation code, experiment scripts, and a small summary of the best validation result. Large datasets, checkpoints, logs, and virtual environments are intentionally excluded.

## Best Validation Result

Baseline CE softmax on the multi-feature dataset:

- Macro F1: `0.440290`
- MCC: `0.198533`

Best observed variant:

- Method: CE with `label_smoothing=0.02` and `inverse_sqrt` class weights
- Variant name: `smooth002_invsqrt`
- Epoch: `6`
- Macro F1: `0.445608`
- MCC: `0.199982`

Remote checkpoint path used in the experiment:

```text
/home/ubuntu/quant_project/checkpoints/ce_regularized_F23_smooth002_invsqrt.pt
```

## Repository Layout

```text
src/        Training, models, losses, datasets, evaluation, KNN sweeps
scripts/    Remote experiment runner scripts
results/    Small JSON summary files only
```

## Notes

Do not commit generated market data, `.npy` arrays, `.parquet` files, checkpoints, logs, or `.venv` directories.
