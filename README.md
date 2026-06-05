# AI Quant Mini Project

Transformer-based BTCUSDT minute-level classification experiments.

This repository contains the training/evaluation code, experiment scripts, and a small summary of the best validation result. Large datasets, checkpoints, logs, and virtual environments are intentionally excluded.

## Final Selected Version

Baseline CE softmax on the multi-feature dataset:

- Macro F1: `0.440290`
- MCC: `0.198533`

Final model:

- Method: CE + SupCon joint loss
- Training objective: `loss = CE loss + lambda * SupCon loss`
- Selected lambda: `0.02`
- Epoch: `12`
- Macro F1: `0.441857`
- MCC: `0.188430`

Checkpoint path used in the experiment:

```text
/home/ubuntu/quant_project/checkpoints/joint_ce_supcon_F23_lam0p02.pt
```

## Repository Layout

```text
src/        Training, models, losses, datasets, evaluation, KNN sweeps
scripts/    Experiment runner scripts
results/    Small JSON summary files only
```

See `FINAL_EXPERIMENT_REPORT.md` for the full experiment report.

## Notes

Do not commit generated market data, `.npy` arrays, `.parquet` files, checkpoints, logs, or `.venv` directories.
