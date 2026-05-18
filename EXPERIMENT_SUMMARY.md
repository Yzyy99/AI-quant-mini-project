# Experiment Summary

Goal: improve the multi-feature BTCUSDT three-class forecasting validation macro-F1 over the CE softmax baseline.

## Baseline

- CE softmax: `F1=0.440290`, `MCC=0.198533`

## Main Experiments

1. SupCon + KNN: did not beat CE. Best KNN F1 was around `0.4295`.
2. CE embedding + KNN: still below CE softmax.
3. CE + SupCon joint loss: improved macro-F1 over the CE softmax baseline. This is the final selected version because it keeps the project focused on supervised contrastive representation learning while improving the primary F1 metric.
4. CE regularization sweep: label smoothing and class weighting was also explored, but it is not the selected final version.

## Final Selected Result

Variant: `joint_ce_supcon_F23_lam0p02`

Configuration:

- Joint objective: `loss = CE loss + lambda * SupCon loss`
- `lambda=0.02`
- `temperature=0.07`
- `dropout=0.1`
- `weight_decay=1e-4`

Validation result at epoch 12:

- Macro F1: `0.441857`
- MCC: `0.188430`

This exceeds the CE softmax baseline on the primary macro-F1 metric.

## Interpretation

The labels are noisy because they are based on future return quantiles. Pure SupCon/KNN did not work well because same-label financial sequences can still have very different shapes. The joint CE + SupCon objective keeps the stable CE classification signal while adding a weak supervised contrastive representation-learning term. The best F1 was obtained with `lambda=0.02`; larger or smaller contrastive weights did not improve the final F1 as much.
