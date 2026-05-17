# Experiment Summary

Goal: improve the multi-feature BTCUSDT three-class forecasting validation metrics over the CE softmax baseline.

## Baseline

- CE softmax: `F1=0.440290`, `MCC=0.198533`

## Main Experiments

1. SupCon + KNN: did not beat CE. Best KNN F1 was around `0.4295`.
2. CE embedding + KNN: still below CE softmax.
3. CE + SupCon joint loss: improved F1 slightly but reduced MCC. Best F1 reached around `0.4419`, while MCC stayed below baseline.
4. CE regularization sweep: label smoothing and class weighting produced the best result.

## Best Result

Variant: `smooth002_invsqrt`

Configuration:

- Cross entropy loss
- `label_smoothing=0.02`
- `class_weight=inverse_sqrt`
- `dropout=0.1`
- `weight_decay=1e-4`

Validation result at epoch 6:

- Macro F1: `0.445608`
- MCC: `0.199982`

This exceeds the CE softmax baseline on both metrics.

## Interpretation

The labels are noisy because they are based on future return quantiles. Light label smoothing reduces overconfident fitting to noisy boundary labels, while inverse-sqrt class weights lightly compensate for the 40/30/30 class distribution without overcorrecting.
