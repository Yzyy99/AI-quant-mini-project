# Project Implementation Guide (Full Version)
# Time-Series Representation Learning via Transformer + SupCon + KNN for Financial Forecasting

---

## 1. Project Overview

### 1.1 What This Project Does

Build a financial time-series classification system that predicts whether BTC price will go **Up**, **Down**, or stay **Neutral** in the next H minutes, using the past T minutes of **multi-feature** market data (price, volume, technical indicators).

The key innovation: instead of training a model to directly output class probabilities (Cross-Entropy), we train a Transformer encoder with **Supervised Contrastive Loss** to produce embeddings where same-class samples cluster together. At inference time, we use **KNN with a rolling memory bank** — this allows adaptation to market regime shifts without retraining.

### 1.2 Core Innovation (vs. Traditional Approach)

| | Traditional Approach | Our Approach |
|---|---|---|
| Training objective | Cross-Entropy (directly predict class probabilities) | Supervised Contrastive Loss (learn an embedding space where same-class samples cluster) |
| Inference | Softmax → argmax | KNN on embedding space with a rolling memory bank |
| Adaptation to regime shift | Retrain the entire model | Just update the memory bank (no retraining needed) |
| Feature representation | Fixed decision boundaries | Flexible distance-based classification |

### 1.3 Architecture Diagram

```
Input: X ∈ R^(T, F)      ← T minutes of F features (e.g., T=60, F=15)
       │
       ▼
[Linear Projection]       ← (T, F) → (T, d_model)
       │
       ▼
[+ Positional Encoding]   ← learnable positional embeddings
       │
       ▼
[Prepend [CLS] token]    ← (T, d_model) → (T+1, d_model)
       │
       ▼
[N-layer Transformer Encoder]  ← Self-attention layers
       │
       ▼
[Extract [CLS] output]   ← (T+1, d_model) → (d_model,)
       │
       ▼
[Projection Head (MLP)]   ← d_model → embed_dim (ONLY used during training)
       │
       ▼
[L2 Normalize]            ← unit vector v ∈ R^embed_dim
       │
       ▼
[SupCon Loss(v, y)]       ← Training: pull same-class together, push different apart


INFERENCE (after training):
[Extract [CLS] output]   ← (d_model,) — NO projection head
       │
       ▼
[KNN against Memory Bank] ← cosine similarity, top-K neighbors, majority vote → prediction
```

---

## 2. Data Requirements

### 2.1 Data Source

**Primary:** Binance BTC/USDT 1-minute K-line data (freely available from https://data.binance.vision/)

Download at least 12-18 months of data. Suggested range: 2024-01-01 to 2026-04-30.

### 2.2 Raw Data Format (from Binance)

Each row = one 1-minute candle:

| Column | Type | Meaning |
|---|---|---|
| open_time | timestamp | Candle start time (UTC) |
| open | float | Opening price |
| high | float | Highest price in this minute |
| low | float | Lowest price in this minute |
| close | float | Closing price |
| volume | float | BTC traded volume |
| close_time | timestamp | Candle end time |
| quote_asset_volume | float | USDT traded volume |
| num_trades | int | Number of trades |
| taker_buy_base_volume | float | Taker buy BTC volume |
| taker_buy_quote_volume | float | Taker buy USDT volume |

### 2.3 Feature Engineering (MUST IMPLEMENT)

From the raw OHLCV data, compute the following features. All features should be **rolling z-score normalized** (window = 1440 minutes = 1 day) to prevent look-ahead bias and ensure stationarity.

| # | Feature | Formula / Description | Intuition |
|---|---|---|---|
| 1 | ret_1m | log(close[t]) - log(close[t-1]) | 1-minute log return |
| 2 | ret_5m | log(close[t]) - log(close[t-5]) | 5-minute momentum |
| 3 | ret_15m | log(close[t]) - log(close[t-15]) | 15-minute momentum |
| 4 | volatility_20 | std(ret_1m, window=20) | Recent 20-min realized volatility |
| 5 | volatility_60 | std(ret_1m, window=60) | 1-hour realized volatility |
| 6 | volume_change | log(volume[t] / rolling_mean(volume, 20)) | Volume relative to recent average |
| 7 | rsi_14 | RSI with period=14 | Relative Strength Index (overbought/oversold) |
| 8 | macd | EMA(close,12) - EMA(close,26) | Trend direction and strength |
| 9 | macd_signal | EMA(macd, 9) | MACD signal line |
| 10 | macd_hist | macd - macd_signal | MACD histogram (momentum change) |
| 11 | bb_width | (upper_band - lower_band) / middle_band, period=20 | Bollinger Band width (volatility) |
| 12 | bb_position | (close - lower_band) / (upper_band - lower_band) | Price position within Bollinger Bands |
| 13 | atr_14 | Average True Range, period=14 | Volatility measure |
| 14 | obv_change | Rate of change of On-Balance Volume | Volume-price confirmation |
| 15 | vwap_deviation | (close - VWAP) / close | Deviation from volume-weighted average price |

**After computing raw features, apply rolling z-score normalization:**
```python
for each feature f:
    f_normalized[t] = (f[t] - rolling_mean(f, window=1440)) / rolling_std(f, window=1440)
```

This gives approximately 15 features (F=15). The exact number can be adjusted — the model architecture is feature-count agnostic.

### 2.4 Label Generation

```python
# Define future return
horizon = 10  # predict 10 minutes ahead
future_return[t] = log(close[t + horizon]) - log(close[t])

# Compute thresholds from TRAINING SET ONLY (no look-ahead)
q_low = np.percentile(future_return_train, 30)   # ~30th percentile
q_high = np.percentile(future_return_train, 70)   # ~70th percentile

# Assign labels
label[t] = 1 (Up)      if future_return[t] > q_high
label[t] = 2 (Down)    if future_return[t] < q_low
label[t] = 0 (Neutral) otherwise
```

This gives approximately 30% Up / 30% Down / 40% Neutral.

### 2.5 Sliding Window Construction

```python
T = 60  # lookback window (past 60 minutes)
F = 15  # number of features

# For each valid time t:
X[i] = feature_matrix[t-T+1 : t+1, :]   # shape (60, 15)
y[i] = label[t]                            # scalar in {0, 1, 2}
```

**Validity constraints:**
- The 60-minute window must be fully contiguous (no gaps)
- The future horizon (10 min) must also be contiguous
- Skip samples where any feature is NaN (warmup period for indicators)

### 2.6 Chronological Train/Val/Test Split

```python
# Sort all samples by time, then split 70/15/15
train: first 70% of samples (earliest period)
val:   next 15% (middle period)
test:  last 15% (most recent period)
```

**CRITICAL:** Never shuffle across splits. This is a time-series problem — future data must not leak into training.

### 2.7 Final Data Format

After processing, save as NumPy arrays:

```
X_train.npy  — shape (N_train, 60, F), dtype=float32
X_val.npy    — shape (N_val, 60, F), dtype=float32
X_test.npy   — shape (N_test, 60, F), dtype=float32
y_train.npy  — shape (N_train,), dtype=int64, values in {0, 1, 2}
y_val.npy    — shape (N_val,), dtype=int64
y_test.npy   — shape (N_test,), dtype=int64
time_train.npy — shape (N_train,), dtype=datetime64[ns]
time_val.npy   — shape (N_val,), dtype=datetime64[ns]
time_test.npy  — shape (N_test,), dtype=datetime64[ns]
meta.json    — all parameters (T, F, horizon, thresholds, feature_list, split_times)
```

---

## 3. Implementation Tasks (In Order)

### Task 0: Data Pipeline

**File:** `data_pipeline.py`

**Input:** Raw Binance 1-min K-line CSV/Parquet files

**Output:** X_train.npy, y_train.npy, etc. (as described in Section 2.7)

**Steps:**
1. Load and concatenate all monthly Binance data files
2. Verify continuity (no time gaps)
3. Compute all 15 features from raw OHLCV
4. Apply rolling z-score normalization (window=1440)
5. Compute future returns and labels (thresholds from train split only)
6. Build sliding windows (T=60)
7. Filter out NaN/invalid samples
8. Split chronologically 70/15/15
9. Save as .npy files + meta.json

**Quality checks:**
- Print feature statistics (mean ≈ 0, std ≈ 1 after normalization)
- Print class distribution per split
- Verify no NaN in final arrays
- Verify time ordering is preserved

---

### Task 1: PyTorch Dataset & DataLoader

**File:** `dataset.py`

**Requirements:**
- Load X_*.npy and y_*.npy into a PyTorch Dataset
- Return (x, y) where x is (T, F) float32 tensor and y is int64 scalar
- DataLoaders:
  - Train: batch_size=512, shuffle=True, drop_last=True (IMPORTANT for SupCon)
  - Val/Test: batch_size=1024, shuffle=False

**Why drop_last=True:** SupCon requires large batches to have enough positive pairs. An incomplete final batch might have too few samples of one class.

---

### Task 2: Transformer Encoder Model

**File:** `model.py`

**Architecture:**

```python
class TransformerEncoder(nn.Module):
    """
    Input:  x of shape (batch_size, seq_len=T, n_features=F)
    Output (training): L2-normalized embedding of shape (batch_size, embed_dim)
    Output (inference): [CLS] representation of shape (batch_size, d_model)
    """
```

| Component | Specification |
|---|---|
| Input projection | nn.Linear(F, d_model) where F = number of features |
| Positional encoding | Learnable, shape (1, T+1, d_model) |
| [CLS] token | Learnable parameter, shape (1, 1, d_model) |
| Transformer layers | nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True) × n_layers |
| Representation | Take [CLS] position (index 0) from Transformer output → (batch_size, d_model) |
| Projection head | nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, embed_dim)) |
| Normalization | F.normalize(z, dim=1) — L2 normalize to unit sphere |

**Two forward modes:**
```python
def forward(self, x, return_embedding=False):
    # ... transformer encoding ...
    cls_output = ...  # shape (B, d_model)
    
    if return_embedding:
        return cls_output  # For KNN inference (no projection head)
    else:
        z = self.projection_head(cls_output)
        z = F.normalize(z, dim=1)
        return z  # For SupCon training
```

**Default hyperparameters:**

```python
config = {
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 3,
    "dim_feedforward": 256,
    "embed_dim": 64,
    "dropout": 0.1,
}
```

---

### Task 3: Supervised Contrastive Loss

**File:** `loss.py`

Implement SupCon loss (Khosla et al., NeurIPS 2020):

```
L = Σ_i  (-1/|P(i)|) * Σ_{p∈P(i)} log[ exp(v_i · v_p / τ) / Σ_{a∈A(i)} exp(v_i · v_a / τ) ]
```

Where:
- v_i = L2-normalized embedding of sample i
- P(i) = indices in the batch with same label as i (excluding i itself)
- A(i) = all indices except i
- τ = temperature (default: 0.07)

**Implementation pseudocode:**
```python
def supcon_loss(features, labels, temperature=0.07):
    # features: (B, embed_dim), already L2-normalized
    # labels: (B,)
    
    # 1. Compute similarity matrix
    sim_matrix = features @ features.T / temperature  # (B, B)
    
    # 2. Create masks
    labels_equal = (labels.unsqueeze(0) == labels.unsqueeze(1))  # (B, B) bool
    self_mask = ~torch.eye(B, dtype=bool, device=device)         # exclude diagonal
    positive_mask = labels_equal & self_mask                      # same class, not self
    negative_mask = ~labels_equal                                 # different class
    
    # 3. For numerical stability, subtract max
    sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True).values.detach()
    
    # 4. Compute log-softmax over all non-self entries
    exp_sim = torch.exp(sim_matrix) * self_mask  # zero out diagonal
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
    
    # 5. Average over positive pairs
    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_mask.sum(dim=1).clamp(min=1)
    
    # 6. Loss
    loss = -mean_log_prob_pos.mean()
    return loss
```

---

### Task 4: Training Loop

**File:** `train.py`

**Hyperparameters:**
```python
training_config = {
    "batch_size": 512,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "epochs": 50,
    "patience": 10,          # early stopping
    "temperature": 0.07,     # SupCon temperature
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR",
}
```

**Training procedure:**
```python
for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        v = model(X_batch)                    # (B, embed_dim), normalized
        loss = supcon_loss(v, y_batch, tau)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    scheduler.step()
    
    # Evaluate on val set using KNN
    val_f1 = evaluate_knn(model, val_loader, memory_bank_loader, K=20)
    
    # Early stopping
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        save_checkpoint(model)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break
```

**Logging (print per epoch):** epoch, train_loss, val_macro_f1, val_mcc, lr

---

### Task 5: KNN Inference & Evaluation

**File:** `evaluate.py`

**Step 1: Build Memory Bank**
```python
# Encode all validation samples (or a recent subset) using trained model
model.eval()
memory_embeddings = []  # will be (N_memory, d_model)
memory_labels = []      # will be (N_memory,)

with torch.no_grad():
    for X_batch, y_batch in memory_loader:
        emb = model(X_batch, return_embedding=True)  # (B, d_model), NO projection head
        memory_embeddings.append(emb)
        memory_labels.append(y_batch)

memory_embeddings = torch.cat(memory_embeddings)  # (N_memory, d_model)
memory_labels = torch.cat(memory_labels)           # (N_memory,)
# Normalize for cosine similarity
memory_embeddings = F.normalize(memory_embeddings, dim=1)
```

**Step 2: KNN Prediction**
```python
K = 20

predictions = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        emb = model(X_batch, return_embedding=True)
        emb = F.normalize(emb, dim=1)  # (B, d_model)
        
        # Cosine similarity with memory bank
        sim = emb @ memory_embeddings.T  # (B, N_memory)
        
        # Top-K neighbors
        topk_sim, topk_idx = sim.topk(K, dim=1)  # (B, K)
        topk_labels = memory_labels[topk_idx]     # (B, K)
        
        # Majority vote (or distance-weighted vote)
        pred = torch.mode(topk_labels, dim=1).values  # (B,)
        predictions.append(pred)
```

**Step 3: Compute Metrics**
```python
from sklearn.metrics import f1_score, matthews_corrcoef, classification_report, confusion_matrix

macro_f1 = f1_score(y_true, y_pred, average='macro')
mcc = matthews_corrcoef(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=['Neutral', 'Up', 'Down'])
cm = confusion_matrix(y_true, y_pred)
```

**Performance note:** If memory bank is very large (>100k), computing full similarity matrix may be memory-intensive. Batch the computation or use faiss for approximate nearest neighbors.

---

### Task 6: Baselines

**File:** `baselines.py`

Implement these for comparison:

| Baseline | Training | Inference | Purpose |
|---|---|---|---|
| **Raw KNN** | None (no training) | Flatten X to (T*F,), sklearn KNeighborsClassifier | Shows that raw features + KNN don't work well |
| **Transformer + CE** | Same Transformer + nn.Linear(d_model, 3) + CrossEntropyLoss | Softmax → argmax | The "standard" deep learning approach |
| **Transformer + CE + KNN** | Train with CE | Use penultimate features for KNN (same as our method's inference) | Isolates the effect of SupCon vs CE |

The most critical comparison is: **Ours (SupCon + KNN) vs. Transformer + CE (Baseline 2)**.

For Baseline 2 (Transformer + CE):
- Same architecture, same hyperparameters
- Replace SupCon loss with `nn.CrossEntropyLoss()`
- Add a classification head: `nn.Linear(d_model, 3)`
- Train to directly predict labels
- Inference: softmax output → argmax

---

### Task 7: Ablation Studies

| Ablation | Values to Test | What it Shows |
|---|---|---|
| Memory bank size M | 1000, 5000, 10000, 50000, full val | How much recent context is needed |
| K (neighbors) | 5, 10, 20, 50, 100 | Sensitivity to neighborhood size |
| Temperature τ | 0.05, 0.07, 0.1, 0.2, 0.5 | How tight/loose the clusters should be |
| Embedding dim | 32, 64, 128 | Representational capacity |
| Number of features | 1, 5, 10, 15 | Whether more features help |
| Lookback T | 30, 60, 120 | How much history the model needs |

---

### Task 8: Visualization

**File:** `visualize.py`

1. **t-SNE / UMAP of embedding space**: Encode test samples, plot in 2D, color by true label. Compare SupCon model vs. CE model embeddings — SupCon should show cleaner clusters.

2. **Training curves**: Plot loss vs. epoch, val_F1 vs. epoch for both SupCon and CE models.

3. **Confusion matrix**: Heatmap for test predictions.

4. **Temporal embedding drift**: Take embeddings from different time periods (e.g., month by month) and show how they shift in the space — demonstrates why a rolling memory bank helps.

---

### Task 9: Backtesting (Stretch Goal)

**File:** `backtest.py`

**Strategy:**
- Predict "Up" → long position
- Predict "Down" → short position  
- Predict "Neutral" → flat (no position)

**Metrics to compute:**
- Cumulative PnL curve (using actual realized returns)
- Annualized Sharpe Ratio
- Maximum Drawdown
- Win rate (% of trades that are profitable)
- Total number of trades

**Transaction costs:** Assume 5 basis points (0.05%) per trade (entry + exit).

---

## 4. Expected Results & Success Criteria

### 4.1 Quantitative Targets

| Metric | Random Baseline | Reasonable | Good |
|---|---|---|---|
| Macro F1 | ~0.33 | 0.35-0.38 | >0.38 |
| MCC | ~0.00 | 0.03-0.08 | >0.08 |

**Important context:** Financial prediction is extremely noisy. Even 1-2% improvement over random in Macro F1 is considered meaningful in this domain.

### 4.2 Key Hypotheses to Validate

1. **Primary:** SupCon + KNN achieves higher Macro F1 than Transformer + CE on the test set.
2. **Secondary:** The rolling memory bank allows better adaptation — performance degrades less over time compared to the CE model.
3. **Tertiary:** The embedding space shows clear class separation (visible in t-SNE plots).

---

## 5. Technical Notes

### 5.1 GPU Requirements

- Model: ~2M parameters (small Transformer) → negligible memory
- Batch: (512, 61, 128) float32 → ~16 MB
- Total: <4 GB → any modern GPU works (even a free Colab T4)
- Can also train on CPU if needed, just slower (~10x)

### 5.2 Training Time Estimate

- ~1000 steps/epoch, ~50 epochs max, with early stopping likely ~20-30 epochs
- On GPU: ~2-3 min/epoch → total ~1 hour
- On CPU: ~20-30 min/epoch → total ~10 hours

### 5.3 Dependencies

```
torch>=2.0
numpy
pandas
scikit-learn
matplotlib
seaborn
tqdm
umap-learn (for visualization)
faiss-cpu (optional, for fast KNN with large memory banks)
ta (for technical indicator computation, pip install ta)
```

### 5.4 Critical Pitfalls to Avoid

| Pitfall | Explanation | Correct Approach |
|---|---|---|
| Using projection head for KNN | Projection head is only for training loss calculation | Use [CLS] output (d_model dim) for KNN |
| Small batch size | SupCon needs many positive pairs per batch | batch_size ≥ 256, ideally 512 |
| Shuffling test data | Financial data is time-ordered | Never shuffle; splits are chronological |
| Look-ahead in normalization | Cannot use future data for z-score | Rolling window uses only past data |
| Look-ahead in thresholds | Label thresholds must come from train set only | Compute q_low, q_high from training data |
| Forgetting L2 normalization | SupCon assumes unit vectors | Always normalize before loss computation |
| Memory bank from training set | Training data is too old, doesn't represent current regime | Use validation set or recent subset as memory bank |

---

## 6. Project File Structure

```
project/
├── data/
│   ├── raw/                    # Raw Binance CSV/Parquet files
│   └── processed/              # X_train.npy, y_train.npy, etc.
├── src/
│   ├── data_pipeline.py        # Task 0: Raw data → processed arrays
│   ├── dataset.py              # Task 1: PyTorch Dataset/DataLoader
│   ├── model.py                # Task 2: Transformer Encoder
│   ├── loss.py                 # Task 3: SupCon Loss
│   ├── train.py                # Task 4: Training loop
│   ├── evaluate.py             # Task 5: KNN inference + metrics
│   ├── baselines.py            # Task 6: Baseline models
│   ├── ablation.py             # Task 7: Ablation experiments
│   ├── visualize.py            # Task 8: Plots and figures
│   └── backtest.py             # Task 9: Trading simulation
├── checkpoints/                # Saved model weights
├── results/                    # Metrics tables, plots, figures
├── configs/                    # Hyperparameter configs (JSON/YAML)
└── README.md
```

---

## 7. Deliverables Checklist

### Must Have (Core)
- [ ] Data pipeline: raw → processed multi-feature arrays
- [ ] Trained SupCon model with best val checkpoint
- [ ] Trained CE baseline with best val checkpoint
- [ ] Results table: SupCon+KNN vs. CE vs. Raw KNN (Macro F1, MCC, per-class metrics)
- [ ] t-SNE/UMAP plot showing embedding quality
- [ ] Training curves (loss, val_F1)

### Should Have
- [ ] All 3 baselines compared
- [ ] Ablation on K, temperature, memory bank size
- [ ] Confusion matrices
- [ ] Classification reports

### Nice to Have
- [ ] Ablation on number of features
- [ ] Temporal analysis of embeddings
- [ ] Backtest with PnL curve and Sharpe Ratio

---

## 8. Execution Priority

**If time is limited, do tasks in this exact order:**

1. **Task 0** — Data pipeline (everything depends on this)
2. **Task 1** — DataLoader (needed for training)
3. **Task 2** — Transformer model
4. **Task 3** — SupCon loss
5. **Task 4** — Training loop → get a trained model
6. **Task 5** — KNN evaluation → get results
7. **Task 6** — Baselines (at minimum: Transformer+CE)
8. **Task 8** — Visualization (t-SNE + training curves)
9. **Task 7** — Ablations
10. **Task 9** — Backtesting

The **single most important result** is the comparison table:

```
| Model                  | Macro F1 | MCC   |
|------------------------|----------|-------|
| Raw KNN                | ???      | ???   |
| Transformer + CE       | ???      | ???   |
| Transformer + CE + KNN | ???      | ???   |
| Ours (SupCon + KNN)    | ???      | ???   |  ← should be best
```

If this table shows SupCon+KNN > CE, the project is a success.
