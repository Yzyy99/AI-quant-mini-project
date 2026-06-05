# AI Quant Mini Project 完整实验报告

## 1. 项目背景与目标

本项目围绕 BTCUSDT 1 分钟级别行情数据，尝试构建一个基于深度学习的短期方向预测模型。输入为过去一段时间的多维市场特征，输出为未来短期收益方向的三分类标签：

```text
Neutral / Up / Down
```

项目最初的核心目标不是直接构建一个可实盘交易系统，而是验证两件事：

1. 多特征分钟级时间序列能否提供比简单 baseline 更好的分类信号。
2. Supervised Contrastive Learning（SupCon）能否改善金融时间序列的表征学习，从而提高方向分类性能。

最终版本选择 **CE + SupCon joint loss**。训练目标为：

```text
loss = CE loss + lambda * SupCon loss
```

最终选定：

```text
lambda = 0.02
```

项目当前以 validation macro-F1 作为主要选择指标。原因是三分类任务中类别并非完全均衡，单纯 accuracy 容易被多数类影响；macro-F1 更关注 `Neutral`、`Up`、`Down` 三类整体表现。

## 2. 数据集与标签设计

### 2.1 数据形态

最终使用的是 multi-feature BTCUSDT 数据集。每个样本由过去 60 个 1-minute bar 的特征组成，预测未来 10 分钟收益方向。

```text
lookback window T = 60
prediction horizon H = 10
feature dimension F = 23
input shape = (60, 23)
```

数据划分如下：

| Split | Samples |
|---|---:|
| Train | 550,189 |
| Validation | 117,936 |
| Test | 117,926 |

训练集标签分布大致为：

| Class | Approx. Ratio |
|---|---:|
| Neutral | 40% |
| Up | 30% |
| Down | 30% |

这说明类别并非严重不均衡，但 `Neutral` 明显更多，因此模型容易倾向于预测中性类别。

### 2.2 标签构造

标签来自未来收益率的分位数划分。具体地，基于 future return 的 30/70 分位数构造三分类：

```text
future return <= 30% quantile      -> Down
30% quantile < future return < 70% -> Neutral
future return >= 70% quantile      -> Up
```

这种标签设计合理地把预测问题转成分类问题，但也带来一个明显问题：**边界附近样本标签 noisy**。例如两个历史走势几乎相同的窗口，只要未来收益率一个略高于 70% 分位、另一个略低于 70% 分位，它们就会被分到不同类别。这也是 SupCon 类方法在金融数据上不一定稳定的原因之一。

### 2.3 数据质量检查

在正式训练前，对数据做了基础检查：

- 多特征数据维度为 `F=23`，Transformer 代码能够自适应 1-feature 和 23-feature 输入。
- `X_train / X_val / X_test`、`y_train / y_val / y_test` 形状匹配。
- 检查过数组中没有明显 NaN/inf。
- 时间戳为连续 1-minute 序列。
- 特征标准化采用过去信息，避免使用未来信息造成 leakage。滚动标准化使用 `shift(1)`，并设置合理 rolling window 与 `min_periods`。

因此，后续实验主要关注模型与损失函数，而不是数据清洗问题。

## 3. 模型结构

基础模型是一个 Transformer Encoder，用于处理 `(T, F)` 的时间序列输入。

主要结构如下：

| Component | Setting |
|---|---:|
| Input projection | `F -> d_model` |
| `d_model` | 128 |
| Attention heads | 4 |
| Transformer encoder layers | 3 |
| Feedforward hidden dim | 256 |
| Dropout | 0.1 |
| Projection head output dim | 64 |
| Classification classes | 3 |

模型使用 CLS token 汇聚整个时间窗口的表示。对于 CE baseline，CLS embedding 直接进入 linear classifier：

```text
CLS embedding -> Linear(128, 3) -> logits
```

对于 SupCon 或 joint loss，模型还会通过 projection head 得到 normalized embedding：

```text
CLS embedding -> projection head -> normalized contrastive embedding
```

这样同一个 encoder 同时服务于分类任务和对比学习任务。

## 4. 损失函数与评估指标

### 4.1 Cross Entropy

CE 即 Cross Entropy loss，是三分类任务最直接的监督学习目标。模型输出三个 logits，对应 `Neutral / Up / Down`，CE 会鼓励真实类别的概率变高。

CE baseline 训练目标为：

```text
loss = CE(logits, label)
```

### 4.2 Supervised Contrastive Loss

SupCon 的目标是在 embedding 空间中拉近同标签样本、推远不同标签样本。简化表达如下：

```text
L_supcon = - mean_i mean_p log exp(sim(z_i, z_p) / tau) / sum_a exp(sim(z_i, z_a) / tau)
```

其中：

- `z_i` 是样本 embedding
- `p` 是与 `i` 同标签的 positive sample
- `a` 是 batch 中其它样本
- `tau` 是 temperature，本项目使用 `0.07`

在金融三分类任务中，SupCon 的挑战在于：同标签样本不一定具有相似历史形态，而标签又有分位数边界噪声，因此 SupCon loss 不一定像图像分类那样稳定下降。

### 4.3 Joint Loss

最终版本使用 CE + SupCon joint loss：

```text
loss = CE loss + lambda * SupCon loss
```

CE 保留强监督分类能力，SupCon 提供额外的类别结构约束。`lambda` 控制 SupCon 对总 loss 的影响。实验扫了：

```text
lambda = 0.01, 0.02, 0.05
```

### 4.4 评估指标

主要评估指标：

| Metric | Meaning |
|---|---|
| Macro-F1 | 三个类别 F1 的平均值，主要选择指标 |
| MCC | Matthews Correlation Coefficient，衡量整体三分类相关性 |
| KNN-F1 | 用 embedding 的 KNN 分类结果评估表征质量 |

当前最终版本按 macro-F1 选择，因为项目希望优先展示 CE + SupCon 对主要分类指标的改善。

## 5. 训练设置

主要训练参数如下：

| Parameter | Value |
|---|---:|
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Weight decay | `1e-4` |
| Scheduler | CosineAnnealingLR |
| Batch size | 1024 |
| Eval batch size | 4096 |
| Epochs | 30 |
| SupCon temperature | 0.07 |
| KNN memory size | 50,000 |
| KNN K | 200 |

训练设备为单卡 GPU。此前也讨论过 M4 Pro 和多卡 A100 的资源需求，结论是该模型参数量和显存占用都不算大，用 8 卡 A100 明显过度；单卡高端 GPU 已足够完成主要实验。

## 6. Baseline 实验

### 6.1 Raw KNN

先测试不训练深度模型，直接用原始窗口特征做 KNN。

| Method | Macro-F1 | MCC |
|---|---:|---:|
| Raw KNN | 0.3790 | 0.0788 |

结果明显低于深度模型，说明原始窗口空间中的欧氏/余弦邻近关系不足以直接表达未来收益标签。

### 6.2 CE Softmax Baseline

CE softmax 是主要 baseline。

| Method | Best Macro-F1 | Best MCC |
|---|---:|---:|
| Transformer + CE softmax | 0.440290 | 0.198533 |

该 baseline 是后续实验的主要对照。它的优点是训练稳定、目标直接、对 noisy label 的鲁棒性比纯 SupCon 更好。

## 7. SupCon 与 KNN 实验

### 7.1 SupCon + KNN

第一条主线是训练 SupCon 表征，然后用 memory bank + KNN 做分类。memory bank 使用最近一段训练集 embedding 作为检索库，validation 样本根据最近邻标签投票得到预测。

初始 SupCon + KNN 结果约为：

| Method | Macro-F1 | MCC |
|---|---:|---:|
| SupCon + KNN initial | 0.4215 | 0.1466 |

之后进一步扫了：

- embedding 类型：CLS embedding / projection embedding
- memory size：10k / 50k / 100k / 200k
- K：5 / 10 / 20 / 50 / 100 / 200
- vote：majority / similarity weighted / exp weighted

较好的结果来自 SupCon projection embedding：

| Method | Best Macro-F1 | MCC |
|---|---:|---:|
| SupCon projection KNN sweep | 0.4295 | 0.1645 |

仍然低于 CE softmax baseline。

### 7.2 CE Embedding + KNN

也测试了 CE 模型 embedding 再接 KNN 的方案。结果大致为：

| Method | Best Macro-F1 | MCC |
|---|---:|---:|
| CE embedding + KNN | 0.4118 | 0.1274 |

这说明对于当前任务，softmax classifier 学到的决策边界比 KNN 邻近投票更有效。embedding 的局部结构并不一定直接对应 future return 的三分类标签。

## 8. CE + SupCon Joint Loss 实验

### 8.1 实验设计

纯 SupCon/KNN 不够好，但 SupCon 仍可能作为辅助约束改善 encoder 表征。因此引入 joint loss：

```text
loss = CE loss + lambda * SupCon loss
```

该方案的直觉是：

- CE 负责直接优化分类边界。
- SupCon 负责让同类样本在 embedding 空间中更集中。
- 通过较小的 `lambda` 控制 SupCon 不要压过 CE。

### 8.2 Lambda Sweep

实验结果如下：

| Lambda | Best F1 Epoch | Macro-F1 | MCC | KNN-F1 | KNN-MCC |
|---:|---:|---:|---:|---:|---:|
| 0.01 | 12 | 0.441679 | 0.187209 | 0.419116 | 0.155039 |
| 0.02 | 12 | **0.441857** | 0.188430 | 0.420048 | 0.155872 |
| 0.05 | 12 | 0.440703 | 0.188844 | 0.421225 | 0.158235 |

`lambda=0.02` 的 macro-F1 最高，因此作为最终版本。

最终模型：

```text
joint_ce_supcon_F23_lam0p02
```

最终 checkpoint 对应：

```text
epoch = 12
validation macro-F1 = 0.441856711735245
validation MCC = 0.18842999414872402
```

相比 CE softmax baseline：

```text
0.441857 - 0.440290 = +0.001567
```

虽然 MCC 低于 CE baseline，但本项目最终以 macro-F1 为主指标，因此选择该模型作为最终版本。

## 9. CE Regularization 补充实验

在后续探索中，还尝试了 label smoothing、class weight、focal loss 等 CE regularization。该部分不作为最终版本，但对理解任务很有帮助。

### 9.1 Label Smoothing

label smoothing 会把 one-hot 标签变得不那么极端。例如真实类别为 `Up` 时，普通 one-hot 是：

```text
[0, 1, 0]
```

`label_smoothing=0.02` 后会变成近似：

```text
[0.0067, 0.9867, 0.0067]
```

这能降低模型对 noisy boundary label 的过度自信。

### 9.2 Inverse-Sqrt Class Weight

训练集约为 `Neutral 40% / Up 30% / Down 30%`，因此可以给少数类轻微加权。但完全 balanced weight 可能太激进，所以采用更温和的 inverse-sqrt class weight：

```text
weight_c ∝ 1 / sqrt(count_c)
```

实际权重大致为：

```text
Neutral: 0.906
Up:      1.047
Down:    1.047
```

这会轻微降低 Neutral 权重、轻微提高 Up/Down 权重。

### 9.3 补充结果

较好的 CE regularization 结果为：

| Variant | Epoch | Macro-F1 | MCC |
|---|---:|---:|---:|
| smooth005 | 6 | 0.4357 | 0.1998 |
| smooth005 | 12 | 0.4410 | 0.1897 |
| smooth005_dropout02_wd5e4 | 12 | 0.4378 | 0.1930 |
| smooth002_invsqrt | 6 | 0.445608 | 0.199982 |

其中 `smooth002_invsqrt` 数值上同时超过 CE baseline 的 macro-F1 和 MCC。但由于项目最终希望围绕 supervised contrastive learning 展开，并且当前报告以 `CE + SupCon joint loss` 作为最终版本，因此该结果作为补充发现记录，而不是最终模型。

这个补充实验说明：如果未来目标转向纯分类效果最大化，`label_smoothing=0.02 + inverse_sqrt class_weight` 是非常值得继续深入的方向。

## 10. 为什么 SupCon/KNN 不如 Softmax

本项目中 KNN 效果不如 softmax classifier，主要有几个原因：

1. 金融标签由未来收益分位数生成，同标签样本不一定历史形态相似。
2. 边界样本噪声较强，SupCon 会强迫这些 noisy 同标签样本聚在一起，可能反而伤害表征。
3. KNN 依赖 embedding 的局部邻域结构，但 validation 紧接 train，memory bank 可能混入时间局部性信号，不能完全代表泛化能力。
4. Softmax classifier 可以直接学习非线性分类边界，比固定 KNN 投票更灵活。

因此，最终没有选择纯 SupCon + KNN，而是采用 CE + SupCon joint loss。

## 11. 结果解释

实验结果可以总结为：

1. 原始特征 KNN 明显不够。
2. 纯 SupCon/KNN 能学到一些结构，但低于 CE baseline。
3. CE softmax 是强 baseline。
4. CE + SupCon joint loss 在 macro-F1 上小幅超过 CE baseline。
5. SupCon 权重不能太大，`lambda=0.02` 是当前 sweep 中最合适的点。
6. 金融标签 noisy，任何强制聚类或强监督边界都容易受标签噪声影响。

从方法角度看，CE + SupCon joint loss 的价值在于它没有完全依赖对比学习，而是把 SupCon 作为轻量辅助项加入 CE 训练。这样可以保留 CE 的稳定分类能力，同时让 encoder 学到更有类别结构的 embedding。

## 12. 局限性

当前实验仍有明显限制：

1. 服务器不可用后，无法重新跑 test split 复现实验。
2. GitHub 中没有保存完整训练 log、checkpoint 和原始数据，因此报告中的部分实验细节来自当时实验记录。
3. 当前只评估分类指标，没有做交易回测。
4. 没有考虑手续费、滑点、仓位管理和信号阈值。
5. 验证集上的 F1 提升幅度较小，需要 test split 和 walk-forward validation 进一步确认。
6. MCC 不是最终模型的优势指标；若考核标准改成 MCC，应重新选择模型。

## 13. 后续工作

后续建议按以下顺序推进：

1. 恢复服务器或换新环境，重新下载/生成数据。
2. 在 test split 上重新评估 `joint_ce_supcon_F23_lam0p02`。
3. 对 `lambda` 做更细粒度 sweep，例如：

```text
0.01, 0.015, 0.02, 0.025, 0.03
```

4. 保留 CE regularization 作为备选方向，尤其是 `smooth002_invsqrt`。
5. 将分类输出转为交易信号，做手续费和滑点下的回测。
6. 使用 rolling/walk-forward validation，验证模型在不同市场 regime 下的稳定性。
7. 尝试 confidence threshold，只在模型高置信度时开仓。

## 14. 最终结论

本项目最终选择 **CE + SupCon joint loss** 作为最终模型。该方法符合项目探索 supervised contrastive learning 的方向，并在 validation macro-F1 上相较 CE softmax baseline 取得小幅提升。

最终版本如下：

```text
Model: Transformer Encoder
Objective: CE loss + 0.02 * SupCon loss
Validation Macro-F1: 0.441857
Validation MCC: 0.188430
Best epoch: 12
```

虽然补充实验中 CE regularization 得到过更高的分类数值，但本报告最终版本以 `CE + SupCon joint loss` 为主，因为它更贴合项目主题，并且已经在主要指标 macro-F1 上超过 baseline。

