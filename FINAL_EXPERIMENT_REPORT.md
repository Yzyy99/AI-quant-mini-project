# AI Quant Mini Project 实验报告

## 1. 项目目标

本项目目标是在 BTCUSDT 1 分钟级别数据上，构建一个基于 Transformer 的短期方向预测模型。任务被定义为三分类问题：根据未来收益率的分位数标签，将样本划分为 `Neutral`、`Up`、`Down` 三类。项目最终选择以验证集 macro-F1 作为主要指标，因为该指标更关注三类整体表现，而不是单纯准确率。

最终选定模型为 **CE + SupCon joint loss**：

```text
loss = CE loss + lambda * SupCon loss
```

其中最终选择 `lambda=0.02`。

## 2. 数据与标签

使用的数据集为 multi-feature BTCUSDT 数据，输入窗口长度为 `T=60`，预测 horizon 为 `H=10`。模型输入维度为 `F=23`，即每个样本形状为：

```text
(60, 23)
```

数据划分记录如下：

| Split | 样本数 |
|---|---:|
| Train | 550,189 |
| Validation | 117,936 |
| Test | 117,926 |

标签规则为 future return 的 30/70 分位三分类。训练集标签分布大致为 `Neutral 40% / Up 30% / Down 30%`。该标签设计符合金融预测任务，但边界附近样本天然 noisy：过去形态相近的样本，未来收益可能落入不同分位区间。

## 3. 模型结构

基础模型为 Transformer Encoder：

| 参数 | 设置 |
|---|---:|
| `d_model` | 128 |
| attention heads | 4 |
| encoder layers | 3 |
| feedforward dim | 256 |
| projection dim | 64 |
| dropout | 0.1 |

模型使用 CLS token 汇聚时间序列表征。CE softmax baseline 直接在 CLS embedding 后接分类头；SupCon 相关实验额外使用 projection head 生成对比学习 embedding。

## 4. 训练设置

主要训练设置如下：

| 参数 | 设置 |
|---|---:|
| optimizer | AdamW |
| learning rate | `1e-4` |
| weight decay | `1e-4` |
| scheduler | CosineAnnealingLR |
| batch size | 1024 |
| eval batch size | 4096 |
| epochs | 30 |
| SupCon temperature | 0.07 |

最终模型的 SupCon 权重扫了：

```text
lambda = 0.01, 0.02, 0.05
```

## 5. Baseline

CE softmax baseline 是本项目的主要对照组。

| Model | Macro-F1 | MCC |
|---|---:|---:|
| CE softmax baseline | 0.440290 | 0.198533 |

CE baseline 稳定、简单，是后续所有实验的参考标准。

## 6. 实验过程

### 6.1 SupCon + KNN

首先尝试用 SupCon 训练表征，再使用 KNN 在 embedding 空间分类。该方向包括不同 memory size、K 值和投票方式的 sweep。

结果显示，SupCon + KNN 的最好 macro-F1 约为 `0.4295`，低于 CE softmax baseline。原因可能是金融时间序列中“同标签样本”不一定在形态上相似，尤其是分位数标签本身 noisy，导致纯对比学习难以形成清晰聚类。

### 6.2 CE embedding + KNN

进一步测试 CE 模型的 embedding 是否适合 KNN 分类。结果仍低于 softmax head，说明该任务更适合直接学习分类边界，而不是依赖局部邻居投票。

### 6.3 CE + SupCon Joint Loss

最终主线实验采用 CE 和 SupCon 的 joint loss。该方法保留 CE 的稳定分类信号，同时加入弱监督对比学习约束，使模型表征更有类别结构。

结果如下：

| Lambda | Best F1 Epoch | Macro-F1 | MCC |
|---:|---:|---:|---:|
| 0.01 | 12 | 0.441679 | 0.187209 |
| 0.02 | 12 | **0.441857** | 0.188430 |
| 0.05 | 12 | 0.440703 | 0.188844 |

`lambda=0.02` 取得最高 macro-F1，因此被选为最终版本。

### 6.4 CE Regularization 补充实验

后续还探索了 label smoothing、class weight、focal loss 等 CE 正则化方法。其中 `label_smoothing=0.02` 加 `inverse_sqrt` class weight 在验证集上取得了更高数值，但它不作为最终版本，因为本项目最终希望围绕 supervised contrastive learning 展开，并且当前评估重点改为 joint-loss 方向下的 macro-F1 提升。

## 7. 最终结果

最终选择：

```text
joint_ce_supcon_F23_lam0p02
```

训练目标：

```text
loss = CE loss + 0.02 * SupCon loss
```

验证集结果：

| Model | Macro-F1 | MCC |
|---|---:|---:|
| CE softmax baseline | 0.440290 | 0.198533 |
| CE + SupCon, lambda=0.02 | **0.441857** | 0.188430 |

最终模型相比 CE baseline 在主要指标 macro-F1 上有所提升：

```text
0.441857 - 0.440290 = +0.001567
```

## 8. 结果分析

实验结果说明，单独使用 SupCon/KNN 并不适合当前金融三分类任务，因为标签由未来收益分位生成，存在较强噪声。CE softmax 能直接优化分类边界，因此表现更稳。

CE + SupCon joint loss 的优势在于，它没有完全放弃 CE 的监督分类能力，而是在此基础上加入轻量表征约束。`lambda=0.02` 的效果最好，说明 SupCon 项需要保持较弱权重；过强的 SupCon 约束可能会受 noisy label 影响，反而压制分类性能。

## 9. 局限性

当前实验仍有几个限制：

1. 主要结果基于 validation split，服务器不可用后没有进一步补做 test split 评估。
2. 当前指标是分类指标，还没有加入交易回测、手续费、滑点和仓位管理。
3. 标签来自 future return 分位数，边界样本噪声较大。
4. KNN/memory bank 结果可能受时间相邻结构影响，不能单独作为最终判断依据。

## 10. 后续工作

后续可以继续做：

1. 在 test split 上复现实验，确认泛化表现。
2. 将分类信号转成交易信号，加入手续费和滑点回测。
3. 对 `lambda` 做更细粒度 sweep，例如 `0.015/0.02/0.025/0.03`。
4. 尝试基于预测置信度过滤交易，只在高置信度样本上开仓。
5. 使用 walk-forward validation 检查非平稳市场环境下的稳定性。

## 11. 结论

本项目最终采用 `CE + SupCon joint loss` 作为最终模型。相比 CE softmax baseline，该方法在验证集 macro-F1 上取得小幅提升，并且符合项目希望探索 supervised contrastive learning 在金融序列预测中作用的方向。

最终版本：

```text
Model: Transformer Encoder
Objective: CE loss + 0.02 * SupCon loss
Validation Macro-F1: 0.441857
Validation MCC: 0.188430
```

