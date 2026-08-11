# SpikeCS 消融实验建议

## 结论先行

当前项目的核心主张不是单纯 PSNR 最优，而是 **SpikeCS 在可接受重建质量下，用更低脉冲活动和能耗完成压缩感知重建**。因此消融实验应围绕三件事展开：

1. 脉冲化是否带来能耗收益，而不是只是换了激活函数。
2. 时间步、阈值、衰减和 spike regularization 如何形成 PSNR/SSIM/FR/Energy 权衡。
3. 网络容量和测量率变化时，SpikeCS 的收益是否稳定。

不要把主要篇幅放在与外部 SOTA 的 PSNR 对比上。已有结果中 ISTA-Net++、PLV-CSNet 的 PSNR 明显高于当前实现，而 SpikeCS 的优势主要在参数量和估算能耗；消融更适合证明设计选择和效率趋势。

## 已读项目依据

- 主模型在 `src/models.py`：
  - `UnrolledCSNet` 使用 `stages=5` 的展开式 CS 重建。
  - `ANN-Ours` 使用 `AnnPrior`。
  - `SpikeCS` 使用 `SpikingPrior`，每阶段包含输入卷积、若干隐藏卷积、LIF 节点和膜电位 readout。
  - SpikeCS 关键超参：`time_steps`、`threshold`、`decay`、`channels`、`blocks_per_stage`、`stages`。
- 脉冲节点在 `src/spiking.py`：
  - LIF 更新为 `mem = decay * mem + x`。
  - 超阈值产生 spike，并从膜电位减去 threshold。
  - 使用 surrogate gradient，反传窗口为 `abs(x) < 1.0`。
- 能耗估计在 `src/energy.py`：
  - ANN MAC 能耗：`E_MAC_PJ = 4.6`。
  - SNN AC 能耗：`E_AC_PJ = 0.9`。
  - SpikeCS 报告 `acs = macs * firing_rate` 和 `energy = acs * E_AC_PJ`。
- 已有 MR=0.25 结果：
  - `channels=32`：SpikeCS 在 Set11/BSD68/Urban100 的 PSNR 高于 ANN-Ours，能耗约 `1161.708 uJ`，ANN 约 `4284.840 uJ`。
  - `channels=80`：ANN-Ours PSNR 更高，SpikeCS 能耗约 `8245.167 uJ`，ANN 约 `21034.671 uJ`。
  - 说明 SpikeCS 的消融重点应报告质量差距、能耗下降比例和 firing rate，而不是只看 PSNR。

## 必做消融

### 1. ANN prior vs Spiking prior

目的：证明收益来自脉冲 prior，而不是展开结构或参数量本身。

对照：

| 组别 | 配置 | 需要固定 |
|---|---|---|
| ANN-Ours | `configs/ann_mr025.yaml` | MR=0.25, stages=5, channels=32, blocks=2 |
| SpikeCS | `configs/spikecs_mr025.yaml` | 同上，替换为 LIF spiking prior |

报告：

- Set11 / BSD68 / Urban100 的 PSNR、SSIM。
- Params(M)、FR、Energy(uJ)。
- Energy reduction：`1 - SpikeCS_energy / ANN_energy`。
- 每个数据集 PSNR 差值：`SpikeCS - ANN`。

已有结果已经能支持这一项，但建议重新用统一脚本生成表：

```bash
python scripts/run_fair_table.py \
  --mr 0.25 \
  --reconnet runs/reconnet_mr025/best.pt \
  --csnet runs/csnet_mr025/best.pt \
  --ann runs/ann_mr025/best.pt \
  --spike runs/spikecs_mr025/best.pt \
  --out-csv results/ablation_ann_vs_spike_mr025.csv \
  --out-md results/ablation_ann_vs_spike_mr025.md
```

### 2. 时间步 T 消融

目的：证明增加时间步是否提升重建质量，以及能耗/FR 如何变化。

建议设置：

| 实验名 | time_steps | threshold | decay | spike_reg | 说明 |
|---|---:|---:|---:|---:|---|
| SpikeCS-T3 | 3 | 0.45 | 0.7 | 0.0005 | 极低时延 |
| SpikeCS-T4 | 4 | 0.45 | 0.7 | 0.0005 | 低时延 |
| SpikeCS-T6 | 6 | 0.45 | 0.7 | 0.0005 | 中等时延 |
| SpikeCS-T8 | 8 | 0.45 | 0.7 | 0.0005 | 当前主模型 |

注意：现有 `configs/spikecs_mr025_t6.yaml` 同时改了 `epochs=60`、`lr=3e-5`、`spike_reg=5e-5`，不能作为严格 T 消融。严格消融必须只改 `model.time_steps`，训练轮数和学习率保持一致。

建议新建：

- `configs/abl_spikecs_mr025_t3.yaml`
- `configs/abl_spikecs_mr025_t4.yaml`
- `configs/abl_spikecs_mr025_t6.yaml`
- `configs/abl_spikecs_mr025_t8.yaml`

每个配置只改：

```yaml
run_name: abl_spikecs_mr025_t{T}
model:
  time_steps: {T}
train:
  save_dir: runs/abl_spikecs_mr025_t{T}
```

其余保持 `configs/spikecs_mr025.yaml` 一致。

### 3. spike regularization 强度消融

目的：证明 `train.spike_reg` 能控制 firing rate 和能耗，并量化质量损失。

建议设置：

| 实验名 | spike_reg | 预期 |
|---|---:|---|
| no-reg | 0 | PSNR 可能最高，FR/Energy 最高 |
| weak-reg | 0.0001 | 轻度省能 |
| default-reg | 0.0005 | 当前主模型 |
| strong-reg | 0.001 | 更省能，可能损失 PSNR |
| very-strong-reg | 0.002 | 低能耗边界 |

严格要求：

- 固定 `T=8, threshold=0.45, decay=0.7, stages=5, channels=32`。
- 不要同时改 threshold 或 T。
- 画一张 PSNR vs Energy 或 PSNR vs FR 曲线，比单表更有说服力。

### 4. LIF 阈值 threshold 消融

目的：证明阈值影响 spike 稀疏性，是 SpikeCS 能耗控制的关键。

建议设置：

| threshold | 预期 |
|---:|---|
| 0.35 | firing rate 高，质量可能高，能耗高 |
| 0.45 | 当前主配置 |
| 0.55 | 更稀疏 |
| 0.65 | 低能耗 |
| 0.75 | 极低能耗，可能明显掉点 |

严格要求：

- 固定 `T=8, decay=0.7, spike_reg=0.0005`。
- 不要直接使用 `spikecs_mr025_lowenergy.yaml` 或 `spikecs_mr025_ultralow.yaml` 当阈值消融，因为它们还改了 T、decay 和 spike_reg。

### 5. LIF 衰减 decay 消融

目的：判断膜电位记忆长度对重建质量和脉冲率的影响。

建议设置：

| decay | 预期 |
|---:|---|
| 0.3 | 记忆短，可能 FR 低但质量差 |
| 0.5 | 中等记忆 |
| 0.7 | 当前主配置 |
| 0.9 | 记忆长，可能质量高但 FR 高 |

严格要求：

- 固定 `T=8, threshold=0.45, spike_reg=0.0005`。
- 如果 `decay=0.9` 训练不稳定，保留日志并报告为稳定性边界，不要悄悄删掉。

## 建议做的消融

### 6. 网络容量消融：channels / stages

目的：说明 SpikeCS 的效率优势在小模型和宽模型下是否都成立。

已有可用组：

| 组别 | 配置 | 参数量 |
|---|---|---:|
| SpikeCS-C32-S5 | `configs/spikecs_mr025.yaml` | 约 0.096M |
| ANN-C32-S5 | `configs/ann_mr025.yaml` | 约 0.096M |
| SpikeCS-C80-S4 | `configs/spikecs_mr025_c80.yaml` | 约 0.468M |
| ANN-C80-S4 | `configs/ann_mr025_c80.yaml` | 约 0.468M |

建议补充：

| 实验名 | channels | stages | 用途 |
|---|---:|---:|---|
| C16-S5 | 16 | 5 | 小模型下是否仍有效 |
| C32-S3 | 32 | 3 | 更少展开阶段 |
| C32-S7 | 32 | 7 | 更多展开阶段 |

报告时将 ANN 和 SpikeCS 分别配对，不要把 C32 和 C80 混在一起直接比较。

### 7. 测量率 MR 消融

目的：证明 SpikeCS 在不同压缩率下是否稳定。

已有配置：

- `configs/spikecs_mr010.yaml`
- `configs/spikecs_mr025.yaml`
- `configs/spikecs_mr050.yaml`
- `configs/ann_mr010.yaml`
- `configs/ann_mr025.yaml`
- `configs/ann_mr050.yaml`

建议报告：

| MR | 对比 |
|---:|---|
| 0.10 | 高压缩，重建更难 |
| 0.25 | 主实验 |
| 0.50 | 低压缩，质量上限更高 |

每个 MR 都报告 ANN vs SpikeCS 的 PSNR、SSIM、FR、Energy。重点看能耗下降比例是否稳定。

### 8. 低能耗组合配置消融

目的：展示实际可选 operating points，而不是单因素因果。

已有组合配置：

- `configs/spikecs_mr025_lowenergy.yaml`
- `configs/spikecs_mr025_balanced_energy.yaml`
- `configs/spikecs_mr025_ultralow.yaml`

这些不是严格单因素消融，因为同时改了 T、threshold、decay、spike_reg。建议把它们单独命名为 **operating point study**：

| 组别 | T | threshold | decay | spike_reg |
|---|---:|---:|---:|---:|
| quality | 8 | 0.45 | 0.7 | 0.0005 |
| balanced | 6 | 0.52 | 0.7 | 0.0002 |
| low-energy | 4 | 0.65 | 0.6 | 0.002 |
| ultra-low | 3 | 0.75 | 0.55 | 0.004 |

报告建议：

- 用一张表展示 PSNR/SSIM/FR/Energy。
- 用一张散点图展示 Energy vs PSNR。
- 论文里不要把它写成“证明 threshold 的作用”，而要写成“不同能耗预算下的配置选择”。

## 可选消融

### 9. ISTA proximal prior 消融

代码里已有 `IstaProx`，但当前配置列表没有对应 YAML。可以增加：

```yaml
model:
  kind: ista
  stages: 5
  channels: 32
  blocks_per_stage: 2
```

目的：

- 区分普通 CNN prior、soft-threshold prior、spiking prior 的影响。
- 作为“展开网络内部 prior 类型”的结构消融。

优先级低于 T、spike_reg、threshold，因为当前主张更偏 SNN efficiency。

### 10. surrogate gradient 形状消融

当前 surrogate gradient 是矩形窗口：

```python
mask = (x.abs() < 1.0).to(x.dtype)
```

可以比较：

- window=0.5
- window=1.0
- window=2.0

但这需要改代码并保证实现可复现。只有在前面消融已经完成后再做。

## 不建议优先做

- 不建议优先做大量外部 SOTA 对比消融。当前 SpikeCS PSNR 与 ISTA-Net++、PLV-CSNet 差距较大，外部对比更适合作为背景表，而不是消融主线。
- 不建议把 `lowenergy`、`balanced_energy`、`ultralow` 直接当成单因素消融，因为它们同时改了多个变量。
- 不建议只在 Set11 上做消融。Set11 太小，至少同时报告 BSD68；如果时间允许，加 Urban100。
- 不建议只报告能耗，不报告 FR。当前 SNN 能耗估计直接依赖 firing rate，必须一起给出。

## 推荐实验顺序

1. 先整理已有 ANN vs SpikeCS、C32 vs C80、MR=0.10/0.25/0.50 结果。
2. 新跑严格 T 消融：T=3/4/6/8。
3. 新跑 spike_reg 消融：0/0.0001/0.0005/0.001/0.002。
4. 新跑 threshold 消融：0.35/0.45/0.55/0.65/0.75。
5. 新跑 decay 消融：0.3/0.5/0.7/0.9。
6. 最后整理 operating point study：quality/balanced/low-energy/ultra-low。

## 统一评估指标

每个实验都用同一套指标：

| 指标 | 来源 |
|---|---|
| PSNR | `scripts/eval.py` / `scripts/collect_results.py` |
| SSIM | `scripts/eval.py` / `scripts/collect_results.py` |
| Params(M) | `src.utils.count_params` |
| MACs / ACs | `src.energy.estimate_ops` |
| FR | `model.firing_rate()` |
| Energy(uJ) | `src.energy.estimate_ops` |

建议额外计算：

```text
Energy reduction vs ANN = 1 - Energy(SpikeCS) / Energy(ANN)
PSNR delta vs ANN = PSNR(SpikeCS) - PSNR(ANN)
```

## 表格命名建议

建议输出以下文件：

- `results/ablation_ann_vs_spike_mr025.md`
- `results/ablation_time_steps_mr025.md`
- `results/ablation_spike_reg_mr025.md`
- `results/ablation_threshold_mr025.md`
- `results/ablation_decay_mr025.md`
- `results/ablation_capacity_mr025.md`
- `results/ablation_measurement_rate.md`
- `results/operating_points_mr025.md`

## 论文写法建议

可以把消融章节组织成：

1. **Effect of Spiking Prior**：ANN-Ours vs SpikeCS，同参数、同 MR。
2. **Temporal Dynamics**：T 和 decay 消融，说明时序积分对重建的影响。
3. **Sparsity-Energy Trade-off**：threshold 和 spike_reg 消融，说明 FR/Energy 可控。
4. **Capacity and Measurement Rate**：channels/stages 和 MR 消融，说明结论是否稳健。
5. **Operating Points**：quality/balanced/low-energy/ultra-low，给出实际部署选择。

核心论点应写成：SpikeCS 通过稀疏脉冲活动降低估算能耗，在部分配置下保持或接近 ANN 重建质量；当容量增大时，ANN 的质量优势增强，但 SpikeCS 仍保持显著能耗优势。
