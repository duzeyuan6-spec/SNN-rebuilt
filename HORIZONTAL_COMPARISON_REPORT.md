# 横向对比实验报告：3 个同参数量模型 vs SpikeCS

## 实验目标

在同一任务、同一数据入口、同一训练/评估代码下，选择 3 个非 SNN 模型与 SpikeCS 做横向对比。要求：

- 所有模型能在当前项目环境中运行。
- 与 SpikeCS 参数量差距不超过 10%。
- 指标完整，不缺 PSNR、SSIM、Params、FR、Energy。
- 使用 MR=0.25，测试集为 Set11、BSD68、Urban100。

## 数据与环境

- 服务器 GPU：NVIDIA GeForce RTX 4090。
- 训练集：`data/train/Train400`，链接到 `baseline_data/images/Train400`，共 400 张。
- 测试集：`data/test/Set11`、`data/test/BSD68`、`data/test/Urban100`，分别 11、68、100 张。
- 统一训练脚本：`python scripts/train.py --config <config>`。
- 统一评估脚本：`python scripts/collect_results.py ...`。

## 模型选择

SpikeCS 主模型参数量为 `0.095540M`，因此允许范围为约 `0.085986M` 到 `0.105094M`。

| Method | Config / Checkpoint | Params(M) | 与 SpikeCS 差距 | 是否满足 10% |
|---|---|---:|---:|---|
| SpikeCS | `configs/spikecs_mr025.yaml` / `runs/spikecs_mr025/best.pt` | 0.095540 | 0.00% | reference |
| ANN-Ours | `configs/ann_mr025.yaml` / `runs/ann_mr025/best.pt` | 0.095530 | 0.01% | yes |
| ISTA-Prox | `configs/istaprox_mr025_c32.yaml` / `runs/istaprox_mr025_c32/best.pt` | 0.095540 | 0.00% | yes |
| CSNet-style-C36B8 | `configs/csnet_mr025_c36_b8.yaml` / `runs/csnet_mr025_c36_b8/best.pt` | 0.094295 | 1.30% | yes |

没有选择默认 `configs/csnet_mr025.yaml`，因为它约 `0.056M`，与 SpikeCS 差距超过 10%。没有选择 `ReconNet`，因为它约 `0.309M`，同样超过 10%。

## 新增并训练的模型

新增配置：

- `configs/istaprox_mr025_c32.yaml`
- `configs/csnet_mr025_c36_b8.yaml`

训练日志：

- `results/logs/horizontal_istaprox_mr025_c32_train.log`
- `results/logs/horizontal_csnet_mr025_c36_b8_train.log`

训练输出：

- `runs/istaprox_mr025_c32/best.pt`
- `runs/csnet_mr025_c36_b8/best.pt`

两个新增模型均跑满 200 epoch。

## 统一评估命令

```bash
python scripts/collect_results.py \
  --data-root data/test \
  --sets Set11 BSD68 Urban100 \
  --checkpoint ANN-Ours:0.25:runs/ann_mr025/best.pt \
  --checkpoint ISTA-Prox:0.25:runs/istaprox_mr025_c32/best.pt \
  --checkpoint CSNet-style-C36B8:0.25:runs/csnet_mr025_c36_b8/best.pt \
  --checkpoint SpikeCS:0.25:runs/spikecs_mr025/best.pt \
  --out-csv results/horizontal_three_models_vs_snn_mr025.csv \
  --out-md results/horizontal_three_models_vs_snn_mr025.md \
  --device cuda
```

完整性检查：

```bash
python scripts/check_table_completeness.py --csv results/horizontal_three_models_vs_snn_mr025.csv
```

检查结果：`complete`。

## 横向对比结果

| Method | MR | Set | PSNR | SSIM | Params(M) | FR | Energy(uJ) | Source |
|---|---:|---|---:|---:|---:|---:|---:|---|
| ANN-Ours | 0.25 | BSD68 | 21.31 | 0.4726 | 0.096 | 0.0000 | 4284.840 | runs/ann_mr025/best.pt |
| CSNet-style-C36B8 | 0.25 | BSD68 | 20.84 | 0.4184 | 0.094 | 0.0000 | 4236.149 | runs/csnet_mr025_c36_b8/best.pt |
| ISTA-Prox | 0.25 | BSD68 | 22.53 | 0.5476 | 0.096 | 0.0000 | 4284.840 | runs/istaprox_mr025_c32/best.pt |
| SpikeCS | 0.25 | BSD68 | 22.32 | 0.4775 | 0.096 | 0.1755 | 1161.708 | runs/spikecs_mr025/best.pt |
| ANN-Ours | 0.25 | Set11 | 19.81 | 0.4817 | 0.096 | 0.0000 | 4284.840 | runs/ann_mr025/best.pt |
| CSNet-style-C36B8 | 0.25 | Set11 | 19.39 | 0.4146 | 0.094 | 0.0000 | 4236.149 | runs/csnet_mr025_c36_b8/best.pt |
| ISTA-Prox | 0.25 | Set11 | 21.35 | 0.5766 | 0.096 | 0.0000 | 4284.840 | runs/istaprox_mr025_c32/best.pt |
| SpikeCS | 0.25 | Set11 | 21.34 | 0.5101 | 0.096 | 0.1755 | 1161.708 | runs/spikecs_mr025/best.pt |
| ANN-Ours | 0.25 | Urban100 | 18.05 | 0.3671 | 0.096 | 0.0000 | 4284.840 | runs/ann_mr025/best.pt |
| CSNet-style-C36B8 | 0.25 | Urban100 | 17.95 | 0.3409 | 0.094 | 0.0000 | 4236.149 | runs/csnet_mr025_c36_b8/best.pt |
| ISTA-Prox | 0.25 | Urban100 | 19.36 | 0.4766 | 0.096 | 0.0000 | 4284.840 | runs/istaprox_mr025_c32/best.pt |
| SpikeCS | 0.25 | Urban100 | 19.32 | 0.4245 | 0.096 | 0.1755 | 1161.708 | runs/spikecs_mr025/best.pt |

## 结论

- 参数公平性满足要求：三个横向模型与 SpikeCS 参数量差距最大约 1.30%，低于 10%。
- 指标完整性满足要求：所有行都有 PSNR、SSIM、Params(M)、FR、Energy(uJ)、Source。
- SpikeCS 的能耗为 `1161.708 uJ`，显著低于三个非 SNN 模型约 `4236-4285 uJ`。
- 在重建质量上，ISTA-Prox 在三个测试集上略高于 SpikeCS；SpikeCS 高于 ANN-Ours 和参数匹配 CSNet-style。
