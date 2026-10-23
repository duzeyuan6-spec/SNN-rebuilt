# 冻结压缩感知实验代码（2026-10-23）

本仓库根目录现为当前冻结实验实现。从仓库根目录运行所有命令。旧版原型、配置和报告已从当前版本移除，可通过 Git 历史查阅。

主模型：49 通道、11 阶段、前 8 阶段半分辨率、后 3 阶段全分辨率，块大小 32，MR=.25 时 748894 参数。主模型实现和训练源文件原样保存，摘要见 `SOURCE_MANIFEST.json`。`time_steps=8` 在该模型中表示有符号计数级数，不能表述为 8 个时序仿真步。连续 ANN 与计数 SNN 从公共 ANN teacher 初始化后分别分支训练。

## 已完成实验结果

以下表格由已归档结果生成，更新于 2026-10-23。展示范围为本仓库当前维护的方法与实验，不是所有曾运行模型的穷尽列表；原始记录另行保留。移出维护范围不等于算法无效。数据源 CSV 位于 [results](results/)，不包含显存指标。

### MR=25%：三数据集重建质量

PSNR 单位 dB，越高越好；SSIM 越高越好。每个数据集按图等权平均，样本数分别为 11、68、100。除下方三种子表外，均为 seed123 的冻结检查点，按验证集完整图 SSIM 选模型后测试。

|模型|Set11 PSNR / SSIM|BSD68 PSNR / SSIM|Urban100 PSNR / SSIM|
|---|---:|---:|---:|
|主 SNN|34.1920 / 0.952370|31.4235 / 0.916453|29.8391 / 0.932085|
|固定全分辨率 ANN|34.6145 / 0.955388|31.6405 / 0.919366|30.3285 / 0.937399|
|ISTA-Net+|33.9999 / 0.949882|31.3438 / 0.911435|29.6079 / 0.926284|
|OCTUF-CS|35.1324 / 0.956766|31.9446 / 0.918974|31.0963 / 0.941432|
|ESDNet-CS|33.7305 / 0.946818|31.2327 / 0.909598|29.4428 / 0.924556|
|SpikeRain-CS|32.7711 / 0.944735|30.6194 / 0.910541|28.5451 / 0.920001|

主 SNN 在三集上均优于本表中的 ISTA-Net+、ESDNet-CS 和 SpikeRain-CS；固定 ANN 和 OCTUF 的 PSNR 更高。不得将 CS 适配结果称为原论文复现或严格等参数比较。

### 统一效率复测

同一物理 RTX A6000，FP32、batch1、256×256，cuDNN deterministic=True，包含采样与重建，排除文件 IO 和主机传输。每项三次交替测量，每次至少20秒，并检查无其他计算进程。均值 ± 样本标准差；这是硬件技术重复，不是训练种子不确定性。板卡 J/图由采样功率估计，包含基础功耗，不是整机或神经形态硬件能耗。不能与历史非确定性测量混用。

|模型|延迟 ms/图 ↓|吞吐 图/s ↑|板卡 J/图 ↓|
|---|---:|---:|---:|
|主 SNN|45.743 ± 0.030|21.861 ± 0.014|6.192 ± 0.302|
|固定全分辨率 ANN|48.898 ± 0.175|20.451 ± 0.073|7.717 ± 0.247|
|ISTA-Net+|9.800 ± 0.034|102.039 ± 0.354|2.837 ± 0.018|
|OCTUF-CS|68.942 ± 0.145|14.505 ± 0.031|15.081 ± 0.296|
|ESDNet-CS|2155.755 ± 6.005|0.464 ± 0.001|254.324 ± 7.278|
|SpikeRain-CS|175.411 ± 0.243|5.701 ± 0.008|50.064 ± 0.715|

主 SNN 相对固定全分辨率 ANN 的延迟降低约6.5%、板卡能耗降低约19.8%；但 ISTA-Net+ 更快、更省能，不能声称普遍最快或最低能耗。当前结果未证明计数激活本身单独带来 GPU 效率收益。

### 分辨率与机制重训消融

|模型|Set11 PSNR / SSIM|BSD68 PSNR / SSIM|Urban100 PSNR / SSIM|
|---|---:|---:|---:|
|全分辨率 SNN|34.4488 / 0.954119|31.5723 / 0.918025|30.1669 / 0.935378|
|半分辨率 SNN|33.8238 / 0.949052|31.2476 / 0.913145|29.4314 / 0.926610|
|主 SNN|34.1920 / 0.952370|31.4235 / 0.916453|29.8391 / 0.932085|
|无脉冲正则|34.1865 / 0.952262|31.4213 / 0.916369|29.8269 / 0.931916|
|无量化预热|34.0793 / 0.951237|31.3727 / 0.915528|29.7414 / 0.930549|

主布局为8半分辨率+3全分辨率。两项机制消融各额外训练200轮，seed123，其余训练与采样协议保持一致。逐阶段中间输出诊断不是独立早退重训。

### 多采样率

MR=10%实际为102/1024，MR=50%为512/1024。每率使用独立teacher，同率三种方法共享其冻结采样矩阵。每新增率预算为teacher200 + SNN200 + ANN200 + OCTUF400轮。

|MR|模型|Set11 PSNR / SSIM|BSD68 PSNR / SSIM|Urban100 PSNR / SSIM|
|---|---|---:|---:|---:|
|10%|主 SNN|28.9525 / 0.890019|27.6252 / 0.819087|25.0636 / 0.829752|
|10%|固定全分辨率 ANN|29.0817 / 0.893086|27.6916 / 0.821919|25.1804 / 0.833877|
|10%|OCTUF-CS|29.7247 / 0.896612|28.0201 / 0.818779|25.7924 / 0.839485|
|50%|主 SNN|40.1168 / 0.982109|36.6011 / 0.971432|35.7858 / 0.978211|
|50%|固定全分辨率 ANN|40.5390 / 0.983267|36.9042 / 0.972818|36.5413 / 0.980583|
|50%|OCTUF-CS|40.7451 / 0.983485|37.1094 / 0.973095|37.1756 / 0.981893|

### 三个分支训练种子

种子123、124、125共用teacher初始化和采样矩阵，反映分支训练方差，不能称完整流程独立重复。不挑最好种子，以下为三次均值±样本标准差。

|数据集|PSNR dB|SSIM|
|---|---:|---:|
|Set11|34.211387 ± 0.025339|0.952397 ± 0.000048|
|BSD68|31.434761 ± 0.010047|0.916458 ± 0.000069|
|Urban100|29.853242 ± 0.014063|0.932109 ± 0.000021|

### 脉冲正则活动量

每模型全部179图，按神经元元素数加权汇总，包含块补齐区域；不是时间发放率。正则仅小幅降低活动量，并未证明稳定板卡节能。

|数据集|主SNN非零比例|无正则非零比例|主SNN平均绝对计数|无正则平均绝对计数|
|---|---:|---:|---:|---:|
|Set11|0.799547|0.809337|1.999873|2.044552|
|BSD68|0.792490|0.802540|1.840621|1.882464|
|Urban100|0.792743|0.801992|2.025527|2.062062|

训练/测试界限：360张训练、40张验证；测试不调参。公共teacher含200轮训练预算，外部解码器容量与训练预算不同。Set11曾在项目历史Hybrid开发中使用过，此项目级限制需披露。上述实验完成不代表全部论文要求已经满足。

## 安装与数据

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

根据设备安装兼容 CUDA 的 PyTorch。核心 CPU 单元测试不需要 GPU；实际训练/功率测量需要 NVIDIA GPU 和 nvidia-smi。

自行准备合法获取的数据，数据、检查点和凭据不在本次代码上传范围：

```text
data/train/Train400/       # 400 张训练源图，固定划分 360/40
data/test/Set11/           # 11 张
data/test/BSD68/           # 68 张
data/test/Urban100/        # 100 张
```

划分由灰度像素摘要排序、固定划分种子 20261013 生成，记录至 split.json；训练随机种子为 123。文件内容必须与原数据一致，才能复现原划分。禁止训练/验证与测试像素重复。检查点根据完整验证图 SSIM 选择，测试不用于调参。

## 主模型及固定全分辨率 ANN

以下为 Bash 命令。先训练公共 ANN teacher，再从同一个 best.pt 分别训练两个分支。`--seconds 0` 表示按完整轮数运行；执行前自行评估计算预算。三个任务各 200 轮。

```bash
mkdir -p runs
COMMON=(--architecture repository --train-root data/train/Train400 --test-root data/test --manifest runs/split.json --channels 49 --stages 11 --time-steps 8 --mr .25 --block-size 32 --seed 123 --epochs 200 --seconds 0 --selection-metric ssim --structure-weight .4 --full-val --stability-guard --memory-fraction .15 --structure-mode metric --intermediate-weight .05 --spatial-dilation 2 --burst-calibration rms --learnable-sensing)
python scripts/train_reconstruction_v2.py "${COMMON[@]}" --kind ann --out runs/teacher
python scripts/train_reconstruction_v2.py "${COMMON[@]}" --kind spike --out runs/spike --freeze-sensing --init-from runs/teacher/best.pt --quant-warmup 40 --spike-reg .0001 --snn-spatial-stride 2 --snn-fullres-tail 3
python scripts/train_reconstruction_v2.py "${COMMON[@]}" --kind ann --out runs/ann --freeze-sensing --init-from runs/teacher/best.pt --quant-warmup 40
```

`--memory-fraction` 只是训练显存使用上限，不是显存指标测量。完整推理是 `model(model.cs(image))`，不得省略采样操作。

```bash
ANN_SHA=$(sha256sum runs/ann/best.pt | cut -d ' ' -f 1)
for DS in Set11 BSD68 Urban100; do
  python scripts/evaluate_reconstruction_v3.py --spike runs/spike/best.pt --ann runs/ann/best.pt --fixed-ann-sha256 "$ANN_SHA" --manifest runs/split.json --split test --test-root "data/test/$DS" --out "outputs/$DS" --device cuda
done
```

训练脚本有 `--resume`；只在核对日志与检查点后显式使用。不同数据路径会改变 manifest 摘要，不能直接把旧检查点与重写后的 manifest 混用。原权重未上传，因此本仓库只提供代码复现入口，不承诺未经新训练即可得到原分数。

## 多率、分辨率、重复与消融

- 多率：将 COMMON 的 MR 改为 .10/.50，每率单独输出目录、独立 teacher 与分支，但率内共享 teacher 的冻结 Phi。实际 MR 为 102/1024、256/1024、512/1024。每新增率 teacher200 + SNN200 + ANN200 + OCTUF400，总 1000 轮。
- 分辨率：SNN 全分辨率使用 stride1/tail0；纯半分辨率 stride2/tail0；主模型 stride2/tail3。只有显式独立训练后的检查点才可作重训对照。`evaluate_resolution_stages.py` 的中间输出诊断不是独立早退训练。
- 分支重复仅 123、124、125，公共 teacher 不变，不能称为全流程独立重复。
- 无正则：seed123，`--spike-reg 0`；无量化预热：seed123，`--quant-warmup 0`；其余主模型协议不变，各增加 200 轮。
- `run_repeats_ablations.py`、`run_resolution_multirate.py`、`run_ann_comparison_queue.py` 沿用原始队列目录约定（外部方法列表已按当前范围更新），`--base` 下仍要求原实验相对目录布局；可读作精确配置参考，不能直接指向任意数据文件夹。

## 外部方法

仅上传适配器，不重新分发第三方源码。将官方源代码放入自行指定的位置，参考各 trainer 的 `--help` 传入 `--source`、teacher、manifest 等参数。保留上游许可。

|方法|官方来源|本实验变化|
|---|---|---|
|ISTA-Net+|https://github.com/jianzhangcs/ISTA-Net-PyTorch|33→32 块；Qinit 只用训练图拟合|
|OCTUF|https://github.com/songjiechong/OCTUF|冻结公共 Phi，不联合优化采样器|
|ESDNet|https://github.com/MingTian99/ESDNet|灰度 CS 前端、官方解码器；torch 后端|
|SpikeRain-M|https://github.com/tanvirnwu/SpikeRain_WACV_2025|灰度 CS 前端、官方 SSIM 损失|

ESDNet/SpikeRain 还需与上游实现匹配的 SpikingJelly（实验使用 2024 activation_based 源码）；ESDNet 不能误用缺少相应 attention 实现的旧版本。依赖没有完整锁定文件，部署前需复核版本与检查点加载；不把适配结果称为原论文复现。第三方模型保留原生容量，与主模型不是严格等参数比较。Burst 转换代码不在本次上传范围，DehazeSNN 未部署。

## 统一效率测量（不测显存）

编辑 `configs/hardware_plan.example.json` 的检查点列表，然后：

```bash
python scripts/unified_hardware.py --plan configs/hardware_plan.example.json --out outputs/hardware
```

单组在同一物理卡，FP32、batch1、256×256、确定性 cuDNN，包含采样和重建，排除 IO/主机传输；每模型单独进程，3 轮交替、每轮至少20秒。等待无其他计算进程的 GPU；竞争时整组无效并保留，不拼接片段、不停止他人任务。板卡功率积分估计 J/图包含基础功耗，不是整机/神经形态能耗。输出吞吐与延迟、原始功率；没有显存统计。三次是技术重复，不是训练不确定性。不同确定性设置的历史测量不能合并。

外部检查点保存的 source 路径必须在当前机器有效；单模型 worker 可通过 `SNN_EXTERNAL_SOURCE` 覆盖上游源码位置。不同来源模型混合队列需分别确保各检查点内的来源路径有效，不能对混合队列统一覆盖成同一个源码目录。只加载可信检查点（torch pickle）。

## 正则活动量

```bash
python scripts/analyze_activity.py --main runs/spike/best.pt --no-regularizer runs/no_regularizer/best.pt --manifest runs/split.json --test-root data/test --out outputs/activity
```

全部179图逐层非零比例、绝对计数、归一化计数、饱和比例；活动量包含块补齐区域，质量只算原图。活动量 hook 不参与计时。活动量降低不能直接推断稠密 GPU 卷积节能。

## 验证与结论边界

```bash
python -m compileall -q src scripts
python scripts/test_repository_reconstruction.py
python scripts/test_metric_structure_loss.py
python scripts/test_multirate_extension.py --source /path/to/baselines
```

原项目历史 Hybrid 开发使用过 Set11，应披露此限制。所有有效结果保留，无胜负筛选。本代码包不包含数据集、权重、运行日志、自动监测状态或服务器信息；旧版报告仅保留于 Git 历史，不代表当前协议。当前证据不支持“普遍最快/最低能耗”，也不证明计数激活单独造成 GPU 效率提升。
