# 冻结压缩感知实验代码（2026-10-23）

此目录独立于仓库根目录的历史原型。从本目录运行所有命令。

主模型：49 通道、11 阶段、前 8 阶段半分辨率、后 3 阶段全分辨率，块大小 32，MR=.25 时 748894 参数。主模型实现和训练源文件原样保存，摘要见 `SOURCE_MANIFEST.json`。`time_steps=8` 在该模型中表示有符号计数级数，不能表述为 8 个时序仿真步。连续 ANN 与计数 SNN 从公共 ANN teacher 初始化后分别分支训练，不是 SNN 转 ANN。

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
- `run_repeats_ablations.py`、`run_resolution_multirate.py`、`run_ann_comparison_queue.py` 是原始队列脚本，`--base` 下仍要求原实验相对目录布局；可读作精确配置参考，不能直接指向任意数据文件夹。

此次主发布目录不纳入新增同布局连续 ANN 实验。其原始证据在本地归档保留，删除正文展示并不证明其无效；不能据此声称计数激活自身已证明 GPU 节能。

## 外部方法

仅上传适配器，不重新分发第三方源码。将官方源代码放入自行指定的位置，参考各 trainer 的 `--help` 传入 `--source`、teacher、manifest 等参数。保留上游许可。

|方法|官方来源|本实验变化|
|---|---|---|
|ISTA-Net+|https://github.com/jianzhangcs/ISTA-Net-PyTorch|33→32 块；Qinit 只用训练图拟合|
|ISTA-Net++|https://github.com/jianzhangcs/ISTA-Netpp|单 MR=.25 条件|
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

原项目历史 Hybrid 开发使用过 Set11，应披露此限制。所有有效结果保留，无胜负筛选。本代码包不包含数据、权重、运行日志、自动监测状态或服务器信息；历史根目录报告不代表本目录最新协议。当前证据不支持“普遍最快/最低能耗”，也不证明计数激活单独造成 GPU 效率提升。
