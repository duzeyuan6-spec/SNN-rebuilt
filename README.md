# SpikeCS First Experiment

Minimal code for the first experiment:

> Spiking unrolled network for grayscale image compressed sensing reconstruction.

The default setting is:

- Task: block-based grayscale image CS reconstruction
- Measurement rate: `MR=0.25`
- Block size: `33`
- Unrolled stages: `K=5`
- Spiking time steps: `T=4`
- Train: `Train400` or any folder of natural images
- Test: `Set11`, `BSD68`, `Urban100`

## 1. Environment

Create an environment:

```bash
conda create -n spikecs python=3.10 -y
conda activate spikecs
pip install -r requirements.txt
```

If you have CUDA, install the matching PyTorch build from:

```text
https://pytorch.org/get-started/locally/
```

## 2. Dataset

Use this folder layout:

```text
data/
  train/
    Train400/
      001.png
      ...
  test/
    Set11/
      barbara.png
      ...
    BSD68/
      ...
    Urban100/
      ...
```

Recommended datasets:

- Train400: common training set used by ISTA-Net style CS reconstruction papers.
- Set11: small classic CS test set.
- BSD68: standard grayscale restoration test set.
- Urban100: harder urban structure images.
- Optional training replacement: DIV2K train images.

The code accepts `.png`, `.jpg`, `.jpeg`, `.bmp`, and `.tif`.

## 3. Train ANN Baseline

```bash
python scripts/train.py --config configs/ann_mr025.yaml
```

## 4. Train SpikeCS

```bash
python scripts/train.py --config configs/spikecs_mr025.yaml
```

## 5. Evaluate

```bash
python scripts/eval.py --checkpoint runs/spikecs_mr025/best.pt --data-root data/test --sets Set11 BSD68 Urban100
```

## 6. First Paper Table

For the first experiment, report:

- PSNR / SSIM on Set11, BSD68, Urban100
- Params
- MACs for ANN
- ACs/SOPs for SNN
- Average firing rate
- Estimated energy

Use `ANN-Ours` and `SpikeCS` as the first internal comparison before adding external baselines.

