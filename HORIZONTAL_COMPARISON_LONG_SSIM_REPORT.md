# Horizontal Comparison After Long SSIM-Oriented Training

This report records the extended horizontal comparison requested for SpikeCS at MR=0.25. The comparison uses three non-SNN models that run in the same code environment and keep parameter counts within 10% of SpikeCS.

## Protocol

- Task: image compressed sensing reconstruction, MR=0.25.
- Environment: same repository, same data layout, same evaluation script, same GPU runtime.
- Training target: extended training up to 1000 epochs, saving the checkpoint with the best validation Set11 SSIM as best_ssim.pt.
- Early-stop settings: patience 150, min_delta 0.0005. All four runs reached the 1000 epoch cap.
- Evaluation sets: Set11, BSD68, Urban100.
- Required metrics checked: PSNR, SSIM, Params(M), FR, Energy(uJ), checkpoint source. The result table passed completeness checking.

## Compared Models

| Method | Checkpoint | Params(M) | Param delta vs SpikeCS |
|---|---|---:|---:|
| SpikeCS-LongSSIM | runs/spikecs_mr025_ssim_ext/best_ssim.pt | 0.096 | reference |
| ANN-Ours-LongSSIM | runs/ann_mr025_ssim_ext/best_ssim.pt | 0.096 | within 1% |
| ISTA-Prox-LongSSIM | runs/istaprox_mr025_c32_ssim_ext/best_ssim.pt | 0.096 | within 1% |
| CSNet-style-C36B8-LongSSIM | runs/csnet_mr025_c36_b8_ssim_ext/best_ssim.pt | 0.094 | within 2% |

Note: the earlier runs/ann_mr025/best.pt was found to contain an MR=0.10 configuration, so ANN-Ours was retrained from scratch with configs/ann_mr025.yaml for this long-training comparison.

## Best Validation SSIM During Training

| Method | Best epoch | Set11 validation PSNR | Set11 validation SSIM | Last epoch |
|---|---:|---:|---:|---:|
| ANN-Ours-LongSSIM | 997 | 27.15 | 0.8276 | 1000 |
| ISTA-Prox-LongSSIM | 999 | 26.40 | 0.8058 | 1000 |
| CSNet-style-C36B8-LongSSIM | 995 | 22.71 | 0.6500 | 1000 |
| SpikeCS-LongSSIM | 980 | 27.30 | 0.8277 | 1000 |

## Final Test Results

| Method | MR | Set | PSNR | SSIM | Params(M) | FR | Energy(uJ) | Source |
|---|---:|---|---:|---:|---:|---:|---:|---|
| ANN-Ours-LongSSIM | 0.25 | BSD68 | 26.57 | 0.7674 | 0.096 | 0.0000 | 4284.840 | runs/ann_mr025_ssim_ext/best_ssim.pt |
| CSNet-style-C36B8-LongSSIM | 0.25 | BSD68 | 23.57 | 0.6199 | 0.094 | 0.0000 | 4236.149 | runs/csnet_mr025_c36_b8_ssim_ext/best_ssim.pt |
| ISTA-Prox-LongSSIM | 0.25 | BSD68 | 26.15 | 0.7478 | 0.096 | 0.0000 | 4284.840 | runs/istaprox_mr025_c32_ssim_ext/best_ssim.pt |
| SpikeCS-LongSSIM | 0.25 | BSD68 | 26.64 | 0.7646 | 0.096 | 0.2573 | 1702.701 | runs/spikecs_mr025_ssim_ext/best_ssim.pt |
| ANN-Ours-LongSSIM | 0.25 | Set11 | 27.11 | 0.8264 | 0.096 | 0.0000 | 4284.840 | runs/ann_mr025_ssim_ext/best_ssim.pt |
| CSNet-style-C36B8-LongSSIM | 0.25 | Set11 | 22.71 | 0.6500 | 0.094 | 0.0000 | 4236.149 | runs/csnet_mr025_c36_b8_ssim_ext/best_ssim.pt |
| ISTA-Prox-LongSSIM | 0.25 | Set11 | 26.40 | 0.8053 | 0.096 | 0.0000 | 4284.840 | runs/istaprox_mr025_c32_ssim_ext/best_ssim.pt |
| SpikeCS-LongSSIM | 0.25 | Set11 | 27.30 | 0.8277 | 0.096 | 0.2573 | 1702.701 | runs/spikecs_mr025_ssim_ext/best_ssim.pt |
| ANN-Ours-LongSSIM | 0.25 | Urban100 | 23.56 | 0.7487 | 0.096 | 0.0000 | 4284.840 | runs/ann_mr025_ssim_ext/best_ssim.pt |
| CSNet-style-C36B8-LongSSIM | 0.25 | Urban100 | 20.15 | 0.5392 | 0.094 | 0.0000 | 4236.149 | runs/csnet_mr025_c36_b8_ssim_ext/best_ssim.pt |
| ISTA-Prox-LongSSIM | 0.25 | Urban100 | 23.02 | 0.7222 | 0.096 | 0.0000 | 4284.840 | runs/istaprox_mr025_c32_ssim_ext/best_ssim.pt |
| SpikeCS-LongSSIM | 0.25 | Urban100 | 23.62 | 0.7467 | 0.096 | 0.2573 | 1702.701 | runs/spikecs_mr025_ssim_ext/best_ssim.pt |

## Figures

The updated comparison figures are saved in:

results/figures/horizontal_three_models_vs_snn_mr025_long_ssim/

- psnr_by_set.png
- ssim_by_set.png
- energy_by_set.png
- params_by_method.png

## Takeaways

- SpikeCS reaches the highest Set11 SSIM: 0.8277, slightly above ANN-Ours at 0.8264 in the final best-checkpoint evaluation.
- ANN-Ours is strongest on Urban100 SSIM: 0.7487 vs SpikeCS 0.7467, with nearly identical parameters but higher estimated energy.
- SpikeCS has much lower estimated energy than the non-spiking baselines in this evaluation: 1702.701 uJ vs about 4236-4285 uJ.
- CSNet-style-C36B8 remains parameter-matched but underperforms in SSIM after long training, so it is a fair-size baseline but not a strong accuracy baseline for this setup.
