# MR=0.25 Horizontal Comparison Figures After 2000 Epochs

This file summarizes the figures generated from the 2000-epoch training logs and the final evaluation from each model's best_ssim.pt checkpoint.

## Training Curve Figures

Directory:

results/figures/ssim_2000_training_curves/

Combined epoch-wise curves:

- all_models_ssim_by_epoch.png
- all_models_best_ssim_by_epoch.png
- all_models_psnr_by_epoch.png
- all_models_loss_by_epoch.png

Per-model epoch-wise curves are stored under:

- results/figures/ssim_2000_training_curves/ANN-Ours/
- results/figures/ssim_2000_training_curves/ISTA-Prox/
- results/figures/ssim_2000_training_curves/CSNet-style-C36B8/
- results/figures/ssim_2000_training_curves/SpikeCS/

Each model directory contains:

- ssim_by_epoch.png
- best_ssim_by_epoch.png
- psnr_by_epoch.png
- loss_by_epoch.png

## Best Epoch Summary From Training Logs

| Method | Epochs | Best epoch | Best SSIM | Best PSNR at best SSIM | Last SSIM | Last PSNR |
|---|---:|---:|---:|---:|---:|---:|
| ANN-Ours | 2000 | 1994 | 0.8866 | 29.80 | 0.8860 | 29.78 |
| ISTA-Prox | 2000 | 1994 | 0.8813 | 29.54 | 0.8809 | 29.53 |
| CSNet-style-C36B8 | 2000 | 1992 | 0.7002 | 23.64 | 0.6997 | 23.64 |
| SpikeCS | 2000 | 10 | 0.8273 | 27.24 | 0.7152 | 24.59 |

## Final Test Metric Figures

Directory:

results/figures/horizontal_three_models_vs_snn_mr025_2000/

- psnr_by_set.png
- ssim_by_set.png
- energy_by_set.png
- params_by_method.png

Final metric tables:

- results/horizontal_three_models_vs_snn_mr025_2000.csv
- results/horizontal_three_models_vs_snn_mr025_2000.md

The CSV completeness check passed, so PSNR, SSIM, Params(M), FR, Energy(uJ), and checkpoint source are all present.

## Key Observations

- ANN-Ours-2000 and ISTA-Prox-2000 improve substantially after long training and lead the final SSIM results.
- SpikeCS-2000 keeps much lower estimated energy, but its best validation SSIM appears very early in the continued 2000-epoch run and then degrades.
- CSNet-style-C36B8-2000 remains parameter-matched but is still weaker in PSNR and SSIM.
