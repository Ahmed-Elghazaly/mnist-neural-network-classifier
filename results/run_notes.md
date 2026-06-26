Experiment Execution Notes
==========================

The updated `main.py` was run locally on June 26, 2026 using the workspace virtual environment.

Regenerated during this cleanup run:

- `baseline_curves.png`
- `baseline_confusion.png`
- `lr_sweep_loss.png`
- `lr_sweep_acc.png`
- `bs_sweep_loss.png`
- `bs_sweep_acc.png`
- `arch_sweep_loss.png`
- `arch_sweep_acc.png`
- `best_fnn_confusion.png`

The CNN ablation section was started but stopped because it was taking substantially longer than the rest of the execution run. The existing CNN output images from the original completed run folder were copied into `results/` so the repository still contains the full expected output set:

- `cnn_ablation_acc.png`
- `cnn_ablation_loss.png`
- `cnn_ablation_train_loss.png`
- `cnn_confusion.png`

The updated script still contains the CNN ablation code and can regenerate those images when allowed to run to completion.
