# SingMOS Model Weaknesses and Fixes

## Overview

This short report summarizes the main weaknesses identified in the original SingMOS training/inference pipeline and the changes implemented to address them.

## Key Weaknesses Identified

1. **Inference normalization mismatch risk**
  - Training computed `MOS_MEAN`/`MOS_STD`, but checkpoints did not persist them.
  - Inference could fall back to approximate defaults, reducing prediction reliability.
2. **Limited adaptation capacity**
  - The MERT encoder was fully frozen and wrapped with `no_grad`, preventing encoder fine-tuning.
3. **Optimizer excluded encoder parameters**
  - Only head layers were optimized, so unfreezing would not be effective without optimizer changes.
4. **Potential training instability**
  - No LR scheduler, no early stopping, and no reproducibility seed.
  - Correlation-based loss term could be noisy across epochs without scheduling.
5. **Possible calibration/generalization issues**
  - Weighted sampling was always on and could bias calibration.
  - No train-time augmentation or amplitude normalization controls.
6. **Evaluation protocol gap**
  - No optional test-set reporting path after training.

## Changes Implemented

### 1) Checkpoint and Inference Reliability

- Added `mos_mean` and `mos_std` to checkpoint payloads in training.
- Updated inference to require valid MOS stats (from checkpoint or CLI args), removing approximate fallback defaults.

### 2) Staged Encoder Fine-Tuning

- Removed `@torch.no_grad()` from encoder forward.
- Added support to unfreeze top encoder layers:
  - `--unfreeze_last_n` (default `2`)
  - `--unfreeze_epoch` (default `3`)

### 3) Optimizer and Regularization Improvements

- Switched to `AdamW`.
- Added parameter groups with differential learning rates:
  - Head: `--lr`
  - Encoder: `--encoder_lr`
- Added `--weight_decay`.

### 4) Training Stability and Reproducibility

- Added `ReduceLROnPlateau` scheduler (`--scheduler_patience`).
- Added early stopping (`--early_stop_patience`).
- Added alpha schedule for hybrid loss:
  - `--alpha_start` to `--alpha_end`.
- Added deterministic seeding (`--seed`) across Python, NumPy, and PyTorch.

### 5) Sampling and Robustness Controls

- Made weighted sampling optional via `--no_weighted_sampler`.
- Added `--sampler_max_ratio` to control reweighting strength.
- Added optional train-time waveform augmentation via `--train_augment`.
- Added optional RMS normalization via `--rms_norm`.

### 6) Evaluation Coverage

- Added optional test split evaluation using `--eval_test` when a test split is present.

## Files Updated

- `train_singmos.py`
- `predict_mos.py`
- `README.md`

## Expected Impact

- More reliable inference behavior across environments.
- Better fine-tuning flexibility and potentially higher correlation metrics.
- Improved training stability and reproducibility.
- Better robustness and evaluation hygiene for model selection.

