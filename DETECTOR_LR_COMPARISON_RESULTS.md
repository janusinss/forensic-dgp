# Full-training-split detector learning-rate comparison — 28 September 2026

Completed a bounded four-epoch local CPU comparison between learning rates `1e-5` and `1e-4`
starting from the original completion epoch 2 detector (`outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth`).
The generator remained bitwise unchanged in all evaluations and epochs.

Both arms used seed 42, 21 updates/epoch, batch size 8 (2 real covered, 2 real uncovered, 2 synthetic covered,
2 synthetic uncovered), AdamW, gradient clipping (max norm 1.0), and the existing BCE + Dice + 0.25 hard-visible penalty.
All 68 real training images were confirmed present in every epoch. Synthetic replay used the fixed 200-image
reservoir from the original training split. Evaluation ran against all 68 real training images, all 25 V2 real validation
images (reporting the known mannequin separately), and the fixed 400-case synthetic validation benchmark.
No deployment weights were saved; neither arm produced a qualified replacement checkpoint.

## Measured results

| Arm & Epoch | Real Train IoU | Real Val IoU (all 25) | Real Val IoU (excl. mannequin, 24) | Real Val Missed % | Syn Val IoU (400 cases) | Syn Val Missed % | Syn Uncovered False Masks | Qualified / Selected |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| **Baseline (Initialization)** | 0.0754 | 0.0608 | 0.0644 | 93.18% | **0.9747** | 1.54% | 0/80 | Baseline |
| `lr_1e5` Ep 1 | 0.0654 | 0.0640 | 0.0698 | 92.83% | 0.9731 | 1.63% | 0/80 | No |
| `lr_1e5` Ep 2 | 0.1421 | 0.1038 | 0.1130 | 88.66% | 0.9744 | 1.64% | 0/80 | No |
| `lr_1e5` Ep 3 | 0.2166 | 0.1739 | 0.1897 | 81.17% | 0.9728 | 1.69% | 0/80 | No |
| `lr_1e5` Ep 4 | 0.2424 | 0.2020 | 0.2195 | 77.83% | 0.9724 | 1.79% | 1/80 | No |
| `lr_1e4` Ep 1 | 0.2037 | 0.2078 | 0.2258 | 78.52% | 0.8700 | 11.66% | 2/80 | No |
| `lr_1e4` Ep 2 | 0.3756 | **0.3536** | **0.3792** | 62.37% | 0.8614 | 11.57% | 4/80 | No |
| `lr_1e4` Ep 3 | 0.3867 | 0.3152 | 0.3457 | 66.56% | 0.8416 | 15.05% | 0/80 | No |
| `lr_1e4` Ep 4 | 0.3874 | 0.2663 | 0.2919 | 71.93% | 0.8587 | 13.56% | 2/80 | No |

## Analysis and findings

1. **Trade-off between real adaptation and synthetic retention:**
   - **`lr_1e5` (conservative rate):** Preserves synthetic detection almost completely (IoU `0.9724` vs. baseline `0.9747`; only 1/80 uncovered synthetic false-positive case). However, real mask adaptation is slow, reaching only `0.2020` real validation IoU (`0.2195` excl. mannequin) and still missing 77.83% of covered pixels after four epochs.
   - **`lr_1e4` (aggressive rate):** Rapidly accelerates real covering adaptation, reaching peak real validation IoU of `0.3536` (`0.3792` excl. mannequin) at Epoch 2—surpassing both the 10-epoch plain replay (`0.3204`) and consistency (`0.3157`) VM checkpoints. However, it causes immediate catastrophic forgetting on synthetic coverings (synthetic IoU plunges to `0.8700` at Ep 1 and `0.8587` at Ep 4; missed synthetic coverage surges from 1.54% to >11.5%; up to 4/80 uncovered synthetic cases trigger false masks).
2. **Overfitting on real training data at `1e-4`:**
   - At `lr_1e4`, real validation IoU peaked at Epoch 2 (`0.3536`) and then degraded across Epoch 3 (`0.3152`) and Epoch 4 (`0.2663`), while real training IoU plateaued at `~0.3874`. This indicates that without stronger regularization or more real data, higher learning rates overfit to the 68 training crops and generalize poorly.
3. **Selection gate outcomes:**
   - Neither arm passed the predeclared dual selection gates: `lr_1e5` failed the strict synthetic retention requirement (narrowly, by 0.0023 IoU), while `lr_1e4` failed synthetic retention catastrophically.
   - Zero checkpoints were promoted to `best_detector.pth`.

## Artifacts and next steps

- `outputs/detector_lr_comparison/results.json`: Full configuration, split hashes, per-epoch metrics across all five validation loaders, and timing.
- `outputs/detector_lr_comparison/validation_masks.jpg`: Ten-row visual comparison contact sheet (Input, V2 Ground Truth, `lr_1e5` Ep 4, `lr_1e4` Ep 4).
- `outputs/detector_lr_comparison/PROTOCOL.md`: Explicit protocol documentation and experimental bounds.

### Concrete next steps before VM training:
1. Do not adopt `1e-4` globally for full 10-epoch GPU runs; it destroys synthetic covering detection.
2. If pursuing a higher learning rate, test an intermediate learning rate (e.g., `3e-5`) or a staged schedule (warmup at `1e-4` for 1-2 epochs decaying to `1e-5`) combined with stronger synthetic replay weighting or teacher consistency.
3. Incorporate the expanded real mask dataset candidate ([Kaggle `hughiephan/face-mask`](https://www.kaggle.com/datasets/hughiephan/face-mask/data)) to expand the real training pool beyond 68 crops before committing to another cloud GPU run.
