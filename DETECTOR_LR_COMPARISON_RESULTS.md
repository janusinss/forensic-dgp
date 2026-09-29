# Full-real-training learning-rate comparison — reviewed 29 September 2026

The local process completed both arms after the conversation interruption. It was
not restarted. Each arm ran four epochs of 21 updates, batch eight, on all 68 real
training images plus synthetic replay drawn from the same 200 original training
sources (178 FFHQ and 22 Asian-source). Every real training image was seen each
epoch. Not every synthetic variant was necessarily sampled within this budget.

Same original completion epoch-2 initialization, seed 42, sampler order,
BCE/Dice plus 0.25 hard-visible loss, and no teacher consistency term. Only the
learning rate differed. The corrected V2 25-image validation set and fixed
400-case synthetic validation set were evaluated every epoch. The known mannequin
was reported separately; aggregate selection kept it included. No test cases were
used. Original split hash, replay-source membership and source hashes were checked;
replay sources were excluded by hash from original validation and known benchmark/
review sources. This establishes these checks, not identity-level independence.

| Metric | Initial | lr 1e-5, epoch 4 | lr 1e-4, epoch 4 |
|---|---:|---:|---:|
| Real training IoU | 0.07509 | 0.24240 | 0.38737 |
| Real validation IoU | 0.06078 | 0.20196 | 0.26627 |
| Synthetic validation IoU | 0.97469 | 0.97244 | 0.85874 |
| Real validation missed pixels | 93.14% | 77.83% | 72.36% |
| Synthetic negative false-mask cases | 0/80 | 1/80 | 4/80 |

The high-rate arm peaked in real-validation IoU at epoch 2 (0.35362), while
synthetic IoU was already down to 0.86137. This is a diagnostic peak, not a selected
best model. All eight epoch candidates failed the unchanged synthetic retention
gate. No weights were saved or deployed; generator integrity assertions passed.
Recorded arm runtimes were approximately 413 and 444 seconds, excluding shared
initial checks and baseline evaluation.

## Interpretation

The four-example overfit success did not transfer to the full mixed-data problem.
Increasing the learning rate alone worsens forgetting. Real training fit also
remains limited at this budget, so these results do not prove that the only issue
is generalization. Four epochs cannot establish the optimal schedule or rule out
other training strategies. Visual review of ten fixed validation rows shows
fragmented white-mask coverage, dark-covering omissions and false positives on
caps, hair or background. No convincing completion-quality improvement is shown.

## Artifacts and next step

Local root `c:\xampp\htdocs\YEAR 4\Testing\`; VM root `~/forensic-dgp/`.
Artifacts exist locally under `outputs/detector_lr_comparison/`:
`results.json`, `PROTOCOL.md`, and `validation_masks.jpg`.
The reproduction script is `outputs/run_detector_lr_comparison.py`. Transfer
explicitly if needed on the VM; Git ignores these experiment artifacts.

Do not launch another unchanged learning-rate run or extend the budget blindly.
Next diagnostic: isolate the real-only supervised objective from replay with a
matched, bounded training-fit comparison, measuring both real and synthetic
validation throughout. This would determine whether replay interaction is limiting
real-mask fitting before changing architecture or commissioning further GPU runs.
It is a proposed diagnostic, not yet run. Keep current application weights and
manual region correction; all selection gates remain in force.
