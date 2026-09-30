# Refinement initialization audit — 30 September 2026

The original VM runtime recreated the reported initialization digest exactly:
`fb7e3260d745c7ce93a8cf393303cfa343fffeb48b4408bf47ed5d6f138cbdae`.
Local loading of the exported tensors independently reproduces that digest.
Exported file hash, original protocol file hash and executed export script match.
Frozen parent and gate tensors equal their originals; residual output weights
and bias are zero. Zero optimizer updates occurred in the export or local audit.

Evidence: `outputs/downloaded_refinement_initial/results.json`, `initial.pth`,
`verification.json`. These are recreated initial tensors, not independently saved
historical states from the training run. The executed training code's same-state
assertions remain the evidence for matching the two arms at run time.

Comparison with local seeded recreation (torch 2.13.0+cpu versus VM 2.9.1+cu129):

| Tensor | Different elements | Maximum absolute difference |
|---|---:|---:|
| refine.0.weight | 3,035 | 7.45058e-9 |
| refine.0.bias | 17 | 7.45058e-9 |
| refine.2.weight | 2,425 | 3.72529e-9 |
| refine.2.bias | 8 | 3.72529e-9 |

Every other tensor matches exactly. These tiny floating-point differences explain
why the byte digest differs; this audit does not isolate the library/kernel
implementation responsible. It supplies no evidence that the initialization
discrepancy caused the quality failures. No repeat training is justified by it.

Decision: initialization reconstruction no longer blocks interpretation of the
returned comparison. Neither refinement candidate meets the original quality
requirements, and RGB did not outperform the semantic-only control. Do not rerun
the same experiment. Future runners should archive their actual initial tensors
before optimization rather than relying only on seeded cross-runtime recreation.

Next: consolidate the existing matched experiments into an architecture decision,
including the common false-positive/recall tradeoff, known frozen-gate misses and
glare data coverage. Select a different, bounded hypothesis only after that review;
do not continue serial loss-weight, width or learning-rate sweeps against the same
development set. No new training package or promotion is approved by this audit.

Local root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM audit: `~/forensic-dgp/expanded_feature_bundle/outputs/refinement_initial_audit/`.
