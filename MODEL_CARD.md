# Frozen restoration models in IMPLEMENTATION

**Default: U96.** Original terminal-update-64 post-fusion heads (u64) are selectable for comparison. No failed successor replaces either artifact.

Both use the trained R2 CodeFormer base with the 24-tensor U96 residual skip adapter. u64 adds 36 post-fusion correction tensors from the retained update-64 experiment. The numbers denote training updates, not image sizes. Internal processing is singleton CPU FP32 at 512×512, fidelity 1 and AdaIN, followed by explicit 256×256 delivery. PNG uses floor quantization without extra sharpening or brightness display processing.

| Asset | Bytes | SHA-256 |
| --- | ---: | --- |
| `codeformer_r2.pth` | 376609465 | `d4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896` |
| `codeformer_u96_adapter.pth` | 212827 | `958e8338d4bc1a0204c314c3af239ab0ab3d783c67e3e532d9fe23c58f8a73d0` |
| `terminal_u0064_heads.npz` | 685818 | `77c0e1f236eb70d37f3f81059b75580e92d606bb2f8b5dd5eeb00da0305dca17` |

The implementation comparison freshly verified 642 saved RAW/PNG outputs across 321 paired DEV cases and retained adverse cases. On the broader 285-case subset, u64 improved mean PNG MAE from 0.06130135 to 0.06052539 and mean PSNR by 0.12043 dB. It worsened normal-blur MAE in 46/48 cases and all four fixed severe-Gaussian cases. Existing preservation checks recorded 137 violating scalar comparisons for U96 and 141 for u64; these are scalar checks, not counts of people or entire images. Neither model passes all research requirements. The incomplete broader bank cannot supply a full-bank qualification.

These measurements use known synthetic paired targets. Native CCTV and wild images without aligned clean references remain unpaired. No ethnicity, Zamboanga performance, exact hidden identity or independent FINAL qualification is established.

The application preserves neutral padding and reviewed input preparation. Generated facial details may still change; clearer/less-covered input is required when visible information is insufficient. Completion and covering removal are outside this restoration implementation.

Model lineage and failed experiments remain in the research checkout; packaging does not relax their gates. Further training remains manual on the designated L4 VM under a new justified finite protocol.
