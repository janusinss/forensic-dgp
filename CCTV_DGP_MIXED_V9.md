# V9: broader balanced restoration replay

Status on 4 October 2026: completed, returned and independently audited.
7,820 updates finished in1,356.35 seconds; training/VM audit/export took1,475.78
seconds. All four snapshots failed unchanged appearance safeguards;
`best.pth` retains the V2 baseline, selected epoch0. All five original-cell grids
were reviewed. After collection and idle checks, the VM is confirmed TERMINATED.
Detailed results and fingerprints: `C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_MIXED_V9_RESULTS.md`
↔ intended `~/forensic-dgp/CCTV_DGP_MIXED_V9_RESULTS.md` after document sync.
All actual training, including short pilots, uses the existing L4 VM. Local
preparation, inference and arithmetic audits remain permitted. Do not launch or
resume an existing V9 root automatically.

## Why this experiment

V8 demonstrated fitting capacity on two blurred training pairs, with severe
motion, clear-face and other-face regressions. Its overfit checkpoint is not a
starting state for V9. V9 starts from the audited V2 tensor state and trains all
five clear/degradation profiles for both reviewed training sources. This changes
coverage, exposure and weighting jointly; it does not isolate a sole cause.

The source-only Asian review inspected all 451 original training candidates on
29 exact-target contact pages, plus 49 original-resolution checks. Accepted:
390; excluded for coverings: 28; excluded for source/pose quality: 33. Two
mistyped IDs and two omitted obvious exclusions were corrected with the original
pages and native photos; initial observations and corrections remain preserved.
All Asian originals have a minimum edge below 256. They are low-resolution
replay targets, not newly captured HQ detail. Source review is development review,
not an independent final human assessment.

| Cohort | Training references | Validation references |
| --- | ---: | ---: |
| Reviewed HQ FFHQ counterparts | 391 | 53 |
| Reviewed Asian original replay | 390 | 51 unchanged sentinels |
| Total | 781 | 104 |

Original roles are preserved. Validation includes all 520 unchanged V6 cases;
no metric-based exclusions or validation-to-training reassignment were made.
Direct source hashes are disjoint across roles; subject identity and complete
historical exposure are unestablished. The 32 reserved native cases remain unused.

## Frozen finite recipe

Twenty epochs, 391 steps per epoch, batch 10, 7,820 optimizer updates and 78,200
case exposures. Every batch contains one case from each of the ten source/profile
cells. Each epoch sees all 3,905 unique prepared cases. The 390-item Asian cells
each repeat one shuffled case to match 391 steps; this is recorded explicitly in
the frozen schedule. Profiles: clear, blur/24, low-light/32, motion/48, compound/24.

Fresh Adam uses backbone `2e-5`, remaining parameters `1e-4`, weight decay `1e-5`
and clipping norm 1. Five frozen normalization adapters preserve original buffers;
AMP/EMA are disabled. The objective is observed RGB pixel MSE with fixed
source/profile weights calibrated only on the starting model's 3,905 training
cases. Raw weights are clipped to `[0.25,4]` before mean-one normalization.
ArcFace remains a frozen fixed-affine validation metric and receives no training
loss gradients in this pixel-foundation experiment.

Snapshots: baseline and epochs 2, 5, 10, 20. Trainer cap: 2,400 seconds (40 minutes).
Complete training/audit/export cap: 3,600 seconds (60 minutes). At update 32,
measured training timing plus measured baseline evaluation and remaining snapshots
must project within the trainer cap. Nonfinite gradients/loss, altered files,
unexpected normalization state, wrong exposure or timeout stop the run. Preserve
failure receipts/partial state; do not repeat or resume automatically.

`best.pth` uses the unchanged strict PNG selection: at least 0.1 dB degraded PSNR
gain over the current best, with no MSE, SSIM or fixed ArcFace degradation in
every source/profile aggregate. If no trained snapshot passes, it retains the
starting V2 baseline. Metric eligibility still requires original-cell visual
inspection, native development review and independent final review. This pilot
cannot automatically promote a production checkpoint or establish Zamboanga CCTV
performance.

## Paths and fingerprints

| Artifact | Windows local | Linux VM |
| --- | --- | --- |
| This runbook | `C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_MIXED_V9.md` | Intended `~/forensic-dgp/CCTV_DGP_MIXED_V9.md` after document sync; no copy claimed |
| Executable root | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_mixed_vm_v9_r2\` | `~/forensic-dgp/cctv_dgp_mixed_vm_v9_r2/` after verified upload/extraction |
| Source-review evidence | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_asian_source_review_v9\` | Required ledger copies under executable `lineage/`; full contact pages remain local |
| Package | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv-dgp-mixed-v9.tar.gz` | `/home/janusdominic0/cctv-dgp-mixed-v9.tar.gz` |
| Returned archive | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv-dgp-mixed-v9-results.tar.gz` verified locally | `~/forensic-dgp/cctv_dgp_mixed_vm_v9_r2/cctv-dgp-mixed-v9-results.tar.gz` verified after success |

Executable protocol SHA256:
`6cd9ac8d5f3bf27be773979c2f628e33f5d3cf09748452eeab91a65b05b01c70`.
Package: 305,689,318 bytes, 6,232 exact verified members; SHA256:
`f8fb4196a1515c451ca1f77b1d73e81f50e6335e3280b2287de7b78f48090a99`.
The original weight files are omitted from the thin upload and copied from
hash-verified V6 VM caches. No runtime installation/change is planned.

Manifest SHA256: `ec25bf9dc37b32b07fb426cf7875477e70181bf55398f6fdfe420bd4ca550f5d`.
Design SHA256: `a7b365f97810b11aa35f2707a1ddeff8c609805394e82dac9198d16a9cd5e30b`.
Schedule SHA256: `ce125d1a19d4880437758ad16b0ed9c6edfe7c63397b0ad37ab3e78c66e5a5eb`.
Local asset audit SHA256: `e4a1987964d83c955b9b27275568c77fa7730cf4ba119ba090de42b910981656`.

The first assets-only preparation stopped on V6's retained legacy thumbnail pixel
metadata. Current HQ target files matched both V6 protocol and reviewed-source
byte pins. A separate `r2` preparation preserves the original reference records,
pins actual canonical pixels separately and passed all 3,905 proxy reconstructions
and 885 target checks. The failed script/root/receipt remain intact.

## Autonomous execution and return audit

Configured Google Cloud CLI can start/stop, SSH and SCP the existing
`forensic-dgp-thesis`, project `forensic-dgp-thesis`, zone `us-central1-a`.
Use `WINDOWS_GCLOUD_TRANSFER.md` with the configured TLS trust file and SSH host
key pin. The assistant uploads the archive, LF checksum and hash-verified launcher
`scratch/launch_mixed_v9_v1.py`, verifies an idle L4/CUDA, and launches dedicated
tmux `dgp_mixed_v9`. No user paste is required when this access remains available.

After success, download the archive and `.sha256` separately using gcloud SCP.
Local audit command, from `C:\xampp\htdocs\YEAR 4\Testing\`:

```powershell
.\venv\Scripts\python.exe -X utf8 -u scripts/audit_cctv_dgp_mixed_v9.py --root outputs/cctv_dgp_mixed_vm_v9_r2 --expected-protocol-sha 6cd9ac8d5f3bf27be773979c2f628e33f5d3cf09748452eeab91a65b05b01c70 --archive outputs/cctv-dgp-mixed-v9-results.tar.gz --extract-to outputs/cctv_dgp_mixed_return_v9 --receipt outputs/cctv_dgp_mixed_return_v9/local_independent_audit.json
```

The auditor independently rebuilds all 2,600 validation PNG metrics, 50 raw
preview exports, 3,120 embedding cosines, 3,905 reported calibration errors/group
weights, 20 raw calibration errors, 7,820 traces/78,200 exposures and four trained
states. It reconstructs the unchanged selection rather than trusting the receipt.
It does not replay CUDA gradients or recognizer forwards locally. View all five
ten-row grids at original cells before a new training decision. After collection,
check competing jobs and restore a VM started solely for this pilot to its prior
stopped state. Update `PROJECT_HANDOFF.md`; the full DGP-first Goal remains active.
