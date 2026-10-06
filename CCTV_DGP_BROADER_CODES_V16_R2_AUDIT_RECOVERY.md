# V16 r2: recover the arithmetic audit without retraining

**Executed 5 October 2026:** transfer/import verification passed. The separate
corrected full audit passed in 73.05 seconds under its unchanged 240-second
limit. Original failure, frozen source, protocol and checkpoints are preserved.
Both learned snapshots fail appearance-preservation guards; no adoption.
Report: `CCTV_DGP_BROADER_CODES_V16_R2_RESULTS.md`.
Receipt: `outputs/cctv_dgp_broader_codes_v16_r2_audit_recovery_1/local_full_audit.json`.
Do not repeat the completed import/audit commands into existing destinations.

## Historical preparation and recovery instructions

5 October2026. Local workspace `C:\xampp\htdocs\YEAR 4\Testing`;
VM pilot `~/forensic-dgp/cctv_dgp_broader_codes_vm_v16_r2`.
This document and recovery script have not been uploaded. No assistant cloud
connection or VM execution is authorized by this preparation.

The user reports3,128 training updates/epoch8 completion in1,055.125922286s,
followed by `ValueError: Timing stop receipt differs` at auditor line143.
The frozen checker requires20 cache timing references after already verifying
the R2 timing helper, which requires30. The original audit loop rejects valid
30-reference receipts in a local regression. This is an auditor migration bug;
it does not establish the quality or validity of the trained images.

CE5.0193 is observed-token cross entropy at logged update3100.
Accuracy0.097 is9.7% matching code-token classifications for that training batch,
not face recognition, whole-run evaluation accuracy or a restoration quality score.

## Scope of the correction

The original R2 source, manifest/protocol, archive, training recipe, checkpoints,
failed audit log and supervisor failure stay unchanged. A separate recovery
checks their original hashes, requires all3,128 completed trace records and
complete trained results, and creates a separate corrected auditor file.

Only two tuple literals change:

| Timing receipt | Frozen checker | Corrected checker |
| --- | --- | --- |
| Cache |20 references /900s |Protocol's30 references /900s |
| Fit |25 updates /1,200s |Protocol's25 updates /1,200s |

Counts/caps are read from the unchanged checksum-bound protocol design. All
other provenance, teacher/split, timing, trace, snapshot, metric, parity, raw
output, grid and quality-report checks remain byte-for-byte intact. The new
audit has the original240s limit. No neural/training call, resume, best.pth,
checkpoint selection, cap increase or application promotion occurs.

Protocol SHA256:
`4f64cf2204458f8fadcab1824fc4bc293c65803e123fdebb304f7b5bd3a502a0`.
Original auditor SHA256:
`0ec24b386ccfc210f4efe126d48d1e3e4064c38574bf158f139f310b7d6472fd`.
The new request receipt records the exact original/corrected auditor and recovery
script hashes, results/log/failure/import hashes and the unchanged scope.

Seven meaningful regressions passed in0.124s. They execute the original and
corrected timing loops, bind counts to design, reject changed source, preserve
all other checks and reject wrong counts/caps, nonfinite or out-of-budget timing.
No neural libraries or models are imported. Boundary/preparation evidence:
`outputs/cctv_dgp_v16_r2_audit_recovery_preparation.json`.

## Collection and local audit

Exact manual gcloud downloads are at the top of
`CCTV_DGP_BROADER_CODES_V16_R2_VM.md`. Download the **failure** archive/sidecar
and use local receipt name `failure_export_v16_r2.json`.

After the user reports those downloads, the assistant executes these locally:

1. Import into a fresh return, preserving the original failure:

   ```powershell
   .\venv\Scripts\python.exe -X utf8 -u scripts/import_cctv_dgp_broader_codes_v16_r2.py --archive outputs/cctv-dgp-broader-codes-v16-r2-failure.tar.gz --completion outputs/failure_export_v16_r2.json --extract-to outputs/cctv_dgp_broader_codes_failure_return_v16_r2
   ```

   This verifies the archive/hash/size/sidecar and all frozen source assets;
   `local_failure_import.json` checks completed traces. It is not the full output
   audit and retains `success:false` because the original supervisor failed.

2. Run the full corrected audit into a fresh, separate directory:

   ```powershell
   .\venv\Scripts\python.exe -X utf8 -u scripts/recover_cctv_dgp_broader_codes_v16_r2_audit.py --return-root outputs/cctv_dgp_broader_codes_failure_return_v16_r2 --recovery-to outputs/cctv_dgp_broader_codes_v16_r2_audit_recovery_1
   ```

   Any failed gate remains a failure with a separate retained log; no retry or
   training is automatically authorized. Do not overwrite an existing return or
   recovery directory. Passed receipts require the actual archive; no full audit
   result is claimed by the preparation regressions.

3. Inspect all ten original256-cell grids and epoch0/4/8 structure/appearance
   reports before proposing adoption or another processing change.

The full audit checks1,710 PNGs,300 raw previews,1,710 embedding cosines,
150 training code probes,100 fresh image parity cases,781 teacher arrays,
4,425 cache bindings,3,128 update records/31,280 exposures and600 grid cells.
The full9GB cache remains on the VM; recorded bindings/probes are audited locally.
This is serialized arithmetic/provenance verification, not a CUDA gradient replay
or independent final human review. Synthetic paired metrics stay separate from
native unpaired CCTV evidence. The DGP-led useful-output/app/covering-family Goal
remains incomplete.
