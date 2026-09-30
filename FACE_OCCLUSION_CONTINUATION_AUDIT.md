# Continuation return verification — 30 September 2026

The return checker is ready. The continuation result archive is **absent** locally;
no new VM execution, model-quality result or promotion is claimed. Actual training
uses the Linux CUDA runner already packaged in `FACE_OCCLUSION_CONTINUATION_VM.md`.
This checker performs read-only CPU inference after return, with zero local
optimizer updates. It does not retry or extend the VM training budget.

## 1. Return the VM artifact

Linux source:
`/home/janusdominic0/forensic-dgp/coverage_vm_bundle/face-occlusion-continuation-results.tar.gz`.

Windows destination:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\face-occlusion-continuation-results.tar.gz`.

The VM result directory is
`~/forensic-dgp/coverage_vm_bundle/outputs/face_occlusion_continuation_vm/`.
Do not substitute the earlier `face-occlusion-results.tar.gz`: that archive is
the completed initialization comparison, not this additional-fitting experiment.

## 2. Verify the archive, update log and masks

From Windows PowerShell:

```powershell
Set-Location -LiteralPath 'C:\xampp\htdocs\YEAR 4\Testing'
& .\venv\Scripts\python.exe -u scripts/audit_face_occlusion_continuation_results.py
```

The checker validates the fixed sent package, old source inventories, reviewed
dataset membership and prior VM dependency setup. It permits only expected
regular files/directories, rejects links, duplicate/escaping paths and changed
code, and bounds extraction to 600 MiB. Model files have a 64 MiB limit; the
final AdamW snapshot has a separate 128 MiB limit for approximately115 MB of
moments. Existing extraction/audit outputs are preserved.

It reconstructs all420 additional batches from two cycles of the original
schedule, distinguishes fresh420 from cumulative630 updates, checks candidate
epochs11/15/20/30 and recomputes the original selection decisions. All2,915 saved
masks are recounted:425 original-parent masks,498 source masks and four sets
of498 candidate masks. Source bytes must match pretrained epoch10 exactly;
logged source metrics must match that checkpoint's original selection record.
Final model/optimizer hashes must match the completion record.

Extracted evidence:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\downloaded_face_occlusion_continuation\`.
Audit output:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\face_occlusion_continuation_validation\`.
`results.json` records counts/gates and `members.json` records every returned
regular-file hash. `preview.png` contains ten fixed validation rows:four covered,
four clear, the strong-lens-glare case and the mannequin. Columns are input,
target, original parent, source epoch10 and all four continuation candidates.

This first pass validates saved masks/logs. It does **not** establish that those
masks came from the returned model or that the optimizer moments are valid.

## 3. Verify checkpoints, optimizer and CPU predictions

```powershell
& .\venv\Scripts\python.exe -u scripts/audit_face_occlusion_continuation_results.py --reproduce
```

Pinned isolated dependencies in
`C:\xampp\htdocs\YEAR 4\Testing\outputs\face_extraction_dependencies\` are reused;
there is no weight download, optimizer construction or fitting. Each checkpoint
is safely loaded with `weights_only=True` and strict adapter state validation.
Original pilot metadata remains original; `continuation_metadata` must match the
new run. Frozen reference-head/BatchNorm tensors must match the copied source.
Declared encoder/decoder/new-head components must contain changed tensors.

The final AdamW snapshot must bind to the epoch30 hash, cover every expected
parameter in group order, retain the fixed rates/decay/settings, and contain
finite, correctly shaped moments with step420 for every parameter. Intermediate
and best optimizer snapshots were not saved. If an eligible best detector exists,
its state/metadata must match the last selected candidate; serialization bytes
need not match because it was saved under a different filename.

CPU inference compares all2,915 saved masks and independently recomputes scores
and selection. Report any CPU/VM pixel or gate differences explicitly; do not
claim bitwise identity by default. `reproduction.json` contains the completed
checks and `reproduction_partial.json` records progress. Read-only inference must
leave model tensors unchanged. Remote parent/generator invariance remains an
executed-code assertion/log claim because its VM tensors were not returned.

## 4. Review before the next model decision

Inspect the ten-row mask grid and strong lens glare separately. Advance only an
eligible detector to reviewed end-to-end restoration/completion; synthetic and
real selection guards are unchanged. Metric eligibility alone does not promote
the application or complete the overall goal. Hidden facial parts remain
plausible estimates. External FFHQ pretraining overlap is unresolved and these
reused development cases do not establish final generalization.

If the pilot fails its guards, preserve the result and inspect fit/precision
before specifying another intervention. Do not repeat the same run automatically.

## 5. Verified preparation

Seventeen new checker tests pass;25 pass including the earlier return-audit and
VM-runner regressions. Tests cover changed schedule/counters, invalid selection,
nonfinite/incorrect losses, altered metadata/frozen state, optimizer binding and
moment errors, unsafe archive members and preservation of existing evidence.
Small archive fixtures verify actual extraction/hash behavior; they are not a
model-quality or CUDA-training result.

The already sent seven-member package was reverified without modification:
11,145 bytes, SHA256
`93bab8a34d23a48cd8f9232dcbe1cd65c91bd024b2cd6e8480d76e863f82e3f7`.
Its source files and checksum remain unchanged. This new local auditor/document
is separate from the immutable VM package and has not been pushed.

Read-only package check:

```powershell
& .\venv\Scripts\python.exe scripts/audit_face_occlusion_continuation_results.py --verify-package
```

Regression command:

```powershell
& .\venv\Scripts\python.exe -m unittest tests.test_face_occlusion_continuation_results tests.test_face_occlusion_results tests.test_face_occlusion_continuation
```
