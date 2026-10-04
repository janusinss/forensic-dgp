# Grayscale detector pilot readiness — 3 October 2026

Ready for the existing VM's setup checks and one-batch CUDA preflight. Actual
execution and output improvement remain pending. No local training occurred.
The frozen commands are in `C:\xampp\htdocs\YEAR 4\Testing\GRAY_COVERING_VM.md`
(bundled counterpart `~/forensic-dgp/gray_covering_vm_bundle/GRAY_COVERING_VM.md`).

| Evidence | Windows local | Linux counterpart after the stated transfer |
| --- | --- | --- |
| Upload | `C:\xampp\htdocs\YEAR 4\Testing\outputs\gray-covering-vm-bundle.tar.gz` and `.sha256` | `/home/janusdominic0/gray-covering-vm-bundle.tar.gz` and `.sha256` |
| Protocol | `C:\xampp\htdocs\YEAR 4\Testing\outputs\gray_covering_protocol_v1\protocol.json` | `~/forensic-dgp/gray_covering_vm_bundle/inputs/gray_covering_protocol.json` |
| Package audit | `C:\xampp\htdocs\YEAR 4\Testing\outputs\gray_covering_package_validation_v1\verification.json` | Intended `~/forensic-dgp/outputs/gray_covering_package_validation_v1/verification.json` after a separate evidence transfer |
| Shell / contract evidence | `C:\xampp\htdocs\YEAR 4\Testing\outputs\gray_covering_package_validation_v1\shell_verification.json` | Intended `~/forensic-dgp/outputs/gray_covering_package_validation_v1/shell_verification.json` after a separate evidence transfer |
| Diagnostic | `C:\xampp\htdocs\YEAR 4\Testing\outputs\varied_covering_footprint_diagnostic_v1\` | Only its frozen JSON receipts are bundled; not the evaluation images |

The archive is 50,299,398 bytes (50.3 MB), SHA256
`212b91935e548ad1423f92f71014681be25abd052bad1d7bf2785e8b2b702d18`.
Its LF sidecar, 1,022 regular members and full inventory pass independent checks.
All 1,009 original V2 members remain byte-identical in this separate bundle;
the original V2 archive, source checkpoints, datasets and protocols are untouched.
Twenty-three consumed Python files compile, both shell scripts pass installed
Git Bash `-n`, and setup's inline Python and Windows-path normalization are checked.
Seven meaningful local contract tests pass, including the actual Windows training
refusal and the executed V2 first128-step schedule regression. No models or
optimizers are constructed for these tests/audits; tensor preprocessing is permitted.

This is a matched grayscale hypothesis test, not a claim that gray input caused
the failure. Both branches keep the previously balanced family sampler, the
same 133 sources/266 paired inputs and 280 unchanged replay fixtures. The first 128
RGB updates must reproduce all previous treatment-model states exactly on the VM.
Both branches receive 512 updates; `gray133` adds 970 gray real-input occurrences
among 2,048 real occurrences. All 133 sources receive every native/degraded and
RGB/gray condition. Labels, unknown support, split membership and crop geometry
are unchanged. Previous failed retention/hair/visibility results are preserved.

The budget is 1,024 optimizer updates across both branches, 11,440 training plus
measurement image forwards and one separate zero-update preflight. The four
measured states save 3,248 masks. A 30-minute processing cap starts after loading;
new runtime is not measured yet. No `best.pth`, generator/identity training or
app swap is enabled. Both final `last.pth` files remain diagnostic artifacts.

SHA256 references: protocol
`c29eabcc23b07e2f9353116068c623b35258a56ed5ab259126b13b9c781be47a`;
package audit `4a341c260c4ed39df3af562136c4da41fd2bdbbfa58bd6dbda05048b86288899`;
shell/contracts `ec5056207ea8eeda196846228b73c0d9a7750effa2f0a0d6135b110bb9ff298d`.

Next: upload the two files with the configured Windows SDK, extract into the new
VM directory and run `bash scripts/run_gray_covering_vm.sh` inside `dgp_training`.
Return `gray-covering-results.tar.gz` and its checksum for independent audit and
the fixed practical gallery before checkpoint selection. The user-directed SSH
command workflow continues; no agent SSH session or new VM job is open, and no
git commit/push occurred.

The independent return auditor and 14 passing integrity tests are now prepared.
Its read-only contract check confirms the frozen sources, schedules and 812
measurement targets. `GRAY_COVERING_RETURN_REVIEW.md` contains the Windows SDK
download and local audit commands (intended VM counterpart
`~/forensic-dgp/GRAY_COVERING_RETURN_REVIEW.md` after separate transfer). The new
auditor is outside the frozen training bundle; no package/hash/recipe changes are
needed. The actual gray return remains absent locally, so result verification,
practical inference and output-quality review remain pending.
