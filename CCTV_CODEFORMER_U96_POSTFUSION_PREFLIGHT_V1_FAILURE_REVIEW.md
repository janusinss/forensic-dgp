# U96 post-fusion V1: returned failure audit and R1 repair

## Material Passport

21 October 2026. Saved-only archive/source review; no local model, gradient,
optimizer, training or VM action. V1 is a failed infrastructure preflight, not
model-quality evidence. R1 is separately prepared and remains unrun.

## Location, cause and effect

Location: `guarded_dependencies_and_CUDA_configuration`, worker line 236.
Cause: `ModuleNotFoundError: No module named 'psutil'` in the VM environment.
Torch was imported, but model construction had not been attempted. All model
forward, hook-site, gradient-query and optimizer counters are zero. No generated
image, head state or quality metric exists. The supervisor recorded exit 1,
no timeout and 14.349 seconds. Successful export preserved the failure; it did
not establish CUDA/model parity or useful restoration.

The 1,297,655-byte archive SHA256 is
`40bcb262dd1c3f879fd3087c37b104275c7de57242d9975b05fb9ce1487c09b5`.
Checksum, export receipt and original protocol matched. Independent readback
verified all 45 members and all 38 returned nonweight static files against the
sealed stage. The auditor's successful exit means the failed evidence was
validated; full runtime artifact audit and model qualification remain false.
Original archives, code, protocol, failures, checkpoints, splits and inputs stay
retained. No unchanged run was relaunched.

[Saved-only failure readback](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_preflight_v1_return_audit/returned_failure_readback_v1.json>) ·
[Independent peer audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_preflight_v1_return_peer/independent_return_review_v1.json>).

## R1 correction

A separate R1 worker removes the optional `psutil` dependency and reads current
RSS from Linux `/proc/self/statm` resident pages times the OS page size. The
existing `resource.getrusage` peak-RSS guard remains. Numerical dependencies
are checked for discoverability before Torch import. Source review confirms
unchanged model/data/math, parity/objective/gradient logic and all 16 resource
and call limits; only resource/dependency handling and namespaces changed.
This addresses the observed import failure, without claiming the rest of the
L4 run has passed.

The current-RSS method follows the [Linux kernel's statm documentation](https://docs.kernel.org/filesystems/proc.html).
It is OS accounting, which the kernel documents as potentially asynchronous,
not an exact physical-memory profiler. Peak usage remains available through
[Python's resource interface](https://docs.python.org/3.10/library/resource.html).
These are resource controls, not restoration measurements.

The 1.25 MiB thin transfer includes every nonweight source, input and evidence
file, including the retained V1 failure/protocol/supervisor. A reviewed installer
verifies the original zero-call failure, all original static files and the fresh
R1 namespace before copying the two exact incumbent weights with exclusive
writes and before/after hashes. It changes no original file, downloads nothing
and creates no hardlinks. Full local-stage verification and independent readback
of all 44 archive members passed. The real Linux installer and L4 numerical
execution have not run. Stop if the original stage or sufficient disk space is
unavailable; do not silently substitute another model or erase a failure.

[New five-step guide](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_CODEFORMER_U96_POSTFUSION_PREFLIGHT_V1_R1_VM.md>).

## Continuation

The user manually launches R1 in tmux. Independently audit its actual return,
including raw/PNG parity and all gradient/head arrays, before considering a
finite actual-step training pilot. No additional epochs are prescribed from
this failure. U96 remains the selected restorer. The paired surroundings study
still does not support always-on preparation; its optional protection/confirmation
and bypass remain separate from model learning. Native/unpaired evidence,
independent FINAL, user acceptance, app integration and later covering scope
remain outstanding. The full goal stays active and incomplete.
