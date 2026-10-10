# V42 preparation milestone — 9 October 2026

The distinct finite V42 packet is prepared and independently checked. Training
has not started. The next action is the user's manual transfer and tmux launch
from [the five-step guide](CCTV_DGP_RESIDUAL_EPOCHS_V42_VM.md).

## Change being tested

The preceding [arithmetic review](CCTV_DGP_RESIDUAL_SUPERVISION_V1_REVIEW.md)
supports testing direct supervision of the mean-centered spatial correction
before the final image clamp. V42 keeps the current app DGP checkpoint,
normalization and input geometry, plus our previous own spatial architecture
and its fixed initial reference. Only the new spatial decoder learns. The
original DGP's weights remain frozen. This is not a claim that its complete
encoder or ancestral training was extended by five epochs.

The reconstruction objective now supervises observed RGB, landmark RGB and
multiscale RGB correction targets. Clear cases explicitly learn zero correction
to the retained DGP. Absolute degraded identity supervision and the retained
identity, pixel and SSIM regression penalties accompany the reconstruction terms.
They are training signals, not guarantees of preserved appearance. No clean
target or teacher is a model forward input. No pretrained restorer supplies
targets or substitutes for our DGP.

The current accepted checkpoint is SHA256
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
It includes original Phase3 training plus two selected identity-v2 fine-tuning
epochs; the complete lifetime epoch count remains unconfirmed. Every previous
rejected checkpoint and stop stays preserved. V42 does not resume V40 or V41.

## Frozen finite study

Each epoch visits the same 781 approved TRAIN references and five photographic
profiles exactly once. Five epochs equal 3,905 updates and 19,525 case exposures.
Native CCTV, DEV and reserved final identities are excluded from optimization.
The approved photographic roles and historical overlap limitations are retained.
Reuse the 391 accepted HQ targets and 390 lower-resolution source replay
references; no new acquisition is introduced here.

Compare snapshots at updates 0, 50, 781, 1,562 and 3,905. The 1% structure
requirement applies at update 50 and the intermediate epoch snapshots; the final
necessary structure requirement remains 10%. All 17 pixel/SSIM/embedding
preservation groups, both-source nonregression and 20% brightness-only limit
apply separately to raw outputs and delivered PNGs. The first failure stops and
exports; additional epochs cannot override it. A passed capacity study would
still need paired development, unpaired native visual review and independent
final review before app adoption.

AdamW uses 0.0003, weight decay 0.01 and gradient clipping 1. After update 1,562,
MultiStepLR reduces the rate to 0.00009. The initial-40 degraded loss scales are
frozen and independently audited; clear correction targets are zero. The same
17,952 parameters and 57 tensors must have finite nonzero reconstruction
gradients on the fixed 50-case TRAIN cohort before constructing an optimizer.
These declared rates and weights are not a claim of optimal hyperparameters.
Their actual learning behavior is a manual VM test, not a local training result.

Finite limits are 900 seconds for cache, 6,300 for fit, 7,200 for the worker,
7,230 external worker seconds with a 30-second kill grace, and 900 export seconds
with 930 external seconds and grace. Peak allocated VRAM is limited to 20 GiB;
return contents to 3.5 GiB; initial free disk to at least 8 GiB with a protected
512 MiB reserve. Cache, update-20 timing and initial output/export storage
projections stop the run when exceeded. No cleanup or automatic retry occurs.

Snapshots and stops preserve the decoder, optimizer, scheduler, Python/NumPy/
Torch/CUDA random state, next schedule index and source hashes. Environment
versions and failure receipts accompany the export. A failed scientific gate
cannot automatically resume from this saved state. The existing local research
cache backup remains separate; no disk snapshot purchase or billing query occurred.

## Completed preparation checks

The producer bound 5,493 packet assets, verified the source copies, produced a
444,114,939-byte archive and made zero neural, gradient or optimizer calls.
Preparation took 100.18 seconds. A separate checker verified archive contents,
hashes, unchanged TRAIN roles and full coverage of all five schedules.

Independent local inference proves exact initial output equality with the
retained DGP on 50 prospective TRAIN cases, with zero centered corrections and
exact outside-support pixels. The original and initial decoder states remain
unchanged. All 781 references pass the three fixed-scale support checks, 2,343
checks in total. The checker also verifies six scientific-gate failure
regressions, six return archive-boundary regressions, schedule mutations,
unchanged gate definitions, Python 3.10 parsing and rejection of local learning
before neural imports. No local autograd, backward or optimizer operation ran.
The independent packet check took 94.39 seconds.

The first checker failed before neural imports because a string search confused
Python's AST field `targets` with a clean-image input. Its source and failure
receipt are retained. R1 inspects actual variable names and forward arguments,
and passes. This changes the independent checker only: the packet, recipe,
scientific gates and archive hashes remain unchanged. Git Bash's first sandbox
syntax check failed on its Windows signal-pipe permissions; the read-only check
outside that sandbox passes. Both outcomes are retained.

## Audit scope and open work

The prospective return auditor is frozen before training. It rechecks every
delivered PNG pixel metric per complete snapshot, all saved raw aggregates and
embedding dot products, full-state fields and schedule/rate logs, and replays
50 raw previews through independent CPU inference per snapshot. Other raw float
arrays are hashed rather than retained. Their complete pixel metrics cannot be
independently recomputed by this checker; it does not claim a full raw audit or
proof of identity preservation. Cross-runtime inference tolerances do not relax
the scientific selection gates. Every preview and native development output
still requires visual review.

No training, VM connection, cleanup, candidate promotion, reserved final
evaluation or app modification occurred. All visible facial features remain
required. Insufficient inputs still need a clearer or less-covered crop. The
seven-family automatic/assisted completion scope and full app/final verification
remain incomplete. This milestone is progress toward the existing goal, not a
smaller definition of completion.

## Transfer and evidence bindings

Protocol SHA256:
`c19ca1790ae9ebe7f99000a81678c8fc756d70ce2114debb6f81691c741a6f2d`.

Execution archive SHA256:
`9988d57e0c3733df0b255c93282b284dd13b98e296fe605de101f3c2972ddf4c`.

- Manual guide: `CCTV_DGP_RESIDUAL_EPOCHS_V42_VM.md`.
- Packet and protocol: `outputs/cctv_dgp_residual_epochs_vm_v42/`.
- Transfer archive: `outputs/cctv-dgp-residual-epochs-v42-execution.tar.gz`
  and its adjacent `.sha256` file.
- Preparation and independent checks:
  `outputs/cctv_dgp_residual_epochs_v42_preparation/`.
- Prospective return checker:
  `scripts/audit_cctv_dgp_residual_epochs_v42_return.py`.
