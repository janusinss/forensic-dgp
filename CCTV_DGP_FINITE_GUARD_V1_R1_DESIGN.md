# Finite preservation mechanics, storage-corrected packet

Preparation revision, 10 October 2026. Follow the frozen design in
CCTV_DGP_FINITE_GUARD_V1_DESIGN.md. The user chose the existing NVIDIA L4 VM.
The VM is stopped at the last verified API inspection; no startup, guest
inventory, deletion or actual training has occurred during this preparation.

The first finite-guard packet passes baseline CPU inference parity and local
training denials. An additional file-size check found its redundant
preceding-anchor diagnostic PNG would make the conservative projected return
3,245,653,754 bytes, above the3GiB cap. Retain that original packet, its source
hashes and its audit. It is superseded and must not be run. The correction is
recorded in outputs/cctv_dgp_finite_guard_v1_preparation/storage_preflight_correction.json.

R1 preserves every original and proposal raw/PNG output, embeddings, the
original-anchor mean-only diagnostic PNG, and both anchor metrics/shifts.
Reconstruct the preceding-anchor mean-only image exactly from the saved preceding
raw image and proposed raw image rather than saving an additional duplicate PNG.
Independent audit recomputes that image and its delivered metric from the saved
inputs. The finite acceptance rule, all scientific thresholds and original1%
quality requirement remain unchanged. Add2MiB per100-case variant for the new
anchor metadata to the prospective storage projection. No primary evidence,
rejected placement, gradient matrix, accepted trained copy or failure is omitted.

Both named sampled TRAIN cohorts participate in fitting, with64 distinct
cohort/source/profile derivative groups,320 queries per state, at most960 queries,
three accepted training changes and nine finite proposals. Recompute after every
accepted change. Compare actual raw and delivered metrics against original and
preceding state with the appropriate mean-only anchor. All rejected proposals
are retained. Accepting a microchange counts as actual training; it is not an
epoch,1% quality pass or qualified model. No resume or automatic continuation.

Step fractions are2e-5,1e-5,5e-6 times the fixed original decoder L2. Worker1800s,
external1830s, per-state derivatives240s/solver120s/proposals180s, cache120s,
export360s/external390s and30s termination grace are finite. Require7GiB free
after installation,512MiB reserve,3GiB retained return cap and20GiB allocated
VRAM. Runtime is estimated10–20 minutes plus1–6 minutes export using the prior
L4 receipts; the revised64-group trace has not been timed. Dynamic projection
and conservative per-file stops apply. No cleanup is included in the packet.

The new sources/protocol/transfer and prospective return checker need independent
verification before manual launch. The current app checkpoint remains exact.
Useful native DEV outputs, full-corpus/paired DEV preservation, independent final
review, all covering families and bundled inline Playwright flow are still
required. No native CCTV, DEV, final identities or new data enter this mechanics
study. Saved aggregate/certificate arithmetic can be audited locally; individual
autograd queries remain unreplayed locally. The full goal remains incomplete.
