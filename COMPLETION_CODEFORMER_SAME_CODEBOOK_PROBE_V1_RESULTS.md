# Same-codebook completion probe: code prediction is worth testing

26 October 2026. **Completed representation diagnostic; R1 remains selected.**
No training update, new checkpoint or app change occurred. The result justifies
a different learning-path experiment; it is not a new deployable model.

[All 12 comparison sheets and audits](outputs/completion_codeformer_same_codebook_probe_v1_return/review.html)
show all 48 fixed cases and 192 outputs. These are 32 clear mask cases and 16
synthetic challenges from eight adaptively exposed TRAIN photographs, not 48
independent identities, real CCTV clean references or independent final review.

## What changed in the comparison

Every arm uses the same covered-input context, reviewed removal support, R1
weights and exact app delivery at 256×256. A uses R1's ordinary predicted codes
and fusion at w=1. B switches fusion off. C keeps fusion at w=1 but substitutes
retained clean-reference codes from the **same inpainting codebook**. D combines
those clean-reference codes with fusion off.

**C and D are ORACLES: they use unavailable clean information.** They cannot be
run on a real covered face without its unobscured reference. They test one
representation, not an optimal ceiling or exact hidden identity. No clean RGB
was pasted into the covered-input conditioning. The source-image column is
shown beside the mask overlay; those selected pixels are removed before the
network sees the input.

## Independently recomputed results

Numbers below are relative reductions in delivered PNG **hole-L1 error versus
ordinary R1 (A)**. They are not the restoration model's feature-structure score.
Positive means less average error against the paired reference.

| Condition | 32 clear cases | 16 challenges | Existing aggregate hole/boundary checks |
|---|---:|---:|---|
| A: ordinary R1, fusion w=1 | Reference | Reference | Reference |
| B: actual codes, fusion off | −24.180% | −30.156% | Fail; every individual hole and boundary worsens |
| C: ORACLE clean codes, fusion w=1 | +16.488% | +16.364% | Pass all declared RAW/PNG aggregate subgroups |
| D: ORACLE clean codes, fusion off | +26.016% | +30.160% | Fail boundary checks |

C still worsens three individual holes and ten individual boundaries. Strict
gray inputs can acquire colored eyes/lips: mean generated PNG channel spread
on the eight gray challenges rises from 4.671 to 6.260 levels under C. An
aggregate pass does not make every result correct or resolve that defect.

Root inspected every row/column of all 12 sheets at original detail. C often
improves expression, eye shape or glasses relative to A, but generated regions
remain soft, and some mouths change incorrectly. D sometimes gives clearer
detail while producing hard seams or color/texture patches. B is a worse
ordinary-input alternative. Clear-reference glasses and expressions in oracle
outputs do not prove those hidden details can be inferred from a real covering.

## Integrity and actual execution

- Manual NVIDIA L4 run: 200 forward calls, zero backwards, gradients, teacher
  calls, optimizer objects or optimizer steps. Supervisor elapsed 106.579s;
  no timeout. This is VM diagnostic timing, not Windows CPU app timing.
- Full archive/hash/namespace audit: 414 regular members, 413 payload files.
  Archive SHA256: `e68faa3360b0421c15598a12d472db191395a1790b3b81833f7d611da71577d9`.
- All 497 before/after tensor records exactly match safely reconstructed R1;
  all 165 original prior tensors are unchanged. All 192 code selections match
  their declared actual or same-codebook oracle sources.
- All 196 NPZs pass numeric/schema/CRC checks. All 192 RAW→PNG outputs replay
  exactly, with 33,551,076 outside-mask RGB bytes preserved. Four fresh app
  parity pairs and all 48 historical R1 outputs match exactly.
- A separate root pass rehashed all 413 payloads, recomputed all 192 PNG hole
  and boundary endpoints, and verified all 22 R1 app/model pins unchanged.
- Windows MAX_PATH interrupted the first local extraction. Its incomplete
  attempt remains separately identified; extended paths admitted the complete
  return without changing archive contents or scientific settings.

The empty worker stdout log is consistent with this quiet diagnostic's source,
zero captured bytes and complete per-case/state receipts. It is not substituted
for execution evidence. No learned checkpoint was planned or returned.

## Next decision

Retain R1, its encoder, fusion at w=1 and original prior. Review a **predictor-only
CE experiment** that can change selected codes while keeping those context and
generation paths fixed. The proposed path has 114 existing tensors; the other
383 remain frozen. It must train only on existing FIT identities and labels,
never the 16 monitor-label sets used for this oracle diagnostic.

[Independent predictor design review](research/completion_codeformer_post_probe_predictor_review_v1/independent_design_research.md)
explains the graph and optimizer mapping. A finite 32-step design is under
review; no new execution packet or launch command is ready. Runtime gradient
ownership, mapped Adam state, balanced FIT schedule, exact call/save/resource
bounds and independent source/transfer checks are still required. The 16%
oracle improvement is not a promised or already achieved training gain.

Actual training stays under the user's manual L4/tmux workflow. No historical
probe rerun, failed V2/V3/V4 continuation, app promotion or IMPLEMENTATION port
is authorized by this diagnostic. Restoration files remain separate.

The full goal remains active: useful real covering-family outputs, separately
evaluated automatic proposals, user acceptance, blinded independent final
review, Windows CPU minimum-hardware/fresh-install checks and selective
IMPLEMENTATION integration remain incomplete.

## Evidence

- [Independent transport](outputs/completion_codeformer_same_codebook_probe_v1_return/transport_audit.json)
- [Independent frozen state and code provenance](outputs/completion_codeformer_same_codebook_probe_v1_return/independent_state_code_audit.json)
- [Independent saved-output audit](outputs/completion_codeformer_same_codebook_probe_v1_return/independent_saved_output_audit.json)
- [Root PNG replay and R1 preservation](outputs/completion_codeformer_same_codebook_probe_v1_return/root_endpoint_and_preservation_closure.json)
- [All-case visual observations](outputs/completion_codeformer_same_codebook_probe_v1_return/root_visual_review.json)
