# Original-u64 gradient diagnostic: completed and independently audited

23 October 2026. **The manual diagnostic completed. It made zero model updates.**
The original trained u64 is unchanged; there is no new restoration checkpoint to
promote. Its purpose was to identify why the previous continuation improved an
edge score while worsening clear/blurred photographic fidelity.

## The useful finding

The current four-role training loss couples low-light correction to clear/blurred
appearance changes. The low-light objective has the largest separate gradient:
its share of the sum of four gradient norms averages **62.23%** over the 16 matched
TRAIN photographs, ranging from32.14% to79.17%. Its mean per-photograph norm is
0.301123, against0.044625 localized blur,0.045148 global blur and0.090754 clear.
All these objective gradients already include their single division by four.

Low-light objective gradients oppose localized-blur objectives in14/16 matched
photographs, global blur in15/16, and clear in12/16. The three nondark objective
pairs have positive cosine in every matched photograph. At the pooled role means,
the old equal-role objective direction predicts photographic MAE increasing for
local/global/clear and decreasing for low light. The hypothetical fresh-Adam
first-step formula gives the same four signs. This is measured **initializer-local
coupling**, not a replay or unique causal proof of the failed eight-update history.

Counts below are positive MAE directional derivatives, meaning a predicted local
increase in error. Each column has16 photographs per role. Directions are array
calculations; none was applied to a model.

| Role | Matched-photo descent | Matched-photo fresh-Adam formula | Old offset schedule, descent | Old offsets, fresh-Adam formula |
| --- | ---: | ---: | ---: | ---: |
| Local blur | 4 | 4 | 10 | 11 |
| Global blur | 8 | 10 | 13 | 14 |
| Clear | 7 | 8 | 7 | 9 |
| Low light | 0 | 1 | 1 | 1 |

Recombining the old offsets at the same unchanged initializer does not recreate
its changing parameters or Adam moments. Matching photographs improves several
sign counts but leaves substantial interference. Removing low light is also a
tradeoff: nondark-only descent predicts better MAE for48/48 nondark rolecases,
but worse low-light MAE for15/16. Low-light-only descent improves low light16/16
while worsening local/global/clear MAE14/16,15/16 and12/16 respectively.

The same-role objective is aligned with its photographic MAE gradient in all64
rolecases. Its alignment with the **adjacent-error structure gate** is inconsistent:
opposed directions occur in local7/16, global8/16, clear12/16 and low light8/16.
Thus changing the sampler or dropping the dark loss does not by itself establish
a solution for both fidelity and the structure gate. Gradient magnitude and
angle are not percentages of recovered identity or visible improvement.

## What the return audits establish

Returned archive: **1,226,001,016 bytes**, SHA256
`c052f2b8c09e42d6ee82c263e5a9b1777b3eeb114c6e111a82ba46b4b1862be7`.
Protocol SHA256
`07d00785b4aef82c2e915a032fa2808064d0f728972f8c518694698a13f5d5e3`.

- Independent transport checks verify single-gzip CRC/ISIZE, complete USTAR
  checksums/framing,796 unique confined regular members,1,332,595,869 payload
  bytes and3,072 final zero bytes. Actual member hashes match the export index;
  all20 static files match the reviewed seal. All469 parent bindings were
  rehashed locally, using only the two previously declared local model aliases.
- All799 events replay to264 exact reservation/return pairs:72 forwards and192
  gradient queries, no pending/unknown operations. Worker exit0, no timeout and
  reaped process-group cleanup are saved. The worker completed in approximately
 112 seconds;682 resource samples and77 phase returns respect their limits.
  Maximum sampled RSS3,353,632,768 bytes, CUDA reserved5,484,052,480 bytes and
  minimum free disk151,453,745,152 bytes. Final supervisor/export process exit
  receipts are excluded by the export design and are not claimed here.
- Saved-pixel audit passes2,466 checks:72 exact RAW-to-floor-PNG quantizations,
 216 stock-array shape/value hashes, four historical original-u64 RAW/PNG
  witnesses and four six-output repeats. All64 photographic objectives and192
  query scalars independently agree. Maximum query arithmetic difference is
 4.1444e-8, below the declared5e-7 numerical readback bound. Native/unclipped
  resize differences are at most one FP32 epsilon. Quality tolerance stays zero.
- All192 gradient NPZ files have the exact36 finite FP32 tensors/168,960
  coordinates, correct canonical/array-description hashes, slot/query/region
  joins and scalar bindings. No unused or zero gradients were reported or found.
  Independent reduction arithmetic crosschecks8,682 numeric values in34
  summaries, maximum difference8.88e-16. All326 consumed inputs remain exact.
- Initial and closing575-tensor NPZ files are byte-identical to the independently
  audited original-u64 initializer,377,504,068 bytes each. All575 array values
  and canonical base515/adapter24/head36 namespaces verify; all539 frozen values
  and the trained36 heads remain unchanged. Root inspected all18 planned saved
  sheets, covering72 actual outputs once. Independent readback verifies every
  one of216 source RGB cells without resampling or enhancement.

The photographs are16 reused optimization TRAIN references, plus four-role
historical/repeat evidence on one adaptive TRAIN monitor photograph. There are
no native CCTV, wild-image, DEV or reserved FINAL gradient queries. Correlated
rolecases are not independent identities. The unchanged u64 images retain their
existing smoothing, appearance differences and incomplete low-light correction.
This is not independent human FINAL review, identity proof or Zamboanga evidence.

## Next improvement direction supported by the measurements

A **separate, read-only design exploration**, after the diagnostic audit, tested
the eight pooled photographic gate gradients: four roles × MAE/adjacent error.
A minimum-norm convex combination of their unit-normalized vectors gives a
direction with strictly negative dot products against all eight original pooled
gradients in the current head space. Minimum normalized descent margin is
0.182066884. This is mathematical evidence that adding another architecture is
premature; the existing heads admit a locally favorable pooled direction.

The same direction improves116/128 individual derivative signs but worsens12/128.
Pooled descent is not an all-case guarantee. The naive mean of unit gradients
fails the low-light MAE direction. Monitor and legacy-guard gradients were not
measured, and no finite step or PNG outcome was tested. The exploration is not
part of the sealed run, a model optimizer, a parameter application, an accepted
training recipe or permission to skip original gates.

The next source-only design therefore prioritizes **gate-aligned head updates**
from exact trained u64: jointly account for photographic fidelity and adjacent
structure, prevent the low-light loss from dominating through its magnitude,
and test actual RAW/PNG and unchanged preservation requirements after every
finite update. It must use a fresh reviewed manual-VM protocol. No worker,
transfer packet or launch guide for that new design is ready from this return
audit. The executed gradient diagnostic must not be rerun or resumed.

This design direction is informed by multi-objective optimization, rather than
a claim that those papers prove useful CCTV restoration.
[Sener and Koltun, NeurIPS 2018](https://papers.nips.cc/paper_files/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html)
formulate shared-task training as multiple objectives; their task benchmarks are
different from ours. [Liu et al., NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html)
study conflicting task gradients and worst-task local improvement. Neither is a
guarantee about our finite steps, PNGs, identities or generalization.

## Retained status and evidence

**Status: COMPLETED_ZERO_UPDATE_DIAGNOSTIC_AUDITED_NOT_A_NEW_MODEL.** Original
trained u64 stays a retained challenger; U96 stays the experimental app incumbent.
The failed eight-update continuation and broader inherited DEV failures remain
binding. Automatic goal advancement did not trigger. Original weights, splits,
all states, gradient evidence, stop records and predecessor models stay retained.

Reviewed Protect/Apply/bypass stays separate. Automatic blur matching and later
completion development remain deferred. Native/wild are unpaired; reserved FINAL
is unopened/execution-unfrozen with independent reviewers pending. All actual
parameter application/training remains manual on the L4 VM. These local audits
used file/NumPy/PIL arithmetic only: no Torch, model construction, new neural
forward, actual gradient, optimizer, parameter application or VM call. Full goal
remains ACTIVE/incomplete.

Authoritative receipts are under
`outputs/codeformer_u64_role_gradients_v1_return_transport/`,
`outputs/codeformer_u64_role_gradients_v1_return_pixels/`,
`outputs/codeformer_u64_role_gradients_v1_return_geometry/`, and
`outputs/codeformer_u64_role_gradients_v1_return_audit/`.
