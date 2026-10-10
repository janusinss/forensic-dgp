# Source and degradation conflicts before further DGP training

Prepared diagnostic design, 10 October 2026. The current DGP and application
remain unchanged. New training is not authorized for automatic execution.

The reviewed original-loss-balance return raised delivered structure gain above
1% on two small TRAIN cohorts, but none of its nine directions passed appearance
preservation. Adding a mean identity gradient reduced failures without protecting
clear-image MSE, SSIM and identity consistently. A mean objective is therefore
an inadequate proxy for the existing source/profile preservation requirements.
These results do not establish that more epochs will improve useful CCTV faces.

Use the unchanged isolated copy of checkpoint
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`,
with the same 158 connected feature/decoder tensors, frozen normalization,
recognizer, masks, geometry, targets, 100 cases and two photographic TRAIN
cohorts. The first 50 cases supply derivatives; the other 50 are a cross-check,
not an independent DEV or final set. No native CCTV or reserved final identities
enter this diagnostic. Existing targets, overlap limits and terms remain pinned.

1. Check forward loss values against the retained metrics before derivatives.
   All 100 cached initial outputs are checked. MSE uses the retained float32
   squared differences; SSIM uses a differentiable float64 7-pixel uniform-window
   surrogate with sample covariance; structure uses the retained luma and
   13-pixel Gaussian on eroded landmark support with exact uint8/255 targets.
   SSIM is a surrogate, with prospective absolute parity limit 3e-5. The separate
   scientific SSIM preservation allowance remains 1e-6. The local failed target
   rounding check and corrected forward-only check are retained and hash-bound.
2. Query 32 group derivatives: each of two sources and five profiles has MSE,
   SSIM loss and fixed observed ArcFace loss; each source additionally has a
   degraded landmark structure loss. Ten five-profile reference batches produce
   160 autograd queries, averaged across five references per source. Save all
   32 float64 aggregate vectors and each query's value, connection/zero status,
   vector hash and parameter norms. Individual query vectors are not exported:
   their autograd results and aggregation cannot be independently replayed on
   the local host under the manual-VM gradient boundary. The audit verifies
   source binding, metadata, saved-vector arithmetic and fresh forward outputs.
3. Compare the previous balanced direction against all 32 group gradients.
   Independently calculate their norms and cosine matrix. A bounded simplex
   minimum-norm solve proposes a direction only if every normalized group has
   a descent cosine of at least 1e-7. This is a local derivative certificate for
   the sampled losses, not a proof of general feasibility or useful restoration.
   A failed certificate completes a diagnostic without trials or training.
4. If a direction is certified, test three independent reset displacements:
   decoder weight L2 times 1e-5, 1e-4 and 1e-3. Save float32 parameter vectors,
   quantization-aware derivative predictions, all 100 raw and delivered outputs
   per variant and the unchanged scientific decisions. Each trial resets to the
   original state. No trajectory, optimizer, epoch or trained checkpoint exists.
5. Independently audit the return and visually review every saved output before
   considering a new training recipe. All 17 aggregate/source/profile groups
   retain MSE, SSIM and ArcFace preservation checks in both raw and PNG stages.
   Require 1% early structure gain, nonnegative gain in each degraded source and
   at most 20% brightness-only explanation. Final useful DEV/native outputs,
   separate final identities and the full application remain mandatory.

Finite limits: 1500 seconds worker, 1530 external plus 30 seconds kill grace;
cache 120, gradients 480, dual solve 120, finite trials 300 seconds. Export is
300 seconds, external 330 plus 30 seconds grace. Require 4 GiB free after
installation, keep 512 MiB reserve, bound uncompressed return to 1.5 GiB and
allocated GPU memory to 20 GiB. At most 20 original, 90 candidate and 110
recognizer forward calls. Preserve partial failures and refuse reruns/resume.
The manual tmux wrapper exports evidence and starts no follow-on process.

The rationale follows the common-descent formulation of
[Sener and Koltun, Multi-Task Learning as Multi-Objective Optimization](https://arxiv.org/abs/1810.04650)
and the concern about individual objectives in
[Liu et al., Conflict-Averse Gradient Descent for Multi-task Learning](https://papers.nips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html).
This packet is a bounded diagnostic inspired by those formulations, not an
implementation or performance claim for either paper. Neither source establishes
CCTV facial usefulness, preservation or forensic identity recovery here.

This addresses milestone 3 only. The five restoration milestones, native CCTV
review and all seven completion families remain active and incomplete. The app
continues to use its retained DGP; no diagnostic direction is promoted.
