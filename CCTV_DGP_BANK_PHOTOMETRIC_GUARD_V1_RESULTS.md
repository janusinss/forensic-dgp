# Brightness-guard return and next diagnostic — 11 October 2026

The downloaded experiment trained correctly and stopped on the retained quality
requirements after **50 updates**. It is not a qualified replacement. The user
judged the fifth column on the first fixed native sheet **much better than the
original DGP** and requested the best next approach. That development feedback
is retained separately from numerical gates and independent final review.

| Candidate | Raw / PNG landmark-detail error reduction | Raw / PNG overlapping group-metric failures |
|---|---:|---:|
| Earlier B r4 | 0.625591% / 0.627191% | 2 / 3 |
| Brightness guard | 0.680385% / 0.681828% | 4 / 4 |

The early detail requirement is **1%**. Brightness-only contribution is
**34.582900% raw / 34.508277% PNG**, above
the retained **20%** maximum. Both source-label blur groups regress in MSE;
the Asian-source-label compound group regresses in observed ArcFace similarity.
The older low-light MSE failures are absent, but the new tradeoffs prevent a pass.
Failure counts overlap groups and metrics; they are not counts of failed people.
This is an intentional supervisor quality exit **2**, not a loading exception.

The structure score is **landmark_high_frequency_MSE**, a paired photographic
TRAIN proxy. It is not recognizer feature MSE or a percentage of recovered
identity. Recognition similarity is a separate preservation measure. The 24
native ChokePoint DEV crops have no aligned clean targets and no native PSNR/SSIM.
Source labels do not establish ethnicity or Zamboanga performance.

## Execution and independent checks

The L4 made 50 optimizer updates using 250 distinct TRAIN references, 125 per
source, and 1,250 profile exposures: **zero complete epochs**, 0.3201 of the
781-reference pool. Actual fitting took **2.62 minutes**;
the **23.88-minute worker** also included baseline,
preflight and snapshot work. Archive export took about **4.20 minutes**.
This does not establish the original checkpoint's complete ancestral epoch count
or authorize the conditional additional-epoch proposal.

The 5,217,289,325-byte returned archive has SHA256
`f2ff55481a68d1690a3f7463384cd337c4dad238fcf8b8a600362bed54e2d356`.
Import verified 7,702 regular members. Independent checks covered **7,910 paired
raw/PNG compositions**, **72 native compositions**, strict full state, six
CPU/CUDA prior fixtures and two limited fresh CPU replay batches. This is not
a full fresh neural replay of every saved case. All **781 baseline RGB/embedding
packs** and 24 native baselines match B r4. All **37 trainable tensors changed**;
755 frozen state tensors stayed exact. Full model/optimizer/scheduler/RNG state
is retained; failed-state training continuation stays prohibited.

The new guard contribution was positive in **48 of 50 updates**. Independently
recomputed final per-image active hinges fell from **2,184 to 1,717 of 3,124**
degraded cases. It changed learning pressure, but did not enforce every final
preservation and brightness requirement. Changing-batch objectives do not prove
plateau, convergence or a unique cause of failure. Exposed and unexposed members
of this same TRAIN pool remain TRAIN evidence, not independent DEV identities.

## Visible outputs and the selected next approach

All **26 sheets** were displayed at their original resolution and all **844
source cells** were independently checked. They cover 100 paired TRAIN previews
and all 24 fixed native DEV inputs. Clearer native faces retain hair, glasses and
visible facial texture better than the original baseline's smoothing. Darker or
turned cases remain soft around eyes, nose and mouth. Differences from B r4 are
mostly subtle; the user's statement compares the guard against the original,
not against B r4. The review was by the implementing assistant and the user on
the first native sheet; independent final review remains pending. Input-only
usable labels are retained. Model softness cannot relabel a usable input as
insufficient information.

On the fixed 100-case bank ablation, **80 degraded TRAIN cases** have raw detail
gains of **0.617096% enabled / 0.716408% disabled**. Disabling
the bank also reduces pooled degraded MSE, but worsens clear MSE, SSIM, similarity
and detail relative to the enabled model. Some individual profile effects differ
from the pooled result. Only nine of the twenty preview references were exposed
in the 50-update schedule. This is the same rejected jointly trained B50 decoder
with its bank switched off, not an independently trained decoder control or a
full 3,905-case bank-disabled gate. Global bank disabling is not selected.

**Next: a frozen feature-path diagnostic before another training recipe.** Keep
the saved weights fixed; compare enabled, disabled, and dropping one of the five
16/32/64/128/256 readouts. Each uses the same 100 TRAIN cases and 24 native DEV
crops. It must prove zero-readout parity and exact same-L4 enabled/disabled saved
output and embedding replay. Seven variants mean 700 paired and 168 native
compositions. No gradient, backward, optimizer update, new checkpoint, epoch,
inference strength selection or app promotion occurs. One-path removal measures
finite responses of this checkpoint; it cannot prove the capacity of a separately
trained architecture or select a deployable model by itself.

GLEAN studies how multiresolution encoder, bank and decoder features affect
restoration. That supports inspecting the contribution of each bank path rather
than assuming that a frozen generative prior automatically helps every input.
This is a design inference from its ablations, not a reproduction or evidence
of our CCTV performance. [Primary GLEAN paper, section 5](https://arxiv.org/html/2012.00739).
The prior DC-centering probe remains rejected; this diagnostic does not repeat
global feature centering or alter a scientific gate. No new clean-target
acquisition is justified by a 128-source-size metadata field: the existing 391
HQ FFHQ counterpart targets remain bound in the audited packet.

After the three unsuccessful A/B/guard comparisons, the AGENTS.md diagnostic
question was asked. The user chose the visibly better fifth column and the best
next approach. That resolves the design discussion for this bounded diagnostic;
it does not waive preservation or authorize unchanged failed training.

## Prepared manual diagnostic and remaining goal

The independently checked transfer is **253,174 bytes**, with
three members and **5,852 hash-bound parent assets**. No large weight/data copy
is needed. Source/data/archive checks, Python 3.10-compatible hashing, readonly
Bash syntax and actual Windows rejection before neural imports passed. The first
Bash check was blocked by the Windows sandbox signal pipe; its log is retained
and the same readonly check passed outside the sandbox. This was an audit-runtime
issue, not a training or model-source change.

Actual CUDA diagnostic parity, timing, outputs and visual review are **unrun**.
Require **4 GiB free after installation**, an idle L4 and manual tmux. Estimate
**5–15 minutes inference plus 1–5 minutes export**; this new timing is unmeasured.
Finite limits: worker 20 minutes/external 20.5, inference 15, export 5/external
5.5, 30-second kill grace, VRAM 20 GiB, output 1 GiB, disk reserve 1 GiB,
312 B forwards and 140 recognizer forwards. Preserve any stop. No agent VM
connection, training, gradient work, cleanup or VM power change occurred in
this return audit/preparation turn. Old free-space snapshots are not current.

[Five exact manual upload/tmux/download steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_PATH_ABLATION_V1_VM.md>).
[Numerical and state audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_photometric_guard_v1_analysis/independent_audit.json>);
[Matched trial comparison](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_photometric_guard_v1_analysis/matched_guard_comparison.json>);
[Visual closure](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_photometric_guard_v1_analysis/visual_review.json>);
[Saved bank attribution and user feedback](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_photometric_guard_v1_analysis/saved_bank_attribution.json>).

The permanent app checkpoint remains
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
The custom replacement trains our reconstruction and conditioning while declaring
its external frozen StyleGAN2 bank; it is not an own-trained StyleGAN generator
or an exact published-DGP implementation. Preserve app design, source/split
roles, terms, caches, all historical failures and original states. Pretrained
restorers remain comparators. No real Zamboanga samples exist; final identities
remain outside tuning. All five milestones stay required, including identical
resize/Phase3/pretrained comparisons, useful DEV and independent final review,
qualified primary DGP Auto/override integration, meaningful regressions and
bundled inline Playwright app checks.

Completion remains separate and must cover masks, sunglasses, strong glare,
hands, obstructing hair, scarves and objects, with automatic removal-area
preview/correction, preserved clear glasses/ordinary hair/visible appearance,
original/mask/one plausible estimate, PNG and optional bundle downloads, clearer
or less-covered requests and separate automatic/assisted review. This experiment
qualifies none of those covering families. Goal **ACTIVE and incomplete**.
