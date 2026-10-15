# Completed loss-direction R1 diagnostic — 11 October 2026

**R1 completed and its layout repair passed on the L4.** The run measured loss
gradients without any optimizer updates. It produced useful diagnostic evidence,
but no newly trained or qualified restoration model.

The diagnostic took **112.07 seconds** (about 1 minute 52 seconds),
with **3.162 GiB peak allocated VRAM**. It made 80 B
forwards, 40 fixed-recognizer forwards and **360 gradient queries**, with zero
original-model forwards, backwards calls, optimizer updates or epochs. Export
took 20.93 seconds. The 437,166,689-byte return SHA256 is
`976dc758f21f1f45572e65ed12d94eb3658ec35f7d4be70a2455d59b1b42d376`. Python 3.10.12/PyTorch 2.9.1+cu129/NVIDIA L4 and
human manual tmux were recorded. No agent VM compute or connection occurred.

Independent checks covered **134 returned files**, all member paths/sizes/hashes,
**5,852 parent assets**, exact input strides and **40 raw blocks/200 case replays**
against the saved initial/B50 controls. The initial and rejected state hashes
match. The old layout stop remains retained. All replay errors are exactly zero.
These are frozen controls of already reviewed outputs: no new PNG or visual-quality
comparison was generated, and no independent final review is claimed.

All **40 scalar component rows** were independently reconstructed from saved
pixels/targets/embeddings, with maximum error 4.62e-13.
All summary matrices were checked for partition sums, symmetry, finite values
and positive semidefinite consistency. **Four full gradient samples**, one per
source per state, retained **370 arrays each**. Their Gram matrices, direction
products, norms/clipping and hypothetical Adam directions were independently
recomputed in NumPy. This checks the arithmetic on saved derivatives; it does
**not** independently re-run all 40 neural derivatives. Numerical tolerances
apply to arithmetic checks, not scientific gates or exact raw replay.

One local audit initially required exact equality of a derived seven-component
sum. Python 3.10 versus newer float summation produced differences at most
8.88e-16; that failed check and its differences were
retained. The derived-sum audit now allows 2e-14. The 1% structure, 20% brightness,
source/profile preservation and raw exact replay requirements did not change.

## Why the next change addresses a demonstrated cause

At the initial model, every brightness penalty value was **exactly zero**, but
every measured reference had a **nonzero brightness-loss gradient**. The assumed
relationship “zero penalty means inactive gradient” was false for this helper.
The helper used three `clamp_min(0)` hinges. PyTorch 2.9.1 defines a passing
gradient at equality for that operation; the boundary convention differs from
an inactive ReLU hinge. [Recorded-version PyTorch derivative source](https://github.com/pytorch/pytorch/blob/v2.9.1/tools/autograd/derivatives.yaml#L407-L408).

| Measured state | Zero brightness value with active gradient | Median brightness/detail gradient-norm ratio | Nominal Adam detail-error ascents | Nominal Adam degraded-MSE ascents |
|---|---:|---:|---:|---:|
| Pinned initial | 20 / 20 | 54.28× | 1 / 20 | 20 / 20 |
| Rejected B50 | 0 / 20 | 75.85× | 4 / 20 | 19 / 20 |

These counts are **20 photographic TRAIN references per state**, ten per source,
five frozen profiles each. They are not native CCTV outcomes or failed-person
counts. Positive directional products mean predicted local increases in a loss,
not an actually applied parameter step. The measured full objective's nominal
direction descends on all 20 references at both states while degraded MSE rises
on 20/20 initially and 19/20 at B50. A lower combined objective can therefore
conceal competing reconstruction directions.

At the initial state, the guard is about **54.28×**
the detail gradient by median norm, and its median cosine with degraded-MSE
gradient is **-0.973**.
The B50 ratio is **75.85×**.
Correcting boundary behavior is justified before increasing epochs. It does not
prove this is the only cause: summed gradients without the guard still oppose
degraded MSE on 15/20 initial and 20/20 B50 references. Clear/blur anchors and
other objective interactions remain unresolved. No guarantee of useful improvement
or global convergence follows from these first-order findings.

Analytical first-step Adam counterfactuals use the two **initial** saved gradient
samples, setting only the inactive guard derivative to zero and preserving the
optimizer math. The FFHQ-source sample then descends in detail, clear MSE and
degraded MSE; the sample from `dataset/asian_faces` still predicts a degraded-MSE ascent.
These calculations use saved arrays with no parameter displacement or new
gradients. They are not an accumulated five-reference update or a trained method.

## Selected finite prerequisite and decision

The next manual packet changes the **three nonnegative hinges from clamp_min to
ReLU**, making the guard inactive at exactly zero. The scalar formula, coefficient
10, fraction .2, masks, targets and normalization remain identical. Independent
forward-only checks on all **40 rows/200 case compositions** verified **exactly
equal penalty values and metadata** before and after the change. Eight synthetic
sign/boundary inputs also matched. No local gradients were computed.

The **258,033-byte packet** retains both initial/B50 states and the
same 20 TRAIN references, 100 distinct cases, raw replay, scientific gates, loss
weights and hypothetical Adam configuration. It is a declared derivative change,
not an unchanged failed recipe, a guard removal, or failed-state continuation.
It requires zero initial guard gradients on all 20 references and checks scalar
parity against the retained helper during execution. It makes **zero optimizer
updates**, with no checkpoint or automatic follow-on training.

Transfer checks passed for four packet assets, five archive members, all 5,852
parent hashes, source guards, LF checksum, readonly Bash syntax and actual
Windows rejection before neural/gradient imports. Independent return scripts are
prepared for archive safety, raw/scalar replay, sampled gradient arithmetic and
the inactive-guard condition. Their pure boundary validator rejects the actual
old defect and accepts a declared synthetic zero-gradient summary; this fixture
is not model evidence. Actual corrected L4 gradients and the new return remain unrun.

Success of this diagnostic confirms the corrected derivative condition and lets
us assess remaining directions. It does **not** qualify extra epochs. Persistent
reconstruction conflicts require a recorded supervision/anchor review before any
new training recipe. A failed invariant/resource stop stays retained. No old
diagnostic is automatically rerun, and the rejected B50 state is diagnostic only.

Require **4 GiB free after installation**, idle L4 and manual tmux. Unmeasured
estimate: **2–5 minutes measurement plus 1–3 minutes export**, informed by R1's
112-second measurement and 21-second export; new checks add work. Worker,
measurement and export limits remain **900/600/300 seconds**; measurement stop
is checked between reference batches. External caps **930/330** have 30-second
kill grace. VRAM 20 GiB, output/reserve 1 GiB each; combined raw/gradient payload
upper bound 712,406,496 bytes. First two batches retain the finite projection stop.
No current VM power, free-space or live credit balance is claimed.

[Five exact upload, tmux and download steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_HINGE_BOUNDARY_V1_VM.md>).
[Independent R1 return audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_loss_alignment_v1_r1_analysis/independent_audit.json>);
[Loss-direction analysis](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_loss_alignment_v1_r1_analysis/loss_direction_review.json>);
[New scalar-loss equivalence audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_hinge_boundary_v1_loss_audit.json>);
[Verified next transfer](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_hinge_boundary_v1_transfer_audit.json>).

## Full goal remains required

The permanent app checkpoint remains
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`:
the fine-tuned DeblurGAN-v2-compatible baseline. Candidate B declares the external
frozen StyleGAN2 bank and our conditioning/reconstruction; no own-trained StyleGAN
or exact published-DGP claim is made. Checkpoints, full states, splits, research
caches, provenance and failures remain. No model or existing app design changed.

No native/final pixels entered this diagnostic. Native CCTV remains unpaired,
paired TRAIN stays separate, and final identities are outside tuning. Source
labels establish neither ethnicity nor Zamboanga performance; no real Zamboanga
samples exist. All five milestones still apply: frozen data/criteria, identical
256 baseline/candidate comparisons, finite justified improvements, qualified
primary DGP Auto/override integration, and independent DEV/final review plus
meaningful regressions and bundled inline Playwright app verification.

Completion remains separate for masks, sunglasses, strong glare, hands,
obstructing hair, scarves and objects. Automatic removal-area preview with
optional correction, preserved clear glasses/ordinary hair/visible appearance,
a small documented margin, original/mask/one plausible estimate, PNG/optional
bundle downloads and separate automatic/assisted results remain required.
Request clearer or less-covered inputs when information is insufficient; model
softness cannot exclude an input already labelled usable. Exact hidden identity
is not claimed. All seven covering families and independent final review remain
pending. Manual VM training, finite stops, migration/credit notices and the
November 2026 deadline remain governed by the canonical plan. Goal **ACTIVE**.
