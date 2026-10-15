"""Publish completed R1 measurement, boundary diagnosis and unrun manual test."""
from datetime import datetime, timezone
from pathlib import Path

from import_cctv_dgp_bank_loss_alignment_v1_r1_return import ROOT, OUT, RETURN, ANALYSIS, PARENT, read, sha, write
from publish_cctv_dgp_bank_AB_and_guard_v1_review import prepend_locked


def link(label, name):
    return '['+label+'](<'+(ROOT/name).as_posix()+'>)'


def new(path, text):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)


def main():
    receipt = ANALYSIS / 'publication.json'
    assert not receipt.exists()
    evidence = [ANALYSIS / 'import.json', ANALYSIS / 'independent_audit.json',
                ANALYSIS / 'audit_scalar_sum_roundoff.json', ANALYSIS / 'loss_direction_review.json',
                OUT / 'cctv_dgp_bank_hinge_boundary_v1_loss_audit.json',
                OUT / 'cctv_dgp_bank_hinge_boundary_v1_preparation.json',
                OUT / 'cctv_dgp_bank_hinge_boundary_v1_transfer_audit.json',
                OUT / 'cctv_dgp_bank_hinge_boundary_v1_return_checker_preparation.json']
    assert all(read(path)['complete'] for path in evidence)
    imported, audit, roundoff, review, scalar, prep, transfer, checker = [read(path) for path in evidence]
    assert audit['raw_case_replays_checked'] == 200 and audit['sampled_full_gradient_batches_checked'] == 4
    assert transfer['preparation_sha256'] == sha(evidence[5])
    assert checker['protocol_sha256'] == prep['protocol_sha256']
    assert sha(OUT / (prep['stem']+'-execution.tar.gz')) == prep['archive_sha256']
    assert not (OUT / prep['name'] / 'outputs').exists()
    protected = read(ANALYSIS / 'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT / name) == expected for name, expected in protected.items())
    report_name = 'CCTV_DGP_BANK_LOSS_ALIGNMENT_V1_R1_RESULTS.md'
    guide_name = 'CCTV_DGP_BANK_HINGE_BOUNDARY_V1_VM.md'
    states = review['group_summaries']['all']
    table = []
    for state, title in [('initial', 'Pinned initial'), ('joint50', 'Rejected B50')]:
        r = states[state]
        table.append(f"| {title} | {r['scalar_zero_brightness_penalty_with_nonzero_gradient']} / 20 | {r['median_brightness_gradient_norm_over_detail_gradient_norm']:.2f}× | {r['nominal_single_reference_Adam_direction_detail_ascents']} / 20 | {r['nominal_single_reference_Adam_direction_degraded_MSE_ascents']} / 20 |")
    report = f"""# Completed loss-direction R1 diagnostic — 11 October 2026

**R1 completed and its layout repair passed on the L4.** The run measured loss
gradients without any optimizer updates. It produced useful diagnostic evidence,
but no newly trained or qualified restoration model.

The diagnostic took **{audit['VM_seconds']:.2f} seconds** (about 1 minute 52 seconds),
with **{audit['VM_peak_VRAM']/1024**3:.3f} GiB peak allocated VRAM**. It made 80 B
forwards, 40 fixed-recognizer forwards and **360 gradient queries**, with zero
original-model forwards, backwards calls, optimizer updates or epochs. Export
took 20.93 seconds. The {imported['archive_bytes']:,}-byte return SHA256 is
`{imported['archive_sha256']}`. Python 3.10.12/PyTorch 2.9.1+cu129/NVIDIA L4 and
human manual tmux were recorded. No agent VM compute or connection occurred.

Independent checks covered **134 returned files**, all member paths/sizes/hashes,
**5,852 parent assets**, exact input strides and **40 raw blocks/200 case replays**
against the saved initial/B50 controls. The initial and rejected state hashes
match. The old layout stop remains retained. All replay errors are exactly zero.
These are frozen controls of already reviewed outputs: no new PNG or visual-quality
comparison was generated, and no independent final review is claimed.

All **40 scalar component rows** were independently reconstructed from saved
pixels/targets/embeddings, with maximum error {audit['maximum_scalar_component_error']:.3g}.
All summary matrices were checked for partition sums, symmetry, finite values
and positive semidefinite consistency. **Four full gradient samples**, one per
source per state, retained **370 arrays each**. Their Gram matrices, direction
products, norms/clipping and hypothetical Adam directions were independently
recomputed in NumPy. This checks the arithmetic on saved derivatives; it does
**not** independently re-run all 40 neural derivatives. Numerical tolerances
apply to arithmetic checks, not scientific gates or exact raw replay.

One local audit initially required exact equality of a derived seven-component
sum. Python 3.10 versus newer float summation produced differences at most
{roundoff['maximum_absolute_error']:.3g}; that failed check and its differences were
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
{chr(10).join(table)}

These counts are **20 photographic TRAIN references per state**, ten per source,
five frozen profiles each. They are not native CCTV outcomes or failed-person
counts. Positive directional products mean predicted local increases in a loss,
not an actually applied parameter step. The measured full objective's nominal
direction descends on all 20 references at both states while degraded MSE rises
on 20/20 initially and 19/20 at B50. A lower combined objective can therefore
conceal competing reconstruction directions.

At the initial state, the guard is about **{states['initial']['median_brightness_gradient_norm_over_detail_gradient_norm']:.2f}×**
the detail gradient by median norm, and its median cosine with degraded-MSE
gradient is **{states['initial']['median_guard_vs_degraded_MSE_gradient_cosine']:.3f}**.
The B50 ratio is **{states['joint50']['median_brightness_gradient_norm_over_detail_gradient_norm']:.2f}×**.
Correcting boundary behavior is justified before increasing epochs. It does not
prove this is the only cause: summed gradients without the guard still oppose
degraded MSE on 15/20 initial and 20/20 B50 references. Clear/blur anchors and
other objective interactions remain unresolved. No guarantee of useful improvement
or global convergence follows from these first-order findings.

Analytical first-step Adam counterfactuals use the two **initial** saved gradient
samples, setting only the inactive guard derivative to zero and preserving the
optimizer math. The FFHQ-source sample then descends in detail, clear MSE and
degraded MSE; the Asian-capture-source sample still predicts a degraded-MSE ascent.
These calculations use saved arrays with no parameter displacement or new
gradients. They are not an accumulated five-reference update or a trained method.

## Selected finite prerequisite and decision

The next manual packet changes the **three nonnegative hinges from clamp_min to
ReLU**, making the guard inactive at exactly zero. The scalar formula, coefficient
10, fraction .2, masks, targets and normalization remain identical. Independent
forward-only checks on all **40 rows/200 case compositions** verified **exactly
equal penalty values and metadata** before and after the change. Eight synthetic
sign/boundary inputs also matched. No local gradients were computed.

The **{prep['archive_bytes']:,}-byte packet** retains both initial/B50 states and the
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

{link('Five exact upload, tmux and download steps', guide_name)}.
{link('Independent R1 return audit', 'outputs/cctv_dgp_bank_loss_alignment_v1_r1_analysis/independent_audit.json')};
{link('Loss-direction analysis', 'outputs/cctv_dgp_bank_loss_alignment_v1_r1_analysis/loss_direction_review.json')};
{link('New scalar-loss equivalence audit', 'outputs/cctv_dgp_bank_hinge_boundary_v1_loss_audit.json')};
{link('Verified next transfer', 'outputs/cctv_dgp_bank_hinge_boundary_v1_transfer_audit.json')}.

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
"""
    stem, pin = prep['stem'], prep['protocol_sha256']
    guide = f"""# Inactive-guard diagnostic: five exact manual steps

**Prepared and independently transfer-verified; corrected L4 gradients are unrun.**
R1 completed and exposed a brightness guard with zero value but active gradient.
This packet tests the corrected boundary derivative with **zero optimizer updates**.
Scalar penalties, checkpoints, case lists and scientific gates remain preserved.
{link('Audited R1 findings and finite decision', report_name)}.

Upload **{prep['archive_bytes']:,} bytes**. Preserve both earlier loss diagnostics
and `~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm` (5,852 required assets).
Require **4 GiB free after installation** and idle L4. Estimate **2–5 minutes
measurement plus 1–3 minutes export**, unmeasured for this packet. Worker/
measurement/export caps 900/600/300 seconds (measurement checked between batches),
external 930/330 with 30-second kill grace; VRAM 20 GiB, output/reserve 1 GiB each.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{stem}-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{stem}-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
tr -d '\\r' < {stem}-execution.tar.gz.sha256 | sha256sum -c - &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm &&
test ! -e ~/forensic-dgp/{prep['name']} &&
tar -xzf {stem}-execution.tar.gz -C ~/forensic-dgp
```

Archive SHA256: `{prep['archive_sha256']}`.
Protocol SHA256: `{pin}`. Installation deletes nothing.

3. Open **tmux from the normal VM SSH shell**:

```bash
tmux new-session -A -s dgp_bank_hinge_boundary_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/{prep['name']} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_bank_hinge_boundary_v1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha {pin} --verify-transfer &&
bash scripts/run_boundary.sh {pin}
```

Expected transfer receipt: `complete: true`, `packet_assets: 4`, `parent_assets: 5852`,
with zero neural calls/gradients/updates in that transfer check. Successful
measurement ends with **360 gradient queries and zero optimizer updates**.
Detach with **Ctrl+B, then D**; reattach with
`tmux attach -t dgp_bank_hinge_boundary_v1`. A stop is preserved/exported too.
Export `complete` confirms packaging only; there is no automatic training.

5. Download from **Windows Google Cloud SDK Shell**, one remote source per call:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{stem}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{stem}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{stem}-export.json" "."
```

Keep the three files together. The prepared independent return checker verifies
raw replay, scalar penalty parity and sampled gradient arithmetic before a new
training decision. Diagnostic success alone qualifies no model or extra epochs.
All five milestones and seven automatic/assisted completion families remain.
"""
    new(ROOT / report_name, report)
    new(ROOT / guide_name, guide)
    prefix = f"""# Latest verified status — 11 October 2026: R1 gradients audited; inactive-guard boundary test prepared

The manual R1 loss-direction diagnostic completed in {audit['VM_seconds']:.2f}
seconds with 80 B/40 recognizer forwards, 360 gradient queries and zero optimizer
updates/backwards/epochs. All 134 returned files, 5,852 parent assets, 40 exact
raw blocks/200 case replays and 40 scalar rows passed independent checks.
The layout repair is confirmed on L4. Four full gradient samples/370 arrays each
passed NumPy Gram/clipping/Adam arithmetic checks; neural derivatives were not
freshly replayed for all 40 rows. Derived scalar sum roundoff was retained and
audited separately from unchanged scientific gates. No checkpoint was created.

The original brightness helper has zero penalty value but nonzero gradients on
all 20 initial TRAIN references. Its median gradient norm is {states['initial']['median_brightness_gradient_norm_over_detail_gradient_norm']:.2f}× detail;
nominal single-reference Adam predicts degraded-MSE ascent on 20/20 initially,
19/20 at rejected B50, despite full-objective descent on all 20. PyTorch's
clamp_min equality derivative explains the boundary pressure. This is not an
actually applied step or native performance. Anchor conflicts remain; removing
the guard from gradient sums does not resolve them. More epochs are unqualified.

Next is a declared ReLU hinge derivative test, preserving exactly the scalar
penalty values on all 40 saved rows/200 case compositions, coefficient .2/10,
states, cases, weights, scientific gates and resource limits. It requires zero
inactive initial guard gradients. The {prep['archive_bytes']:,}-byte packet passed
source/archive/LF/Bash/Windows checks and all 5,852 parent hashes; independent
return checkers are prepared. Actual corrected L4 gradients remain unrun.
Scope stays 20 TRAIN references/100 cases/two states, 360 queries, zero updates,
no native/final inputs and no checkpoint. Successful diagnosis is not qualification.

Require 4 GiB free, idle L4 and human tmux. Worker/measurement/export stops are
900/600/300 seconds, external 930/330 with 30-second kill grace. Measurement is
checked between batches. VRAM 20 GiB, output/reserve 1 GiB; payload bound
712,406,496 bytes. No agent VM connection, training, cleanup or power action ran.
{link('Audited R1 findings', report_name)}.
{link('Five exact next upload/tmux/download steps', guide_name)}.

Goal ACTIVE. Permanent checkpoint
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`,
existing design, five milestones and seven completion families remain required.
Native CCTV is unpaired; paired TRAIN/final identities/completion stay separate.
No new incumbent, extra epochs, failed-state resume or app promotion is qualified.
No ethnicity, local Zamboanga-performance or exact hidden-identity claim is made.
Input-only usable labels remain. Historical sections retain exact bytes below.

---

"""
    backup = ANALYSIS / 'publication_before'
    backup.mkdir()
    documents = {}
    for name in ['PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md', 'CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md']:
        documents[name] = prepend_locked(ROOT / name, prefix, backup / name)
    old_guide = 'CCTV_DGP_BANK_LOSS_ALIGNMENT_V1_R1_VM.md'
    warning = '# R1 executed and independently audited — historical commands\n\nR1 completed; do not rerun this immutable diagnostic unchanged.\n' + link('Read the audited loss-direction result', report_name) + '.\n' + link('Use the separately verified inactive-guard diagnostic guide', guide_name) + '.\n\n---\n\n'
    documents[old_guide] = prepend_locked(ROOT / old_guide, warning, backup / old_guide)
    assert all(sha(ROOT / name) == expected for name, expected in protected.items())
    write(receipt, dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), publisher_sha256=sha(__file__),
        evidence_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in evidence},
        documents=documents, new_documents_sha256={name:sha(ROOT / name) for name in [report_name, guide_name]},
        R1_complete_and_independently_audited=True, numerical_layout_repair_confirmed=True,
        next_manual_boundary_diagnostic_unrun=True, return_checkers_prepared=True,
        protected_bindings_unchanged=True, no_local_gradient_or_optimizer=True, no_agent_VM_launch=True,
        five_milestones_and_seven_covering_families_retained=True, goal_status='active', goal_complete=False))
    print(dict(complete=True, status_documents=5, new_report_and_guide=2, next_manual_diagnostic_unrun=True))


if __name__ == '__main__':
    main()
