"""Publish audited diagnostic failure and verified unrun layout revision."""
from datetime import datetime, timezone
from pathlib import Path

from import_cctv_dgp_bank_loss_alignment_v1_return import ROOT, OUT, RETURN, ANALYSIS, PARENT, read, sha, write
from publish_cctv_dgp_bank_AB_and_guard_v1_review import prepend_locked


def link(label, name):
    return '[' + label + '](<' + (ROOT / name).as_posix() + '>)'


def new(path, text):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)


def main():
    receipt = ANALYSIS / 'r1_publication.json'
    assert not receipt.exists()
    evidence = [ANALYSIS / 'import.json', ANALYSIS / 'independent_failure_audit.json',
                ANALYSIS / 'layout_reproduction/run_results.json',
                OUT / 'cctv_dgp_bank_loss_alignment_v1_r1_preparation.json',
                OUT / 'cctv_dgp_bank_loss_alignment_v1_r1_transfer_audit.json']
    assert all(read(path)['complete'] for path in evidence)
    imported, failed, probe, prep, transfer = [read(path) for path in evidence]
    assert transfer['preparation_sha256'] == sha(evidence[3])
    assert sha(OUT / (prep['stem'] + '-execution.tar.gz')) == prep['archive_sha256']
    assert not (OUT / prep['name'] / 'outputs').exists()
    protected = read(ANALYSIS / 'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT / name) == digest for name, digest in protected.items())
    report_name = 'CCTV_DGP_BANK_LOSS_ALIGNMENT_V1_RESULTS.md'
    guide_name = 'CCTV_DGP_BANK_LOSS_ALIGNMENT_V1_R1_VM.md'
    delta = probe['differences']['initial_legacy_vs_initial_canonical']
    report = f"""# Loss-direction diagnostic failure and layout repair — 11 October 2026

**The manual diagnostic stopped before measuring any loss gradients.** It is a
processing failure, not evidence that more training worsened the model. The export
completed, but the loss-direction measurement did not. No new checkpoint exists.

Location: the first frozen-output replay at line 155 of the original diagnostic,
initial state, reference batch 180 (five profiles of v9_tr_ffhq_00084). Its assertion
was `Same L4 frozen raw control differs`. The VM ran {failed['VM_seconds']:.2f} seconds:
**one model forward, zero recognizer forwards, gradients, backwards, updates or
epochs**. Python 3.10.12, PyTorch 2.9.1+cu129, NVIDIA L4 and manual tmux were recorded.

The archive SHA256 is `{imported['archive_sha256']}`. All **nine returned regular
files**, member paths/sizes/hashes, prepared source and **5,852 retained parent
assets** passed independent checks. The log and exit code agree with the failure.
There are no measured loss rows, sampled gradients or timing-projection results.
The failed script recorded no raw error size; that omission is retained explicitly.
The old return, protocol, code, log and failure remain immutable.

## Demonstrated processing difference

The new diagnostic stacked HWC arrays and permuted them into NCHW dimensions,
leaving channels-last storage. The parent concatenated individual tensors into
contiguous NCHW storage. Pixel values and dimensions were identical, but strides
were **(196608, 1, 768, 3)** versus **(196608, 65536, 256, 1)** for the five-image
batch. This was an unintended input-layout change in our diagnostic preparation.

A finite frozen CPU reproduction used the same pinned initializer and first five
TRAIN cases, one original forward and two B forwards, **no gradients or updates**,
and unchanged model/buffer hashes. Restoring the parent's layout made initial B
**exactly equal to the original model on the same CPU batch**. Changing only layout
changed **{delta['changed_values']:,} values**, with maximum raw error
**{delta['maximum_absolute_error']:.12g}**. Independent saved-array checks reproduced
both results. The three saved raw arrays remain retained. This was a numerical
regression reproduction, not a new visual-quality comparison.

PyTorch describes channels-last storage as a different stride order with the same
dimensions and notes that floating-point calculations are not generally guaranteed
to be bitwise identical across implementations/devices. These references explain
why identical values alone are insufficient for an exact replay; the causal evidence
here is our controlled local reproduction. [Memory-format documentation](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html),
[PyTorch 2.9 numerical-accuracy documentation](https://docs.pytorch.org/docs/2.9/notes/numerical_accuracy.html).

The failed L4 worker did not retain its raw output, so its exact difference size
cannot be reconstructed. CPU reproduction demonstrates the layout defect;
**confirmation of the repaired exact L4 replay remains pending**. We do not treat
CPU/CUDA differences as model quality, or loosen the control check to force a pass.

## Corrected manual R1 diagnostic

The only numerical-path change is `.contiguous()` before CUDA transfer in the RGB
batch constructor. This matches the parent input and target values/strides for
**all 200 case tensors across 40 input/target batches**. Loss helpers, objective
weights, gradient queries, hypothetical Adam arithmetic, states, source data,
case lists, checkpoints, thresholds and resource limits remain byte/value matched
to the failed protocol. The exact L4 frozen replay and 1e-6 gradient/frozen forward
check remain. No optimizer is constructed and no training is resumed.

R1 also saves every frozen raw replay and error/stride receipt **before checking**
it. A future stop will therefore retain actual output/error evidence. All 40 planned
blocks add 157,286,400 raw payload bytes. The combined gradient/forward upper bound
is **712,406,496 bytes**, including the existing gradient metadata allowance,
below the unchanged 1 GiB output cap. R1 uses a new root/archive/protocol and never
overwrites the failed run.

The **{prep['archive_bytes']:,}-byte packet** passed independent archive/source,
5,852 parent asset, LF checksum, readonly Bash syntax, input-layout regression,
three saved output-array and actual Windows launch-rejection checks. The Windows
worker rejects the run before neural or gradient imports. These are preparation
checks; actual repaired CUDA execution and loss-direction measurements are unrun.

Scope remains **20 TRAIN references, 100 distinct TRAIN cases, initial and joint50
states, 360 gradient queries, 80 B forwards, 40 recognizer forwards**, no native/final
input pixels and no checkpoint. Full gradient samples cover four reference/state
batches; the other 36 gradient summaries have narrower independent audit scope.
Hypothetical single-reference Adam directions cannot establish finite-step quality,
the actual accumulated five-reference update or restored identity.

Require **4 GiB free after installation** and an idle L4. Estimate **3–10 minutes
measurement plus 1–5 minutes export**, still unmeasured. Worker cap 900 seconds,
measurement stop 600 checked between reference batches, export cap 300; external
930/330 with 30-second kill grace. First two batch durations determine the retained
projection stop. VRAM 20 GiB, output 1 GiB, disk reserve 1 GiB. No current VM power,
free-space or credit claim is made. Actual runs remain human manual tmux.

{link('Five exact upload, tmux and download steps', guide_name)}.
{link('Independent failure audit', 'outputs/cctv_dgp_bank_loss_alignment_v1_analysis/independent_failure_audit.json')};
{link('Frozen layout reproduction', 'outputs/cctv_dgp_bank_loss_alignment_v1_analysis/layout_reproduction/run_results.json')};
{link('Independent R1 transfer and regression audit', 'outputs/cctv_dgp_bank_loss_alignment_v1_r1_transfer_audit.json')}.

## Goal and preservation

The app checkpoint remains the permanent fine-tuned DeblurGAN-v2-compatible
baseline `646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
Our candidate declares the external frozen StyleGAN2 bank and our learned
conditioning/reconstruction. No own-trained StyleGAN or exact published-DGP claim
is made. No candidate is qualified, extra epochs are not justified by this stop,
and no failed-state continuation or app promotion is authorized. The first
diagnostic failure is preserved and will not be rerun unchanged.

All five milestones, seven covering families, final identity separation and
independent final review remain required. Native CCTV remains unpaired; paired
TRAIN metrics remain separate. Clear glasses, ordinary hair, visible appearance,
automatic mask-area review/correction and original/mask/one plausible estimate
with PNG/bundle downloads remain in scope. Automatic/assisted completion is not
tested here. No local Zamboanga data, ethnicity inference or exact hidden identity
claim is introduced. Research caches, checkpoints, split/gate records, app bindings
and design remain retained. No agent VM connection, cleanup or power action occurred;
local work used frozen inference and audits only. Goal **ACTIVE and incomplete**.
"""
    stem, pin = prep['stem'], prep['protocol_sha256']
    guide = f"""# Corrected loss-direction diagnostic R1: five manual steps

**Prepared and independently verified; actual L4 replay/gradients are unrun.**
The original diagnostic stopped before gradients because its input layout differed
from the retained baseline. R1 restores that layout and retains the exact-output
check. This is **zero-update diagnosis, not training or failed-state resume**.
{link('Audited failure and repair', report_name)}.

Upload **{prep['archive_bytes']:,} bytes**. Preserve the original failed diagnostic
and `~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm` (5,852 bound assets).
Require **4 GiB free after installation** and idle L4. Unmeasured estimate:
**3–10 minutes measurement plus 1–5 minutes export**. Worker/measurement/export
limits are 900/600/300 seconds; measurement stop is checked between batches.
External stops 930/330 with 30-second kill grace. VRAM 20 GiB; output/reserve
1 GiB each. Full raw replay evidence is retained before any parity stop.

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

Expected archive SHA256: `{prep['archive_sha256']}`.
Expected protocol SHA256: `{pin}`. Installation deletes nothing.

3. Open **tmux from the normal VM SSH shell**:

```bash
tmux new-session -A -s dgp_bank_loss_alignment_v1_r1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/{prep['name']} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_bank_loss_alignment_v1_r1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha {pin} --verify-transfer &&
bash scripts/run_alignment.sh {pin}
```

Expected transfer check: `complete: true`, `packet_assets: 3`,
`parent_assets: 5852`, `neural_calls: 0`, `gradient_queries: 0`,
`optimizer_updates: 0`. Successful measurement ends with **360 gradient queries,
zero optimizer updates**. Exact raw replay must pass first; no threshold is relaxed.
Detach with **Ctrl+B, then D**. Reattach with
`tmux attach -t dgp_bank_loss_alignment_v1_r1`. The shell exports a retained failure
too; export completion confirms packaging. It launches no subsequent training.

5. Download from **Windows Google Cloud SDK Shell**, one remote source per call:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{stem}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{stem}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{stem}-export.json" "."
```

Keep all three together for independent hash, frozen-replay and sampled-gradient
audits. Do not rerun the original packet, overwrite its failure, or promote a model
from diagnostic/export completion. Restoration, native DEV/final review, app flow
and all seven automatic/assisted completion families remain required.
"""
    new(ROOT / report_name, report)
    new(ROOT / guide_name, guide)
    prefix = f"""# Latest verified status — 11 October 2026: zero-gradient layout stop audited; corrected R1 diagnostic prepared

The manual loss-direction diagnostic stopped at its first initial-state frozen
forward after {failed['VM_seconds']:.2f} seconds: one B forward, zero gradients,
recognizer forwards, updates or epochs. All nine return files and 5,852 retained
parent assets passed independent checks. No loss measurements or model resulted.
The failure/log/source/protocol remain immutable; export success is packaging.

An unintended channels-last input layout differed from the parent's contiguous
NCHW layout despite identical pixels. Three frozen CPU forwards reproduced a
{delta['maximum_absolute_error']:.12g} raw difference from layout alone; canonical
initial B exactly equaled the original model on that CPU batch. State/buffer and
12 protected app/checkpoint/example bindings remained exact. The failed VM omitted
its raw error; actual repaired exact L4 replay still needs manual confirmation.

R1 restores only the input/target layout and saves every frozen raw control/error
before checking. All 200 input/target case tensors match parent values/strides;
three saved CPU arrays and packet/source/archive/LF/Windows/Bash guards passed
independent checks. The {prep['archive_bytes']:,}-byte packet is prepared, unrun.
Losses, states, weights, scientific gates, exact control parity and finite limits
remain unchanged. Scope: 20 TRAIN references/100 distinct cases/two states,
360 gradient queries, 80 B/40 recognizer forwards, zero optimizer updates,
no native/final pixels and no checkpoint. Four sampled gradient batches have
raw independent audit scope; single-reference Adam directions are hypothetical.

Require 4 GiB free and idle L4. Worker/measurement/export limits 900/600/300 seconds
(measurement checked between batches), external 930/330 with 30-second kill grace;
VRAM 20 GiB, output/reserve 1 GiB. Combined gradient/raw payload bound 712,406,496
bytes. Human manual tmux only; no agent VM action occurred. No failed-state resume,
new incumbent, extra epoch justification, model qualification or app promotion.
{link('Audited failure and demonstrated repair', report_name)}.
{link('Five exact corrected R1 upload/tmux/download steps', guide_name)}.

Goal ACTIVE: permanent checkpoint
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`,
existing design, five milestones and seven completion families remain required.
Native CCTV is unpaired; paired TRAIN/final identities/completion stay separate.
No ethnicity, Zamboanga-performance or exact hidden-identity inference is added.
All historical sections below retain exact bytes and authorize no reruns.

---

"""
    backup = ANALYSIS / 'r1_publication_before'
    backup.mkdir()
    documents = {}
    for name in ['PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
                 'CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md']:
        documents[name] = prepend_locked(ROOT / name, prefix, backup / name)
    old_guide = 'CCTV_DGP_BANK_LOSS_ALIGNMENT_V1_VM.md'
    warning = '# Executed diagnostic failure — preserved historical commands\n\nDo not rerun this V1 packet unchanged. The first raw replay failed before any\ngradients. ' + link('Use the corrected, independently verified R1 guide', guide_name) + '.\n' + link('Audited failure and layout reproduction', report_name) + '.\n\n---\n\n'
    documents[old_guide] = prepend_locked(ROOT / old_guide, warning, backup / old_guide)
    assert all(sha(ROOT / name) == digest for name, digest in protected.items())
    write(receipt, dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), publisher_sha256=sha(__file__),
        evidence_sha256={path.relative_to(ROOT).as_posix(): sha(path) for path in evidence},
        documents=documents, new_documents_sha256={name: sha(ROOT / name) for name in [report_name, guide_name]},
        diagnostic_failure_preserved=True, numerical_layout_repair_verified_locally=True,
        actual_L4_repair_confirmation_unrun=True, no_agent_VM_connection_or_training=True,
        five_milestones_and_seven_completion_families_retained=True,
        protected_app_and_checkpoint_bindings_unchanged=True, goal_status='active', goal_complete=False))
    print(dict(complete=True, status_documents=len(documents), new_reports_and_guides=2,
               original_failed_return_preserved=True, corrected_manual_diagnostic_unrun=True))


if __name__ == '__main__':
    main()
