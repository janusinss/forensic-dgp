"""Publish the verified unrun manual comparison; preserve exact old documents."""
from datetime import datetime, timezone
from pathlib import Path

from cctv_dgp_spatial_readout_v1_contract import NAME, STEM, read, write, sha
from publish_cctv_dgp_png_group_gradients_v1_r2 import finish_prepend

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/'outputs'
REVIEW = OUT/'cctv_dgp_reconstruction_design_review_v2'


def new(path, text):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)


def link(label, name):
    return '['+label+'](<'+(ROOT/name).as_posix()+'>)'


def main():
    prepared, audited = read(REVIEW/'preparation.json'), read(REVIEW/'independent_preparation_audit.json')
    assert prepared['complete'] and audited['complete'] and audited['packet_status'] == 'VERIFIED_PREPARED_UNRUN'
    pin, digest = prepared['protocol_sha256'], prepared['archive_sha256']
    assert sha(OUT/NAME/'protocol.json') == pin and sha(OUT/(STEM+'-execution.tar.gz')) == digest
    assert not (OUT/NAME/'outputs').exists()
    guide_name = 'CCTV_DGP_SPATIAL_READOUT_V1_VM.md'
    prep_name = 'CCTV_DGP_SPATIAL_READOUT_V1_PREPARATION.md'
    guide = f'''# Spatial reconstruction comparison V1 — manual L4 commands

**VERIFIED PREPARED / UNRUN.** This tests a reconstruction change; no improved
checkpoint or useful CCTV performance is established. All14 CPU boundary tests,
16 archive members, 203 required parent assets, actual hash-only CLI and host
Bash syntax pass. The first preparation metadata error and checker order/
sandbox errors are retained; the final sealed archive was unchanged by the
checker corrections.

Route A fits the corrected current decoder; B adds learned spatial bank
connections with frozen conditioning. Each has at most50 updates on20 exposed
TRAIN references /100 views. These are2.5 small-sample passes per route, not
additional epochs of the full DGP. The original checkpoint remains protected.

Require **8 GiB free after installation** and8 GiB available system RAM before
model loading. No current VM capacity or power state is claimed. Estimated
**20–40 minutes** including checks/export; this is not a new runtime measurement.
Hard supervised phase limits, including kill grace, total at most78.5 minutes
plus short hash/metadata checks: training51, replay6, saved audit10.5, export11.
Each fit itself is capped at10 minutes; cache4 and each snapshot5 minutes.
VRAM20 GiB / RSS12 GiB / outputs3 GiB / retained disk reserve1 GiB.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{STEM}-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{STEM}-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c {STEM}-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -f ~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm/protocol.json &&
test ! -e ~/forensic-dgp/{NAME} &&
tar -xzf {STEM}-execution.tar.gz -C ~/forensic-dgp
```

The SHA sidecar is verified LF UTF-8, with no CR or BOM. The203 reused parent
files must still exist and match their hashes. Hash-only verification below
stops before models if a file is missing; it never launches an older pilot.

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_spatial_readout_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/{NAME} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_spatial_readout_v1_vm.py --root . --protocol-sha {pin} --verify-transfer &&
bash scripts/run_spatial_readout.sh {pin}
```

The supervisor displays live output and saves trainer/replay/audit/export logs.
At the endpoint it prints each route's actual RAW/PNG gain and failed checks.
Training exit2 is the retained quality stop; it is not a model-loading traceback.
The archive is still exported for independent review. Export `complete:true`
means packaging finished; `training_success_not_implied:true` remains binding.
Initial parity/nonzero derivatives and new CUDA timing remain unverified until
this manual execution. No failed recipe automatically resumes or gains epochs.

Detach with **Ctrl+B, then D**. Reattach with
`tmux attach -t dgp_spatial_readout_v1`. The process runs within that session.

5. Download after export from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-export.json" "."
```

Each PuTTY-compatible download has **one remote source**. Keep the returned
archive, sidecar and export receipt together for import, independent saved-array
audit and direct review of all20 sheets. No app adoption follows automatically.

Protocol SHA256: `{pin}`.
Execution archive SHA256: `{digest}`.
{link('Design and unchanged requirements', 'CCTV_DGP_RECONSTRUCTION_DESIGN_REVIEW_V2.md')}.
'''
    new(ROOT/guide_name, guide)
    report = f'''# Spatial reconstruction V1 preparation — 14 October2026

**Verified and UNRUN.** The human diagnostic answer resolved the required design
discussion. The selected comparison changes reconstruction capacity, instead
of extending a rejected conditioning-only state or repeating a gradient solver.
The original app DGP SHA256 remains646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b.

Route A has15 decoder tensors /609,219 elements. B adds five anchored3×3 feature
connections for25 tensors /2,176,419 elements. Existing decoder/readout weights
are reused. Observation features, input-to-prior conditioning, normalization,
external pretrained StyleGAN2 and recognizer stay frozen. The bank is declared
external; it is not an own-trained GAN or a pretrained restoration substitution.
The base feature-provider state hash is98675d10be6f3405dd4bd0950d9251a61b26b9b161ae62a9ce48f775bc43e2bd;
this is the original B initializer, not the later anchored wrapper's state hash.

Both routes start at identical original outputs, subject to exact manual CUDA
parity on all100 controls. Each then gets at most50 bounded Adam steps on the
same20 exposed TRAIN references /100 views. All selected tensors need finite
nonzero objective derivatives before any optimizer. Adam computes proposals on
shadow values; only the projected result enters the live decoder. The common
0.001 distance allowance cannot grow with B's added parameters. Full states,
RNG, optimizer moments and all100 parameter-journal steps remain in the return.
This is research fitting, not accepted development updates or full-corpus epochs.

All17 RAW and17 PNG groups, original RGB/SSIM/recognizer tolerances, ≤20%
brightness-only contribution, ≥1% global and both-source structure gate and the
later full-TRAIN10% requirement remain. Original float32 embedding scoring and
cached baseline controls are retained. Actual output preservation, useful native
DEV images and independent final review remain necessary; local derivative
predictions are not acceptance. The previous V6/V9/original-feature/bank failures
are explicitly reconciled in the design review, rather than being treated as
untried unfreezing ideas.

The independent preparation audit checked all16 archive members,203 required
parent file bindings,20 canonical targets,100-case membership and unchanged
historical block ordering,5 readout rank/shape checks,14 CPU boundary regressions,
11 altered-protocol rejections, actual hash-only CLI, Windows pre-neural training
denial, Python3.10 AST and read-only host Bash syntax. No model construction,
forwards, derivatives, optimizer updates or VM calls ran during preparation.
Local tensor deserialization was read-only. New CUDA behavior/runtime remains
unverified and is explicitly guarded in the manual test.

The first preparer assumed an HQ-only metadata field existed on replay records.
The first checker assumed case-ID and block lists shared an order; all100 unique
members actually match, while the exact block/schedule order is separately pinned.
Git Bash then failed to spawn within the Windows sandbox; the unchanged launcher
passed `bash -n` with normal host permissions. All three preparation/checker
failure records are retained. None is a training or model-quality result.
The sealed archive/protocol/executable sources were not changed for checker fixes.

Require8 GiB free and8 GiB available RAM before imports; estimate20–40 minutes
including checks/export with finite caps in the guide. This is not current VM
inventory. No cleanup, automatic VM access or automatic training occurred.
Checkpoint/caches/splits/provenance/full states/failures and original app design
remain. Native unpaired, synthetic paired, source-specific and untouched final
roles stay separate. The seven automatic/assisted covering families and detector
limitations remain independent; reconstruction fitting does not qualify completion.
Goal active/incomplete; no new incumbent, extra epochs or app promotion.

{link('Exact upload, tmux and download commands', guide_name)}.
{link('Chosen hypothesis and research sources', 'CCTV_DGP_RECONSTRUCTION_DESIGN_REVIEW_V2.md')}.
{link('Independent preparation audit', 'outputs/cctv_dgp_reconstruction_design_review_v2/independent_preparation_audit.json')}.
Protocol `{pin}`; execution archive `{digest}`, {prepared['bytes']:,} bytes.
'''
    new(ROOT/prep_name, report)
    stage = REVIEW/'publication'
    stage.mkdir()
    (stage/'before').mkdir()
    expected = {
        'SYSTEM_WORKFLOW_AND_GOAL.md': '83f2d87a1ab95f5fd53ec4264b2d5035582463f91293669fce4cbf8cf6c43181',
        'PRACTICAL_OUTPUT_SCOPE.md': '3a1af665a447d079cb31ab954bb47164082068250e9bf5f50e48668ff7837ca8',
        'CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md': '87cfa670acf51c8b86a8d9e2cb7d48254295d5c8c0277bee333322fddc1d69dc',
        'CCTV_DGP_NEXT_PILOT.md': 'd091bc51bfbc21589e710ed552e974b3d2eed0f8140dc0772a94cb5a3ddbe10c',
        'PROJECT_HANDOFF.md': '37557383187ac13d7c70204b6732aef0e43dc38c8a6428e68894529f61e3c200',
    }
    assert all(sha(ROOT/name) == digest for name, digest in expected.items()), 'Preserve concurrent document changes'
    prefix = f'''# Reconstruction discussion resolved; finite spatial test verified, UNRUN —14 October2026

The human answered “apply the best  approach to improve”. The required diagnostic
discussion is resolved; earlier pending/blocked records below are historical.
{link('Current evidence-based design decision', 'CCTV_DGP_RECONSTRUCTION_DESIGN_REVIEW_V2.md')}.
Do not repeat the rejected nearby/joint/RAW-only recipes or append their epochs.

The new comparison fits an isolated corrected current decoder (A) against a
decoder with learned3×3 frozen-bank feature connections (B). Conditioning,
observation features, normalization, external StyleGAN2 and recognizer remain
frozen. Existing weights are reused; exact original-output parity and connected
derivatives must pass on the L4 before an optimizer. At most50 updates per route
on the same20 exposed TRAIN references /100 views. These are2.5 small-sample
passes, not full-corpus extra epochs or accepted model updates.

All16 archive/203 parent bindings,20 canonical targets,14 meaningful CPU tests,
11 protocol mutations, actual hash-only CLI, Windows guard, Python3.10 AST and
read-only Bash syntax pass. The99,256-byte packet is **VERIFIED PREPARED / UNRUN**.
{link('Exact Windows upload, tmux and single-source downloads', guide_name)}.
{link('Verification, finite limits and retained preparation failures', prep_name)}.
New CUDA parity, gradients, timing, training and useful output are unverified.
Require8 GiB free /8 GiB available RAM before models; estimate20–40 minutes
including checks/export. Current VM capacity/power/credits were not queried.

Original RAW/PNG/source/profile preservation, brightness≤20%, ≥1% global and
both-source endpoint, common0.001 original-weight distance bound and later
full-TRAIN10% requirement remain. Keep full states, optimizer/RNG, proposals,
originals, masks, outputs, source terms, splits, caches and all failures. No
automatic extra epochs, incumbent selection or app adoption. The permanent
DGP checkpoint remains646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b.

Actual training/gradients/parameter changes remain human-manual L4 tmux; direct
agent VM access remains inventory-first hash-bound maintenance only. Local
preparation used read-only tensor/file/array operations and made no VM calls.
Original app design, all five milestones, native unpaired versus synthetic paired
roles, untouched final identities, clearer/less-covered requests and all seven
automatic/assisted covering families remain. Detector/completion localization
failures are separate and unqualified. Goal active/incomplete.
This dated decision supersedes older discussion-pending/prepared statuses;
exact previous document follows.

'''
    docs = {}
    for name, digest in expected.items():
        docs[name] = finish_prepend(ROOT/name, prefix, stage/'before'/name, digest)
    record = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), documents=docs,
        new_reports_sha256={name: sha(ROOT/name) for name in [guide_name, prep_name, 'CCTV_DGP_RECONSTRUCTION_DESIGN_REVIEW_V2.md']},
        proof_sha256={path.relative_to(ROOT).as_posix(): sha(path) for path in
            [REVIEW/'preparation.json', REVIEW/'independent_preparation_audit.json', REVIEW/'user_decision.json',
             REVIEW/'preparation_attempt1.json', REVIEW/'preparation_audit_attempt1.json', REVIEW/'preparation_audit_attempt2.json']},
        protocol_sha256=pin, archive_sha256=prepared['archive_sha256'], original_checkpoint_sha256='646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b',
        goal_status='active_incomplete', human_discussion_resolved=True, training_launched=False, model_qualified=False)
    write(stage/'publication.json', record)
    print('Published verified UNRUN manual comparison and five preserved status documents.')


if __name__ == '__main__':
    main()
