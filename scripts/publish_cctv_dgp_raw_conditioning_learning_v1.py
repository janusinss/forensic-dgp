"""Publish the verified, unrun optimizer comparison and exact manual commands."""
from datetime import datetime, timezone

from import_cctv_dgp_conditioning_learning_v1_r4_return import ROOT, OUT, ANALYSIS, read, sha, write
from publish_cctv_dgp_bank_saved_direction_forward_v1_r1_failure import link
from publish_cctv_dgp_png_group_gradients_v1_r2 import retained_new, finish_prepend


def main():
    stage = OUT / 'cctv_dgp_raw_conditioning_learning_v1_publication'
    assert not (stage / 'publication.json').exists()
    prepared = read(OUT / 'cctv_dgp_raw_conditioning_learning_v1_preparation.json')
    transfer = read(OUT / 'cctv_dgp_raw_conditioning_learning_v1_transfer_audit.json')
    syntax = read(OUT / 'cctv_dgp_raw_conditioning_learning_v1_bash_syntax.json')
    prior = read(ANALYSIS / 'publication/publication.json')
    assert prepared['complete'] and transfer['complete'] and syntax['complete']
    packet = OUT / prepared['name']
    p = read(packet / 'protocol.json')
    pin = sha(packet / 'protocol.json')
    assert pin == prepared['protocol_sha256'] == transfer['protocol_sha256']
    stem = 'cctv-dgp-raw-conditioning-learning-v1'
    archive = OUT / (stem + '-execution.tar.gz')
    assert sha(archive) == prepared['archive_sha256'] == transfer['archive_sha256']
    assert syntax['exit_code'] == 0 and syntax['checked_shell_sha256'] == sha(packet / 'scripts/run_learning.sh')
    assert transfer['RAW_optimizer_and_unchanged_quality_gate_design_audit']['complete']
    assert len(transfer['all_three_actual_R4_trial_decisions_replayed_and_still_rejected']) == 3
    assert not (packet / 'outputs').exists() and not (packet / 'review.log').exists()
    guide_name = 'CCTV_DGP_RAW_CONDITIONING_LEARNING_V1_VM.md'
    report_name = 'CCTV_DGP_RAW_CONDITIONING_LEARNING_V1_PREPARATION.md'
    guide = f'''# RAW conditioning learning V1 — verified manual L4 comparison

**Prepared and UNRUN. At most 2 accepted updates, 3 trials per update.**
The previous R4 run rejected all three changes. This separate test optimizes
32 RAW loss groups and checks every delivered PNG before accepting a change.
The starting model, 100 paired TRAIN cases, masks, 12 conditioning tensors and
all actual RAW/PNG preservation rules remain fixed. More epochs depend on useful
preserved learning and later development review.

Require **9 GiB free after installation**. Estimated **5–12 minutes plus export**,
based on R4's measured timings; the new CUDA runtime is not yet measured.
Maximum 640 gradient queries, 180 candidate and 320 recognizer forwards.
Learning cap 1,350s; worker 1,500s; outer 1,530s; export 420s/outer 450s;
each outer timeout has 30s kill grace. Total outer allowance is **34 minutes**.
Gradient round 600s, solver 60s, VRAM 20 GiB, RSS 12 GiB, output 3.5 GiB,
disk reserve 1 GiB. Runtime space checks retain failures. Upload size:
**{prepared['archive_bytes']:,} bytes**. No automatic follow-on or app promotion.

1. Upload from **Windows Google Cloud SDK Shell**:
```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{stem}-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{stem}-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:
```bash
cd ~ &&
sha256sum -c {stem}-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm &&
test ! -e ~/forensic-dgp/{packet.name} &&
tar -xzf {stem}-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux** in VM SSH:
```bash
tmux new-session -A -s dgp_raw_conditioning_learning_v1
```
Detach an older attached session with **Ctrl+B,D** before opening this one.

4. Launch **inside tmux**:
```bash
cd ~/forensic-dgp/{packet.name} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_raw_conditioning_learning_v1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha {pin} --verify-transfer &&
bash scripts/run_learning.sh {pin}
```
Detach **Ctrl+B,D**; reattach with `tmux attach -t dgp_raw_conditioning_learning_v1`.
Progress is saved in `review.log` and printed when the finite worker ends.
Export runs after a retained worker stop. `complete: true` describes archive
creation; `accepted_updates` describes actual learning. Keep the used directory,
original checkpoints/splits/caches and all failed trial outputs.

5. Download from **Windows Google Cloud SDK Shell** after export:
```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{stem}-results.tar.gz" "{stem}-results.tar.gz"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{stem}-results.tar.gz.sha256" "{stem}-results.tar.gz.sha256"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{stem}-export.json" "{stem}-export.json"
```
Use one remote source per PuTTY download. Return all 3 files for independent
state/vector/RAW/PNG/preservation audit and direct review of every output sheet.
No qualified new DGP or longer-epoch continuation is claimed by preparation.

Protocol SHA256: `{pin}`.
Execution archive SHA256: `{prepared['archive_sha256']}`.
{link('Transfer and scientific-gate audit', 'outputs/cctv_dgp_raw_conditioning_learning_v1_transfer_audit.json')}.
{link('Why this comparison is justified', report_name)}.
'''
    report = f'''# RAW conditioning V1 preparation — 14 October 2026

**The optimizer comparison is verified and UNRUN.** It tests the R4 finding that
joint RAW/approximate-PNG constraints have a poorly conditioned Gram (about
885 million), while RAW32 alone is about 4,164. R4's largest actual trial gained
0.688521% RAW / 0.705580% PNG and failed 31 preservation checks in each; every
R4 trial remains rejected. Conditioning is a hypothesis, not proof of a unique
cause. No success or useful trained image is promised by this new preparation.
{link('R4 results and full saved-output review', 'CCTV_DGP_CONDITIONING_LEARNING_V1_R4_RESULTS.md')}.

The new recipe computes 32 RAW derivative groups per current state and solves
all of their constraints with the original float64 Gram/rank/dot math and
float32 direction. It refreshes them after an accepted change. Approximate PNG
derivatives no longer select the update direction. Each proposed change is
still evaluated on all 100 cases with the unchanged 17 RAW and 17 PNG groups,
clear-image MSE/SSIM/appearance preservation, positive global/per-source detail,
brightness fraction <=0.2, protected-pixel copies and cumulative parameter
distance <=0.1%. The endpoint still needs >=1% structure globally and per source
in both delivered forms. Actual PNG quality gates are not removed or relaxed.

The same 20 photographic TRAIN references, five profiles and 100 prepared 256
inputs give identical evaluation opportunity to retained R4 and original DGP.
RAW neural exposures remain full-cohort; derivative queries and runtime differ
and must be reported separately. Original R4's complete RAW+PNG64 recipe is
immutable and failed. This is a changed optimizer comparison, not proof of a
new architectural cause or an unchanged failed rerun.

Only the 12 already activated style/spatial conditioning tensors
(2,580,864 parameters) learn. The same original trained restorer, decoder,
deterministic readouts, same-observed-input reference, external StyleGAN2 prior,
fixed recognizer and buffers remain frozen. Initial output parity and all
full-state/source fingerprints are retained; fresh CUDA parity is checked
inside the manual run. The prior is attributed external pretrained material,
not an own-trained generative prior or a pretrained restorer promoted as primary.
RAW SSIM retains its declared approximate derivative and canonical scientific
forward; gradient correctness and finite quality still require evidence.

Local preparation checked all {len(p['assets_sha256']) + 1} archive files,
ASCII/LF checksum, 5,852 parent bindings, initial virtual state, all 100 raw/PNG
baseline cases, nonfinite rejection, six acceptance-boundary fixtures, all three
actual R4 failures and four older rejected proposals. Prospective return
auditors check RAW32 gradient means/vectors/KKT arithmetic, actual rounded
updates, component states and every RAW/PNG score/mask/composition. Exact saved
receipt binding and independent gate branches are retained from R4's corrected
checker. Windows training rejects before Torch import; Bash syntax passes.
{link('Independent preparation audit', 'outputs/cctv_dgp_raw_conditioning_learning_v1_transfer_audit.json')}.

Two meaningful synthetic solver tests check full rank, nearly correlated full
rank, rank-deficiency rejection, nonfinite losses and cumulative distance.
An unsealed dimension-substitution draft also changed float64 tokens; the
precision regression caught it. That source/stop is retained, and the corrected
solver passes before transfer. Original float64 calculations and the complete
actual scientific decision function are preserved by an inverse source audit.
These are synthetic arithmetic and CPU pixel-utility checks, not neural
training, actual model proposals or model-usefulness evidence.

Maximum two accepted updates, three trials each, 640 gradient queries, 180
candidate/320 recognizer forwards; no automatic continuation. Runtime uses R4
measurements for an estimated 5–12 minutes plus export and retains finite
time/projection/VRAM/RSS/storage/output/export stops. Require 9 GiB free after
installation; current guest free space or billing is not claimed. The used
pilot cannot be resumed or rerun unchanged. Any partial accepted component and
all failed trials are exported with true counts and remain research artifacts.

Protocol SHA256 `{pin}`; execution archive {prepared['archive_bytes']:,} bytes,
SHA256 `{prepared['archive_sha256']}`.
{link('Exact manual upload, tmux and single-source downloads', guide_name)}.

The permanent baseline remains
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
No new incumbent, app change, full epoch or CUDA learning occurred locally.
Useful visual review/full TRAIN/native DEV/final qualification remains required
before adoption or the separately frozen maximum-five-epoch proposal. Paired
photographic metrics remain separate from unpaired native CCTV; final identities
stay outside tuning. No ethnicity, hidden identity or Zamboanga-performance
claim is supported. Existing app design and Auto/override remain preserved.
All five milestones and seven automatic/assisted covering families remain
required; this restoration pilot does not establish completion usefulness.
The full goal stays active and incomplete.
'''
    retained_new(ROOT / guide_name, guide)
    retained_new(ROOT / report_name, report)
    docs = ['PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
            'CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md', 'CCTV_DGP_DELIVERED_LOSS_LEARNING_REVIEW_V1.md']
    design_name = 'CCTV_DGP_POST_R4_GRADIENT_DESIGN_REVIEW_V1.md'
    expected = {n: prior['documents'][n]['after_sha256'] for n in docs}
    expected[design_name] = prior['new_documents_sha256'][design_name]
    prefix = f'''# RAW32 conditioning comparison verified, UNRUN — 14 October 2026

R4's audited failure remains binding: best0.688521% RAW/0.705580% PNG, all three
changes rejected, zero accepted updates. Saved prediction review and the human's
“apply best approach” decision justify a separate optimizer comparison.
{link('Audited R4 result', 'CCTV_DGP_CONDITIONING_LEARNING_V1_R4_RESULTS.md')}.

The new recipe chooses a direction from all32 RAW constraints and independently
checks all17 RAW/all17 PNG scientific groups before accepting any change.
The same100 prepared TRAIN cases, initial full state,12 conditioning tensors,
frozen components, original checkpoint and scientific/quality thresholds remain.
No PNG appearance/preservation or1% global/per-source endpoint gate is relaxed.
At most2 accepted updates/3 trials each, refreshing32 RAW derivatives; maximum
640 queries,180 candidate/320 recognizer forwards and finite resource/export
stops. No automatically appended epochs or app adoption. Verified and UNRUN.
{link('Current manual upload, tmux and single-source download commands', guide_name)}.
{link('Design, component origins, preparation and limits', report_name)}.

Preparation passes16 archive bindings,5,852 parent bindings, all100 RAW/PNG
baseline cases, exact state/source checks, declared synthetic solver/precision
fixtures, invalid-score/gate tests and all3 R4 plus4 older rejected proposals.
An unsealed precision-draft stop is retained and repaired before transfer;
original float64 math and all actual RAW/PNG gates are unchanged. CPU utilities,
Windows pre-neural rejection and Bash syntax pass. These are preparation facts,
not a new CUDA result or model usefulness claim. Require9GiB free at launch.

Original model/app design, checkpoints/splits/caches/provenance/full states and
failures remain preserved. Paired TRAIN/native unpaired/final roles stay separate;
useful DEV/final review and all five milestones/seven automatic and assisted
covering families remain. Actual training stays human-manual L4 tmux; no agent
VM launch or local model training occurred. Full goal active/incomplete.
Exact previous document follows.

'''
    (stage / 'before').mkdir(parents=True, exist_ok=True)
    changed = {}
    for name in docs + [design_name]:
        changed[name] = finish_prepend(ROOT / name, prefix, stage / 'before' / name, expected[name])
    evidence_names = ['preparation', 'source_preparation_r1', 'checker_preparation', 'transfer_audit',
                      'solver_red', 'solver_green', 'precision_red', 'precision_green',
                      'pixel_fixture_green', 'bash_syntax']
    evidence = {('outputs/cctv_dgp_raw_conditioning_learning_v1_' + name + '.json'):
                sha(OUT / ('cctv_dgp_raw_conditioning_learning_v1_' + name + '.json')) for name in evidence_names}
    evidence['outputs/cctv_dgp_conditioning_learning_v1_r4_analysis/publication/publication_audit.json'] = sha(
        ANALYSIS / 'publication/publication_audit.json')
    write(stage / 'publication.json', dict(
        complete=True, UTC=datetime.now(timezone.utc).isoformat(), publisher_sha256=sha(__file__), documents=changed,
        new_documents_sha256={n: sha(ROOT / n) for n in [guide_name, report_name]}, evidence_sha256=evidence,
        protocol_sha256=pin, archive_sha256=prepared['archive_sha256'], archive_bytes=prepared['archive_bytes'],
        previous_R4_rejections_and_stop_retained=True, actual_RAW_and_PNG_quality_gates_unchanged=True,
        RAW32_optimizer_change_explicitly_declared=True, manual_training_unrun=True,
        no_local_models_gradients_or_actual_weight_proposals=True,
        no_new_incumbent_or_epochs_or_app_changes=True, model_qualified=False, goal_complete=False))
    print(dict(complete=True, manual_guide=guide_name, exact_history_documents=len(changed),
               verified_transfer_bytes=prepared['archive_bytes'], manual_training_unrun=True))


if __name__ == '__main__':
    main()
