"""Publish independently verified, UNRUN saved-path commands; never run a model."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from publish_cctv_dgp_png_group_gradients_v1_r2 import finish_prepend, retained_new

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs'
DESIGN = OUT / 'cctv_dgp_spatial_path_replay_v1_design'
NAME = 'cctv_dgp_spatial_path_replay_v1_vm'
STEM = 'cctv-dgp-spatial-path-replay-v1'
PIN = '6f9a8757466578ef57457994ded7fe9b6b170657f2ee7dd3b8b23f0bd2cc1bc7'
ARCHIVE = '228e7bb9c1a957021c86be471f0eea1f3f0a5c38ada99dedf4fb69756b8c7f88'
CHECKPOINT = '646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def link(label, name):
    return '[' + label + '](<' + (ROOT / name).as_posix() + '>)'


def main():
    stage = DESIGN / 'verified_publication'
    stage.mkdir(exist_ok=True)
    (stage / 'before').mkdir(exist_ok=True)
    assert not (stage / 'publication.json').exists(), 'Retain completed publication'
    proof = read(DESIGN / 'independent_preparation_audit.json')
    compare = read(DESIGN / 'progress_comparison_audit.json')
    assert proof['complete'] and proof['status'] == 'VERIFIED_PREPARED_UNRUN'
    assert proof['CPU_tests'] == 15 and proof['actual_saved_array_scorer_calibrations'] == 30
    assert compare['complete'] and compare['baseline_RAW_controls'] == 100
    assert compare['same100_input_ids_in_order'] and compare['same_scientific_thresholds']
    packet = OUT / NAME
    assert sha(packet / 'protocol.json') == PIN == proof['protocol_sha256']
    archive = OUT / (STEM + '-execution.tar.gz')
    assert sha(archive) == ARCHIVE == proof['archive_sha256']
    assert archive.stat().st_size == 85745
    expected_sidecar = (ARCHIVE + '  ' + archive.name + '\n').encode('utf-8')
    assert Path(str(archive) + '.sha256').read_bytes() == expected_sidecar
    assert not (packet / 'outputs').exists() and not (packet / 'review.log').exists()
    protected = read(DESIGN / 'protected_before_preparation.json')['files']
    documents = ['SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
                 'CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md', 'CCTV_DGP_NEXT_PILOT.md',
                 'PROJECT_HANDOFF.md', 'CCTV_DGP_POST_SPATIAL_READOUT_REVIEW_V1.md']
    for name, expected in protected.items():
        backup = stage / 'before' / name
        assert sha(ROOT / name) == expected or (
            name in documents and backup.is_file() and sha(backup) == expected
        ), 'Preserve concurrent changes: ' + name
    old_metrics_path = OUT / 'cctv_dgp_bank_group_direction_forward_v1_analysis/finite_metrics.json'
    old = read(old_metrics_path)['proposals']['balanced_0.02']['gates']
    result_path = OUT / 'cctv_dgp_spatial_readout_v1_return/outputs/spatial_readout_v1/run_results.json'
    outcomes = read(result_path)['outcomes']
    rows = [dict(trial='Earlier temporary direction, step 0.02',
                 RAW_gain_percent=100 * old['raw']['relative_feature_gain'],
                 PNG_gain_percent=100 * old['png']['relative_feature_gain'],
                 RAW_failed_group_metric_checks=len(old['raw']['preservation_failures']),
                 PNG_failed_group_metric_checks=len(old['png']['preservation_failures']),
                 new_optimizer_updates=0, accepted=False, checkpoint_created=False)]
    for outcome in outcomes:
        gates = outcome['gates']
        rows.append(dict(trial='Spatial reconstruction ' + outcome['route'],
                         RAW_gain_percent=100 * gates['raw']['global_structure_gain'],
                         PNG_gain_percent=100 * gates['png']['global_structure_gain'],
                         RAW_failed_group_metric_checks=len(gates['raw']['preservation_failures']),
                         PNG_failed_group_metric_checks=len(gates['png']['preservation_failures']),
                         new_optimizer_updates=outcome['updates'], accepted=False,
                         stopped_research_state_retained=True))
    assert [r['PNG_failed_group_metric_checks'] for r in rows] == [37, 8, 18]
    assert all(r['PNG_gain_percent'] < 1 for r in rows)
    ledger = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
                  rows=rows, original_checkpoint_sha256=CHECKPOINT,
                  comparison_receipt_sha256=sha(DESIGN / 'progress_comparison_audit.json'),
                  old_metrics_sha256=sha(old_metrics_path), new_results_sha256=sha(result_path),
                  accepted_development_updates=0, app_model_replaced=False,
                  same100_paired_TRAIN_inputs_and_original_baseline=True,
                  neural_calls=0, gradient_queries=0, optimizer_updates=0,
                  agent_VM_calls=0, model_qualified=False)
    ledger_path = DESIGN / 'experiment_status_ledger.json'
    if ledger_path.exists():
        saved = read(ledger_path)
        assert saved['rows'] == rows and saved['old_metrics_sha256'] == sha(old_metrics_path)
        assert saved['new_results_sha256'] == sha(result_path)
    else:
        with ledger_path.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(ledger, stream, indent=2); stream.write('\n')

    progress_name = 'CCTV_DGP_EXPERIMENT_PROGRESS_20261014.md'
    guide_name = 'CCTV_DGP_SPATIAL_PATH_REPLAY_V1_VM.md'
    prep_name = 'CCTV_DGP_SPATIAL_PATH_REPLAY_V1_PREPARATION.md'
    table = '\n'.join(
        f"| {r['trial']} | {r['RAW_gain_percent']:.6f}% | {r['PNG_gain_percent']:.6f}% | "
        f"{r['RAW_failed_group_metric_checks']}/{r['PNG_failed_group_metric_checks']} | Rejected |"
        for r in rows
    )
    progress = f'''# Experiment results and the retained model — 14 October 2026

**The latest experiment did not improve the model.** The earlier approximately
0.87% result was also rejected. Neither replaced the original app checkpoint.
These are different candidate experiments, not successive accepted checkpoints.

| Experiment | RAW structure gain | Delivered PNG structure gain | Failed RAW/PNG preservation checks | Decision |
|---|---:|---:|---:|---|
{table}

The comparison uses the same 100 synthetic photographic TRAIN inputs, the same
original RAW controls and unchanged scientific thresholds. All 203 shared parent
bindings match; every original RAW batch matches the earlier exact control hash.
These are 20 already exposed references, not native CCTV or reserved final faces.

Structure gain means relative reduction in landmark-region high-frequency error
against the paired clean TRAIN target. It is not percentage recovered identity,
percentage of the face reconstructed, or proof that an image looks better.
The 1% early gate also requires per-source gain and preservation of MSE, SSIM,
fixed-alignment ArcFace and visible appearance, with a brightness control.
Failed-check counts refer to group/metric checks, not people or images.

The earlier 0.877397% PNG proposal performed zero optimizer updates or epochs;
it temporarily displaced conditioning parameters, failed 37 preservation checks,
restored its initial state and created no replacement checkpoint. Route A
trained a separate copy of selected decoder tensors for 50 research updates;
it reduced failed checks to eight but achieved only 0.132699% PNG structure gain.
Route B added spatial prior connections, achieved only 0.111965%, failed 18 checks
and failed the brightness control. All are rejected, with failures retained.
The frozen external StyleGAN2 bank is declared; it is not our own trained GAN.

What is demonstrated: A fits its clear-image anchor with little added facial
detail; B worsens the saved objective on 19/20 references. Both stayed on the
original-relative parameter bound at every update. All selected tensors changed
and B's spatial neighbors learned. Connectivity alone did not solve the problem.
A unique architecture, data, step-size or loss cause has not been isolated.
More epochs are not yet justified by these results.

The selected next check replays already saved A/B states at updates 0, 1, 10,
20, 30, 40, 50. It measures where structure or preservation first fails before
choosing another training change. It adds zero updates, gradients or epochs.
Every planned sheet must be directly reviewed after the manual VM return; a
retrospectively promising TRAIN state is not selected as an app incumbent.

Original retained checkpoint: `{CHECKPOINT}`.
There is no qualified development replacement or cumulative accepted improvement.
The five milestones and seven automatic/assisted covering families remain pending.

{link('Earlier audited 0.87% proposal', 'CCTV_DGP_BANK_GROUP_DIRECTION_FORWARD_V1_RESULTS.md')}.
{link('Current audited reconstruction result', 'CCTV_DGP_SPATIAL_READOUT_V1_RESULTS.md')}.
{link('Exact comparison controls', 'outputs/cctv_dgp_spatial_path_replay_v1_design/progress_comparison_audit.json')}.
{link('Evidence-bound experiment ledger', 'outputs/cctv_dgp_spatial_path_replay_v1_design/experiment_status_ledger.json')}.
{link('Next manual diagnostic commands', guide_name)}.
'''
    guide = f'''# Saved learning-path replay — manual L4 commands

**VERIFIED / PREPARED / UNRUN.** This measures existing A/B reconstruction states;
it performs zero new training updates, gradients or epochs. The original app
checkpoint stays retained. This is the diagnostic selected after both endpoints
failed. The agent has not uploaded or launched it.

Require **8 GiB free after install** and **8 GiB available RAM before models**.
Use the existing L4 VM, `forensic-dgp-thesis`, in `us-central1-a`. Keep the existing
`cctv_dgp_spatial_readout_v1_vm`, `cctv_dgp_bank_photometric_guard_v1_vm` and shared
venv. The thin packet reuses their hash-bound saved states and research assets.
Estimate **15–30 minutes including audit/export**; the new runtime is unmeasured.
Enforced maxima including kill grace: measurement 41 minutes, saved audit
11 minutes, export 11 minutes; hash-only verification adds time separately.
VRAM 20 GiB, RSS 12 GiB, output 3 GiB, retained free reserve 1 GiB.

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
test -f ~/forensic-dgp/cctv_dgp_spatial_readout_v1_vm/protocol.json &&
test -f ~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm/protocol.json &&
test ! -e ~/forensic-dgp/{NAME} &&
tar -xzf {STEM}-execution.tar.gz -C ~/forensic-dgp
```

Expected checksum: `OK`. The SHA256 sidecar is verified LF UTF-8 without BOM or
carriage returns. If the destination exists, stop and preserve that previous run;
do not delete it or bypass the guard.

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_spatial_path_replay_v1
```

4. Measure **inside tmux**:

```bash
cd ~/forensic-dgp/{NAME} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_spatial_path_replay_v1_vm.py --root . --protocol-sha {PIN} --verify-transfer &&
bash scripts/run_spatial_path.sh {PIN}
```

The hash-only check should report `complete: true`, 10 packet assets, 410 source
assets, 203 parent assets, 14 saved states and zero neural calls/optimizer updates.
The supervised measurement then prints progress for both routes, the saved audit
and export. `Ctrl+B`, then `D`, detaches tmux without stopping it. `Ctrl+C` in the
running worker interrupts it. Do not rerun any historical training worker.

Expected diagnostic completion: `review_exit: 0`, `audit_exit: 0`, `export_exit: 0`
and `optimizer_updates: 0`. Numeric gates can still fail. Export `complete: true`
confirms packaging only; it is not training or model-quality success. If any stage
fails, preserve the printed traceback and all export files for independent audit.

5. Download from **Windows Google Cloud SDK Shell**, one remote source per command:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-export.json" "."
```

This avoids the Windows PuTTY multiple-remote-sources error. Keep all three files
in the named local `outputs` directory. Do not delete the VM research assets while
the run or transfer is active. No cleanup is included in these commands.

Execution archive SHA256: `{ARCHIVE}`; **85,745 bytes**.
Protocol SHA256: `{PIN}`.

After return: verify the archive/manifest, independently check all saved outputs,
all 17 RAW/PNG groups and exact 0/50 controls, then directly inspect every one of
the 40 planned sheets. This packet does not qualify native CCTV, final identities,
completion families or a replacement app checkpoint.

{link('Verified preparation and limits', prep_name)}.
{link('Why 0.87% and 0.13% are rejected experiments', progress_name)}.
'''
    preparation = f'''# Saved learning-path packet verification — 14 October 2026

**VERIFIED / PREPARED / UNRUN.** The independent preparation audit passed.
No agent VM connection, new neural evaluation, local candidate construction,
gradient query, optimizer update, epoch extension or app change occurred.
Original checkpoints, source states, caches, split roles and failures remain.

The 85,745-byte execution archive contains 11 regular files: 10 bound assets
and its protocol. It reuses 410 exact source bindings and 203 parent bindings
from the manually completed, rejected spatial reconstruction comparison.
Fourteen finite saved states are frozen: A/B at 0, 1, 10, 20, 30, 40, 50 updates.
All saved vectors are finite, hash-bound and inside the unchanged original bound.
Read-only tensor-byte streaming matched the original and final full-state hashes;
it did not construct or modify a local candidate network.

All 100 paired TRAIN inputs and 20 canonical targets are bound. The diagnostic
plans 1,400 saved-state compositions plus 100 fresh original controls. It requires
500 exact original/initial/endpoint controls and separate RAW and delivered PNG
scores for the same 17 source/profile groups. Original, recognizer and declared
external prior states must remain unchanged. It has no train or resume option.

Final local CPU verification passed **15 tests**, including 30 actual saved-array
RAW/PNG scorer calibrations, invalid protocols, vector/geometry boundaries, exact
state hashes and Windows rejection before any model or output construction.
The sealed packet retains the earlier 13-test receipt; the final expanded suite
and separate independent preparation receipt are retained locally. Actual
hash-only CLI, Python 3.10 parsing, read-only Bash syntax and all transfer hashes
passed. The old guarded source working-directory contract remains enforced;
the new manual worker enters that directory only for guarded dependency calls.
No historical training worker, optimizer or learning-mode helper is called.

The planned 40 sheets are 2304×1320: one per reference per route, nine columns
(input, paired TRAIN target, original, 1, 10, 20, 30, 40, 50), five degradation rows,
and 1,800 exact 256×256 cells. Direct review of all 40 is required after return.
Even a numerically passing exposed TRAIN snapshot cannot retrospectively become
an incumbent. Unsampled updates remain unmeasured. New CUDA parity and timing
are unverified until the human manually executes the L4 diagnostic.

Finite resource rules: 8 GiB free after install, 8 GiB available RAM before models,
1 GiB disk reserve, 3 GiB output cap, 20 GiB peak allocated VRAM, 12 GiB RSS.
Worker 2,400 s / external 2,430 s; cache 240 s; each snapshot 180 s; saved audit
600 s / external 630 s; export 600 s / external 630 s. Each external stage has
30 s kill grace and only terminates its own child process group. Planned call
limits are 20 original batches, 20 prior-feature batches, 280 decoder batches
and 600 recognizer batches. New updates, backwards and gradients are all zero.
Estimated runtime including audit/export is 15–30 minutes, not a fresh benchmark;
supervised upper bound is 63 minutes plus separate hash-only verification.

Retained preparation stops: an early CPU fixture used a Windows temporary-folder
permission pattern; ordinary retained workspace fixtures fixed it and the final
suite passed. The sandboxed read-only Bash syntax command could not create its
signal pipe; the identical `-n` check passed with approved host execution. No
script or model was executed by that syntax check. These are local preparation
issues, not new VM training failures. All retained logs and receipts remain.

Archive SHA256: `{ARCHIVE}`.
Protocol SHA256: `{PIN}`.
Original checkpoint SHA256: `{CHECKPOINT}`.

{link('Independent preparation audit', 'outputs/cctv_dgp_spatial_path_replay_v1_design/independent_preparation_audit.json')}.
{link('Final CPU test receipt', 'outputs/cctv_dgp_spatial_path_replay_v1_design/cpu_tests_final.json')}.
{link('Bash syntax receipt', 'outputs/cctv_dgp_spatial_path_replay_v1_design/bash_syntax.json')}.
{link('Frozen design', 'outputs/cctv_dgp_spatial_path_replay_v1_design/design.json')}.
{link('Exact five manual upload/install/tmux/measure/download steps', guide_name)}.

The full five-milestone restoration and seven-family automatic/assisted scope,
native/unpaired and synthetic/paired boundaries, untouched final identities and
app design remain unchanged. No qualified development replacement is claimed.
'''
    reports = {progress_name: progress, guide_name: guide, prep_name: preparation}
    new_hashes = {}
    for name, text in reports.items():
        retained_new(ROOT / name, text)
        new_hashes[name] = sha(ROOT / name)
    prefix = f'''# Saved learning-path diagnostic: verified, PREPARED / UNRUN — 14 October 2026

Neither the earlier 0.877397% PNG proposal (37 failed preservation checks) nor
the newer decoder A (0.132699%, eight failures) qualified. This is a measured
loss of structure gain in a different rejected experiment, not accepted model
progress. The same 100 TRAIN inputs, original RAW controls and gates match.
Original checkpoint `{CHECKPOINT}` remains retained; no app promotion occurred.
{link('Experiment/incumbent distinction and exact comparison', progress_name)}.

The new finite packet is independently verified and still UNRUN. It replays
existing A/B states at 0, 1, 10, 20, 30, 40, 50, using zero new optimizer updates,
gradients or epochs. Fourteen states / 1,500 compositions / 500 exact controls /
40 planned sheets remain to be measured and directly reviewed. Preparation
passed 15 CPU tests, 30 scorer controls, all 10/410/203 file bindings, hash-only
CLI, Python 3.10 and Bash syntax. New CUDA parity/timing are not yet verified.
{link('Exact manual L4 transfer/tmux/download commands', guide_name)}.
{link('Independent preparation and finite limits', prep_name)}.

Require 8 GiB free and 8 GiB available RAM before models. Estimate 15–30 minutes
including audit/export; supervised limits total 63 minutes plus hash checks.
No automatic launch, historical training call, failed optimizer resume,
retrospective TRAIN-incumbent selection, gate weakening or extra epochs.
Keep all checkpoints, journals, caches, splits and failures. Five milestones,
native/final review and seven automatic/assisted covering families stay pending.
Earlier DESIGN ONLY / unprepared statements below are superseded only for this
new certified transfer. Exact previous document follows.

'''
    bindings = {}
    for name in documents:
        bindings[name] = finish_prepend(ROOT / name, prefix, stage / 'before' / name, protected[name])
    unchanged = {name: sha(ROOT / name) for name in protected if name not in documents}
    assert all(unchanged[name] == protected[name] for name in unchanged)
    assert sha(archive) == ARCHIVE and sha(packet / 'protocol.json') == PIN
    receipt = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
                   documents=bindings, new_document_sha256=new_hashes,
                   unchanged_protected_files=unchanged,
                   experiment_ledger_sha256=sha(ledger_path),
                   preparation_audit_sha256=sha(DESIGN / 'independent_preparation_audit.json'),
                   progress_comparison_sha256=sha(DESIGN / 'progress_comparison_audit.json'),
                   protocol_sha256=PIN, archive_sha256=ARCHIVE,
                   status='VERIFIED_PREPARED_UNRUN', agent_VM_calls=0,
                   neural_calls=0, gradient_queries=0, optimizer_updates=0,
                   app_promotion=False, model_qualified=False, goal_complete=False)
    with (stage / 'publication.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    print(json.dumps(dict(complete=True, documents=len(bindings), new_documents=len(new_hashes),
                         status=receipt['status'], original_checkpoint_unchanged=True,
                         agent_VM_calls=0, optimizer_updates=0)))


if __name__ == '__main__':
    main()
