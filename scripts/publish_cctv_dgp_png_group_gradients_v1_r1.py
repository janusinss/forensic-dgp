"""Publish only a verified, unrun manual packet and preserve status history."""
from datetime import datetime, timezone
from pathlib import Path

from import_cctv_dgp_png_group_gradients_v1_r1_return import ROOT, OUT, NAME, STEM, PIN, read, sha, write
from publish_cctv_dgp_bank_saved_direction_forward_v1_r1_failure import prepend_expected, new, link


def main():
    stage = OUT / 'cctv_dgp_png_group_gradients_v1_r1_publication'
    assert not stage.exists()
    packet = OUT / NAME
    prep_path = OUT / 'cctv_dgp_png_group_gradients_v1_r1_preparation.json'
    audit_path = OUT / 'cctv_dgp_png_group_gradients_v1_r1_transfer_audit.json'
    bash_path = OUT / 'cctv_dgp_png_group_gradients_v1_r1_bash_syntax.json'
    checker_path = OUT / 'cctv_dgp_png_group_gradients_v1_r1_return_checker_preparation/verified.json'
    proof_paths = [prep_path, audit_path, bash_path, checker_path]
    assert all(read(path)['complete'] for path in proof_paths)
    prep, audit, bash, checker = [read(path) for path in proof_paths]
    p = read(packet / 'protocol.json')
    assert PIN == sha(packet / 'protocol.json') == prep['protocol_sha256'] == audit['protocol_sha256'] == checker['protocol_sha256']
    archive = OUT / (STEM + '-execution.tar.gz')
    assert sha(archive) == prep['archive_sha256'] == audit['archive_sha256']
    assert bash['exit_code'] == 0 and bash['checked_shell_sha256'] == sha(packet / 'scripts/run_png_groups.sh')
    assert checker['checker_sha256'] == sha(ROOT / 'scripts/audit_cctv_dgp_png_group_gradients_v1_r1_return.py')
    assert not (packet / 'outputs').exists() and not (packet / 'review.log').exists()
    protected = read(OUT / 'cctv_dgp_bank_group_step_check_v1_analysis/local_protected_before.json')['files_sha256']
    assert all(sha(ROOT / name) == digest for name, digest in protected.items())
    assert all(sha(ROOT / name) == digest for name, digest in prep['original_files_sha256'].items())
    previous = read(OUT / 'cctv_dgp_bank_group_step_check_v1_analysis/publication.json')
    docs = ['PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
            'CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md']
    expected = {name: previous['documents'][name]['after_sha256'] for name in docs}
    design = 'CCTV_DGP_DELIVERED_LOSS_LEARNING_REVIEW_V1.md'
    expected[design] = previous['new_documents_sha256'][design]
    assert all(sha(ROOT / name) == digest for name, digest in expected.items()), 'Preserve concurrent document edits'
    guide_name = 'CCTV_DGP_PNG_GROUP_GRADIENTS_V1_R1_VM.md'
    assert not (ROOT / guide_name).exists()
    guide = f'''# PNG-loss gradient measurement V1-R1 — manual L4 commands

**Verified locally; R1 has not been uploaded or launched by the agent.**
Use the fresh **V1-R1** files. The original V1 had an incorrect export prefix;
its archive and code stay preserved. Its human VM execution state is unknown.
{link('Retained V1 naming error', 'outputs/cctv_dgp_png_group_gradients_v1_export_name_stop/receipt.json')}.

This is a bounded diagnostic: 100 paired photographic TRAIN cases, 32 delivered
PNG loss groups, 320 gradient queries, 40 candidate and 20 recognizer forwards.
**Zero optimizer updates, parameter proposals, epochs or new checkpoints.**
It tests exact PNG forward values with an explicitly approximate backward;
it does not establish a true PNG derivative or improved model quality.
Native CCTV, final identities and completion images stay outside this measurement.

Require **4 GiB free after installation**. Estimated measurement/export runtime:
**3–8 minutes**, subject to actual L4 timing. Measurement cap10min, worker15min,
outer15.5min; export5min/outer5.5min, with30s termination grace per stage.
The first two complete batches must project within10min. Limits: VRAM20GiB,
RSS12GiB, output1,280MiB, storage reserve1GiB. Export retains enough free space
for the full payload and reserve. Current VM free space/power is not assumed.
Upload size: **{prep['archive_bytes']:,} bytes**; planned saved arrays about0.94GiB.

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
test -d ~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm &&
test ! -e ~/forensic-dgp/{NAME} &&
tar -xzf {STEM}-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:
```bash
tmux new-session -A -s dgp_png_group_gradients_v1_r1
```

4. Launch **inside tmux**:
```bash
cd ~/forensic-dgp/{NAME} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/{NAME}.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha {PIN} --verify-transfer &&
bash scripts/run_png_groups.sh {PIN}
```

The hash-only preflight reports `packet_assets: 6`, `parent_assets: 5852`,
`groups: 32`, `planned_gradient_queries: 320` and zero actual model/gradient calls.
The run reports `reference_batches` progressing to20 and `gradient_queries` to320.
Export runs after a success or retained stop. `complete: true` in the export
means the archive was created; it is not a successful model or training result.
Detach with **Ctrl+B,D**; reattach with
`tmux attach -t dgp_png_group_gradients_v1_r1`.
A competing GPU workload is preserved. Used packets refuse unchanged reruns;
do not remove an existing outputs directory or bypass any stop.

5. Download from **Windows Google Cloud SDK Shell** after export finishes:
```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz" "{STEM}-results.tar.gz"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz.sha256" "{STEM}-results.tar.gz.sha256"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-export.json" "{STEM}-export.json"
```

Each download uses one remote source for Windows PuTTY. Return all three files
for saved raw/PNG controls, scalar/embedding, gradient-group and timing audits.
The prospective checker is prepared; no returned R1 results exist yet.
The1% structure and all raw/PNG preservation/brightness requirements stay binding.
More epochs need a separate justified finite test; nothing auto-launches here.
Existing app checkpoint, splits, research caches, prior failures and full thesis
restoration/completion requirements are retained. These commands target the
existing named L4 VM; a different VM requires identity/migration verification.

Protocol SHA256: `{PIN}`.
Execution archive SHA256: `{prep['archive_sha256']}`.
The checksum is verified LF text, avoiding the earlier CRLF filename error.
{link('Independent transfer check', audit_path.relative_to(ROOT).as_posix())}.
{link('Delivered-loss design review', design)}.
'''
    new(ROOT / guide_name, guide)
    stage.mkdir()
    backup = stage / 'before'
    backup.mkdir()
    prefix = f'''# Latest next step — 13 October 2026; corrected PNG diagnostic prepared

Fresh V1-R1 is verified for human-manual L4/tmux use; it is unrun locally and
has not been uploaded or launched by the agent. The original V1 export-name
mismatch is retained with its unchanged archive/code; original human VM
execution is unknown. {link('Exact upload, install, tmux and download commands', guide_name)}.

100 fixed paired TRAIN cases / 32 PNG loss groups / 320 approximate-gradient
queries; zero optimizer updates, parameter proposals, epochs or checkpoints.
Exact saved-value forward checks, all7 transfer files/5,852 parent bindings,
100 affines, bounded timing/storage and prospective return fixtures passed.
The return checker explicitly selects100 cases from the3,905 parent rows.
Raw/PNG gates,1% structure and visible appearance remain unchanged. This packet
is preparation, not model qualification or evidence supporting more epochs.
{link('Transfer audit', audit_path.relative_to(ROOT).as_posix())}.

All five milestones and seven covering families remain required; app design,
permanent baseline, source/final splits, failures and migration assets remain
preserved. Native CCTV stays unpaired and final identities stay outside tuning.
Manual training/credit/migration rules remain; agent VM access is maintenance
only. No active workload is stopped and no VM state/free-space claim is made.
The latest signed-step results and full earlier goal remain as exact bytes below.

'''
    documents = {name: prepend_expected(ROOT / name, prefix, backup / name, expected[name]) for name in docs}
    design_prefix = f'''# Verified manual diagnostic now prepared — 13 October 2026

The design below has a corrected V1-R1 transfer packet with exact PNG forward
and explicitly approximate backward measurement. It remains unrun; no optimizer
updates, parameter proposals, extra epochs or model promotion are authorized by
packet verification. {link('Manual guide', guide_name)}.
The original design-review bytes and scientific requirements follow unchanged.

'''
    documents[design] = prepend_expected(ROOT / design, design_prefix, backup / design, expected[design])
    assert all(sha(ROOT / name) == digest for name, digest in protected.items())
    write(stage / 'publication.json', dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        publisher_sha256=sha(__file__), protocol_sha256=PIN, archive_sha256=sha(archive),
        proof_sha256={path.relative_to(ROOT).as_posix(): sha(path) for path in proof_paths},
        documents=documents, new_document_sha256={guide_name: sha(ROOT / guide_name)},
        protected_app_checkpoint_source_files=protected,
        original_packet_files=prep['original_files_sha256'], no_VM_connection_or_launch=True,
        R1_only_local_and_unrun=True, model_qualified=False, more_epochs_qualified=False, goal_complete=False))
    print(dict(complete=True, manual_guide=guide_name, protocol_sha256=PIN, actual_VM_launch=False))


if __name__ == '__main__':
    main()
