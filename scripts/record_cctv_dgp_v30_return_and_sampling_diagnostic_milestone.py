"""Record audited V30 failure and a distinct manual, zero-update diagnostic packet."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/dgp_v30_return_and_sampling_gradient_v1_milestone'
CLEANUP = ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261007_v2'
PREP = ROOT / 'outputs/cctv_dgp_v30_sampling_gradient_v1_preparation'
REVIEW = ROOT / 'outputs/cctv_dgp_broader_mean_v30_failure_review_v1'
RETURNED = ROOT / 'outputs/cctv_dgp_broader_mean_v30_return'
DOCS = ['PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md']


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    assert not OUT.exists(), 'Preserve previous milestone'
    audit_path = ROOT / 'outputs/cctv_dgp_broader_mean_v30_independent_audit_r1.json'
    audit = read(audit_path); packet = read(PREP / 'independent_packet_audit.json')
    assert audit['complete'] and audit['VM_failure_retained'] and audit['original_checker_failure_retained']
    assert audit['members_verified'] == 27622 and audit['completed_snapshot_PNG_metrics_vectors_and_mean_controls_audited'] == 7810
    assert not audit['training_completed800'] and not audit['necessary_capacity_pass']
    assert audit['local_gradient_calls'] == audit['local_backward_calls'] == audit['local_optimizer_updates'] == 0
    assert [r['update'] for r in audit['snapshots_audited']] == [0, 50]
    assert packet['complete'] and packet['optimizer_updates'] == 0 and packet['gradient_queries_bound'] == 280
    assert packet['regressions_passed'] == 11 and not packet['VM_execution_started']
    visual = read(REVIEW / 'visual_review.json'); measured = read(REVIEW / 'preparation.json')
    assert visual['complete'] and visual['cases_reviewed'] == 50 and visual['comparison_cells_reviewed'] == 250
    early = read(RETURNED / 'outputs/early_structure_stop.json'); failure = read(RETURNED / 'outputs/failure.json')
    assert early['minimum'] == .01 and not early['pass'] and early['update'] == failure['optimizer_updates'] == 50
    assert failure['cause'] == 'No one-percent early structural gain; retain stop' and not failure['resume_permitted']
    assert not measured['preservation_regressions_at50']
    roundoff_path = ROOT / 'outputs/cctv_dgp_broader_mean_v30_audit_execution_v1/receipt_roundoff_diagnostic.json'
    roundoff = read(roundoff_path)
    previous = read(CLEANUP / 'closure_manifest.json')
    assert sha(CLEANUP / 'closure_manifest.json') == 'b409d886495f3e9375b54bef1fc0ad6dd32aabfe8e73ea26ce1e280e44b8507c'
    for row in previous['documents']: assert sha(ROOT / row['name']) == row['after_sha256']
    app_path = ROOT / 'outputs/dgp_app_v3_integration_record.json'
    assert sha(app_path) == 'e744f3708f248012469a111e204222f66261af4ca28bbaf9521ee4c5464b5e7a'
    app = read(app_path)
    for name, digest in {**app['sources_sha256'], **app['evidence_sha256']}.items(): assert sha(ROOT / name) == digest
    OUT.mkdir(); (OUT / 'before_docs').mkdir()
    gain = early['relative_feature_error_gain'] * 100
    common = f'''**Latest research milestone - 7 October 2026: V30 independently audited; early structure gate failed.**

The human ran V30 on the existing NVIDIA L4 and returned the archive. The local
independent R1 audit verifies 27,622 files, both complete 3,905-case TRAIN
snapshots, raw/display separation, source and checkpoint provenance, saved
gradients and CPU inference replay. V30 stopped after 50/800 optimizer updates:
structure gain {gain:.6f}% against the unchanged 1% early requirement. No
800-update result exists. Preserve the stopped checkpoint, logs and failed gate;
do not resume, rerun unchanged or promote V30 into the app.

All 17 fixed group preservation checks pass at update50. Source-group gains are
1.072549% for dataset/asian_faces/degraded and 0.742743% for
dataset/thumbnails128x128/degraded. This is paired synthetic photographic TRAIN
evidence; source labels do not establish ethnicity, real CCTV performance or
Zamboanga performance. No new native, development or reserved-final cases were
opened. All 50 fixed preview rows/250 exact comparison cells were visually
reviewed: visible appearance remains broad, but the outputs are soft and show
little useful whole-face clarity gain. All visible facial features remain in scope.

The original local checker failed at its last early-receipt equality comparison
because CPU/VM aggregate arithmetic differed by 1.1102230246251565e-16. Its source
and failure logs remain unchanged. A separate R1 checker allows 1e-12 receipt
arithmetic difference, while retaining the exact 1% threshold and both failed
gate decisions. This checker repair does not change the V30 recipe or qualification.

After the required V28-V30 architecture discussion, the user selected 'Apply the
best approach'. Before another training recipe, a distinct metadata-selected
sampling-gradient diagnostic compares two 50-case matched TRAIN cohorts at the
original and stopped update50 states. It has 280 gradient queries, zero optimizer
updates, a 600-second worker limit, a 900-second external limit and finite export
limits. These subsets are not held-out evaluation. Selection uses frozen schedule
and source/profile metadata only; reference repetition is matched one to one.
No new loss, training recipe or checkpoint is
created. The independently verified packet remains unrun and requires the human
upload/install/tmux/download steps on the existing L4/g2-standard-4 VM. All new
actual training continues under the user's manual command workflow.

The prior authorized cleanup reclaimed 36.75 GiB and measured 43.13 GiB free
before V30 ran; those are historical measurements, not a fresh disk reading.
The actual Windows research-cache backup remains preserved. Research markdown,
provenance, original checkpoints, splits and all prior failed gates remain intact.

The DGP-led app remains unchanged. V29 development/native qualification failures
remain binding. Useful native restoration, automatic and assisted evidence for
all seven covering families, an independent final review and the full bundled
inline Playwright app flow remain required. Preserve clear glasses and visible
appearance; request a clearer/less-covered crop when structure is insufficient.
Goal active/incomplete.

[V30 results](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BROADER_MEAN_V30_RESULTS.md>)
[Next manual diagnostic steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V30_SAMPLING_GRADIENT_V1_VM.md>)
[Independent V30 audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_broader_mean_v30_independent_audit_r1.json>)

The complete earlier document body is preserved below. Its V30-not-started
language is historical and superseded by this audited human run.

'''
    documents = []
    for name in DOCS:
        path = ROOT / name; before = path.read_bytes(); backup = OUT / 'before_docs' / name
        with backup.open('xb') as stream: stream.write(before)
        split = before.index(b'\n') + 1; addition = b'\n' + common.encode('ascii')
        after = before[:split] + addition + before[split:]
        assert path.read_bytes() == before, 'Preserve intervening edits'
        path.write_bytes(after)
        assert path.read_bytes() == after and after[:split] + after[split + len(addition):] == before
        documents.append({'name': name, 'before_path': backup.relative_to(ROOT).as_posix(),
                          'before_sha256': sha(backup), 'after_sha256': sha(path),
                          'addition_bytes': len(addition), 'complete_previous_body_preserved': True})
    sources = ['audit_cctv_dgp_broader_mean_v30_return_r1.py',
               'diagnose_cctv_dgp_broader_mean_v30_receipt_roundoff.py',
               'prepare_cctv_dgp_broader_mean_v30_failure_review.py',
               'record_cctv_dgp_broader_mean_v30_failure_review.py',
               'cctv_dgp_v30_sampling_gradient_v1_vm.py',
               'prepare_cctv_dgp_v30_sampling_gradient_v1.py',
               'cctv_dgp_v30_sampling_gradient_v1_return_audit_template.py',
               'audit_cctv_dgp_v30_sampling_gradient_v1_return.py',
               'verify_cctv_dgp_v30_sampling_gradient_v1_packet.py',
               'record_cctv_dgp_v30_return_and_sampling_diagnostic_milestone.py',
               'verify_cctv_dgp_v30_return_and_sampling_diagnostic_milestone.py']
    paths = [ROOT / 'scripts' / name for name in sources]
    paths.extend(ROOT / name for name in DOCS)
    paths.extend(ROOT / name for name in [
        'tests/test_cctv_dgp_v30_sampling_gradient_v1.py', 'CCTV_DGP_BROADER_MEAN_V30_RESULTS.md',
        'CCTV_DGP_V30_SAMPLING_GRADIENT_V1_VM.md', 'outputs/cctv_dgp_broader_mean_v30_return_import.json',
        'outputs/cctv_dgp_broader_mean_v30_r1_reuse_import.json',
        'outputs/cctv_dgp_broader_mean_v30_independent_audit_r1.json',
        'outputs/cctv-dgp-broader-mean-v30-results.tar.gz',
        'outputs/cctv-dgp-broader-mean-v30-results.tar.gz.sha256',
        'outputs/cctv-dgp-broader-mean-v30-export.json',
        'outputs/cctv-dgp-v30-sampling-gradient-v1-execution.tar.gz',
        'outputs/cctv-dgp-v30-sampling-gradient-v1-execution.tar.gz.sha256'])
    for folder in [OUT / 'before_docs', REVIEW, PREP, ROOT / 'outputs/cctv_dgp_v30_sampling_gradient_v1_vm',
                   ROOT / 'outputs/cctv_dgp_broader_mean_v30_audit_execution_v1',
                   ROOT / 'outputs/cctv_dgp_broader_mean_v30_audit_execution_r1']:
        paths.extend(path for path in folder.rglob('*') if path.is_file())
    bindings = {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(set(paths))}
    receipt = {'complete': True, 'created_utc': datetime.now(timezone.utc).isoformat(),
               'scope': 'Independently audited V30 failure; diagnostic preparation, not restoration qualification',
               'documents': documents, 'new_evidence_sha256': bindings,
               'previous_cleanup_closure_sha256': sha(CLEANUP / 'closure_manifest.json'),
               'previous_cleanup110_original_locations': {r['name']: r['before_path'] for r in documents},
               'previous_research_milestone_sha256': sha(ROOT / 'outputs/dgp_v29_development_and_broader_mean_v30_milestone/milestone.json'),
               'app_record_sha256': sha(app_path), 'app22_bindings_preserved': True,
               'V30_archive_sha256': audit['archive_sha256'], 'V30_members_audited': 27622,
               'V30_complete_snapshot_cases_audited': 7810, 'V30_optimizer_updates': 50,
               'V30_structure_gain_fraction': early['relative_feature_error_gain'], 'unchanged_minimum': .01,
               'V30_capacity_pass': False, 'V30_preservation_group_failures_at50': 0,
               'V30_training_completed800': False, 'V30_fixed_TRAIN_previews_reviewed': 50,
               'all_visible_regions_reviewed': True, 'original_checker_failure_and_source_retained': True,
               'R1_receipt_tolerance': 1e-12, 'diagnostic_protocol_sha256': packet['protocol_sha256'],
               'diagnostic_gradient_queries_bound': 280, 'diagnostic_optimizer_updates': 0,
               'new_training_recipe_created': False, 'diagnostic_actual_VM_run_started': False,
               'manual_VM_execution_required': True, 'local_gradient_calls': 0,
               'local_backward_calls': 0, 'local_optimizer_updates': 0, 'new_VM_actions': [],
               'native_or_reserved_used': False, 'independent_final_review_complete': False,
               'app_promotion': False, 'goal_status': 'active', 'goal_complete': False}
    write(OUT / 'milestone.json', receipt)
    print(json.dumps({key: receipt[key] for key in ['complete', 'V30_optimizer_updates', 'V30_capacity_pass', 'diagnostic_gradient_queries_bound', 'diagnostic_actual_VM_run_started', 'goal_status']}))


if __name__ == '__main__': main()
