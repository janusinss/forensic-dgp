"""Preserve history and bind the V24 audit/review closure; no model or VM calls."""
import ast
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from import_cctv_dgp_degraded_detail_v24 import read, require, sha, write

OUT = ROOT / 'outputs/dgp_degraded_detail_v24_audit_and_architecture_milestone'
PREVIOUS = ROOT / 'outputs/dgp_detail_skip_v23_audit_and_degraded_detail_v24_milestone/milestone.json'
STATUS = ROOT / 'outputs/cctv_dgp_degraded_detail_v24_reported_stop_v1'
DOCS = ['PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
        'CCTV_DGP_DEGRADED_DETAIL_V24_PLAN.md', 'CCTV_DGP_DEGRADED_DETAIL_V24_VM.md']
SCRIPTS = [
    'scripts/import_cctv_dgp_degraded_detail_v24.py',
    'scripts/audit_cctv_dgp_degraded_detail_v24.py',
    'scripts/audit_cctv_dgp_degraded_detail_v24_execution.py',
    'scripts/audit_cctv_dgp_degraded_detail_v24_cohort.py',
    'scripts/diagnose_cctv_dgp_degraded_detail_v24.py',
    'scripts/verify_cctv_dgp_degraded_detail_v24_diagnostic.py',
    'scripts/record_cctv_dgp_degraded_detail_v24_visual_review.py',
    'scripts/audit_cctv_dgp_degraded_detail_v24_loss_v1.py',
    'scripts/verify_cctv_dgp_degraded_detail_v24_loss_v1.py',
    'scripts/record_cctv_dgp_post_v24_architecture_review.py',
    'scripts/record_cctv_dgp_degraded_detail_v24_milestone.py',
    'scripts/verify_cctv_dgp_degraded_detail_v24_milestone.py',
    'tests/test_cctv_dgp_degraded_detail_v24_return_audit.py']
LATEST = '''**Latest research milestone — 6 October 2026: V24 failure audited/reviewed; architecture discussion before another pilot.**

All three downloaded return files are verified:76,492,526-byte archive, SHA256
`1d758362366f3e1f17ddf8239b0f41940b6b7c6ebf349a86ed76e0abdf63dd15`.
Safe import371 members and independent43.42s replay pass all221 frozen assets,
100 raw/PNG pairs,110 recognizer calls, original cohort scalars and execution/
gradient/timing/state receipts. No tolerance or scientific gate changes.
Training correctly stops at update50:0.0033227502% delivered degraded structure
gain against the original1%; raw gain is0.0018034721%. Final800 never executes.
Export complete:true packages the failure; it is not a training success receipt.

All ten original-cell sheets/50 exposed TRAINING cases are visually reviewed;
all200 cells independently verified. No convincing whole-face gain over frozen
own DGP. Typical degraded correction is0.011075 of one byte level. The known-
source50-case stopped-layer trace and fixed saved-objective decomposition are
verified without local backwards/optimization. The V24 clear-reward correction
works, but degraded learning remains far below useful structure. Keep the stop.

Close the V22–V24 small filtered-head sequence. The user's “all are important”
requires eyes/nose/mouth/outline/visible appearance together. A concrete new
architecture review recommends revising our DGP spatial/feature reconstruction
path; the previously permitted declared face-prior extension is an alternative.
The user selects our DGP spatial/feature path. No fourth recipe, executable training
packet or model fix is prepared. Existing training commands for these stopped
recipes are historical; do not rerun or waive them. No further V24 download is
needed. Future training/gradient preflight remains manual on the existing L4.

The original documents, concurrent completed-storage-maintenance entry, original
checkpoints/splits/failures, prior664 research bindings and app22 bindings remain
preserved/verified. The original18 auditor regressions and historical34 app/
bundled inline Playwright checks are retained; no changed app requires a rerun.
Native evidence remains unpaired; paired photographic TRAINING metrics stay
separate. No native/reserved/new covering pixels, local training, assistant VM/
cloud action, app promotion, ethnicity or Zamboanga-performance claim occurs.
The previously useful native input remains usable despite model softness.
Useful native output, canonical app parity, all seven automatic/assisted covering
families and independent final review remain required. Goal active/incomplete.

[V24 audited result](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DEGRADED_DETAIL_V24_RESULTS.md>) ·
[Architecture discussion](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_POST_V24_ARCHITECTURE_REVIEW.md>) ·
[Milestone](<C:/xampp/htdocs/YEAR 4/Testing/outputs/dgp_degraded_detail_v24_audit_and_architecture_milestone/milestone.json>)

Previous bodies below are preserved history, including superseded pending-return
and pending-training notices. They are not current instructions to rerun V24.
'''


def main():
    started = time.monotonic()
    require(not OUT.exists(), 'Preserve previous milestone')
    require(sha(PREVIOUS) == '40f159663d100186dd285ff318680b6201fafb16e38f1b072bb785760ebf94d9', 'Previous milestone changed')
    previous = read(PREVIOUS)
    previous_locations = {}
    for name, expected in previous['new_evidence_sha256'].items():
        if name == 'PROJECT_HANDOFF.md':
            path = ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261006_v1/before_docs_r1/PROJECT_HANDOFF.md'
        elif name in DOCS:
            path = STATUS / 'before_docs' / name
        else:
            path = ROOT / name
        require(sha(path) == expected, 'Previous664 evidence changed: ' + name)
        if path != ROOT / name:
            previous_locations[name] = path.relative_to(ROOT).as_posix()
    reported = read(STATUS / 'status_update_receipt.json')
    for name, expected in reported['bindings_sha256'].items():
        require(sha(ROOT / name) == expected, 'Reported-stop record changed: ' + name)
    app_path = ROOT / 'outputs/dgp_app_v3_integration_record.json'
    app = read(app_path)
    app_bindings = {**app['sources_sha256'], **app['evidence_sha256']}
    for name, expected in app_bindings.items():
        require(sha(ROOT / name) == expected, 'App22 binding changed: ' + name)
    audit_path = ROOT / 'outputs/cctv_dgp_degraded_detail_v24_independent_audit.json'
    audit = read(audit_path)
    require(audit['complete'] and audit['complete_snapshot_updates'] == [0, 50] and
            not audit['early_structure_stop']['pass'] and not audit['necessary_capacity_pass'] and
            audit['counts']['raw_PNG_pairs'] == 100 and audit['cohort_loss_audit']['policy_and_all50_rows_verified'], 'V24 audit closure differs')
    review = read(ROOT / 'outputs/cctv_dgp_degraded_detail_v24_diagnostic/visual_review.json')
    require(review['complete'] and review['cases_reviewed'] == 50 and
            all(not row['convincing_visible_structure_gain'] for row in review['rows']), 'Full review required')
    architecture = read(ROOT / 'outputs/cctv_dgp_post_v24_architecture_review_v1/review.json')
    for name, expected in architecture['source_bindings_sha256'].items():
        require(sha(ROOT / name) == expected, 'Architecture discussion source changed: ' + name)
    require(architecture['architecture_choice_pending'] and not architecture['fourth_recipe_prepared'] and
            architecture['V23_V24_head_AST_exact'], 'Architecture boundary differs')
    decision = read(ROOT / 'outputs/cctv_dgp_post_v24_architecture_review_v1/user_architecture_decision.json')
    require(decision['selected_route'] == 'A' and not decision['architecture_choice_pending'], 'Direct user architecture decision required')
    for name in SCRIPTS:
        ast.parse((ROOT / name).read_text(encoding='utf-8'), feature_version=(3, 10))
    prep = ROOT / 'outputs/cctv_dgp_degraded_detail_v24_return_audit_preparation/preparation_receipt.json'
    preparation = read(prep)
    require(preparation['tests_passed'] == 18, 'Auditor regressions missing')
    for name, expected in preparation['sources_sha256'].items():
        require(sha(ROOT / name) == expected, 'Tested auditor source changed: ' + name)
    OUT.mkdir(); (OUT / 'before_docs').mkdir()
    before = {}
    for name in DOCS:
        shutil.copy2(ROOT / name, OUT / 'before_docs' / name)
        before[name] = sha(OUT / 'before_docs' / name)
    before_handoff = (OUT / 'before_docs/PROJECT_HANDOFF.md').read_bytes()
    require(b'VM storage cleanup completed; future VM use preserved' in before_handoff, 'Concurrent maintenance entry absent')
    write(OUT / 'before_update_plan.json', {'date': '2026-10-06', 'before_docs_sha256': before,
        'previous_milestone_sha256': sha(PREVIOUS), 'previous664_original_locations': previous_locations,
        'reported_status_receipt_sha256': sha(STATUS / 'status_update_receipt.json'),
        'app22_record_sha256': sha(app_path), 'runner_sha256': sha(Path(__file__)),
        'preserve_concurrent_maintenance': True, 'fourth_recipe_prepared': False})
    for name in DOCS:
        require(sha(ROOT / name) == before[name], 'Concurrent document update: ' + name)
        old = (OUT / 'before_docs' / name).read_bytes(); split = old.index(b'\n') + 1
        (ROOT / name).write_bytes(old[:split] + b'\n' + LATEST.encode('utf-8') + b'\n' + old[split:])
        require((ROOT / name).read_bytes().endswith(old[split:]), 'History body changed: ' + name)
    paths = {ROOT / name for name in DOCS + SCRIPTS + [
        'CCTV_DGP_DEGRADED_DETAIL_V24_RESULTS.md', 'CCTV_DGP_POST_V24_ARCHITECTURE_REVIEW.md',
        'outputs/cctv-dgp-degraded-detail-v24-results.tar.gz',
        'outputs/cctv-dgp-degraded-detail-v24-results.tar.gz.sha256',
        'outputs/cctv-dgp-degraded-detail-v24-export.json',
        'outputs/cctv_dgp_degraded_detail_v24_return_import.json',
        'outputs/cctv_dgp_degraded_detail_v24_independent_audit.json']}
    for folder in ['cctv_dgp_degraded_detail_v24_return', 'cctv_dgp_degraded_detail_v24_diagnostic',
                   'cctv_dgp_degraded_detail_v24_loss_audit_v1', 'cctv_dgp_post_v24_architecture_review_v1',
                   'cctv_dgp_degraded_detail_v24_return_audit_preparation']:
        paths.update(path for path in (ROOT / 'outputs' / folder).rglob('*') if path.is_file())
    paths.update(path for path in OUT.rglob('*') if path.is_file())
    bindings = {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(paths)}
    milestone = {'complete': True, 'date': '2026-10-06',
        'scope': 'V24 failure independently audited and reviewed; source/evidence architecture discussion, no fourth model attempt or Goal acceptance',
        'new_evidence_sha256': bindings, 'previous664_evidence_verified': len(previous['new_evidence_sha256']),
        'previous_milestone_sha256': sha(PREVIOUS), 'previous664_original_locations': previous_locations,
        'reported_stop_record_bindings_verified': len(reported['bindings_sha256']),
        'reported_stop_original_doc_locations': {name: (OUT / 'before_docs' / name).relative_to(ROOT).as_posix() for name in DOCS},
        'historical_app22_bindings_verified': len(app_bindings), 'app_record_sha256': sha(app_path),
        'history_bodies_exact': 5, 'concurrent_storage_maintenance_entry_preserved': True,
        'return_members': 371, 'frozen_execution_assets': 221, 'saved_raw_PNG_pairs_replayed': 100,
        'visual_review_cases': 50, 'exact_review_cells': 200, 'auditor_tests_historical_passed': 18,
        'early_gain_percent': 100 * audit['early_structure_stop']['relative_feature_error_gain'], 'early_required_percent': 1,
        'original_early_stop_retained': True, 'necessary_capacity_pass': False,
        'final800_executed': False, 'architecture_choice_pending': False, 'user_selected_architecture': 'A: own-DGP spatial/feature path', 'fourth_recipe_prepared': False,
        'all_facial_features_required_together': True, 'local_backward_calls': 0, 'local_optimizer_updates': 0,
        'local_neural_scope': {'audit_head_forwards': 100, 'diagnostic_exact_head_layer_traces': 50,
            'audit_recognizer_forwards': 110, 'saved_loss_recognizer_forwards': 210, 'DGP_forwards': 0,
            'cohort_fixed_high_pass_calls': 100},
        'VM_cloud_actions': False, 'app_promotion': False, 'native_or_reserved_accessed': False,
        'covering_families_qualified': False, 'independent_final_review': False,
        'goal_status': 'active', 'goal_complete': False, 'seconds': time.monotonic() - started}
    write(OUT / 'milestone.json', milestone)
    print(json.dumps({'complete': True, 'new_bindings': len(bindings), 'previous664_verified': len(previous['new_evidence_sha256']),
        'app22_verified': len(app_bindings), 'history_bodies_preserved': 5,
        'V24_capacity_pass': False, 'fourth_recipe_prepared': False, 'goal_complete': False}, indent=2))


if __name__ == '__main__':
    main()
