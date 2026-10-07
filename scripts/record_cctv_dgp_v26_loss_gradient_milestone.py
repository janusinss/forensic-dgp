"""Bind verified V26 scalar evidence and a finite manual zero-update measurement."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/dgp_v26_corrected_objective_and_gradient_diagnostic_milestone'
PREVIOUS = ROOT / 'outputs/dgp_batchmatched_identity_v26_audit_milestone/milestone.json'
DOCS = ['PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
        'CCTV_DGP_BATCHMATCHED_IDENTITY_V26_RESULTS.md']
NEW_DOCS = ['CCTV_DGP_V26_CORRECTED_OBJECTIVE_REVIEW.md',
            'CCTV_DGP_V26_GRADIENT_DIAGNOSTIC_V1_PLAN.md', 'CCTV_DGP_V26_GRADIENT_DIAGNOSTIC_V1_VM.md']
SOURCES = ['scripts/audit_cctv_dgp_batchmatched_identity_v26_loss_v1.py',
           'scripts/verify_cctv_dgp_batchmatched_identity_v26_loss_v1.py',
           'scripts/prepare_cctv_dgp_v26_gradient_diagnostic_v1.py',
           'scripts/cctv_dgp_v26_gradient_diagnostic_v1_vm.py',
           'scripts/verify_cctv_dgp_v26_gradient_diagnostic_v1.py',
           'scripts/prepare_cctv_dgp_v26_gradient_diagnostic_v1_return_audit.py',
           'scripts/import_cctv_dgp_v26_gradient_diagnostic_v1.py',
           'scripts/audit_cctv_dgp_v26_gradient_diagnostic_v1_return.py',
           'tests/test_cctv_dgp_v26_gradient_diagnostic_v1.py',
           'scripts/record_cctv_dgp_v26_loss_gradient_milestone.py',
           'scripts/verify_cctv_dgp_v26_loss_gradient_milestone.py']
LATEST = '''**Latest research milestone — 6 October 2026: corrected V26 scalar objective audited; finite zero-update L4 gradient measurement verified.**

The unchanged corrected V26 objective decreases from 1.2999999568 to 1.2997388024
between saved states 0/50. Degraded structure/pixel terms improve slightly; clear
controls contribute a small preservation cost. Two degraded cases incur SSIM
costs and one incurs an identity cost at stopped 50. Scalar penalty sizes do not
establish their active gradient strength or the optimizer trajectory's cause.
Do not remove preservation terms, relax gates or invent a new recipe from these
scalar observations. The original V26 failure remains closed: 0.0335635587% versus
the unchanged 1% early requirement, with no convincing whole-face gain in 50 cases.

The 54.273-second local diagnostic uses 30 frozen-recognizer CPU forwards and 20
saved prediction replays, zero head/DGP forwards and zero local derivatives or
optimization. The actual corrected loss/metric function ASTs and five-case batch
contexts are retained. Independent arithmetic checks 280 source bindings, 30 vector
arrays, 100 case states and 34 group states. All seven term assemblies agree exactly.
This is saved-scalar/source evidence, not a GPU derivative or SSIM-field rerun.

The next packet measures actual corrected V26 component gradients at the two
saved heads: 20 head batches, 140 gradient queries, 70 recognizer forwards, zero
DGP forwards and zero optimizer updates/epochs. Its 16,047-byte transfer has four
regular files and uploads no data or weights. Original 240 assets, 118 saved-input
bindings, the prior identity proof and 250 frozen feature arrays are checked.
Every initial batch must retain exactly zero identity value/all 26 gradients.
Worker 420s; supervisor 480s plus 30s grace; export 30s internally/60s externally
plus 10s grace. Require an idle existing L4 and 1 GiB free disk; preserve any stop.
No new checkpoint, unchanged training retry or automatic follow-on is allowed.

Eleven source/archive/matrix tamper checks, Python 3.10 parsing, Windows transfer/
neural-gradient rejection and Bash syntax pass. Full diagnostic-worker AST and
supervisor behavior retain the original measurement except declared objective,
proof, counter and name changes. Prospective safe import/full matrix audit preserve
original guards and require exact-zero corrected identity proof. Actual L4
measurement and independent returned-matrix audit remain pending; no VM/cloud
action occurs. Manual gcloud transfers and tmux launch are in the new five-step
runbook, with three separate PuTTY download calls. No new training recipe exists.

Original V26 and earlier failures/checkpoints/splits, previous 692 bindings and
deeper 299/697/513 evidence, four full document bodies, app 22 bindings and concurrent
completed VM maintenance remain preserved. The user-selected own-DGP spatial path
and all visible facial features together remain required. No app candidate is
promoted. Native CCTV stays unpaired; paired photographic TRAIN evidence stays
separate. No native/reserved-final/new covering pixels or ethnicity/Zamboanga
performance claims enter this milestone. Previously useful inputs remain usable
despite model softness. Useful native output, candidate app parity/full flow,
input-only insufficient-information handling, all seven automatic/assisted
covering families and independent final review remain required. Goal active/incomplete.

[Corrected objective evidence](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V26_CORRECTED_OBJECTIVE_REVIEW.md>) ·
[Finite measurement design](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V26_GRADIENT_DIAGNOSTIC_V1_PLAN.md>) ·
[Five manual VM steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V26_GRADIENT_DIAGNOSTIC_V1_VM.md>) ·
[Milestone](<C:/xampp/htdocs/YEAR 4/Testing/outputs/dgp_v26_corrected_objective_and_gradient_diagnostic_milestone/milestone.json>)

Previous bodies below are preserved history. V26 is a closed training failure;
the new commands measure saved states only. Historical training/diagnostic commands
are not instructions to repeat immutable failed recipes or earlier measurements.
'''


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify(bindings, locations=None):
    for name, digest in bindings.items():
        path = (ROOT / (locations or {}).get(name, name)).resolve()
        assert path.is_relative_to(ROOT) and path.is_file() and sha(path) == digest, name


def main():
    started = time.monotonic()
    assert not OUT.exists(), 'Preserve previous milestone'
    assert sha(PREVIOUS) == '4917d4f2a3e03d9b9da2db737b1329973ece7c418717d1731247d6c98b1542de'
    previous = read(PREVIOUS)
    assert len(previous['new_evidence_sha256']) == 692
    verify(previous['new_evidence_sha256'])
    prior_path = ROOT / 'outputs/dgp_v25_gradient_audit_and_batchmatched_identity_v26_milestone/milestone.json'
    assert sha(prior_path) == previous['previous_milestone_sha256']
    prior = read(prior_path)
    verify(prior['new_evidence_sha256'], previous['previous299_original_locations'])
    old_path = ROOT / 'outputs/dgp_spatial_features_v25_audit_and_gradient_diagnostic_milestone/milestone.json'
    assert sha(old_path) == prior['previous_milestone_sha256']
    old = read(old_path)
    verify(old['new_evidence_sha256'], prior['previous697_original_locations'])
    deep_path = ROOT / 'outputs/dgp_spatial_features_v25_preparation_milestone/milestone.json'
    assert sha(deep_path) == old['previous_milestone_sha256']
    verify(read(deep_path)['new_evidence_sha256'], old['previous513_original_locations'])
    lossfolder = ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_loss_audit_v1'
    loss, losscheck = read(lossfolder / 'results.json'), read(lossfolder / 'independent_saved_loss_audit.json')
    assert loss['complete'] and losscheck['complete']
    assert losscheck['results_sha256'] == sha(lossfolder / 'results.json')
    assert loss['counts'] == {'recognizer_forwards': 30, 'saved_prediction_replays': 20}
    assert losscheck['case_states_verified'] == 100 and losscheck['group_states_verified'] == 34
    verify(loss['source_bindings_sha256'])
    prep = ROOT / 'outputs/cctv_dgp_v26_gradient_diagnostic_v1_preparation'
    packet, checked = read(prep / 'preparation.json'), read(prep / 'independent_packet_and_return_audit.json')
    assert packet['complete'] and checked['complete']
    assert checked['checker_sha256'] == sha(ROOT / 'scripts/verify_cctv_dgp_v26_gradient_diagnostic_v1.py')
    assert packet['archive_bytes'] == checked['archive_bytes'] == 16047
    assert packet['protocol_sha256'] == checked['protocol_sha256']
    assert checked['original_safe_import_and_full_matrix_audit_retained_except_declared_reference_and_name_changes']
    new = ROOT / 'outputs/cctv_dgp_v26_gradient_diagnostic_v1_vm'
    protocol = read(new / 'protocol.json')
    assert sha(new / 'protocol.json') == packet['protocol_sha256']
    verify(protocol['local_basis_sha256'])
    assert protocol['component_gradient_calls'] == 140 and protocol['recognizer_forwards'] == 70
    assert protocol['optimizer_updates'] == 0 and not protocol['new_training_recipe']
    apppath = ROOT / 'outputs/dgp_app_v3_integration_record.json'
    assert sha(apppath) == previous['app_record_sha256']
    app = read(apppath)
    verify({**app['sources_sha256'], **app['evidence_sha256']})
    for name in SOURCES:
        ast.parse((ROOT / name).read_text(encoding='utf-8'), feature_version=(3, 10))
    for name in DOCS:
        assert sha(ROOT / name) == previous['new_evidence_sha256'][name], name
    OUT.mkdir()
    (OUT / 'before_docs').mkdir()
    before = {}
    for name in DOCS:
        shutil.copy2(ROOT / name, OUT / 'before_docs' / name)
        before[name] = sha(OUT / 'before_docs' / name)
        assert before[name] == previous['new_evidence_sha256'][name]
    locations = {name: (OUT / 'before_docs' / name).relative_to(ROOT).as_posix() for name in DOCS}
    assert b'VM storage cleanup completed; future VM use preserved' in (OUT / 'before_docs/PROJECT_HANDOFF.md').read_bytes()
    write(OUT / 'before_update_plan.json', {'date': '2026-10-06', 'before_docs_sha256': before,
        'previous_milestone_sha256': sha(PREVIOUS), 'previous692_original_locations': locations,
        'original_historical_bodies_preserved': True, 'no_new_training_recipe': True,
        'actual_L4_diagnostic_pending': True, 'runner_sha256': sha(Path(__file__))})
    for name in DOCS:
        assert sha(ROOT / name) == before[name], 'Concurrent document update: ' + name
        original = (OUT / 'before_docs' / name).read_bytes()
        split = original.index(b'\n') + 1
        (ROOT / name).write_bytes(original[:split] + b'\n' + LATEST.encode('utf-8') + b'\n' + original[split:])
        assert (ROOT / name).read_bytes().endswith(original[split:])
    files = DOCS + NEW_DOCS + SOURCES + [
        'outputs/cctv-dgp-v26-gradient-diagnostic-v1-execution.tar.gz',
        'outputs/cctv-dgp-v26-gradient-diagnostic-v1-execution.tar.gz.sha256']
    paths = {ROOT / name for name in files}
    for folder in [lossfolder, prep, new, OUT]:
        paths.update(p for p in folder.rglob('*') if p.is_file())
    bindings = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}
    record = {'complete': True, 'date': '2026-10-06',
        'scope': 'Verified corrected scalar loss and finite zero-update manual L4 diagnostic; full Goal incomplete',
        'new_evidence_sha256': bindings, 'previous692_original_locations': locations,
        'previous_milestone_sha256': sha(PREVIOUS), 'app_record_sha256': sha(apppath),
        'corrected_loss_start': loss['groups']['all']['states']['0']['objective'],
        'corrected_loss_stopped50': loss['groups']['all']['states']['50']['objective'],
        'active_penalty_cases_stopped50': loss['groups']['all']['states']['50']['active_penalty_cases'],
        'local_recognizer_inference_calls': 30, 'local_saved_prediction_replays': 20,
        'local_head_DGP_forwards': 0, 'local_gradient_calls': 0, 'local_backward_calls': 0, 'local_optimizer_updates': 0,
        'V26_original_failure_retained': True, 'corrected_optimizer_trajectory_cause_proven': False,
        'diagnostic_protocol_sha256': packet['protocol_sha256'], 'diagnostic_archive_sha256': packet['archive_sha256'],
        'diagnostic_archive_bytes': 16047, 'diagnostic_regular_members': 4,
        'prospective_VM_head_batches': 20, 'prospective_VM_gradient_queries': 140,
        'prospective_VM_recognizer_forwards': 70, 'prospective_VM_optimizer_updates': 0,
        'exact_zero_initial_identity_proof_required': True, 'new_training_recipe': False,
        'actual_L4_diagnostic_pending': True, 'independent_return_audit_pending': True,
        'new_tests_passed': 11, 'new_tests_seconds': .458, 'Bash_n_exit_code': 0,
        'Bash_signal_pipe_sandbox_failure_retained': True,
        'Bash_readonly_unsandboxed_check_passed': True,
        'Python310_and_actual_Windows_transfer_execution_guards_passed': True,
        'all_original_quality_stops_preserved': True, 'no_automatic_follow_on': True,
        'VM_actions': False, 'app_unchanged': True, 'app_promotion': False, 'native_or_reserved_used': False,
        'independent_final_review_complete': False, 'goal_status': 'active', 'goal_complete': False,
        'seconds': time.monotonic() - started}
    write(OUT / 'milestone.json', record)
    print(json.dumps({k: v for k, v in record.items() if k not in ['new_evidence_sha256', 'previous692_original_locations']}, indent=2))
    print('Bound files:', len(bindings))


if __name__ == '__main__':
    main()
