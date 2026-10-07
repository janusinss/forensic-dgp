"""Preserve history; bind audited corrected derivatives and one finite new pilot."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/dgp_v26_gradient_return_and_feature_skips_v27_milestone'
PREVIOUS = ROOT / 'outputs/dgp_v26_corrected_objective_and_gradient_diagnostic_milestone/milestone.json'
DOCS = ['PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
    'CCTV_DGP_BATCHMATCHED_IDENTITY_V26_RESULTS.md', 'CCTV_DGP_V26_CORRECTED_OBJECTIVE_REVIEW.md',
    'CCTV_DGP_V26_GRADIENT_DIAGNOSTIC_V1_PLAN.md', 'CCTV_DGP_V26_GRADIENT_DIAGNOSTIC_V1_VM.md']
SOURCES = ['scripts/analyze_cctv_dgp_v26_gradient_paths_v1.py', 'scripts/verify_cctv_dgp_v26_gradient_paths_v1.py',
    'scripts/prepare_cctv_dgp_feature_skips_v27.py', 'scripts/install_cctv_dgp_feature_skips_v27_vm.py',
    'scripts/verify_cctv_dgp_feature_skips_v27.py', 'scripts/audit_cctv_dgp_feature_skips_v27_interface.py',
    'scripts/verify_cctv_dgp_feature_skips_v27_interface.py', 'scripts/verify_cctv_dgp_feature_skips_v27_host_guard.py',
    'scripts/prepare_cctv_dgp_feature_skips_v27_return_audit.py', 'scripts/verify_cctv_dgp_feature_skips_v27_return_audit.py',
    'scripts/import_cctv_dgp_feature_skips_v27.py', 'scripts/audit_cctv_dgp_feature_skips_v27.py',
    'scripts/audit_cctv_dgp_feature_skips_v27_execution.py', 'scripts/cctv_dgp_feature_skips_v27_return_rules.py',
    'tests/test_cctv_dgp_feature_skips_v27.py', 'scripts/record_cctv_dgp_v26_gradient_return_v27_milestone.py',
    'scripts/verify_cctv_dgp_v26_gradient_return_v27_milestone.py']
LATEST = '''**Latest research milestone — 6 October 2026: corrected V26 gradient return audited; V27 direct own-feature pilot independently prepared.**

The 1,665,230-byte diagnostic return matches SHA256
`bb07f90ef10496a8b464d61d9e446826e4f8d824c12478f35765b3d3b160c201`.
Ten regular files, 240 original assets, 118 saved inputs, 250 frozen own-DGP feature
arrays and two 7×53,781 gradient matrices pass independent source/state/statistic/
batch/count/timing readback. All 752,934 saved gradient values are checked. The L4
measurement takes 19.089s, with 140 gradient queries and zero optimizer updates.
Its snapshot50 replays the old stopped V26 head; no new training occurs. Initial
identity value/all26 gradient tensors are exactly zero in all ten batches.
Preservation/improvement norm ratio at stopped50 is 0.3701499, cosine −0.7547647;
the improvement gradient remains nonzero. No preservation term or margin changes.

Independent pure-array path arithmetic verifies 269 sources/84 term-family rows
and 250 input-feature summaries. Direct RGB carries 98.4251552% of stopped50 landmark
squared-gradient magnitude. This depends on parameterization and does not prove
full optimizer trajectory causality. V26 remains a closed structure failure:
0.0335635587% delivered gain versus the unchanged1% early stop, with no convincing
whole-face gain in50 TRAIN cases. Do not rerun V26 or the completed diagnostic.

Within the user's selected own-DGP spatial direction, V27 adds five independent
normalized zero-initialized 1×1 RGB readouts from frozen own-DGP feature levels.
They shorten the feature path into the same bounded residual, retaining every
original decoder/RGB tensor, frozen DGP/recognizer, corrected seven-term loss,
seed,50 exposed TRAIN cases and original schedule. Add1743 parameters for55524 in
36 groups. All28 original VM state tensors must match V26 initialization exactly.
No fitted statistics, clean target, person/source label or pretrained restoration
output conditions inference. All visible facial features remain in scope together.

The 51,049-byte/seven-file thin packet reuses240 original VM assets with hash-bound
hardlinks and uploads no data or weights. Original files are never modified; require
3GiB free and preserve partial work. Installation60s; neural preflight300s; matched
identity proof180s; fit1500s; worker1800s; external2100s plus30s grace; export120s/
external150s plus30s grace; peak allocated VRAM20GiB. Finite800 updates/80 epochs.
Original update20 timing projection,1% early/10% final structure, all17 group pixel/
SSIM/identity bounds, source gains, brightness limit and final800 selection remain.
All36 initial identity gradients must be exactly zero; all five new skip weight
gradients must be positive before an optimizer. The original five projection
gradients must remain positive at update2. No automatic follow-on or unchanged retry.
If this third spatial-path capacity attempt fails, stop model modifications for
an architecture discussion before another attempt.

Full AST comparisons retain original head/trainer/guard/supervisor and prospective
safe importer/CPU/group/capacity/state/feature/cohort/timing audits except declared
new connections, counts, names, initial proofs and hardlink receipt. Python3.10
parsing,14 meaningful source/archive/receipt regressions and actual Windows transfer/
neural-install rejection pass. All247 bundle files remain unchanged by guard checks.
The shell is byte-equivalent to the original after restoring its worker filename.

Local inference verifies exact initial raw/PNG parity for50 cases and all28 shared
CPU tensors. Five predetermined, unfitted forward controls prove wiring without
clean targets, fitting or derivative estimates; independent NumPy/OpenCV arithmetic
checks their responses. Partial support preserves outside pixels; empty support
is rejected. There are61 valid CPU head forwards, zero DGP/recognizer forwards and
zero local derivatives/backwards/optimizer updates in the new interface audit.
No probe checkpoint/image is created. Actual L4 proof and trained capacity remain
pending. Five pasteable gcloud/SSH/tmux steps include separate PuTTY downloads.
The human performs every VM/cloud action; none is performed by this agent.

Previous66 bindings and deeper692/299/697/513 histories, seven complete document
bodies, app22 bindings and concurrent completed VM maintenance are preserved.
No candidate is promoted and the existing app design/checkpoint is unchanged.
Native CCTV stays unpaired; paired photographic TRAIN metrics remain separate.
Reserved-final identities remain unviewed. No new native/covering pixels, ethnicity
or Zamboanga performance claim enters this milestone. Previously useful native
inputs remain usable despite failed model corrections. Useful native development
output, candidate app parity/full flow, input-only insufficient-information handling,
all seven automatic/assisted covering families and independent final review remain
required. **Goal active/incomplete.**

[Verified diagnostic](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V26_GRADIENT_DIAGNOSTIC_V1_RESULTS.md>) ·
[V27 frozen design](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_FEATURE_SKIPS_V27_PLAN.md>) ·
[Five manual VM steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_FEATURE_SKIPS_V27_VM.md>) ·
[Milestone](<C:/xampp/htdocs/YEAR 4/Testing/outputs/dgp_v26_gradient_return_and_feature_skips_v27_milestone/milestone.json>)

Previous bodies below are preserved history. Their diagnostic-pending statements
and V22–V26/diagnostic commands are historical. Launch only the new V27 protocol
once under its current manual runbook; all earlier failed/completed runs stay closed.
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
    return len(bindings)


def history(current, locations=None):
    assert sha(PREVIOUS) == 'fab3256737334bed2eeb984df4d6deb545654e05dada1d127bbe8716b4382103'
    counts = [verify(current['new_evidence_sha256'], locations)]
    paths = [
        ('outputs/dgp_batchmatched_identity_v26_audit_milestone/milestone.json', 'previous692_original_locations'),
        ('outputs/dgp_v25_gradient_audit_and_batchmatched_identity_v26_milestone/milestone.json', 'previous299_original_locations'),
        ('outputs/dgp_spatial_features_v25_audit_and_gradient_diagnostic_milestone/milestone.json', 'previous697_original_locations'),
        ('outputs/dgp_spatial_features_v25_preparation_milestone/milestone.json', 'previous513_original_locations'),
    ]
    for name, mapping in paths:
        path = ROOT / name
        assert sha(path) == current['previous_milestone_sha256']
        prior = read(path)
        counts.append(verify(prior['new_evidence_sha256'], current[mapping]))
        current = prior
    assert counts == [66, 692, 299, 697, 513]
    return counts


def main():
    started = time.monotonic(); assert not OUT.exists()
    previous = read(PREVIOUS); counts = history(previous)
    returned = ROOT / 'outputs/cctv_dgp_v26_gradient_diagnostic_v1_independent_audit.json'
    audit = read(returned)
    assert audit['complete'] and audit['corrected_initial_identity_exact_zero_verified']
    assert audit['VM_gradient_calls_verified'] == 140 and audit['VM_optimizer_updates_verified'] == 0
    gradient_folder = ROOT / 'outputs/cctv_dgp_v26_gradient_paths_v1'
    analysis, arithmetic = read(gradient_folder / 'analysis.json'), read(gradient_folder / 'independent_readback.json')
    assert arithmetic['complete'] and arithmetic['analysis_sha256'] == sha(gradient_folder / 'analysis.json')
    verify(analysis['source_bindings_sha256'])
    prep = ROOT / 'outputs/cctv_dgp_feature_skips_v27_preparation'
    receipts = {name: read(prep / name) for name in ['preparation.json', 'independent_packet_audit.json',
        'interface_inference_audit.json', 'independent_interface_readback.json', 'windows_execution_guard.json',
        'return_audit_preparation.json', 'independent_return_audit_source_check.json']}
    assert all(value['complete'] for value in receipts.values())
    assert all(value['protocol_sha256'] == '22f76701e317b38d945838a6767239c5636cff534b61ddfbf248cc42ba08a82e'
        for key, value in receipts.items() if 'protocol_sha256' in value)
    verify(receipts['interface_inference_audit.json']['source_bindings_sha256'])
    assert receipts['interface_inference_audit.json']['head_forwards'] == 61
    assert receipts['independent_interface_readback.json']['interface_audit_sha256'] == sha(prep / 'interface_inference_audit.json')
    assert receipts['windows_execution_guard.json']['bundle_files_unchanged'] == 247
    packet = receipts['preparation.json']
    assert packet['archive_bytes'] == 51049 and packet['finite_updates'] == 800 and packet['finite_epochs'] == 80
    bundle = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'; protocol = read(bundle / 'protocol.json')
    verify({(bundle / name).relative_to(ROOT).as_posix(): digest for name, digest in protocol['assets_sha256'].items()})
    verify(protocol['local_basis_sha256'])
    apppath = ROOT / 'outputs/dgp_app_v3_integration_record.json'
    assert sha(apppath) == previous['app_record_sha256']
    app = read(apppath); assert verify({**app['sources_sha256'], **app['evidence_sha256']}) == 22
    for name in SOURCES:
        ast.parse((ROOT / name).read_text(encoding='utf-8'), feature_version=(3, 10))
    for name in DOCS:
        assert sha(ROOT / name) == previous['new_evidence_sha256'][name], name
    OUT.mkdir(); (OUT / 'before_docs').mkdir(); before = {}; locations = {}
    for name in DOCS:
        path = OUT / 'before_docs' / name
        shutil.copy2(ROOT / name, path)
        before[name] = sha(path); assert before[name] == previous['new_evidence_sha256'][name]
        locations[name] = path.relative_to(ROOT).as_posix()
    assert b'VM storage cleanup completed; future VM use preserved' in (OUT / 'before_docs/PROJECT_HANDOFF.md').read_bytes()
    write(OUT / 'before_update_plan.json', {'complete': True, 'date': '2026-10-06',
        'before_docs_sha256': before, 'previous66_original_locations': locations,
        'previous_milestone_sha256': sha(PREVIOUS), 'historical_bodies_will_remain_exact': True,
        'actual_V27_L4_pilot_pending': True, 'runner_sha256': sha(Path(__file__))})
    for name in DOCS:
        assert sha(ROOT / name) == before[name], 'Concurrent update: ' + name
        original = (OUT / 'before_docs' / name).read_bytes(); split = original.index(b'\n') + 1
        (ROOT / name).write_bytes(original[:split] + b'\n' + LATEST.encode('utf-8') + b'\n' + original[split:])
        assert (ROOT / name).read_bytes().endswith(original[split:])
    files = DOCS + SOURCES + ['CCTV_DGP_V26_GRADIENT_DIAGNOSTIC_V1_RESULTS.md',
        'CCTV_DGP_FEATURE_SKIPS_V27_PLAN.md', 'CCTV_DGP_FEATURE_SKIPS_V27_VM.md',
        'outputs/cctv-dgp-v26-gradient-diagnostic-v1-results.tar.gz',
        'outputs/cctv-dgp-v26-gradient-diagnostic-v1-results.tar.gz.sha256',
        'outputs/cctv-dgp-v26-gradient-diagnostic-v1-export.json',
        'outputs/cctv_dgp_v26_gradient_diagnostic_v1_return_import.json',
        'outputs/cctv_dgp_v26_gradient_diagnostic_v1_independent_audit.json',
        'outputs/cctv-dgp-feature-skips-v27-execution.tar.gz',
        'outputs/cctv-dgp-feature-skips-v27-execution.tar.gz.sha256']
    paths = {ROOT / name for name in files}
    for folder in [ROOT / 'outputs/cctv_dgp_v26_gradient_diagnostic_v1_return', gradient_folder, bundle, prep, OUT]:
        paths.update(path for path in folder.rglob('*') if path.is_file())
    bindings = {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(paths)}
    record = {'complete': True, 'date': '2026-10-06',
        'scope': 'Audited corrected fixed-state gradients and independently prepared single-feature-path finite V27; full Goal incomplete',
        'new_evidence_sha256': bindings, 'previous_milestone_sha256': sha(PREVIOUS),
        'previous66_original_locations': locations, 'preserved_history_binding_counts': counts,
        'app_record_sha256': sha(apppath), 'app22_bindings_preserved': True,
        'corrected_gradient_return_sha256': audit['archive_sha256'], 'corrected_gradient_return_bytes': 1665230,
        'corrected_initial_identity_exact_zero': True, 'saved_gradient_values_verified': 752934,
        'actual_VM_diagnostic_gradient_queries': 140, 'actual_VM_diagnostic_optimizer_updates': 0,
        'entire_optimizer_trajectory_cause_proven': False, 'V26_original_quality_failure_retained': True,
        'new_protocol_sha256': packet['protocol_sha256'], 'new_archive_sha256': packet['archive_sha256'],
        'new_archive_bytes': 51049, 'new_archive_regular_files': 7,
        'new_parameters': 1743, 'total_trainable_parameters': 55524, 'new_parameter_groups': 36,
        'original_objective_optimizer_data_schedule_scientific_and_timing_stops_retained': True,
        'initial_parity_cases': 50, 'local_valid_head_inference_forwards': 61,
        'local_DGP_recognizer_forwards': 0, 'local_gradient_calls': 0, 'local_backward_calls': 0, 'local_optimizer_updates': 0,
        'nonfitted_forward_wiring_probes': 5, 'no_probe_checkpoint_or_images': True,
        'new_tests_passed': 14, 'new_tests_seconds': .273, 'Windows_neural_and_install_guards_passed': True,
        'shell_same_as_original_except_worker_filename': True, 'prospective_full_return_audit_verified': True,
        'human_manual_VM_execution_required': True, 'actual_V27_L4_pilot_pending': True,
        'VM_actions': False, 'app_promotion': False, 'native_or_reserved_or_new_covering_pixels_used': False,
        'independent_final_review_complete': False, 'concurrent_completed_maintenance_preserved': True,
        'goal_status': 'active', 'goal_complete': False, 'seconds': time.monotonic() - started}
    write(OUT / 'milestone.json', record)
    print(json.dumps({key: value for key, value in record.items() if key not in ['new_evidence_sha256', 'previous66_original_locations']}, indent=2))
    print('Bound files:', len(bindings))


if __name__ == '__main__':
    main()
