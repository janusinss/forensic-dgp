"""Independent readback of the frozen V22/V23 evidence/docs. No model or VM calls."""
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import time

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'outputs/dgp_detail_prior_v22_r1_audit_and_skip_v23_milestone'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe(name):
    part = PurePosixPath(name)
    assert not part.is_absolute() and '\\' not in name and ':' not in name
    assert part.parts and all(value not in ('', '.', '..') for value in part.parts)
    path = ROOT.joinpath(*part.parts).resolve()
    assert path.is_relative_to(ROOT)
    return path


def main():
    started = time.monotonic()
    milestone_path = EVIDENCE / 'milestone.json'
    milestone = read(milestone_path)
    assert milestone['complete'] and not milestone['goal_complete'] and milestone['goal_status'] == 'active'
    assert milestone['local_optimizer_updates'] == milestone['local_backward_calls'] == milestone['assistant_VM_cloud_actions'] == 0
    assert not any(milestone[k] for k in ('model_changed_in_current_app', 'app_design_source_changed', 'historical_gates_waived',
        'native_usefulness_qualified', 'covering_families_qualified', 'independent_final_review_complete',
        'v23_VM_gradient_training_capacity_quality_verified', 'v22_original_early_gate_pass', 'v22_unexecuted_final_gate_evaluated'))
    assert len(milestone['new_evidence_sha256']) == milestone['new_bindings']
    for name, expected in milestone['new_evidence_sha256'].items():
        assert sha(safe(name)) == expected, name
    previous_path = ROOT / 'outputs/dgp_structure_and_detail_pilot_milestone_v21_v22/milestone.json'
    assert sha(previous_path) == milestone['previous_milestone_sha256']
    previous = read(previous_path)
    assert len(previous['new_evidence_sha256']) == milestone['previous560_bindings_verified'] == 560
    for name, expected in previous['new_evidence_sha256'].items():
        location = milestone['previous_document_locations'].get(name, name)
        assert sha(safe(location)) == expected, name
    app_path = ROOT / 'outputs/dgp_app_v3_integration_record.json'
    assert sha(app_path) == milestone['app_record_sha256']
    app = read(app_path)
    app_bindings = {**app['sources_sha256'], **app['evidence_sha256']}
    assert len(app_bindings) == milestone['app22_bindings_verified'] == 22
    for name, expected in app_bindings.items():
        assert sha(safe(name)) == expected, name
    plan = read(EVIDENCE / 'plan.json')
    for name, expected in plan['before_docs_sha256'].items():
        before = EVIDENCE / 'before_docs' / name
        assert sha(before) == expected
        old = before.read_bytes(); new = (ROOT / name).read_bytes(); split = old.index(b'\n') + 1
        assert new.startswith(old[:split]) and new.endswith(old[split:]), 'Historical document body changed: ' + name
    handoff = (ROOT / 'PROJECT_HANDOFF.md').read_text(encoding='utf-8')
    assert 'Current restoration milestone — 6 October 2026' in handoff
    assert 'Current maintenance — 6 October 2026: cleanup prepared; VM stopped before deletion' in handoff
    bundle = ROOT / 'outputs/cctv_dgp_detail_skip_vm_v23'
    assert sha(bundle / 'protocol.json') == milestone['v23_protocol_sha256']
    protocol = read(bundle / 'protocol.json')
    assert len(protocol['assets_sha256']) == 209 and len(protocol['cases']) == 50
    for name, expected in protocol['assets_sha256'].items():
        path = (bundle / name).resolve()
        assert path.is_relative_to(bundle) and sha(path) == expected
    assert sha(ROOT / 'outputs/cctv-dgp-detail-skip-v23-execution.tar.gz') == milestone['v23_archive_sha256']
    preparation = ROOT / 'outputs/cctv_dgp_detail_skip_v23_preparation'
    check = read(preparation / 'independent_preparation_audit.json')
    padding = read(preparation / 'nonempty_padding_contract_v1.json')
    shell = read(preparation / 'shell_syntax_check.json')
    assert check['complete'] and padding['complete'] and shell['complete'] and shell['exit_code'] == 0
    assert check['new_head_CPU_forwards'] + padding['new_head_CPU_forwards'] == milestone['v23_new_head_CPU_contract_forwards'] == 56
    assert check['closed_v22_CPU_forwards'] == milestone['v23_closed_v22_CPU_sensitivity_forwards'] == 4
    assert padding['padding_pixels'] == 25600 and padding['raw_padding_exact'] and padding['PNG_padding_exact']
    source = bundle / 'scripts/cctv_dgp_detail_skip_v23.py'
    assert sha(source) == shell['source_sha256'] and sha(bundle / 'scripts/run_v23.sh') == shell['shell_sha256']
    text = (ROOT / 'CCTV_DGP_DETAIL_SKIP_V23_VM.md').read_text(encoding='utf-8')
    blocks = re.findall(r'```(?:cmd|bash)\n(.*?)```', text, flags=re.S)
    assert len(blocks) == 5 and all(f'{n}.' in text for n in range(1, 6)), 'Runbook step count differs'
    upload = [line for line in blocks[0].splitlines() if line.startswith('gcloud compute scp')]
    download = [line for line in blocks[4].splitlines() if line.startswith('gcloud compute scp')]
    assert len(upload) == 1 and len(download) == 3
    header = ['gcloud', 'compute', 'scp', '--project=forensic-dgp-thesis', '--zone=us-central1-a']
    dest = 'janusdominic0@forensic-dgp-thesis:/home/janusdominic0/'
    assert shlex.split(upload[0]) == header + ['cctv-dgp-detail-skip-v23-execution.tar.gz', 'cctv-dgp-detail-skip-v23-execution.tar.gz.sha256', dest]
    remote_names = ['cctv-dgp-detail-skip-v23-results.tar.gz', 'cctv-dgp-detail-skip-v23-results.tar.gz.sha256', 'cctv-dgp-detail-skip-v23-export.json']
    for line, name in zip(download, remote_names):
        assert shlex.split(line) == header + [dest + name, '.'], 'Download has multiple remote sources or wrong file'
    pin = milestone['v23_protocol_sha256']
    assert 'tmux new-session -A -s dgp_detail_skip_v23' in blocks[2]
    assert '--protocol-sha ' + pin + ' --preflight' in blocks[3] and 'bash scripts/run_v23.sh ' + pin in blocks[3]
    assert 'sha256sum -c cctv-dgp-detail-skip-v23-execution.tar.gz.sha256' in blocks[1]
    assert 'test ! -e ~/forensic-dgp/cctv_dgp_detail_skip_vm_v23' in blocks[1]
    assert '\r' not in (bundle / 'scripts/run_v23.sh').read_bytes().decode('utf-8')
    for path in (ROOT / 'scripts').glob('*detail*v2[23]*.py'):
        ast.parse(path.read_text(encoding='utf-8'))
    early = read(ROOT / 'outputs/cctv_dgp_detail_prior_v22_r1_return/outputs/early_structure_stop.json')
    assert early['update'] == 50 and not early['pass'] and early['minimum'] == .01
    assert abs(100 * early['relative_feature_error_gain'] - milestone['v22_feature_improvement_percent']) < 1e-12
    receipt = {
        'complete': True, 'date': '2026-10-06', 'checker_sha256': sha(Path(__file__)),
        'milestone_sha256': sha(milestone_path), 'new_bindings_verified': milestone['new_bindings'],
        'previous_bindings_verified': 560, 'app_bindings_verified': 22, 'historical_bodies_exact': 4,
        'existing_maintenance_entry_preserved': True, 'packet_assets_verified': 209,
        'exact_five_steps_and_three_single_remote_downloads': True, 'original_V22_failure_unchanged': True,
        'new_neural_or_backward_or_optimizer_calls': 0, 'VM_actions': False, 'goal_complete': False,
        'seconds': time.monotonic() - started,
    }
    assert receipt['seconds'] <= 120
    with (EVIDENCE / 'independent_milestone_audit.json').open('x', encoding='utf-8', newline='\n') as out:
        out.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
