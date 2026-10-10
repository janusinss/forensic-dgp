"""Independent syntax, boundaries and saved-array audit; no model or VM calls."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from unittest.mock import patch

os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_vm'
PREP = ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_preparation'
OLD = ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_vm'
MATH = ROOT / 'outputs/cctv_dgp_v37_delivered_margins_v1'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


def main():
    started = time.monotonic()
    import numpy as np
    p = read(BUNDLE / 'protocol.json'); old = read(OLD / 'protocol.json'); prep = read(PREP / 'preparation_receipt.json')
    assert sha(BUNDLE / 'protocol.json') == prep['protocol_sha256']
    archive = ROOT / 'outputs/cctv-dgp-delivered-margin-probe-v38-execution.tar.gz'
    assert archive.stat().st_size == prep['archive_bytes'] and sha(archive) == prep['archive_sha256']
    assert Path(str(archive) + '.sha256').read_text().strip().split() == [prep['archive_sha256'], archive.name]
    expected = {'protocol.json': prep['protocol_sha256'], **p['assets_sha256']}
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers(); assert len(members) == 9
        seen = set()
        for member in members:
            assert member.isfile() and not member.issym() and not member.islnk()
            assert member.name.startswith(BUNDLE.name + '/')
            name = member.name.removeprefix(BUNDLE.name + '/')
            assert name in expected and name not in seen
            seen.add(name); assert hashlib.sha256(tar.extractfile(member).read()).hexdigest() == expected[name]
        assert seen == set(expected)
    for name, digest in p['assets_sha256'].items(): assert sha(BUNDLE / name) == digest, name
    for name, digest in p['local_basis_sha256'].items(): assert sha(ROOT / name) == digest, name
    for key, folder in [('guard_return_sha256', 'cctv_dgp_group_guard_grad_v34_return'),
                        ('PNG_gradient_return_sha256', 'cctv_dgp_delivered_guard_grad_v37_return'),
                        ('V36_margin_basis_sha256', 'cctv_dgp_finite_clearance_probe_v36_return'),
                        ('restoration_gradient_sha256', 'cctv_dgp_v32_loss_gradient_v1_return')]:
        for name, digest in p[key].items(): assert sha(ROOT / 'outputs' / folder / name) == digest, name
    assert len(p['guard_return_sha256']) == 230 and len(p['PNG_gradient_return_sha256']) == 28
    assert len(p['V36_margin_basis_sha256']) == 16 and len(p['restoration_gradient_sha256']) == 2
    for key in ['cohorts', 'parameter_layout', 'terms', 'retained_capacity_gates', 'CPU_replay_case_ids',
                'CPU_replay_outputs', 'forward_call_limits', 'budgets', 'guard_return_sha256', 'guard_metrics',
                'proposal_policy', 'same_VM_raw_tolerance', 'CPU_raw_absolute_tolerance', 'CPU_PNG_byte_tolerance',
                'CPU_vector_absolute_tolerance', 'CPU_component_value_tolerance']:
        assert p[key] == old[key], key
    assert p['constraint_labels'][:108] == old['constraint_labels']
    assert p['clearance_targets'][:108] == old['clearance_targets'] and len(p['constraint_labels']) == 210
    assert p['optimizer_updates'] == p['committed_trajectory_updates'] == p['new_gradient_queries'] == p['backwards'] == p['epochs'] == 0
    assert p['candidate_displacement_trials'] == 4 and p['states'] == [0]
    assert [(v['name'], v['scale']) for v in p['variants']] == [('margin_1', 1.), ('margin_half', .5), ('margin_quarter', .25), ('margin_eighth', .125)]
    assert p['before_outputs'] + p['trial_outputs'] == 500 and p['CPU_replay_outputs'] == 100
    checked_sources = 0
    checker = ROOT / 'scripts/audit_cctv_dgp_delivered_margin_probe_v38_return.py'
    for source in [*BUNDLE.rglob('*.py'), checker]:
        text = source.read_text(encoding='utf-8'); ast.parse(text, feature_version=(3, 10)); checked_sources += 1
        assert not any(value in text for value in ['torch.optim', '.backward(', 'autograd.grad', 'torch.save'])
    worker_path = BUNDLE / 'scripts/cctv_dgp_delivered_margin_probe_v38_vm.py'
    worker = module('V38_scope_only', worker_path)
    assert worker.NAME == BUNDLE.name and worker.STEM == 'cctv-dgp-delivered-margin-probe-v38'
    assert worker.verify(BUNDLE, prep['protocol_sha256']) == p
    rejected = 0
    for platform_name, hostname, folder in [('win32', 'forensic-dgp-thesis', BUNDLE),
        ('linux', 'another-vm', Path.home() / 'forensic-dgp' / BUNDLE.name),
        ('linux', 'forensic-dgp-thesis', Path.home() / 'forensic-dgp' / 'wrong-folder')]:
        with patch.object(worker.sys, 'platform', platform_name), patch.object(worker.platform, 'node', return_value=hostname):
            try: worker.scope(folder.resolve())
            except AssertionError: rejected += 1
            else: raise AssertionError('Wrong host/platform/root must stop')
    with patch.object(worker.sys, 'platform', 'linux'), patch.object(worker.platform, 'node', return_value='forensic-dgp-thesis'):
        worker.scope((Path.home() / 'forensic-dgp' / BUNDLE.name).resolve())
    before = set(path.relative_to(BUNDLE).as_posix() for path in BUNDLE.rglob('*') if path.is_file())
    windows = subprocess.run([sys.executable, '-B', str(worker_path), '--root', str(BUNDLE), '--protocol-sha', prep['protocol_sha256'], '--verify-transfer'],
                             capture_output=True, text=True, timeout=15)
    assert windows.returncode != 0 and 'Existing Linux VM only' in windows.stderr
    assert before == set(path.relative_to(BUNDLE).as_posix() for path in BUNDLE.rglob('*') if path.is_file())
    with (PREP / 'Windows_pre_neural_rejection.txt').open('x', encoding='utf-8') as stream: stream.write(windows.stdout + windows.stderr)
    auditor = module('V38_prospective_archive_boundary', checker)
    assert auditor.BUNDLE == BUNDLE and auditor.OUT.name == 'cctv_dgp_delivered_margin_probe_v38_return'
    prior = module('V33_replay_directory_readback', ROOT / 'scripts/audit_cctv_dgp_loss_cone_probe_v33_return_r2.py')
    assert [auditor.PARENT, auditor.ACTIVE, auditor.V32, auditor.MIXED] == [prior.PARENT, prior.ACTIVE, prior.V32, prior.MIXED]
    for folder in [auditor.PARENT, auditor.ACTIVE, auditor.V32, auditor.MIXED]: assert folder.is_dir()

    def item(name, size=0, kind=tarfile.REGTYPE):
        value = tarfile.TarInfo(name); value.size = size; value.type = kind; return value

    prefix = auditor.PREFIX; auditor.safe_members([item(prefix + 'protocol.json')], p)
    bad = [[item(prefix + '../protocol.json')], [item('/' + prefix + 'protocol.json')],
           [item(prefix + 'protocol.json', kind=tarfile.SYMTYPE)], [item(prefix + 'protocol.json', kind=tarfile.LNKTYPE)],
           [item(prefix + 'protocol.json'), item(prefix + 'protocol.json')], [item(prefix + 'C:protocol.json')],
           [item(prefix + 'unknown.npy')], [item(prefix + 'protocol.json', 8 * 1024**2 + 1)]]
    unsafe = 0
    for members in bad:
        try: auditor.safe_members(members, p)
        except AssertionError: unsafe += 1
        else: raise AssertionError('Unsafe member must stop')
    theta, direction, rows, certificate = auditor.geometry_readback(p)
    assert rows.shape == (210, 978243) and certificate['complete'] and certificate['new_PNG_margin_rows'] == 97
    assert certificate['original65_clearances_unchanged'] and certificate['calibration_observations'] == 408
    assert certificate['magnitude_ratio'] <= 2 and certificate['empirical_clearance_not_a_validated_loss_bound']
    assert np.array_equal(theta, np.load(OLD / 'theta_before.npy', allow_pickle=False))
    assert np.array_equal(direction, np.load(MATH / 'candidate_displacement.npy', allow_pickle=False))
    assert not np.array_equal(direction, np.load(OLD / 'projected_displacement.npy', allow_pickle=False))
    raw = np.load(ROOT / 'outputs/cctv_dgp_group_guard_grad_v34_analysis_v1_r1/group_guard_matrix.npy', allow_pickle=False)
    png = np.load(ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_analysis_v1/PNG_group_guard_matrix.npy', allow_pickle=False)
    assert np.allclose(rows[:102], raw, rtol=2e-10, atol=1e-11) and np.allclose(rows[108:], png, rtol=2e-10, atol=1e-11)
    # Separate independent geometry checker calibrated margins directly from408 receipts.
    independent = read(MATH / 'independent_geometry_audit.json')
    assert independent['complete'] and independent['new97_PNG_margins_recalibrated'] and independent['all65_V36_clearances_unchanged']
    assert independent['all210_rows_and408_calibration_observations_verified']
    tree = ast.parse(worker_path.read_text())
    export = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'export')
    prefixes = [node.value for node in ast.walk(export) if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.endswith('_return/')]
    assert prefixes == [prefix]
    oldtree = ast.parse((OLD / 'scripts/cctv_dgp_finite_clearance_probe_v36_vm.py').read_text())

    def snapshot(tree):
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
        return next(node for node in ast.walk(run) if isinstance(node, ast.FunctionDef) and node.name == 'snapshot')

    assert ast.unparse(snapshot(oldtree)).replace('V36_state', 'V38_state') == ast.unparse(snapshot(tree))
    # Preserve all500 finite metric comparisons and all100 frozen CPU replay paths.
    oldaudit = ast.parse((ROOT / 'scripts/audit_cctv_dgp_finite_clearance_probe_v36_return.py').read_text())
    newaudit = ast.parse(checker.read_text())
    for name in ['finite_metrics', 'CPU_replay', 'safe_members', 'proposals']:
        get = lambda t: next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(get(oldaudit), include_attributes=False) == ast.dump(get(newaudit), include_attributes=False), name
    syntax = read(PREP / 'Bash_readonly_syntax.json')
    assert syntax['exit_code'] == 0 and syntax['script_sha256'] == sha(BUNDLE / 'scripts/run_v38_probe.sh')
    guide = (ROOT / 'CCTV_DGP_DELIVERED_MARGIN_PROBE_V38_VM.md').read_text()
    commands = [line for line in guide.splitlines() if line.startswith('gcloud compute scp ')]
    assert len(commands) == 5 and all(line.count('janusdominic0@forensic-dgp-thesis:') == 1 for line in commands)
    assert prep['protocol_sha256'] in guide and prep['archive_sha256'] in guide
    assert not any(name in sys.modules for name in ['torch', 'dgp_frozen_inference_v2', 'cctv_dgp_pilot'])
    receipt = {'complete': True, 'checker_sha256': sha(Path(__file__)), 'protocol_sha256': prep['protocol_sha256'],
               'archive_sha256': prep['archive_sha256'], 'archive_bytes': prep['archive_bytes'], 'archive_files': 9,
               'V34_saved_bindings_verified': 230, 'V37_saved_bindings_verified': 28, 'V36_calibration_bindings_verified': 16,
               'local_bindings_verified': len(p['local_basis_sha256']), 'all210_projection_rows_verified': True,
               'geometry_certificate': certificate, 'original108_rows_and65_clearances_retained': True,
               'new97_PNG_margins_and408_observations_verified': True, 'direction_differs_from_failed_V36': True,
               'exporter_and_checker_prefix_match': True, 'frozen_snapshot_and_replay_AST_preserved': True,
               'Python310_sources_checked': checked_sources, 'Bash_readonly_syntax_verified': True,
               'scope_rejections': rejected, 'Windows_pre_neural_rejection_passed': True, 'unsafe_member_rejections': unsafe,
               'single_source_gcloud_commands_verified': 5, 'all_quality_gates_retained': True,
               'neural_calls': 0, 'gradient_calls': 0, 'optimizer_updates': 0, 'VM_calls': 0,
               'VM_execution_started': False, 'app_promotion': False, 'goal_complete': False, 'seconds': time.monotonic() - started}
    assert receipt['seconds'] < 300
    with (PREP / 'independent_packet_audit.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key:receipt[key] for key in ['complete', 'archive_files', 'all210_projection_rows_verified', 'seconds']}))


if __name__ == '__main__': main()
