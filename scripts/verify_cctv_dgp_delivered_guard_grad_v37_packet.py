"""Independent V37 packet/boundary review, with zero model/gradient/VM calls."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_vm'
PREP = ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_preparation'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


def reject(call):
    try: call()
    except AssertionError: return
    raise AssertionError('Expected boundary rejection')


def main():
    started = time.monotonic(); p = read(BUNDLE / 'protocol.json'); pin = sha(BUNDLE / 'protocol.json')
    prep = read(PREP / 'preparation_receipt.json')
    assert prep['complete'] and prep['protocol_sha256'] == pin and prep['packet_files'] == 4
    checker_path = ROOT / 'scripts/audit_cctv_dgp_delivered_guard_grad_v37_return.py'
    checker = module('prospective_V37_hash_and_boundary_only', checker_path)
    basis = checker.verify_basis(p)
    old = read(ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_vm/protocol.json')
    assert p['retained_capacity_gates'] == old['retained_capacity_gates']
    assert p['cohorts'] == basis['cohorts'] and p['parameter_layout'] == basis['parameter_layout']
    assert sum(len(c['cases']) for c in p['cohorts']) == 100 and all(r['role'] == 'train' for c in p['cohorts'] for r in c['cases'])
    assert p['gradient_queries'] == 300 and p['optimizer_updates'] == p['parameter_updates'] == p['epochs'] == 0
    assert p['states'] == [0] and p['original_batch_size'] == p['delivered_identity_batch_size'] == 5
    assert not p['true_PNG_derivative_claimed'] and p['custom_coarse_backward_inside_autograd_grad']
    assert p['budgets']['worker_seconds'] == 900 and p['budgets']['external_seconds'] == 930
    assert p['budgets']['minimum_free_disk_bytes'] == 6 * 1024 ** 3 and p['budgets']['local_audit_seconds'] == 1200
    assert p['same_VM_PNG_byte_tolerance'] == 0 and p['forward_metric_tolerances'] == [1e-12, 1e-7, 1e-6]
    assert p['CPU_replay_case_ids'] == old['CPU_replay_case_ids']
    assert sum(len(v) for v in p['CPU_replay_case_ids'].values()) == 20
    assert p['assets_sha256']['scripts/cctv_dgp_delivered_png_guard_v37.py'] == sha(ROOT / 'scripts/cctv_dgp_delivered_png_guard_v37.py')
    files = {q.relative_to(BUNDLE).as_posix() for q in BUNDLE.rglob('*') if q.is_file()}
    assert files == set(p['assets_sha256']) | {'protocol.json'} and len(files) == 4
    archive = ROOT / 'outputs/cctv-dgp-delivered-guard-grad-v37-execution.tar.gz'
    assert sha(archive) == prep['archive_sha256'] and archive.stat().st_size == prep['archive_bytes']
    assert Path(str(archive) + '.sha256').read_text().strip().split() == [prep['archive_sha256'], archive.name]
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers(); assert len(members) == 4
        seen = set()
        for item in members:
            assert item.isfile() and not item.issym() and not item.islnk() and item.name.startswith(BUNDLE.name + '/')
            name = item.name[len(BUNDLE.name) + 1:]; assert name in files and name not in seen; seen.add(name)
            assert hashlib.sha256(tar.extractfile(item).read()).hexdigest() == sha(BUNDLE / name)
    worker_path = BUNDLE / 'scripts/cctv_dgp_delivered_guard_grad_v37_vm.py'
    helper_path = BUNDLE / 'scripts/cctv_dgp_delivered_png_guard_v37.py'
    sources = [worker_path, helper_path, checker_path, Path(__file__), ROOT / 'scripts/prepare_cctv_dgp_delivered_guard_grad_v37.py']
    gradient_call_sites = 0
    for path in sources:
        tree = ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 10))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in ['backward', 'AdamW', 'Adam', 'SGD', 'step'], (path.name, node.func.attr)
                if node.func.attr == 'grad':
                    assert path == worker_path and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == 'autograd'
                    gradient_call_sites += 1
                assert not (node.func.attr == 'save' and isinstance(node.func.value, ast.Name) and node.func.value.id == 'torch')
    assert gradient_call_sites == 1
    worker = module('V37_worker_asset_verification_only', worker_path)
    assert worker.verify(BUNDLE, pin) == p
    reject(lambda: worker.verify(BUNDLE, '0' * 64))
    rejected = subprocess.run([sys.executable, '-B', str(worker_path), '--root', str(BUNDLE), '--protocol-sha', pin,
                               '--verify-transfer'], capture_output=True, text=True, timeout=30)
    assert rejected.returncode != 0 and 'Existing Linux VM only; no local gradients' in rejected.stderr
    assert not (BUNDLE / 'outputs').exists()
    with (PREP / 'Windows_gradient_guard.txt').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(rejected.stdout + rejected.stderr)
    tree = ast.parse(worker_path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'scope')
    ns = {'sys': SimpleNamespace(platform='linux'), 'platform': SimpleNamespace(node=lambda: 'forensic-dgp-thesis'),
          'Path': Path, 'NAME': BUNDLE.name}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<isolated-V37-root-guard>', 'exec'), ns)
    expected = (Path.home() / 'forensic-dgp' / BUNDLE.name).resolve(); ns['scope'](expected)
    reject(lambda: ns['scope'](expected.parent / 'cctv_dgp_group_guard_grad_v34_vm'))
    reject(lambda: ns['scope'](expected.parent / (BUNDLE.name + '_r1')))
    ns['platform'] = SimpleNamespace(node=lambda: 'different-vm'); reject(lambda: ns['scope'](expected))
    ns['platform'] = SimpleNamespace(node=lambda: 'forensic-dgp-thesis'); ns['sys'] = SimpleNamespace(platform='win32')
    reject(lambda: ns['scope'](expected))
    # Inspect the surrogate's VM/CUDA assertions without calling autograd locally.
    ht = ast.parse(helper_path.read_text())
    backward = next(n for n in ast.walk(ht) if isinstance(n, ast.FunctionDef) and n.name == 'backward')
    assert sum(isinstance(n, ast.Assert) for n in backward.body) == 2
    assert 'gradient.is_cuda' in ast.unparse(backward) and "'forensic-dgp-thesis'" in ast.unparse(backward)
    assert 'gradient * mask' in ast.unparse(backward)
    member = tarfile.TarInfo(checker.PREFIX + 'protocol.json'); member.size = 1
    assert checker.safe_members([member], p)[1] == 1
    attacks = []
    for name in ['../outside', 'unapproved.pth', 'outputs\\outside', 'C:/outside']:
        bad = copy.copy(member); bad.name = checker.PREFIX + name; attacks.append([bad])
    for kind in [tarfile.SYMTYPE, tarfile.LNKTYPE]:
        bad = copy.copy(member); bad.type = kind; bad.linkname = '/outside'; attacks.append([bad])
    bad = copy.copy(member); bad.size = 65 * 1024 ** 2; attacks.append([bad]); attacks.append([member, member])
    for attack in attacks: reject(lambda attack=attack: checker.safe_members(attack, p))
    import numpy as np
    group_tests = 0
    for cohort in p['cohorts']:
        cases = cohort['cases']; groups = checker.group_indices(cases)
        sources = sorted({c['source'] for c in cases}); g = np.zeros((50, 3, 2), np.float64)
        for index, case in enumerate(cases):
            if case['profile'] == 'blur_lr24': g[index, 2, 0] = 1. if case['source'] == sources[0] else -1.
        assert g.mean(0)[2, 0] == 0
        assert g[groups[sources[0] + '/blur_lr24']].mean(0)[2, 0] == 1
        assert g[groups[sources[1] + '/blur_lr24']].mean(0)[2, 0] == -1
        order = np.arange(50)[::-1]; other = checker.group_indices([cases[i] for i in order])
        for key, ids in groups.items(): assert np.array_equal(g[ids].mean(0), g[order][other[key]].mean(0))
        reject(lambda: checker.group_indices([c for c in cases if c['profile'] != 'clear']))
        group_tests += 3
    expected_arrays = 20 * (15 * 978243 * 4 + 128) + 100 * (256 * 256 * 3 * 4 + 128) + 200 * (512 * 4 + 128)
    assert 15 * 978243 * 4 + 128 < 64 * 1024 ** 2
    assert expected_arrays + 100 * 256 * 256 * 3 + 8 * 1024 ** 2 < p['budgets']['export_uncompressed_bytes']
    shell_path = BUNDLE / 'scripts/run_v37_guard.sh'
    command = [r'C:\Program Files\Git\bin\bash.exe', '--noprofile', '--norc', '-n', shell_path.as_posix()]
    syntax = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert syntax.returncode == 0, syntax.stderr
    write(PREP / 'Bash_readonly_syntax.json', {'complete': True, 'exit_code': syntax.returncode, 'command': command,
          'shell_sha256': sha(shell_path), 'read_only_syntax_check': True, 'launcher_executed': False, 'VM_calls': 0})
    shell = shell_path.read_text(); assert '930s' in shell and '330s' in shell and 'PIPESTATUS[0]' in shell
    assert '$HOME/forensic-dgp/' + BUNDLE.name in shell
    guide = (ROOT / 'CCTV_DGP_DELIVERED_GUARD_GRAD_V37_VM.md').read_text()
    assert pin in guide and prep['archive_sha256'] in guide
    scp = [line for line in guide.splitlines() if line.startswith('gcloud compute scp')]
    assert len(scp) == 5 and all(line.count('janusdominic0@forensic-dgp-thesis:') == 1 for line in scp)
    assert all('--zone=us-central1-a' in line and '--project=forensic-dgp-thesis' in line for line in scp)
    assert 'tmux new-session -A -s dgp_png_guard_v37' in guide
    assert not any(name in sys.modules for name in ['torch', 'dgp_frozen_inference_v2', 'cctv_dgp_pilot'])
    assert not (ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_return').exists()
    result = {'complete': True, 'checker_sha256': sha(Path(__file__)), 'protocol_sha256': pin,
              'archive_sha256': prep['archive_sha256'], 'archive_bytes': prep['archive_bytes'], 'archive_members_verified': 4,
              'basis_VM_bindings_verified': len(p['basis_VM_sha256']), 'V36_original_bindings_verified': len(p['V36_original_readback_sha256']),
              'local_basis_bindings_verified': len(p['local_basis_sha256']), 'Python310_syntax_verified': True,
              'Bash_readonly_syntax_verified': True, 'Windows_rejected_before_neural_imports_or_writes': True,
              'wrong_root_platform_instance_rejections': 4, 'unsafe_archive_boundary_regressions': 8,
              'group_cancellation_reorder_count_regressions': group_tests,
              'original_17_group_and_capacity_gates_retained': True, 'saved_return_array_bytes_expected': expected_arrays,
              'forward_metric_path_audited': True, 'coarse_derivative_tested': False,
              'coarse_gradient_queries_prepared': 300, 'actual_gradient_queries': 0, 'actual_neural_calls': 0,
              'actual_optimizer_updates': 0, 'parameter_assignments': 0, 'VM_launches': 0,
              'new_checkpoint_created': False, 'goal_complete': False, 'seconds': time.monotonic() - started}
    assert result['seconds'] < 300
    write(PREP / 'independent_packet_audit.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
