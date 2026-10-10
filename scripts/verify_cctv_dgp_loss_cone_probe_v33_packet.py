"""Packet and arithmetic checks only; never construct an optimizer or execute VM code."""
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

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs/cctv_dgp_loss_cone_probe_v33_vm'
PREP = ROOT / 'outputs/cctv_dgp_loss_cone_probe_v33_preparation'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


def reject(function):
    try: function()
    except (AssertionError, ValueError): return
    raise AssertionError('Boundary was not rejected')


def main():
    started = time.monotonic(); p = read(BUNDLE / 'protocol.json'); prep = read(PREP / 'preparation_receipt.json'); pin = sha(BUNDLE / 'protocol.json')
    assert prep['complete'] and prep['protocol_sha256'] == pin
    auditor = module('own_finite_probe_return_boundary', ROOT / 'scripts/audit_cctv_dgp_loss_cone_probe_v33_return.py')
    basis = auditor.verify_basis(p)
    assert p['cohorts'] == basis['cohorts'] and p['terms'] == basis['terms'] and p['normalizers'] == basis['normalizers']
    assert p['parameter_layout'] == basis['parameter_layout'] and p['optimizer_updates'] == 8
    assert p['new_gradient_queries'] == p['backwards'] == p['epochs'] == p['committed_trajectory_updates'] == 0
    assert p['retained_capacity_gates'] == {'early_at50_minimum': .01, 'final_at800_minimum': .1, 'preservation_groups': 17,
        'MSE_tolerance': 1e-12, 'SSIM_ArcFace_tolerance': 1e-6, 'source_gains_minimum': 0., 'brightness_fraction_maximum': .2}
    files = {f.relative_to(BUNDLE).as_posix() for f in BUNDLE.rglob('*') if f.is_file()}
    assert files == set(p['assets_sha256']) | {'protocol.json'} and len(files) == 5
    archive = ROOT / 'outputs/cctv-dgp-loss-cone-probe-v33-execution.tar.gz'
    assert sha(archive) == prep['archive_sha256'] and archive.stat().st_size == prep['archive_bytes']
    assert Path(str(archive) + '.sha256').read_text().strip().split() == [prep['archive_sha256'], archive.name]
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers(); assert len(members) == 5
        for item in members:
            assert item.isfile() and not item.issym() and not item.islnk() and item.name.startswith(BUNDLE.name + '/')
            name = item.name[len(BUNDLE.name) + 1:]; assert name in files
            assert hashlib.sha256(tar.extractfile(item).read()).hexdigest() == sha(BUNDLE / name)
    worker = BUNDLE / 'scripts/cctv_dgp_loss_cone_probe_v33_vm.py'
    scripts = [worker, BUNDLE / 'cctv_dgp_loss_cone_v33.py', BUNDLE / 'cctv_dgp_loss_cone_probe_v33_metrics.py', ROOT / 'scripts/audit_cctv_dgp_loss_cone_probe_v33_return.py']
    counts = {}
    for path in scripts:
        tree = ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 10)); calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in ['backward', 'grad'], 'No new gradients/backwards in finite probe'
                assert not (node.func.attr == 'save' and isinstance(node.func.value, ast.Name) and node.func.value.id == 'torch'), 'No checkpoints'
                if node.func.attr in ['AdamW', 'step']:
                    assert path == worker; calls.append(node.func.attr)
        counts[path.name] = calls
    assert counts[worker.name].count('AdamW') == counts[worker.name].count('step') == 1
    assert sum(len(v) for k, v in counts.items() if k != worker.name) == 0
    guard = subprocess.run([sys.executable, '-B', str(worker), '--root', str(BUNDLE), '--protocol-sha', pin, '--run'], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert guard.returncode != 0 and 'Existing Linux VM only; no local parameter updates' in guard.stderr and not (BUNDLE / 'outputs').exists()
    with (PREP / 'Windows_parameter_update_guard.txt').open('x', encoding='utf-8') as stream: stream.write(guard.stderr)
    helper = module('finite_worker_asset_checks_only', worker); assert helper.verify(BUNDLE, pin) == p
    member = tarfile.TarInfo(auditor.PREFIX + 'protocol.json'); member.size = 1
    assert auditor.safe_members([member], p)[1] == 1
    bad = copy.copy(member); bad.name = auditor.PREFIX + '../outside'; reject(lambda: auditor.safe_members([bad], p))
    bad = copy.copy(member); bad.type = tarfile.SYMTYPE; bad.linkname = '/outside'; reject(lambda: auditor.safe_members([bad], p))
    reject(lambda: auditor.safe_members([member, member], p))
    bad = copy.copy(member); bad.name = auditor.PREFIX + 'unapproved.pth'; reject(lambda: auditor.safe_members([bad], p))
    bad = copy.copy(member); bad.size = 17 * 1024**2; reject(lambda: auditor.safe_members([bad], p))
    bad = copy.copy(member); bad.name = auditor.PREFIX + 'outputs\\outside'; reject(lambda: auditor.safe_members([bad], p))
    bad = copy.copy(member); bad.type = tarfile.LNKTYPE; reject(lambda: auditor.safe_members([bad], p))
    # Test the actual proposal geometry with NumPy arithmetic. This is not an AdamW call or parameter assignment.
    import numpy as np
    import torch
    from cctv_dgp_loss_cone_v33 import project_vectors
    original = torch.load(ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27/weights/dgp_v2.pth', map_location='cpu', weights_only=True)
    stopped = torch.load(ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_return/outputs/update50/dgp_candidate_v32.pth', map_location='cpu', weights_only=True)
    primal_checks, max_error = 0, 0.
    for state, weights in [(0, original), (50, stopped)]:
        theta = np.concatenate([weights[r['name']].numpy().reshape(-1) for r in p['parameter_layout']]).astype(np.float64)
        for cohort in p['cohorts']:
            g = np.load(ROOT / f'outputs/cctv_dgp_v32_loss_gradient_v1_return/outputs/state{state}_{cohort["name"]}/gradient_components.npy', allow_pickle=False)
            for raw_gradient in [g.sum(0), g[:3].sum(0)]:
                clipped = raw_gradient / max(1., np.linalg.norm(raw_gradient) + 1e-6)
                hypothetical_displacement = .00003 * .01 * theta + .00003 * clipped / (np.abs(clipped) + 1e-8)
                projected, proof = project_vectors(g, hypothetical_displacement)
                error = auditor.projection_primal(g, hypothetical_displacement, projected); max_error = max(max_error, error); primal_checks += 1
                assert np.all(-(g @ projected) <= proof['KKT_tolerance'] * np.maximum(np.linalg.norm(g, axis=1), 1))
    assert primal_checks == 8
    receipt = {'complete': True, 'protocol_sha256': pin, 'checker_sha256': sha(Path(__file__)), 'archive_sha256': prep['archive_sha256'],
               'archive_members': 5, 'basis_VM_files_verified': len(p['basis_VM_sha256']), 'prior_basis_verified': True,
               'Python310_syntax_verified': True, 'Windows_rejected_before_neural_import_or_writes': True, 'unsafe_return_boundaries_rejected': 7,
               'independent_primal_displacement_checks': primal_checks, 'maximum_primal_L2_error': max_error,
               'checks_use_saved_arrays_only': True, 'actual_optimizer_updates': 0, 'parameter_assignments': 0, 'neural_calls': 0,
               'VM_launches': 0, 'Bash_syntax_requires_separate_readonly_check': True, 'goal_complete': False, 'seconds': time.monotonic() - started}
    with (PREP / 'independent_packet_audit.json').open('x', encoding='utf-8') as stream: stream.write(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__': main()
