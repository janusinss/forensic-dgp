"""Independent portable-packet, scientific-boundary and manual-command verification."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs/cctv_dgp_spatial_decoder_vm_v39'
PREP = ROOT / 'outputs/cctv_dgp_spatial_decoder_v39_preparation'
PARENT = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


def main():
    start = time.monotonic(); import numpy as np
    assert not (PREP / 'independent_packet_audit.json').exists()
    p = read(BUNDLE / 'protocol.json'); prep = read(PREP / 'prepared.json'); pin = sha(BUNDLE / 'protocol.json')
    assert pin == prep['protocol_sha256']
    archive = ROOT / 'outputs/cctv-dgp-spatial-decoder-v39-execution.tar.gz'
    assert sha(archive) == prep['archive_sha256'] and archive.stat().st_size == prep['archive_bytes']
    assert Path(str(archive) + '.sha256').read_text().strip().split() == [prep['archive_sha256'], archive.name]
    expected = {'protocol.json': pin, **p['assets_sha256']}; seen = set(); total = 0
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers(); assert len(members) == len(expected) == prep['files']
        for member in members:
            assert member.isfile() and not member.issym() and not member.islnk()
            assert member.name.startswith(BUNDLE.name + '/')
            name = member.name[len(BUNDLE.name) + 1:]
            assert name in expected and name not in seen and '\\' not in name and ':' not in name
            value = hashlib.sha256()
            with tar.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024**2), b''): value.update(block)
            assert value.hexdigest() == expected[name]; seen.add(name); total += member.size
    assert seen == set(expected) and total < 512 * 1024**2
    for name, digest in p['assets_sha256'].items(): assert sha(BUNDLE / name) == digest, name
    for name, digest in p['local_basis_sha256'].items(): assert sha(ROOT / name) == digest, name
    for name, source in p['copied_source_mapping'].items(): assert sha(BUNDLE / name) == sha(ROOT / source)
    old = read(PARENT / 'protocol.json')
    assert p['cases'] == old['cases'] and p['references'] == old['references']
    assert p['retained_capacity_gates'] == old['prospective_gates']
    for begin in range(0, 50, 5):
        chosen = p['cases'][begin:begin+5]
        assert len({c['source_person_or_reference'] for c in chosen}) == 1
        assert [c['profile'] for c in chosen] == ['clear', 'blur_lr24', 'lowlight_lr32', 'motion_lr48', 'compound_lr24']
    assert all(c['role'] == 'train' for c in p['cases'])
    proof = ROOT / 'outputs/cctv_dgp_spatial_decoder_v39_initial_review_v1'
    initial = read(proof / 'results.json'); checked = read(proof / 'independent_initial_audit.json')
    assert initial['complete'] and checked['complete'] and checked['all50_raw_pairs_exact'] and checked['all100_PNG_compositions_exact']
    assert sha(proof / 'results.json') == checked['results_sha256']
    assert sha(ROOT / 'scripts/verify_cctv_dgp_spatial_decoder_v39_initial_v1.py') == checked['checker_sha256']
    assert p['parameter_layout'] == read(proof / 'plan.json')['parameter_layout']
    assert p['expected_states']['decoder'] == p['expected_states']['reference_decoder'] == initial['states_before_after']['decoder']
    assert p['decoder_parameters'] == 17952 and p['decoder_tensors'] == len(p['parameter_layout']) == 57
    assert p['optimizer_updates'] == p['parameter_updates'] == p['backwards'] == p['epochs'] == 0
    assert p['component_gradient_calls'] == 70 and not p['automatic_follow_on'] and not p['new_training_loss_adopted']
    old_tree = ast.parse((PARENT / 'scripts/cctv_dgp_feature_skips_v27_vm.py').read_text(encoding='utf-8'))
    run = next(n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    head = next(n for n in ast.parse((PARENT / 'cctv_dgp_feature_skips_v27.py').read_text(encoding='utf-8')).body if isinstance(n, ast.ClassDef) and n.name == 'SpatialFeatureHead')
    frozen = ast.parse((BUNDLE / 'frozen_definitions.py').read_text(encoding='utf-8'))
    original_nodes = [next(n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'require_vm')]
    original_nodes += [next(n for n in head.body if isinstance(n, ast.FunctionDef) and n.name == name) for name in ['blur', 'high']]
    original_nodes += [next(n for n in run.body if isinstance(n, ast.FunctionDef) and n.name == name) for name in ['mean', 'feature_errors', 'ssim']]
    assert ast.dump(frozen, include_attributes=False) == ast.dump(ast.Module(body=original_nodes, type_ignores=[]), include_attributes=False)
    worker_path = BUNDLE / 'scripts/cctv_dgp_spatial_decoder_v39_vm.py'
    worker = module('V39_transfer_metadata_only', worker_path)
    assert worker.verify(BUNDLE, pin) == p
    rejected = 0
    for platform_name, hostname, folder in [('win32', 'forensic-dgp-thesis', BUNDLE),
         ('linux', 'another-vm', Path.home() / 'forensic-dgp' / BUNDLE.name),
         ('linux', 'forensic-dgp-thesis', Path.home() / 'forensic-dgp' / 'wrong-folder')]:
        with patch.object(worker.sys, 'platform', platform_name), patch.object(worker.platform, 'node', return_value=hostname):
            try: worker.scope(folder.resolve())
            except AssertionError: rejected += 1
            else: raise AssertionError('Wrong platform/host/root accepted')
    with patch.object(worker.sys, 'platform', 'linux'), patch.object(worker.platform, 'node', return_value='forensic-dgp-thesis'):
        worker.scope((Path.home() / 'forensic-dgp' / BUNDLE.name).resolve())
    before_files = set(path.relative_to(BUNDLE).as_posix() for path in BUNDLE.rglob('*') if path.is_file())
    command = subprocess.run([sys.executable, '-B', str(worker_path), '--root', str(BUNDLE),
                              '--protocol-sha', pin, '--verify-transfer'], capture_output=True, text=True, timeout=15)
    assert command.returncode != 0 and 'Existing Linux VM only' in command.stderr
    assert before_files == set(path.relative_to(BUNDLE).as_posix() for path in BUNDLE.rglob('*') if path.is_file())
    with (PREP / 'Windows_pre_neural_rejection.txt').open('x', encoding='utf-8') as stream:
        stream.write(command.stdout + command.stderr)
    checker = ROOT / 'scripts/audit_cctv_dgp_spatial_decoder_v39_return.py'
    auditor = module('V39_prospective_saved_array_readback', checker)
    assert auditor.BUNDLE == BUNDLE and auditor.OUT.name == 'cctv_dgp_spatial_decoder_v39_return'
    tree = ast.parse(worker_path.read_text(encoding='utf-8'))
    export = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'export')
    prefixes = [n.value for n in ast.walk(export) if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.endswith('_return/')]
    assert prefixes == [auditor.PREFIX]
    assert all(name in p['assets_sha256'] for name in auditor.RETURNED_ASSETS)
    def item(name, size=0, kind=tarfile.REGTYPE):
        value = tarfile.TarInfo(name); value.size = size; value.type = kind; return value
    prefix = auditor.PREFIX; auditor.safe_members([item(prefix + 'protocol.json')], p)
    bad = [[item(prefix + '../protocol.json')], [item('/' + prefix + 'protocol.json')],
           [item(prefix + 'protocol.json', kind=tarfile.SYMTYPE)], [item(prefix + 'protocol.json', kind=tarfile.LNKTYPE)],
           [item(prefix + 'protocol.json'), item(prefix + 'protocol.json')], [item(prefix + 'C:protocol.json')],
           [item(prefix + 'unknown.npy')], [item(prefix + 'protocol.json', 8 * 1024**2 + 1)],
           [item(prefix + 'outputs/initial/' + c['id'] + '_original.npy', 8*1024**2) for c in p['cases'][:25]]]
    unsafe = 0
    for members in bad:
        try: auditor.safe_members(members, p)
        except AssertionError: unsafe += 1
        else: raise AssertionError('Unsafe/oversized return accepted')
    # Known synthetic arrays test boundaries, never VM derivative evidence.
    fixtures = []
    with tempfile.TemporaryDirectory(prefix='V39-array-fixtures-', dir=PREP) as directory:
        folder = Path(directory).resolve(); assert folder.is_relative_to(PREP.resolve()) and folder.is_relative_to(ROOT)
        (folder / 'outputs/gradients').mkdir(parents=True)
        def make(no_improvement=False):
            batches = []; total = np.zeros((7, 17952), np.float64); values = np.zeros(7, np.float64)
            for i in range(10):
                matrix = np.zeros((7, 17952), np.float64)
                if not no_improvement: matrix[:3] = np.asarray([.001, .002, .003])[:, None] * (i + 1)
                path = folder / ('outputs/gradients/batch' + str(i) + '.npy'); np.save(path, matrix, allow_pickle=False)
                v = np.asarray([.1, .025, .005, 0, 0, 0, 0], np.float64); total += matrix; values += v
                batches.append({'batch': i, 'ids': [c['id'] for c in p['cases'][5*i:5*(i+1)]],
                    'gradient_sha256': sha(path), 'norms': np.linalg.norm(matrix, axis=1).tolist(), 'values': v.tolist()})
            path = folder / 'outputs/gradient_components.npy'; np.save(path, total, allow_pickle=False)
            summary = {'complete': True, 'terms': p['terms'], 'parameter_layout': p['parameter_layout'], 'batches': batches,
                'values': values.tolist(), 'component_norms': np.linalg.norm(total, axis=1).tolist(),
                'component_gram': (total @ total.T).tolist(), 'gradient_sha256': sha(path), 'partitions': {}}
            for row in p['parameter_layout']:
                block = total[:, row['start']:row['end']]
                summary['partitions'][row['name']] = {'component_norms': np.linalg.norm(block, axis=1).tolist(),
                    'improvement_gradient_norm': float(np.linalg.norm(block[:3].sum(0)))}
            (folder / 'outputs/gradient_summary.json').write_text(json.dumps(summary), encoding='utf-8')
            return summary
        make(); assert auditor.gradient_summary_check(folder, p, True)['parameter_tensors'] == 57
        for mode in ['nonfinite', 'wrong_dtype', 'wrong_shape', 'preservation_nonzero', 'wrong_order', 'wrong_sum', 'zero_improvement']:
            summary = make(mode == 'zero_improvement'); path = folder / 'outputs/gradients/batch0.npy'
            matrix = np.load(path, allow_pickle=False)
            if mode == 'nonfinite': matrix[0, 0] = np.nan
            elif mode == 'wrong_dtype': matrix = matrix.astype(np.float32)
            elif mode == 'wrong_shape': matrix = matrix[:, :-1]
            elif mode == 'preservation_nonzero': matrix[3, 0] = .1
            elif mode == 'wrong_order': summary['batches'][0]['ids'].reverse()
            elif mode == 'wrong_sum':
                aggregate = folder / 'outputs/gradient_components.npy'; data = np.load(aggregate, allow_pickle=False); data[0, 0] += .1
                np.save(aggregate, data, allow_pickle=False); summary['gradient_sha256'] = sha(aggregate)
            np.save(path, matrix, allow_pickle=False); summary['batches'][0]['gradient_sha256'] = sha(path)
            (folder / 'outputs/gradient_summary.json').write_text(json.dumps(summary, allow_nan=False), encoding='utf-8')
            try: auditor.gradient_summary_check(folder, p, True)
            except AssertionError: fixtures.append(mode)
            else: raise AssertionError('Invalid synthetic gradient payload accepted: ' + mode)
    syntax = read(PREP / 'Bash_readonly_syntax.json')
    assert syntax['exit_code'] == 0 and syntax['script_sha256'] == sha(BUNDLE / 'scripts/run_v39_gradient.sh')
    shell = (BUNDLE / 'scripts/run_v39_gradient.sh').read_text(encoding='utf-8')
    assert shell.count('630s') == 1 and shell.count('330s') == 1 and shell.count('--kill-after=30s') == 2
    assert 'CODE=${PIPESTATUS[0]}' in shell and '"$PIN" --export' in shell
    count = 0
    for source in [*BUNDLE.rglob('*.py'), checker, Path(__file__)]:
        ast.parse(source.read_text(encoding='utf-8'), feature_version=(3, 10)); count += 1
    worker_text = worker_path.read_text(encoding='utf-8'); checker_text = checker.read_text(encoding='utf-8')
    for text in [worker_text, checker_text]:
        assert not any(value in text for value in ['torch.optim', '.backward(', '.step(', 'torch.save('])
    assert worker_text.count('torch.autograd.grad(') == 1 and 'autograd.grad(' not in checker_text
    guide = (ROOT / 'CCTV_DGP_SPATIAL_DECODER_V39_VM.md').read_text(encoding='utf-8')
    commands = [line for line in guide.splitlines() if line.startswith('gcloud compute scp ')]
    assert len(commands) == 5 and all(line.count('janusdominic0@forensic-dgp-thesis:') == 1 for line in commands)
    assert pin in guide and prep['archive_sha256'] in guide and 'tmux new-session -A -s dgp_spatial_decoder_v39' in guide
    app = read(ROOT / 'outputs/completion_input_footprints_comparison_v1_r1/protocol.json')['app_preservation_sha256']
    for name, digest in app.items(): assert sha(ROOT / name) == digest
    assert not any(name in sys.modules for name in ['torch', 'dgp_frozen_inference_v2', 'cctv_dgp_pilot'])
    receipt = {'complete': True, 'checker_sha256': sha(Path(__file__)), 'protocol_sha256': pin,
        'archive_sha256': prep['archive_sha256'], 'archive_bytes': prep['archive_bytes'], 'archive_files': len(expected),
        'uncompressed_packet_bytes': total, 'assets_verified': len(p['assets_sha256']), 'local_basis_verified': len(p['local_basis_sha256']),
        'all50_TRAIN_cases_and10_references_exact': True, 'all57_tensor_layouts_exact': True,
        'all_original_loss_and_filter_definition_AST_exact': True, 'all_quality_gates_retained': True,
        'original_and_initial_weights_portable_and_verified': True, 'seed_is_untrained_asset': True,
        'Windows_pre_neural_rejection_passed': True, 'scope_rejections': rejected,
        'unsafe_archive_rejections': unsafe, 'saved_gradient_rejections': fixtures,
        'synthetic_fixtures_are_not_VM_or_training_evidence': True, 'Python310_sources_checked': count,
        'Bash_readonly_syntax_verified': True, 'single_remote_source_gcloud_commands_verified': 5,
        'all14_app_bindings_unchanged': True, 'neural_calls': 0, 'gradient_calls': 0,
        'optimizer_updates': 0, 'VM_calls': 0, 'V39_VM_execution_started': False,
        'app_promotion': False, 'goal_complete': False, 'seconds': time.monotonic() - start, 'cap_seconds': 300}
    assert receipt['seconds'] <= 300
    with (PREP / 'independent_packet_audit.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: receipt[k] for k in ['complete', 'archive_files', 'archive_bytes', 'scope_rejections', 'seconds']}), flush=True)


if __name__ == '__main__':
    main()
