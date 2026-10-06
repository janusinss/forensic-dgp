"""Independent R2 archive/source/lineage/scope audit; no network forwards."""
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs/cctv_dgp_input_selection_vm_v19_r2'
ORIGINAL = ROOT / 'outputs/cctv_dgp_input_selection_vm_v19'
DIAG = ROOT / 'outputs/cctv_dgp_input_selection_v19_parity_diagnostic_return_r1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def audit():
    started = time.monotonic()
    preparation = read(ROOT / 'outputs/cctv_dgp_input_selection_v19_r2_preparation.json')
    p = read(BUNDLE / 'input_selection_protocol_v19_r2.json')
    archive = ROOT / 'outputs/cctv-dgp-input-selection-v19-r2-execution.tar.gz'
    assert preparation['complete'] and sha(BUNDLE / 'input_selection_protocol_v19_r2.json') == preparation['protocol_sha256']
    assert (BUNDLE / 'protocol.sha256').read_text().strip() == preparation['protocol_sha256']
    assert sha(archive) == preparation['archive_sha256'] and archive.stat().st_size == preparation['archive_bytes']
    assert Path(str(archive) + '.sha256').read_text().split() == [preparation['archive_sha256'], archive.name]
    wanted = set(p['assets_sha256']) | {'input_selection_protocol_v19_r2.json', 'protocol.sha256'}
    members_verified = []
    python_files = 0
    with tarfile.open(archive, 'r:gz') as stream:
        members = stream.getmembers()
        assert len(members) == len({m.name for m in members}) == preparation['members'] == len(wanted)
        assert {m.name for m in members} == wanted and sum(m.size for m in members) <= 128 * 1024**2
        for member in members:
            assert time.monotonic() - started < 120
            part = PurePosixPath(member.name)
            assert member.isfile() and not part.is_absolute() and '..' not in part.parts
            assert str(part) == member.name and ':' not in member.name and '\\' not in member.name
            payload = stream.extractfile(member).read()
            assert payload == (BUNDLE / member.name).read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            if member.name in p['assets_sha256']:
                assert digest == p['assets_sha256'][member.name]
            if member.name.endswith('.py'):
                ast.parse(payload.decode('utf-8'), feature_version=(3, 10)); python_files += 1
            members_verified.append({'name': member.name, 'bytes': member.size, 'sha256': digest})
    hp = read(ORIGINAL / 'input_selection_protocol_v19.json')
    op = sha(ORIGINAL / 'input_selection_protocol_v19.json')
    assert op == p['original_protocol_sha256'] == '2d707ebba97524867c4ce7610666e3abeb29638eb9054a8025983b6476e628d2'
    old_members = set(hp['assets_sha256']) | {'input_selection_protocol_v19.json', 'protocol.sha256'}
    assert {name.removeprefix('original_v19/') for name in p['assets_sha256'] if name.startswith('original_v19/')} == old_members
    for name in old_members:
        assert (BUNDLE / 'original_v19' / name).read_bytes() == (ORIGINAL / name).read_bytes()
    for key in ['references', 'cases', 'preview_reference_ids', 'training_parity_cases', 'training_parity_references',
                'data_assets_sha256', 'terminal_state_hash', 'baseline_results_sha256', 'training',
                'validation_used', 'native_used', 'native_reserved_used', 'teacher_used', 'production_promoted']:
        assert p[key] == hp[key]
    assert len(p['references']) == 104 and len(p['cases']) == 520 and len(p['training_parity_cases']) == 50
    for key, value in hp['design'].items():
        assert p['design'][key] == value
    assert p['design']['original_scientific_gates_changed'] is False
    assert p['design']['additional_canonical_DGP_forwards'] == p['design']['internal_spatial_base_raw_exports'] == 520
    for name, digest in p['assets_sha256'].items():
        if name.startswith('diagnostic/'):
            assert sha(DIAG / name.removeprefix('diagnostic/')) == digest
        elif name.endswith('.py') and not name.startswith('original_v19/'):
            assert sha(ROOT / name) == digest
    assert sha(BUNDLE / 'diagnostic/results.json') == '8a8ac06d6c5948e23c7196c03106a76a1e4902a316952fc47c86fa1a41fc4671'
    assert read(BUNDLE / 'diagnostic/local_independent_audit.json')['normalization_hypothesis_confirmed'] is True
    assert read(BUNDLE / 'diagnostic/results.json')['normalization_hypothesis_confirmed'] is True
    for name, digest in p['data_assets_sha256'].items():
        assert sha(ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2' / name) == digest
    own = read(ROOT / 'outputs/cctv_dgp_broader_codes_vm_v16_r2/broader_codes_protocol_v16_r2.json')
    train = [r for r in own['references'] if r['role'] == 'train']
    for key in ['id', 'source_sha256', 'target_rgb_sha256']:
        assert not ({r[key] for r in train} & {r[key] for r in p['references']})
    runner = (BUNDLE / 'scripts/run_cctv_dgp_input_selection_v19_r2.py').read_text()
    assert 'internal_base, raw, arrays = predict(camera)' in runner
    assert "core.core.dgp(retained_input(camera, 'cuda'))" in runner
    assert runner.index("save('raw/' + key") < runner.index("'Canonical fresh DGP PNG differs")
    assert runner.index("save('predictions/' + key") < runner.index("'Canonical fresh DGP PNG differs")
    assert 'np.array_equal(base_png, expected_png)' in runner
    forbidden = {'backward', 'enable_vm_training', 'step', 'AdamW', 'Adam', 'SGD', 'grad'}
    assert not any(isinstance(n, ast.Call) and (isinstance(n.func, ast.Attribute) and n.func.attr in forbidden
                   or isinstance(n.func, ast.Name) and n.func.id in forbidden) for n in ast.walk(ast.parse(runner)))
    auditor = (BUNDLE / 'scripts/audit_cctv_dgp_input_selection_v19_r2.py').read_text()
    assert 'spatial_cpu_replay_input' in auditor and "counts['internal_spatial_DGP_bases'] == 520" in auditor
    assert "counts['canonical_baseline_raw_previews'] == 50" in auditor
    assert "'CPU cache/internal spatial base binding differs'" in auditor
    for stem in ['run', 'audit', 'supervise', 'launch', 'import']:
        source = (BUNDLE / ('scripts/' + stem + '_cctv_dgp_input_selection_v19_r2.py')).read_text()
        assert '_r2_r2' not in source
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.endswith('.py') and 'cctv_dgp_input_selection' in node.value:
                path = BUNDLE / node.value if node.value.startswith('scripts/') else BUNDLE / 'scripts' / node.value
                assert path.is_file(), node.value
    supervisor = (BUNDLE / 'scripts/supervise_cctv_dgp_input_selection_v19_r2.py').read_text()
    assert "'inference.log', 1200" in supervisor and "'audit.log', 300" in supervisor
    assert "out = root / 'outputs/input_selection_v19_r2'" in supervisor
    launcher = ROOT / 'outputs/launch_cctv_dgp_input_selection_v19_r2.py'
    assert sha(launcher) == sha(BUNDLE / 'scripts/launch_cctv_dgp_input_selection_v19_r2.py') == preparation['launcher_sha256']
    expected = {'dgp': 50 + 520 * 2, 'prior_encoder': 570, 'prior_classifier': 570, 'r2_head': 570,
                'prior_generator': 570, 'prior_RGB_tail': 0, 'residual_head': 570, 'recognizer': 624, 'unused_v11': 0}
    assert preparation['expected_neural_forwards'] == expected
    receipt = {'complete': True, 'protocol_sha256': preparation['protocol_sha256'],
        'archive_sha256': preparation['archive_sha256'], 'archive_bytes': archive.stat().st_size,
        'members': members_verified, 'source_assets': len(p['assets_sha256']), 'Python3_10_files': python_files,
        'unchanged_original_members': len(old_members), 'data_files_verified': len(p['data_assets_sha256']),
        'own_exact_split_overlap_excluded': True, 'pretrained_overlap_excluded': False,
        'original_scientific_gates_changed': False, 'validation_references': 104,
        'validation_cases': 520, 'training_parity_cases': 50, 'expected_neural_forwards': expected,
        'local_neural_forwards': 0, 'backward_calls': 0, 'optimizer_updates': 0, 'VM_launched': False,
        'auditor_sha256': sha(Path(__file__)), 'seconds': time.monotonic() - started}
    with (ROOT / 'outputs/cctv_dgp_input_selection_v19_r2_transfer_audit.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in receipt.items() if k != 'members'}, indent=2))


if __name__ == '__main__':
    audit()
