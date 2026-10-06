"""Independent V19 transfer/scope audit; no neural imports or cloud execution."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs/cctv_dgp_input_selection_vm_v19'


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): h.update(block)
    return h.hexdigest()


def audit():
    started = time.monotonic()
    preparation = read(ROOT / 'outputs/cctv_dgp_input_selection_v19_preparation.json')
    p = read(BUNDLE / 'input_selection_protocol_v19.json')
    archive = ROOT / 'outputs/cctv-dgp-input-selection-v19-execution.tar.gz'
    assert preparation['complete'] and sha(BUNDLE / 'input_selection_protocol_v19.json') == preparation['protocol_sha256']
    assert (BUNDLE / 'protocol.sha256').read_text().strip() == preparation['protocol_sha256']
    assert sha(archive) == preparation['archive_sha256'] and archive.stat().st_size == preparation['archive_bytes']
    assert Path(str(archive) + '.sha256').read_text().split() == [preparation['archive_sha256'], archive.name]
    wanted = set(p['assets_sha256']) | {'input_selection_protocol_v19.json', 'protocol.sha256'}
    verified = []
    with tarfile.open(archive, 'r:gz') as stream:
        members = stream.getmembers()
        assert len(members) == len({m.name for m in members}) == preparation['members'] == len(wanted)
        assert {m.name for m in members} == wanted and sum(m.size for m in members) <= 128 * 1024**2
        for member in members:
            part = PurePosixPath(member.name)
            assert member.isfile() and not part.is_absolute() and '..' not in part.parts
            assert str(part) == member.name and ':' not in member.name and '\\' not in member.name
            payload = stream.extractfile(member).read()
            assert payload == (BUNDLE / member.name).read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            if member.name in p['assets_sha256']: assert digest == p['assets_sha256'][member.name]
            if member.name.endswith('.py'): ast.parse(payload.decode('utf-8'), feature_version=(3, 10))
            verified.append({'name': member.name, 'bytes': member.size, 'sha256': digest})
    original_v18 = ROOT / 'outputs/cctv_dgp_structure_vm_v18'
    for name, digest in p['assets_sha256'].items():
        if name.endswith('.py') and not name.startswith('parent_v18/'):
            assert sha(ROOT / name) == digest
    cp = read(original_v18 / 'structure_protocol_v18.json')
    assert sha(original_v18 / 'structure_protocol_v18.json') == 'e3a7567e9a9c53a1668464ab5c95edbf094c73e9a7c8e550635dcf8d8ef04adb'
    for name in set(cp['assets_sha256']) | {'structure_protocol_v18.json', 'protocol.sha256'}:
        assert (BUNDLE / 'parent_v18' / name).read_bytes() == (original_v18 / name).read_bytes()
    r2 = read(ROOT / 'outputs/cctv_dgp_broader_codes_vm_v16_r2/broader_codes_protocol_v16_r2.json')
    training = [r for r in r2['references'] if r['role'] == 'train']
    validation = [r for r in r2['references'] if r['role'] == 'validation']
    assert p['references'] == validation and len(validation) == 104 and len(training) == 781
    assert p['cases'] == r2['validation_cases'] and len(p['cases']) == len({c['id'] for c in p['cases']}) == 520
    assert p['training_parity_cases'] == cp['training_cases'] and len(cp['training_cases']) == 50
    assert p['training_parity_references'] == cp['references'] and len(cp['references']) == 10
    assert Counter(r['source'] for r in validation) == Counter({'dataset/asian_faces': 51, 'dataset/thumbnails128x128': 53})
    for field in ['id', 'source_sha256', 'target_rgb_sha256']:
        assert not ({r[field] for r in validation} & {r[field] for r in training})
    profiles = ['clear', 'blur_lr24', 'lowlight_lr32', 'motion_lr48', 'compound_lr24']
    for ref in validation:
        assert sorted(c['profile'] for c in p['cases'] if c['reference_id'] == ref['id']) == sorted(profiles)
    assert p['preview_reference_ids'] == r2['validation_preview_reference_ids'] and len(p['preview_reference_ids']) == 10
    names = {c['input'] for c in p['cases']} | {r['target'] for r in validation} | {r['observed'] for r in validation} | set(cp['data_assets_sha256'])
    assert set(p['data_assets_sha256']) == names
    for name, digest in p['data_assets_sha256'].items():
        assert sha(ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2' / name) == digest == r2['data_assets_sha256'][name]
    assert p['training'] is False and p['validation_used'] is True
    assert all(p[key] is False for key in ['native_used', 'native_reserved_used', 'teacher_used', 'production_promoted'])
    lineage = read(BUNDLE / 'lineage/v18_results.json')
    assert sha(BUNDLE / 'lineage/v18_results.json') == 'f0acefaadbf2890627339038371673aeed30cf9a066b9f74ca71d28766e9c206'
    assert p['terminal_state_hash'] == lineage['snapshots'][-1]['state_hash'] and lineage['snapshots'][-1]['update'] == 600
    assert sha(BUNDLE / 'weights/structure_update600.pth') == lineage['artifacts_sha256']['update600/decoder.pth']
    for c in cp['training_cases']:
        for folder, update in [('dgp_base', 'update0'), ('structure_update600', 'update600')]:
            assert sha(BUNDLE / ('parity/' + folder + '/' + c['id'] + '.npy')) == lineage['artifacts_sha256'][update + '/' + c['id'] + '.npy']
    selector_sha = 'cf72354d5092d0aab93dd7ecd646088999dfb009ea5cd4c0b0f19551f0c80180'
    assert sha(BUNDLE / 'dgp_input_selector_v19.py') == sha(ROOT / 'dgp_input_selector_v19.py') == selector_sha
    control = read(BUNDLE / 'lineage/processing_results_v19.json')
    control_audit = read(BUNDLE / 'lineage/processing_audit_v19.json')
    assert control_audit['complete'] and control_audit['results_sha256'] == sha(BUNDLE / 'lineage/processing_results_v19.json')
    assert control_audit['unchanged_quality_guard'] == control['preservation']
    assert control['preservation']['qualified_for_separate_generalization_protocol'] and not control['preservation']['failed_groups_metrics']
    assert control['retained_cases'] == 10 and control['spatial_cases'] == 40
    assert read(original_v18 / 'structure_protocol_v18.json') == cp  # Historical protocol retained.
    design = p['design']
    assert design['runner_cap_seconds'] == 1200 and design['audit_cap_seconds'] == 300 and design['supervisor_cap_seconds'] == 1800
    assert design['export_cap_seconds'] == 180 and design['optimizer_updates'] == design['backward_calls'] == 0
    assert design['fresh_parity_tolerance'] == 2e-6 and design['CPU_cached_head_tolerance'] == 5e-5
    assert design['MSE_preservation_tolerance'] == 1e-12 and design['SSIM_cosine_preservation_tolerance'] == 1e-6
    assert design['minimum_degraded_PSNR_gain_dB'] == .1 and design['minimum_degraded_MSE_improvement'] == .10
    runner = ast.parse((BUNDLE / 'scripts/run_cctv_dgp_input_selection_v19.py').read_text())
    forbidden = {'backward', 'enable_vm_training', 'step', 'Adam', 'AdamW', 'SGD', 'grad'}
    assert not any(isinstance(n, ast.Call) and (isinstance(n.func, ast.Attribute) and n.func.attr in forbidden
                   or isinstance(n.func, ast.Name) and n.func.id in forbidden) for n in ast.walk(runner))
    module = ast.parse((BUNDLE / 'cctv_dgp_input_selection_v19.py').read_text())
    fn = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'expected_counts')
    expected = {key: 50 + 520 for key in ['dgp', 'prior_encoder', 'prior_classifier', 'r2_head', 'prior_generator', 'residual_head']}
    expected.update(prior_RGB_tail=0, unused_v11=0, recognizer=104 + 520)
    assert ast.literal_eval(fn.body[0].value) == expected
    supervisor = (BUNDLE / 'scripts/supervise_cctv_dgp_input_selection_v19.py').read_text()
    assert "call('run_cctv_dgp_input_selection_v19.py'" in supervisor and 'train_cctv' not in supervisor
    assert "call('audit_cctv_dgp_input_selection_v19.py'" in supervisor
    launcher = ROOT / 'outputs/launch_cctv_dgp_input_selection_v19.py'
    assert sha(launcher) == sha(BUNDLE / 'scripts/launch_cctv_dgp_input_selection_v19.py') == preparation['launcher_sha256']
    receipt = {'complete': True, 'protocol_sha256': preparation['protocol_sha256'],
        'archive_sha256': preparation['archive_sha256'], 'archive_bytes': archive.stat().st_size,
        'members': verified, 'source_assets': len(p['assets_sha256']), 'data_files_verified': len(names),
        'own_exact_train_development_overlap_excluded': True, 'pretrained_overlap_excluded': False,
        'training_parity_cases': 50, 'validation_references': 104, 'validation_cases': 520,
        'expected_neural_forwards': expected, 'CPU_replay_cases': 24,
        'neural_calls': 0, 'optimizer_updates': 0, 'backward_calls': 0, 'VM_launched': False,
        'checker_sha256': sha(Path(__file__)), 'seconds': time.monotonic() - started}
    with (ROOT / 'outputs/cctv_dgp_input_selection_v19_transfer_audit.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in receipt.items() if k != 'members'}, indent=2))


if __name__ == '__main__': audit()
