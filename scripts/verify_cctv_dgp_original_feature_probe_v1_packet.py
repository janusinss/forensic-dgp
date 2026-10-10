"""Separate packet readback and initial CPU inference; zero derivatives/updates."""
import ast
import hashlib
from pathlib import Path
import sys
import tarfile
import time
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'outputs/cctv_dgp_original_feature_probe_v1_vm'
PREP = ROOT/'outputs/cctv_dgp_original_feature_probe_v1_preparation'
sys.path.insert(0, str(BUNDLE))
from cctv_dgp_original_feature_probe_v1_contract import NAME, STEM, sha, read, write, verified_assets


def pixels(path, mode='RGB'):
    with Image.open(path) as im: return np.asarray(im.convert(mode)).copy()


def main():
    start = time.monotonic(); prepared = read(PREP/'prepared.json')
    pin = prepared['protocol_sha256']; p = verified_assets(BUNDLE, pin)
    archive = ROOT/'outputs'/(STEM+'-execution.tar.gz')
    assert sha(archive) == prepared['archive_sha256'] and archive.stat().st_size == prepared['archive_bytes']
    assert Path(str(archive)+'.sha256').read_text().split() == [sha(archive), archive.name]
    for n, d in p['local_sources_sha256'].items(): assert sha(ROOT/n) == d, n
    for n, source in p['copied_source_mapping'].items(): assert sha(BUNDLE/n) == sha(ROOT/source)
    archive_files = {}
    with tarfile.open(archive, 'r:gz') as tar:
        for m in tar.getmembers():
            assert m.isfile() and not m.issym() and not m.islnk()
            parts = m.name.split('/'); assert parts[0] == NAME and all(a not in ['', '.', '..'] for a in parts)
            assert '\\' not in m.name and ':' not in m.name
            n = '/'.join(parts[1:]); assert n not in archive_files
            stream = tar.extractfile(m); h = hashlib.sha256()
            for block in iter(lambda: stream.read(1024**2), b''): h.update(block)
            archive_files[n] = h.hexdigest(); assert sha(BUNDLE/n) == h.hexdigest()
    assert set(archive_files) == set(p['assets_sha256']) | {'protocol.json'}
    parsed = {}
    for q in BUNDLE.rglob('*.py'): parsed[q.relative_to(BUNDLE).as_posix()] = ast.parse(q.read_text(), feature_version=(3, 10))
    vm = parsed['scripts/cctv_dgp_original_feature_probe_v1_vm.py']
    source = (BUNDLE/'scripts/cctv_dgp_original_feature_probe_v1_vm.py').read_text()
    assert 'torch.optim' not in source
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'backward' for n in ast.walk(vm))
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'save' and isinstance(n.func.value, ast.Name) and n.func.value.id == 'torch' for n in ast.walk(vm))
    gradient_calls = [n for n in ast.walk(vm) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'grad']
    assert len(gradient_calls) == 1
    keywords = {k.arg: k.value for k in gradient_calls[0].keywords}
    assert ast.literal_eval(keywords['allow_unused']) is True and ast.literal_eval(keywords['materialize_grads']) is False
    # Every trial derives from initial, not prior trial; assignment/restoration and scalar loop are reviewed independently.
    assert 'initial.astype(np.float64)+planned' in source and 'candidate.assign_trial(initial); frozen(True)' in source
    assert 'sys.path.insert(0, str(Path(__file__).resolve().parents[1]))' in source
    shell = (BUNDLE/'scripts/run_feature_probe.sh').read_text()
    assert '\r' not in shell and '1230s' in shell and '330s' in shell and 'PIPESTATUS[0]' in shell
    assert 'TMUX' in shell and NAME in shell and '--record-supervision' in shell
    parent = read(ROOT/'outputs/cctv_dgp_residual_epochs_vm_v42/protocol.json')
    by = {c['id']: c for c in parent['cases']}
    assert all(c == by[c['id']] for c in p['cases'])
    assert p['cohorts'][0]['case_ids'] == parent['preview_case_ids']
    first_refs = {by[cid]['source_person_or_reference'] for cid in parent['preview_case_ids']}
    expected_second_refs = []
    for s in sorted({r['source'] for r in parent['references']}):
        eligible = [r['id'] for r in parent['references'] if r['source'] == s and r['id'] not in first_refs]
        eligible.sort(key=lambda rid: hashlib.sha256(('original-feature-probe-v1:'+rid).encode()).hexdigest())
        expected_second_refs.extend(eligible[:5])
    assert p['cohorts'][1]['case_ids'] == [c['id'] for rid in expected_second_refs for c in parent['cases'] if c['source_person_or_reference'] == rid]
    assert sha(ROOT/'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth') == p['original_checkpoint_sha256']
    refs = {r['id']: r for r in p['references']}
    target_bindings = []
    for r in p['references']:
        target = pixels(BUNDLE/r['target']); mask = pixels(BUNDLE/r['observed'], 'L') > 0
        pixel_sha = hashlib.sha256(target.tobytes()).hexdigest()
        assert np.array_equal(target, pixels(ROOT/'outputs/cctv_dgp_residual_epochs_vm_v42'/r['target']))
        # V6 changed HQ target paths but retained the original reduced-reference pixel field.
        legacy_matches = pixel_sha == r['target_rgb_sha256']
        if not legacy_matches: assert r['evaluation_source_kind'] == 'HQ_FFHQ_counterpart' and r['hq_source_sha256']
        target_bindings.append({'id': r['id'], 'encoded_target_sha256': p['assets_sha256'][r['target']],
            'actual_target_pixel_sha256': pixel_sha, 'legacy_reference_pixel_field_matches': legacy_matches,
            'exact_retained_V42_pixels': True})
        assert hashlib.sha256(mask.tobytes()).hexdigest() == r['observed_sha256']
    import torch
    from cctv_dgp_pilot import state_hash, buffer_hash
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_original_feature_probe_v1_candidate import OriginalFeatureProbe
    torch.set_num_threads(4)
    original, _ = load_frozen_dgp_restorer(BUNDLE/'weights/dgp_v2.pth', expected_sha256=p['original_checkpoint_sha256'])
    candidate = OriginalFeatureProbe(original.net, p['parameter_layout'])
    before = {'original': state_hash(original.net), 'candidate': state_hash(candidate.net), 'buffers': buffer_hash(candidate.net)}
    all_names = dict(candidate.net.named_parameters(remove_duplicate=False))
    for r, (_, value) in zip(p['parameter_layout'], candidate.selected):
        assert {name for name, v in all_names.items() if v is value} == set(r['aliases'])
    assert all(not v.requires_grad and v.grad is None for v in candidate.net.parameters())
    denied = []
    for key, call in [('gradient_enable', lambda: candidate.enable_diagnostic_gradients(BUNDLE)),
                      ('trial_assignment', lambda: candidate.assign_trial(candidate.vector()))]:
        try: call()
        except AssertionError: denied.append(key)
        else: raise AssertionError('Local VM-only operation unexpectedly allowed: '+key)
    maximum_historical_error = 0.; parity_cases = 0; archived_raw_cases = 0
    with torch.no_grad():
        for begin in range(0, 100, 5):
            assert time.monotonic()-start <= 600
            cases = p['cases'][begin:begin+5]
            x = torch.from_numpy(np.stack([pixels(BUNDLE/c['input']) for c in cases]).astype(np.float32)/np.float32(255)).permute(0, 3, 1, 2)
            support = torch.from_numpy(np.stack([pixels(BUNDLE/c['observed'], 'L') > 0 for c in cases]).astype(np.float32))[:, None]
            baseline = original(x).detach().clone(); assert not torch.is_inference(baseline)
            result = candidate(x, support, baseline)
            assert torch.equal(result, torch.where(support.bool(), baseline, x)); parity_cases += len(cases)
            for slot, c in enumerate(cases):
                saved = ROOT/'outputs/cctv_dgp_residual_epochs_vm_v42_return/outputs/update0'/(c['id']+'.npy')
                if saved.is_file():
                    raw = result[slot].permute(1, 2, 0).numpy().copy()
                    maximum_historical_error = max(maximum_historical_error, float(np.abs(raw-np.load(saved, allow_pickle=False)).max()))
                    archived_raw_cases += 1
    assert parity_cases == 100 and archived_raw_cases == 50 and maximum_historical_error <= 3e-6
    after = {'original': state_hash(original.net), 'candidate': state_hash(candidate.net), 'buffers': buffer_hash(candidate.net)}
    assert before == after and before['original'] == before['candidate'] == p['original_state']
    verified_assets(BUNDLE, pin)
    write(PREP/'independent_packet_audit.json', {'complete': True, 'protocol_sha256': pin,
        'archive_sha256': sha(archive), 'archive_members_verified': len(archive_files),
        'source_bindings_verified': len(p['local_sources_sha256']), 'copied_asset_origins_verified': len(p['copied_source_mapping']),
        'actual_target_pixel_bindings': target_bindings,
        'unique_selected_tensors': 158, 'unique_selected_parameters': 1996035,
        'initial_CPU_parity_cases': parity_cases, 'retained_VM_baseline_cases': archived_raw_cases,
        'maximum_historical_raw_error': maximum_historical_error, 'local_guards_denied': denied,
        'states_before': before, 'states_after': after, 'Python310_AST_files': len(parsed),
        'local_neural_forward_batches': 40, 'local_gradient_queries': 0, 'local_optimizer_updates': 0,
        'VM_connections': 0, 'VM_diagnostic_launched': False, 'model_qualification': False,
        'checker_sha256': sha(Path(__file__)), 'seconds': time.monotonic()-start, 'goal_complete': False})
    print({'complete': True, 'initial_parity_cases': parity_cases, 'historical_error': maximum_historical_error,
           'local_gradient_queries': 0, 'VM_diagnostic_launched': False})


if __name__ == '__main__': main()
