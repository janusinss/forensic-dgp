"""Independent archive, split, scientific gate and CPU inference preparation checks."""
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'outputs/cctv_dgp_residual_epochs_vm_v42'
PREP = ROOT/'outputs/cctv_dgp_residual_epochs_v42_preparation'
sys.path.insert(0, str(BUNDLE))
from cctv_dgp_residual_epochs_v42_contract import (NAME, STEM, BUDGETS, sha, read, write,
    validate_schedule, validate_protocol, verified_assets, feature_support, capacity, safe_return_members)


def reject(call):
    try: call()
    except (AssertionError, ValueError): return
    raise AssertionError('Expected a rejection')


def definition(path, name):
    node = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.dump(node, include_attributes=False)


def main():
    start = time.monotonic(); p = read(BUNDLE/'protocol.json'); prepared = read(PREP/'prepared.json')
    pin = sha(BUNDLE/'protocol.json'); verified_assets(BUNDLE, p, pin)
    assert prepared['complete'] and prepared['protocol_sha256'] == pin
    old = read(ROOT/'outputs/cctv_dgp_spatial_fit_vm_v40/protocol.json')
    arithmetic = read(ROOT/'outputs/cctv_dgp_residual_supervision_v1/independent_audit.json')
    assert arithmetic['complete'] and p['arithmetic_audit_sha256'] == sha(ROOT/'outputs/cctv_dgp_residual_supervision_v1/independent_audit.json')
    assert p['cases'] == old['cases'] and p['references'] == old['references']
    assert p['initial_states'] == old['initial_states'] and p['parameter_layout'] == old['parameter_layout']
    assert p['preview_case_ids'] == old['preview_case_ids'] and p['retained_capacity_gates'] == old['retained_capacity_gates']
    for name, digest in p['local_sources_sha256'].items(): assert sha(ROOT/name) == digest
    for name, original in p['copied_source_mapping'].items(): assert sha(BUNDLE/name) == sha(ROOT/original)
    for name in ['capacity', 'exported_pixel_metrics', 'feature_support', 'detail_metric', 'groups']:
        assert definition(BUNDLE/'frozen_capacity_contract.py', name) == \
               definition(ROOT/'outputs/cctv_dgp_spatial_fit_vm_v40/cctv_dgp_spatial_fit_v40_contract.py', name)
    model = (BUNDLE/'cctv_dgp_spatial_decoder_v42.py').read_text()
    restored = model.replace(NAME, 'cctv_dgp_spatial_fit_vm_v40').replace('SpatialDGPCandidateV42', 'SpatialDGPCandidateV40').replace('Distinct V42', 'Distinct V40')
    restored = restored.replace("return {'original_raw': baseline, 'spatial_delta': delta,\n                'centered_correction': torch.where(support.bool(), centered, 0), 'result': result}",
                               "return {'original_raw': baseline, 'spatial_delta': delta, 'result': result}")
    assert restored == (ROOT/'outputs/cctv_dgp_spatial_fit_vm_v40/cctv_dgp_spatial_decoder_v40.py').read_text()
    assert 'target' not in definition(BUNDLE/'cctv_dgp_spatial_decoder_v42.py', 'require_learning_vm')
    for path in BUNDLE.rglob('*.py'): ast.parse(path.read_text(), feature_version=(3, 10))
    schedule = read(BUNDLE/'schedule.json'); validate_schedule(p['cases'], schedule)
    bad = copy.deepcopy(schedule); bad['batches'][781] = bad['batches'][782]
    reject(lambda: validate_schedule(p['cases'], bad))
    bad = copy.deepcopy(p); bad['early_gain'] = .009
    reject(lambda: validate_protocol(bad))
    bad = copy.deepcopy(p); bad['budgets']['fit_seconds'] += 1
    reject(lambda: validate_protocol(bad))
    masks = {}; targets = set()
    for c in p['cases']:
        with Image.open(BUNDLE/c['input']) as im: assert im.mode == 'RGB' and im.size == (256, 256)
        if c['observed'] not in masks:
            with Image.open(BUNDLE/c['observed']) as im:
                a = np.asarray(im); assert im.mode == 'L' and im.size == (256, 256)
            assert set(np.unique(a)) <= {0, 255} and a.any(); masks[c['observed']] = a > 0
        if c['target'] not in targets:
            with Image.open(BUNDLE/c['target']) as im: assert im.mode == 'RGB' and im.size == (256, 256)
            targets.add(c['target'])
        feature_support(masks[c['observed']], c['landmarks5_canvas_xy'])
    archive = ROOT/'outputs'/(STEM+'-execution.tar.gz')
    assert sha(archive) == prepared['archive_sha256'] and archive.stat().st_size == prepared['archive_bytes']
    assert Path(str(archive)+'.sha256').read_text().strip().split() == [prepared['archive_sha256'], archive.name]
    contents = {}
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar.getmembers():
            assert member.isfile() and not member.issym() and not member.islnk()
            assert member.name.startswith(NAME+'/')
            name = member.name[len(NAME)+1:]
            assert name not in contents and all(v not in ['', '.', '..'] for v in name.split('/'))
            with tar.extractfile(member) as stream:
                h = hashlib.sha256()
                for block in iter(lambda: stream.read(1024**2), b''): h.update(block)
            contents[name] = h.hexdigest()
    assert contents == {**p['assets_sha256'], 'protocol.json': pin}
    baseline = {str(i): {'cases': 1, 'MSE': .03, 'SSIM': .65, 'ArcFace_observed_fixed': .3,
                            'landmark_high_frequency_MSE': .02, 'constant_mean_shift_only_MSE': .03} for i in range(15)}
    baseline.update({'a/degraded': dict(baseline['0']), 'b/degraded': dict(baseline['0'])})
    baseline['degraded'] = baseline.pop('0')
    good = copy.deepcopy(baseline)
    for v in good.values(): v['MSE'] = .025; v['landmark_high_frequency_MSE'] = .015
    assert capacity(baseline, good, .1)['pass']
    for metric in ['MSE', 'SSIM', 'ArcFace_observed_fixed']:
        bad = copy.deepcopy(good)
        bad['1'][metric] = baseline['1'][metric]+(.0001 if metric == 'MSE' else -.0001)
        assert not capacity(baseline, bad, .01)['pass']
    bad = copy.deepcopy(good); bad['degraded']['landmark_high_frequency_MSE'] = .0199
    assert not capacity(baseline, bad, .01)['pass']
    bad = copy.deepcopy(good); bad['a/degraded']['landmark_high_frequency_MSE'] = .021
    assert not capacity(baseline, bad, .01)['pass']
    bad = copy.deepcopy(good); bad['degraded']['constant_mean_shift_only_MSE'] = .025
    assert not capacity(baseline, bad, .01)['pass']
    # Malicious return path/link/duplicate/size scopes are rejected without extracting anything.
    valid_names = ['protocol.json', 'export_manifest.json', 'supervisor_receipt.json', 'trainer.log', 'trainer_exit_code.txt']
    def member(name, size=1):
        m = tarfile.TarInfo(NAME+'_return/'+name); m.size = size; return m
    members = [member(name) for name in valid_names]; safe_return_members(members, p)
    reject(lambda: safe_return_members([*members, member('../outside')], p))
    reject(lambda: safe_return_members([*members, member('outputs\\bad')], p))
    reject(lambda: safe_return_members([*members, member('weights/dgp_v2.pth')], p))
    reject(lambda: safe_return_members([*members, members[0]], p))
    link = member('outputs/link'); link.type = tarfile.SYMTYPE
    reject(lambda: safe_return_members([*members, link], p))
    reject(lambda: safe_return_members([*members, member('outputs/oversize', BUDGETS['export_uncompressed_bytes']+33*1024**2)], p))
    if sys.platform != 'linux':
        response = subprocess.run([sys.executable, '-B', str(BUNDLE/'scripts/cctv_dgp_residual_epochs_v42_vm.py'),
            '--root', str(BUNDLE), '--protocol-sha', pin, '--run'], capture_output=True, text=True, timeout=30)
        assert response.returncode != 0 and 'existing manual Linux VM' in response.stderr
        assert not (BUNDLE/'outputs').exists()
        write(PREP/'local_learning_rejection.json', {'complete': True, 'exit_code': response.returncode,
            'stderr': response.stderr, 'neural_calls': 0, 'optimizer_updates': 0})
    # Local inference only; no .backward(), autograd.grad(), or optimizer construction.
    import torch
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_spatial_decoder_v42 import SpatialDGPCandidateV42
    from cctv_dgp_pilot import state_hash
    torch.set_num_threads(4)
    original, provenance = load_frozen_dgp_restorer(BUNDLE/'weights/dgp_v2.pth', expected_sha256=p['original_checkpoint_sha256'], device='cpu')
    seed = torch.load(BUNDLE/'untrained_initial_decoder.pth', map_location='cpu', weights_only=True)
    candidate = SpatialDGPCandidateV42(original, seed)
    assert state_hash(original.net) == p['initial_states']['original']
    assert state_hash(candidate.decoder) == state_hash(candidate.reference_decoder) == p['initial_states']['decoder']
    assert len(list(candidate.decoder.parameters())) == 57
    assert sum(v.numel() for v in candidate.decoder.parameters()) == 17952
    reject(lambda: candidate.enable_vm_learning(BUNDLE))
    chosen = [c for c in p['cases'] if c['id'] in p['preview_case_ids']]
    canonical = lambda a: torch.from_numpy(a.astype(np.float32)/np.float32(255)).permute(2, 0, 1)[None]
    with torch.inference_mode():
        for begin in range(0, 50, 5):
            batch = chosen[begin:begin+5]
            x = torch.cat([canonical(np.asarray(Image.open(BUNDLE/c['input'])).copy()) for c in batch])
            mask = torch.from_numpy(np.stack([masks[c['observed']] for c in batch]))[:, None]
            components = candidate.forward_components(x, mask)
            base = torch.where(mask, original(x), x)
            assert torch.equal(components['result'], base) and torch.equal(components['original_raw'], base)
            assert bool((components['centered_correction'] == 0).all())
            assert torch.equal(components['result'][~mask.expand_as(x)], x[~mask.expand_as(x)])
            print({'V42_CPU_initial_parity': begin+5, 'of': 50}, flush=True)
    assert state_hash(original.net) == p['initial_states']['original']
    assert state_hash(candidate.decoder) == state_hash(candidate.reference_decoder) == p['initial_states']['decoder']
    assert all(not v.requires_grad and v.grad is None for v in candidate.parameters())
    verified_assets(BUNDLE, p, pin)
    write(PREP/'independent_packet_audit.json', {'complete': True, 'protocol_sha256': pin,
        'archive_sha256': prepared['archive_sha256'], 'archive_members': len(contents),
        'training_cases': 3905, 'training_references': 781, 'full_epochs_checked': 5,
        'independent_scientific_gate_regressions': 6, 'archive_boundary_regressions': 6,
        'CPU_initial_parity_cases': 50, 'all_current_checkpoint_states_unchanged': True,
        'python310_parse': True, 'local_learning_rejected': True,
        'no_clean_target_in_model_forward': True, 'training_or_gradient_calls': 0,
        'VM_connections': 0, 'training_launched': False, 'new_quality_claim': False,
        'app_promotion': False, 'goal_complete': False, 'seconds': time.monotonic()-start})
    print({'complete': True, 'prepared_only': True, 'training_calls': 0}, flush=True)


if __name__ == '__main__': main()
