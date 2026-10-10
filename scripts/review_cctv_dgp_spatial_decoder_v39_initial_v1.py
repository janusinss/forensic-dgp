"""Prospective initial-output proof for V39; all local models strictly frozen."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cctv_dgp_spatial_decoder_v39 import SpatialDGPCandidateV39, SEED, WIDTH
from cctv_dgp_pilot import state_hash
from dgp_frozen_inference_v2 import load_frozen_dgp_restorer

PARENT = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'
OUT = ROOT / 'outputs/cctv_dgp_spatial_decoder_v39_initial_review_v1'
ORIGINAL_SHA = '646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def pixels(path, mode='RGB'):
    with Image.open(path) as im:
        return np.asarray(im.convert(mode)).copy()


def tensor(rgb):
    return torch.from_numpy(rgb.astype(np.float32) / np.float32(255)).permute(2, 0, 1)[None]


def main():
    assert not OUT.exists(), 'Preserve previous and partial attempts'
    prior = ROOT / 'outputs/cctv_dgp_v38_return_development_milestone_v1'
    assert read(prior / 'independent_closure_audit.json')['complete']
    parent = read(PARENT / 'protocol.json'); cases = parent['cases']
    assert len(cases) == 50 and all(c['role'] == 'train' for c in cases)
    weight = PARENT / 'weights/dgp_v2.pth'; assert sha(weight) == ORIGINAL_SHA
    for c in cases:
        for k in ['input', 'observed']:
            assert sha(PARENT / c[k]) == parent['assets_sha256'][c[k]]
    OUT.mkdir(); torch.set_num_threads(4); start = time.monotonic()
    original, provenance = load_frozen_dgp_restorer(weight, expected_sha256=ORIGINAL_SHA, device='cpu')
    model = SpatialDGPCandidateV39(original)
    layout = []; offset = 0
    for name, value in model.decoder.named_parameters():
        layout.append({'name': name, 'shape': list(value.shape), 'start': offset, 'end': offset + value.numel()})
        offset += value.numel()
    torch.save(model.decoder.state_dict(), OUT / 'untrained_initial_decoder.pth')
    paths = [Path(__file__), ROOT / 'scripts/cctv_dgp_spatial_decoder_v39.py',
             ROOT / 'dgp_mean_centered_inference_v29.py', ROOT / 'dgp_face_restoration.py',
             ROOT / 'dgp_frozen_inference_v2.py', ROOT / 'cctv_dgp_frozen_norm.py', ROOT / 'cctv_dgp_pilot.py',
             PARENT / 'protocol.json', weight, OUT / 'untrained_initial_decoder.pth',
             prior / 'milestone.json', prior / 'independent_closure_audit.json']
    paths += sorted((ROOT / 'models').glob('*.py'))
    for c in cases:
        paths += [PARENT / c[k] for k in ['input', 'observed']]
    sources = {path.relative_to(ROOT).as_posix(): sha(path) for path in paths}
    app = read(ROOT / 'outputs/completion_input_footprints_comparison_v1_r1/protocol.json')['app_preservation_sha256']
    sources.update(app)
    plan = {'format': 'own-DGP-spatial-decoder-v39-initial-frozen-proof-v1', 'date': '2026-10-08',
            'frozen_before_new_outputs': True, 'case_ids': [c['id'] for c in cases],
            'cases': cases, 'sources_sha256': sources, 'original_checkpoint_sha256': ORIGINAL_SHA,
            'seed': SEED, 'width': WIDTH, 'parameter_layout': layout, 'decoder_parameters': offset,
            'selection': 'All50 already-exposed V27 TRAIN previews; no DEV/final/native inputs or target-based selection.',
            'hypothesis': 'Two spatial scales, full256 local features and fixed-initial-response subtraction give an exactly baseline-preserving start with a different learned spatial path. They do not prove finite useful restoration.',
            'architecture': 'Frozen original DGP; context=observed RGB+original RGB+bilinear256 map0(64)+smooth2(32); new16/32-channel spatial decoder and separately fixed initial copy; half-tanh response difference, same observed mean centering/clamp.',
            'untrained_seed_asset_is_not_a_trained_restorer': True, 'pixel_parity_required': 'Exact fresh original raw and deliveredPNG on all50 cases; exact pixels outside support.',
            'local_device': 'cpu', 'threads': 4, 'budget_seconds': 240,
            'gradient_calls': 0, 'optimizer_updates': 0, 'trained_checkpoint_created': False,
            'native_or_reserved_used': False, 'app_changed': False, 'goal_complete': False}
    write(OUT / 'plan.json', plan)
    states = {'original': state_hash(original.net), 'decoder': state_hash(model.decoder), 'reference_decoder': state_hash(model.reference_decoder)}
    assert states['decoder'] == states['reference_decoder']
    assert not ({v.data_ptr() for v in model.decoder.parameters()} & {v.data_ptr() for v in model.reference_decoder.parameters()})
    counts = {'original_DGP': 0, 'decoder': 0, 'reference_decoder': 0}; handles = []
    for key, net in [('original_DGP', original.net), ('decoder', model.decoder), ('reference_decoder', model.reference_decoder)]:
        def hook(_module, _inputs, _result, key=key): counts[key] += 1
        handles.append(net.register_forward_hook(hook))
    write(OUT / 'execution.json', {'plan_sha256': sha(OUT / 'plan.json'), 'states_before': states,
          'torch': torch.__version__, 'original_provenance': provenance, 'local_gradient_calls': 0,
          'optimizer_updates': 0, 'all_parameters_frozen': True})
    rows = []; (OUT / 'raw').mkdir(); (OUT / 'images').mkdir()
    try:
        with torch.inference_mode():
            for c in cases:
                assert time.monotonic() - start <= 240
                image = pixels(PARENT / c['input']); support = pixels(PARENT / c['observed'], 'L') > 0
                x = tensor(image); s = torch.from_numpy(support)[None, None]
                base = torch.where(s, original(x), x)
                parts = model.forward_components(x, s)
                assert torch.equal(base, parts['original_raw']) and torch.equal(base, parts['result'])
                assert torch.count_nonzero(parts['spatial_delta']) == 0
                for key in ['original_raw', 'result']:
                    raw = parts[key][0].permute(1, 2, 0).numpy().copy()
                    np.save(OUT / ('raw/' + c['id'] + '_' + key + '.npy'), raw, allow_pickle=False)
                    png = np.floor(raw * np.float32(255)).astype(np.uint8); png[~support] = image[~support]
                    Image.fromarray(png).save(OUT / ('images/' + c['id'] + '_' + key + '.png'))
                rows.append({'id': c['id'], 'source': c['source'], 'profile': c['profile'],
                             'raw_maximum_error': 0., 'PNG_maximum_byte_error': 0, 'padding_exact': True})
                if len(rows) % 10 == 0:
                    print(json.dumps({'initial_cases': len(rows), 'of': 50, 'seconds': time.monotonic() - start}), flush=True)
            c = cases[0]; x = tensor(pixels(PARENT / c['input'])); support = torch.zeros((1, 1, 256, 256), dtype=torch.bool)
            support[:, :, 32:224, 48:208] = True
            result = model(x, support); assert torch.equal(result.masked_select(~support), x.masked_select(~support))
            negative = [
                (x.double(), support), (x[:, :, :255], support), (x * float('nan'), support),
                (x, torch.zeros_like(support)), (x, support.float() * .5),
            ]
            before_invalid = counts.copy()
            for image, mask in negative:
                try: model(image, mask)
                except (ValueError, FloatingPointError): pass
                else: raise AssertionError('Invalid input was accepted')
            assert counts == before_invalid
        try: model.enable_vm_learning(ROOT / 'outputs/cctv_dgp_spatial_decoder_vm_v39')
        except AssertionError as e: host_rejection = str(e)
        else: raise AssertionError('Local learning was enabled')
        assert counts == {'original_DGP': 101, 'decoder': 51, 'reference_decoder': 51}
        assert states == {'original': state_hash(original.net), 'decoder': state_hash(model.decoder), 'reference_decoder': state_hash(model.reference_decoder)}
        assert all(not v.requires_grad and v.grad is None for v in model.parameters()) and all(not m.training for m in model.modules())
        for name, digest in sources.items(): assert sha(ROOT / name) == digest
        assert time.monotonic() - start <= 240
        artifacts = {f.relative_to(OUT).as_posix(): sha(f) for f in OUT.rglob('*') if f.is_file()}
        write(OUT / 'results.json', {'complete': True, 'plan_sha256': sha(OUT / 'plan.json'),
              'rows': rows, 'cases': 50, 'exact_initial_raw_and_PNG_parity': True, 'partial_support_exact': True,
              'invalid_inputs_rejected_before_model_calls': 5, 'local_learning_rejection': host_rejection,
              'decoder_parameters': offset, 'decoder_tensors': len(layout), 'states_before_after': states,
              'model_forwards': counts, 'artifacts_sha256': artifacts, 'seconds': time.monotonic() - start,
              'gradient_calls': 0, 'optimizer_updates': 0, 'trained_checkpoint_created': False,
              'local_untrained_seed_asset_created': True, 'gradient_connectivity_established': False,
              'native_or_reserved_used': False, 'app_changed': False, 'restoration_qualified': False, 'goal_complete': False})
        print(json.dumps({'complete': True, 'cases': 50, 'initial_parity_exact': True,
                          'decoder_parameters': offset, 'decoder_tensors': len(layout), 'seconds': time.monotonic() - start}), flush=True)
    except Exception as e:
        write(OUT / 'failure.json', {'type': type(e).__name__, 'error': str(e), 'cases_completed': len(rows),
              'model_forwards': counts, 'seconds': time.monotonic() - start, 'local_gradient_calls': 0, 'optimizer_updates': 0})
        raise
    finally:
        for hook in handles: hook.remove()


if __name__ == '__main__':
    main()
