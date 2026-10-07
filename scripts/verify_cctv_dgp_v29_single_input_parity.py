"""Finite all50 forward-only parity of the candidate's actual single-input path."""
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
from dgp_mean_centered_inference_v29 import load_mean_centered_dgp_v29
from cctv_dgp_pilot import state_hash

OUT = ROOT / 'outputs/cctv_dgp_v29_single_input_parity_v1'
PARENT = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'
RETURNED = ROOT / 'outputs/cctv_dgp_mean_centered_decoder_v29_return'
BUNDLE = ROOT / 'outputs/cctv_dgp_mean_centered_decoder_vm_v29'


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def main():
    assert not OUT.exists(), 'Retain any previous parity evidence'
    audit_path = ROOT / 'outputs/cctv_dgp_mean_centered_decoder_v29_independent_audit.json'
    audit = read(audit_path); assert audit['complete'] and audit['necessary_capacity_pass']
    review = ROOT / 'outputs/cctv_dgp_mean_centered_decoder_v29_visual_review/visual_review.json'
    assert read(review)['cases_reviewed'] == 50
    p = read(BUNDLE / 'protocol.json'); assert sha(BUNDLE / 'protocol.json') == audit['protocol_sha256']
    original = PARENT / 'weights/dgp_v2.pth'
    final = RETURNED / 'outputs/update800/dgp_candidate_v29.pth'
    paths = [Path(__file__), ROOT / 'dgp_mean_centered_inference_v29.py', audit_path, review, BUNDLE / 'protocol.json', original, final,
             ROOT / 'dgp_face_restoration.py', ROOT / 'dgp_frozen_inference_v2.py', ROOT / 'cctv_dgp_frozen_norm.py', ROOT / 'cctv_dgp_pilot.py']
    paths += sorted((ROOT / 'models').glob('*.py'))
    # Keep source/input/raw bindings explicit; never import returned source code.
    for c in p['case_rows']:
        paths.extend([PARENT / c['input'], PARENT / c['observed']])
        paths.extend(RETURNED / 'outputs/update800' / (c['id'] + s) for s in ['.npy', '.png'])
    bindings = {x.relative_to(ROOT).as_posix(): sha(x) for x in paths}
    OUT.mkdir()
    plan = {'format': 'V29-single-input-forward-parity-v1', 'cases': [c['id'] for c in p['case_rows']],
            'CPU_CUDA_raw_tolerance': p['CPU_CUDA_raw_replay_tolerance'], 'PNG_maximum_byte_difference': 1,
            'budget': {'wall_seconds': 300, 'original_DGP_forwards': 50, 'candidate_DGP_forwards': 50},
            'device': 'cpu', 'threads': 4, 'sources_sha256': bindings,
            'batch_size': 1, 'VM_reference_batch_size': 5,
            'numerical_parity_is_not_quality_margin': True, 'native_or_reserved_used': False,
            'local_gradient_calls': 0, 'optimizer_updates': 0, 'app_promotion': False}
    write(OUT / 'plan.json', plan)
    start = time.monotonic(); torch.set_num_threads(4)
    counts = {'original_DGP': 0, 'candidate_DGP': 0}; rows = []; handles = []
    model, provenance = load_mean_centered_dgp_v29(original, final)
    states = {name: state_hash(getattr(model, name)) for name in ['original', 'candidate']}
    for key, net in [('original_DGP', model.original), ('candidate_DGP', model.candidate)]:
        def hook(module, args, out, key=key): counts[key] += 1
        handles.append(net.register_forward_hook(hook))
    try:
        for c in p['case_rows']:
            assert time.monotonic() - start <= 300
            with Image.open(PARENT / c['input']) as im: image = np.asarray(im.convert('RGB')).copy()
            with Image.open(PARENT / c['observed']) as im: support = np.asarray(im).copy() > 0
            values = image.astype(np.float32) / np.float32(255)
            x = torch.from_numpy(values).permute(2, 0, 1)[None]
            s = torch.from_numpy(support)[None, None]
            out = model(x, s)[0].permute(1, 2, 0).numpy().copy()
            ref = np.load(RETURNED / 'outputs/update800' / (c['id'] + '.npy'), allow_pickle=False)
            raw_error = float(np.abs(out - ref).max())
            assert raw_error <= plan['CPU_CUDA_raw_tolerance'], c['id']
            delivered = np.floor(out * np.float32(255)).astype(np.uint8); delivered[~support] = image[~support]
            with Image.open(RETURNED / 'outputs/update800' / (c['id'] + '.png')) as im: ref_png = np.asarray(im.convert('RGB')).copy()
            errors = np.abs(delivered.astype(np.int16) - ref_png.astype(np.int16))
            assert int(errors.max()) <= 1 and np.array_equal(delivered[~support], image[~support])
            rows.append({'id': c['id'], 'raw_maximum_error': raw_error,
                         'PNG_maximum_byte_difference': int(errors.max()),
                         'PNG_changed_channels': int((errors != 0).sum()), 'padding_exact': True})
        assert counts == {'original_DGP': 50, 'candidate_DGP': 50}
        assert states == {name: state_hash(getattr(model, name)) for name in states}
        assert all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters())
        assert time.monotonic() - start <= 300
        for name, digest in bindings.items(): assert sha(ROOT / name) == digest
        write(OUT / 'results.json', {'complete': True, 'plan_sha256': sha(OUT / 'plan.json'),
              'cases': 50, 'rows': rows, 'seconds': time.monotonic() - start,
              'raw_maximum_error': max(r['raw_maximum_error'] for r in rows),
              'PNG_changed_channels_total': sum(r['PNG_changed_channels'] for r in rows),
              'PNG_maximum_byte_difference': max(r['PNG_maximum_byte_difference'] for r in rows),
              'model_states_before_after': states, 'model_forwards': counts, 'provenance': provenance,
              'local_gradient_calls': 0, 'local_optimizer_updates': 0,
              'native_or_reserved_used': False, 'app_promotion': False, 'goal_complete': False})
        print(json.dumps({'complete': True, 'cases': 50, 'seconds': time.monotonic() - start}), flush=True)
    except Exception as e:
        write(OUT / 'failure.json', {'error': str(e), 'type': type(e).__name__, 'cases_completed': len(rows),
              'forwards': counts, 'seconds': time.monotonic() - start, 'local_gradient_calls': 0})
        raise
    finally:
        for h in handles: h.remove()


if __name__ == '__main__': main()
