"""Shared-reference cache for the existing3905 TRAIN cases, existing L4 only."""
from pathlib import Path
import sys
import time


def load_items(root, p, original, identity, canonical_tensor, clock, write):
    assert sys.platform == 'linux', 'Existing Linux L4 VM only; no local training cache'
    assert root.resolve() == (Path.home() / 'forensic-dgp/cctv_dgp_broader_mean_vm_v30').resolve()
    import numpy as np
    from PIL import Image
    import torch
    from cctv_dgp_pilot import grid112
    mixed = root.parent / 'cctv_dgp_mixed_vm_v9_r2'
    started = time.monotonic()
    shared, items, times = {}, [], []
    def check():
        clock()
        assert time.monotonic() - started < p['budgets']['cache_seconds'], 'V30 cache900s cap'
    for ref in p['training_references']:
        check()
        with Image.open(mixed / ref['target']) as im: target = np.asarray(im.convert('RGB')).copy()
        with Image.open(mixed / ref['observed']) as im: support = np.asarray(im).copy() > 0
        assert target.shape == (256, 256, 3) and support.shape == (256, 256) and support.any()
        def erode(radius):
            size = 2 * radius + 1; padded = np.pad(support.astype(np.int64), radius)
            summed = np.pad(padded.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
            return summed[size:, size:] - summed[:-size, size:] - summed[size:, :-size] + summed[:-size, :-size] == size * size
        interior = erode(6); feature = np.zeros((256, 256), bool)
        for point in ref['landmarks5'][0]:
            xx, yy = np.floor(point).astype(int)
            feature[max(0, yy - 12):min(256, yy + 12), max(0, xx - 12):min(256, xx + 12)] = True
        feature &= interior
        assert feature.any()
        t = lambda value: torch.from_numpy(np.asarray(value).copy()).cuda()
        item = {'target8': target, 'mask8': support,
                'target': canonical_tensor(target, 'cuda'),
                'mask': t(support.astype(np.float32))[None, None],
                'feature': t(feature.astype(np.float32))[None, None],
                'interior': t(interior.astype(np.float32))[None, None],
                'valid7': t(erode(3).astype(np.float32))[None, None],
                'grid': t(grid112(ref['matrix112']))[None]}
        with torch.no_grad():
            item['truth'] = identity.embedding(item['target'], item['mask'], item['grid']).detach().clone()
        shared[ref['id']] = item
    assert len(shared) == 781
    reference_seconds = time.monotonic() - started
    for case in p['case_rows']:
        check()
        with Image.open(mixed / case['input']) as im: camera = np.asarray(im.convert('RGB')).copy()
        assert camera.shape == (256, 256, 3)
        item = {**shared[case['source_person_or_reference']], 'case': case, 'camera': camera,
                'x': canonical_tensor(camera, 'cuda')}
        item['degraded_weight'] = item['x'].new_tensor([0. if case['profile'] == 'clear' else 1.25])
        item['clear_weight'] = item['x'].new_tensor([1. if case['profile'] == 'clear' else 0.])
        items.append(item)
    assert len(items) == 3905
    parity = []
    for begin in range(0, 3905, 5):
        check(); step = time.monotonic(); group = items[begin:begin + 5]
        x = torch.cat([i['x'] for i in group]); mask = torch.cat([i['mask'] for i in group])
        with torch.no_grad(): fresh = torch.where(mask.bool(), original(x), x).detach().clone()
        assert not fresh.requires_grad and not torch.is_inference(fresh)
        for slot, item in enumerate(group):
            item['base'] = fresh[slot:slot + 1].clone()
            cid = item['case']['id']
            old = root / 'outputs/initial_baseline' / (cid + '.npy')
            if old.exists():
                cached = np.load(old, allow_pickle=False)
                raw = item['base'][0].permute(1, 2, 0).cpu().numpy()
                error = float(np.abs(raw - cached).max())
                assert error <= p['historical_CUDA_cache_tolerance'], 'Broader/sentinel original-DGP parity: ' + cid
                parity.append({'id': cid, 'maximum_raw_error': error})
        torch.cuda.synchronize(); times.append(time.monotonic() - step)
        if begin == 95:
            # Exclude the first complete batch from the steady sample.
            elapsed = time.monotonic() - started
            projected = elapsed + (781 - 20) * float(np.mean(times[1:])) * 1.25
            write(root / 'outputs/cache_timing.json', {'references': 781, 'cases': 100,
                  'reference_seconds': reference_seconds, 'seconds': elapsed,
                  'steady_sample_seconds': times[1:].copy(), 'remaining_batches': 761,
                  'safety_factor': 1.25, 'projected_seconds': projected,
                  'cap_seconds': p['budgets']['cache_seconds']})
            assert projected <= p['budgets']['cache_seconds'], 'V30 cache projection exceeds900s'
        if begin % 250 == 0: print({'V30_cached_cases': begin + 5, 'of': 3905, 'seconds': time.monotonic() - started}, flush=True)
    assert {r['id'] for r in parity} == set(p['preview_case_ids']) and len(parity) == 50
    check()
    write(root / 'outputs/cache_receipt.json', {'complete': True, 'references': 781, 'cases': 3905,
          'seconds': time.monotonic() - started, 'cap_seconds': p['budgets']['cache_seconds'],
          'initial50_normalizers_retained': True, 'sentinel_parity': parity,
          'target_support_feature_grid_shared_per_reference': True,
          'peak_allocated_VRAM_bytes': torch.cuda.max_memory_allocated(),
          'no_DEV_or_native_or_reserved_cases': True, 'optimizer_updates': 0})
    return items
