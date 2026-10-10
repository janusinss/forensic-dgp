"""Shared TRAIN cache and raw/PNG observations for the finite manual V42 study."""
import hashlib
import time
import numpy as np
from PIL import Image
import torch
from cctv_dgp_pilot import grid112, state_hash
from cctv_dgp_residual_epochs_v42_contract import (BUDGETS, write, sha, erode,
    feature_support, exported_pixel_metrics, detail_metric, groups, capacity)
from frozen_raw_metrics import pixel_metrics, detail_float, deliver, mean_only, review_groups


def pixels(path, mode='RGB'):
    with Image.open(path) as im:
        return np.asarray(im.convert(mode)).copy()


def canonical(a):
    return torch.from_numpy(a.astype(np.float32)/np.float32(255)).permute(2, 0, 1)[None].cuda()


def load_items(root, p, original, identity, clock, out):
    start = time.monotonic(); shared = {}; items = []; samples = []
    t = lambda a: torch.from_numpy(np.asarray(a).copy()).cuda()
    for ref in p['references']:
        clock(); assert time.monotonic()-start < BUDGETS['cache_seconds']
        target = pixels(root/ref['target']); mask = pixels(root/ref['observed'], 'L') > 0
        assert target.shape == (256, 256, 3) and mask.shape == (256, 256)
        item = {'target8': target, 'mask8': mask, 'target': canonical(target),
                'mask': t(mask.astype(np.float32))[None, None],
                'feature': t(feature_support(mask, ref['landmarks5'][0]))[None, None],
                'valid7': t(erode(mask, 3).astype(np.float32))[None, None],
                'grid': t(grid112(ref['matrix112']))[None]}
        with torch.no_grad():
            item['truth'] = identity.embedding(item['target'], item['mask'], item['grid']).detach().clone()
        shared[ref['id']] = item
    for c in p['cases']:
        camera = pixels(root/c['input']); assert camera.shape == (256, 256, 3)
        items.append({**shared[c['source_person_or_reference']], 'case': c, 'camera': camera,
                      'x': canonical(camera), 'clear': t(np.array([c['profile']=='clear'], bool))})
    for begin in range(0, 3905, 5):
        clock(); assert time.monotonic()-start < BUDGETS['cache_seconds']; step = time.monotonic()
        selected = items[begin:begin+5]
        x = torch.cat([i['x'] for i in selected]); mask = torch.cat([i['mask'] for i in selected])
        with torch.no_grad():
            base = torch.where(mask.bool(), original(x), x).detach().clone()
        assert not torch.is_inference(base)
        for slot, item in enumerate(selected): item['base'] = base[slot:slot+1].clone()
        torch.cuda.synchronize(); samples.append(time.monotonic()-step)
        if begin == 95:
            elapsed = time.monotonic()-start
            estimate = elapsed+761*float(np.mean(samples[1:]))*p['timing_safety_factor']
            write(out/'cache_timing.json', {'cases': 100, 'seconds': elapsed,
                  'steady_sample_seconds': samples[1:], 'remaining_batches': 761,
                  'safety_factor': p['timing_safety_factor'], 'projected_seconds': estimate,
                  'cap_seconds': BUDGETS['cache_seconds']})
            assert estimate <= BUDGETS['cache_seconds'], 'Cache timing projection stop'
        if begin % 250 == 0: print({'V42_cached': begin+5, 'of': 3905}, flush=True)
    clock(); elapsed = time.monotonic()-start
    assert elapsed <= BUDGETS['cache_seconds']
    write(out/'cache_receipt.json', {'complete': True, 'seconds': elapsed,
          'references': 781, 'cases': 3905, 'cap_seconds': BUDGETS['cache_seconds'],
          'training_only': True, 'optimizer_updates': 0})
    return items


KEYS = ['x', 'base', 'target', 'mask', 'feature', 'valid7', 'grid', 'truth', 'clear']


def batch(items, ids):
    return {key: torch.cat([items[i][key] for i in ids]) for key in KEYS}


def snapshot(out, update, p, items, candidate, identity, clock, frozen):
    clock(); stamp = time.monotonic(); folder = out/('update'+str(update)); folder.mkdir()
    rows = []; frozen(); torch.save(candidate.decoder.state_dict(), folder/'spatial_decoder.pth')
    with torch.no_grad():
        for begin in range(0, 3905, 5):
            clock(); selected = items[begin:begin+5]; b = batch(items, list(range(begin, begin+5)))
            components = candidate.forward_components(b['x'], b['mask'])
            pred = components['result']
            assert torch.equal(components['original_raw'], b['base'])
            if update == 0:
                assert torch.equal(pred, b['base']), 'Full TRAIN initial parity failed'
                assert bool((components['centered_correction'] == 0).all())
            raw = pred.permute(0, 2, 3, 1).cpu().numpy().copy()
            pngs = [deliver(a, i['camera'], i['mask8']) for a, i in zip(raw, selected)]
            vectors = identity.embedding(torch.cat([pred, torch.cat([canonical(a) for a in pngs])]),
                torch.cat([b['mask'], b['mask']]), torch.cat([b['grid'], b['grid']])).cpu().numpy().copy()
            for slot, (item, a, png) in enumerate(zip(selected, raw, pngs)):
                c = item['case']; cid = c['id']; truth = item['truth'][0].cpu().numpy().copy()
                target = item['target8']; mask = item['mask8']
                np.save(folder/(cid+'_embedding.npy'), vectors[slot+5], allow_pickle=False)
                np.save(folder/(cid+'_raw_embedding.npy'), vectors[slot], allow_pickle=False)
                if update == 0: np.save(folder/(cid+'_target_embedding.npy'), truth, allow_pickle=False)
                if cid in p['preview_case_ids']: np.save(folder/(cid+'.npy'), a, allow_pickle=False)
                Image.fromarray(png).save(folder/(cid+'.png'))
                base = item['base'][0].permute(1, 2, 0).cpu().numpy().copy()
                mean_raw, mean_png, shift = mean_only(a, base, item['camera'], mask)
                Image.fromarray(mean_png).save(folder/(cid+'_mean_only.png'))
                feature = feature_support(mask, c['landmarks5_canvas_xy'])
                raw_metrics = pixel_metrics(a, target, mask)
                raw_metrics.update({'ArcFace_observed_fixed': float(vectors[slot]@truth),
                    'landmark_high_frequency_MSE': detail_float(a, target, feature),
                    'constant_mean_shift_only_MSE': pixel_metrics(mean_raw, target, mask)['MSE']})
                metric = exported_pixel_metrics(png, target, mask)
                metric.update({'ArcFace_observed_fixed': float(vectors[slot+5]@truth),
                    'landmark_high_frequency_MSE': detail_metric(png, target, feature),
                    'constant_mean_shift_only_MSE': exported_pixel_metrics(mean_png, target, mask)['MSE']})
                rows.append({'id': cid, 'source': c['source'], 'profile': c['profile'],
                    'metrics': metric, 'raw_metrics': raw_metrics,
                    'postclip_mean_RGB_shift': shift.tolist(),
                    'raw_float32_sha256': hashlib.sha256(a.tobytes()).hexdigest()})
            if begin % 250 == 0: print({'V42_snapshot': update, 'cases': begin+5, 'of': 3905}, flush=True)
    receipt = {'complete': True, 'update': update, 'completed_epochs': update//781,
        'rows': rows, 'groups': groups(rows),
        'raw_groups': review_groups([{**r, 'raw': r['raw_metrics']} for r in rows], 'raw'),
        'decoder_state': state_hash(candidate.decoder), 'checkpoint_sha256': sha(folder/'spatial_decoder.pth'),
        'snapshot_duration_seconds': time.monotonic()-stamp}
    write(folder/'metrics.json', receipt); frozen(); clock()
    print({'snapshot': update, 'feature_MSE': receipt['groups']['degraded']['landmark_high_frequency_MSE']}, flush=True)
    return receipt


def gates(baseline, current, minimum):
    delivered = capacity(baseline['groups'], current['groups'], minimum)
    raw = capacity(baseline['raw_groups'], current['raw_groups'], minimum)
    return {'update': current['update'], 'delivered': delivered, 'raw': raw,
            'pass': delivered['pass'] and raw['pass']}
