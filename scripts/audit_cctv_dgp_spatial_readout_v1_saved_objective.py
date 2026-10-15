"""Evaluate only saved arrays with fixed loss arithmetic; no neural forwards."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

from cctv_dgp_spatial_readout_v1_contract import read, write, sha

ROOT = Path(__file__).resolve().parent.parent
RETURN = ROOT/'outputs/cctv_dgp_spatial_readout_v1_return'
PARENT = ROOT/'outputs/cctv_dgp_bank_photometric_guard_v1_return'
OUT = RETURN/'outputs/spatial_readout_v1'
ANALYSIS = ROOT/'outputs/cctv_dgp_spatial_readout_v1_analysis'


def pixels(path, mode='RGB'):
    with Image.open(path) as im:
        assert im.size == (256, 256)
        return np.asarray(im.convert(mode)).copy()


def tensor(a):
    return torch.from_numpy(np.stack(a).astype(np.float32)/np.float32(255)).permute(0, 3, 1, 2).contiguous()


def main():
    start = time.monotonic()
    assert read(ANALYSIS/'independent_audit.json')['complete']
    p, old = read(RETURN/'protocol.json'), read(PARENT/'protocol.json')
    bindings = p['parent_required_sha256']
    checked = ['cctv_dgp_group_conflicts_v1_losses.py', 'cctv_dgp_bank_photometric_guard_v1_loss.py', 'frozen_capacity_contract.py', 'cctv_dgp_head4_lossless_v1.py']
    assert all(sha(PARENT/name) == bindings[name] for name in checked)
    sys.path.insert(1, str(PARENT))
    from cctv_dgp_group_conflicts_v1_losses import pixel_and_structure_losses
    from cctv_dgp_bank_photometric_guard_v1_loss import brightness_guard
    from frozen_capacity_contract import feature_support
    from cctv_dgp_head4_lossless_v1 import unpack
    torch.set_num_threads(4)
    refs, cases = {r['id']: r for r in old['references']}, {c['id']: c for c in old['cases']}
    rows = []
    normalizers = p['normalizers']
    with torch.no_grad():
        for serial, block in enumerate(p['reference_blocks']):
            assert time.monotonic()-start <= 360, 'Finite saved-array arithmetic stop'
            ref = refs[block['reference']]
            target_pixels = pixels(PARENT/ref['target'])
            mask_pixels = pixels(PARENT/ref['observed'], 'L') > 0
            target = tensor([target_pixels]*5)
            mask = torch.from_numpy(np.stack([mask_pixels]*5).astype(np.float32))[:, None]
            supports = [feature_support(mask_pixels, cases[cid]['landmarks5_canvas_xy']) for cid in block['case_ids']]
            support = torch.from_numpy(np.stack(supports).astype(np.float32))[:, None]
            with np.load(OUT/'baseline/packs'/f'r{serial:02d}.npz', allow_pickle=False) as pack:
                b = {k: pack[k].copy() for k in pack.files}
            baseline = torch.from_numpy(b['raw']).permute(0, 3, 1, 2).contiguous()
            target_vec = torch.from_numpy(b['target_embeddings']).double()
            retained, retained_embeddings = unpack(PARENT/'outputs/bank_comparison_v1/baseline/packs'/f"b{block['begin']//5:03d}.npz", block['case_ids'])
            assert np.array_equal(retained, b['raw']) and np.array_equal(retained_embeddings[:, 2], b['target_embeddings'])
            old_arc = 1-(torch.from_numpy(retained_embeddings[:, 0]).double()*target_vec).sum(1)
            old_terms = pixel_and_structure_losses(baseline, target, mask, support)
            snapshots = {}
            for label in ['baseline', 'A50', 'B50']:
                with np.load(OUT/label/'packs'/f'r{serial:02d}.npz', allow_pickle=False) as pack:
                    raw, er = pack['raw'].copy(), pack['embeddings_raw'][:5].copy()
                prediction = torch.from_numpy(raw).permute(0, 3, 1, 2).contiguous()
                terms = pixel_and_structure_losses(prediction, target, mask, support)
                arc = 1-(torch.from_numpy(er).double()*target_vec).sum(1)
                # Baseline fixed scores use authoritative cached controls, as in the VM loss.
                if label == 'baseline':
                    arc = old_arc
                averaged = [terms['MSE'].mean(), terms['SSIM_loss'].mean(), arc.mean(), terms['landmark_structure'][1:].mean()]
                components = [float(w*v/n) for w,v,n in zip(p['reconstruction_weights'], averaged, normalizers)]
                barriers = [float(5*torch.relu(terms['MSE']-old_terms['MSE']-1e-12).mean()/normalizers[0]),
                    float(5*torch.relu(terms['SSIM_loss']-old_terms['SSIM_loss']-1e-6).mean()/normalizers[1]),
                    float(5*torch.relu(arc-old_arc-1e-6).mean()/normalizers[2])]
                anchors = [float(terms['MSE'][0]/p['active_anchor_normalizers'][0]),
                           float(terms['MSE'][1]/p['active_anchor_normalizers'][1])]
                photo, details = brightness_guard(prediction, baseline, target, mask, normalizers[0])
                total = sum(components)+sum(barriers)+sum(anchors)+float(photo)
                snapshots[label] = dict(objective=total, normalized_weighted_RGB_SSIM_recognizer_structure=components,
                    regression_barriers_RGB_SSIM_recognizer=barriers, clear_blur_RGB_anchors=anchors,
                    photometric_guard=float(photo), per_image_photometric=details,
                    raw_RGB_MSE=terms['MSE'].tolist(), raw_SSIM_loss=terms['SSIM_loss'].tolist(),
                    raw_recognizer_loss=arc.tolist(), raw_structure_loss=terms['landmark_structure'].tolist())
            rows.append(dict(reference=block['reference'], source=block['source'], snapshots=snapshots))
    source_summaries = {}
    scalar_keys = ['objective', 'photometric_guard']
    vector_keys = ['normalized_weighted_RGB_SSIM_recognizer_structure', 'regression_barriers_RGB_SSIM_recognizer', 'clear_blur_RGB_anchors']
    for source in ['all']+sorted({r['source'] for r in rows}):
        cohort = [r for r in rows if source == 'all' or r['source'] == source]
        summary = {}
        for label in ['baseline', 'A50', 'B50']:
            values = [r['snapshots'][label] for r in cohort]
            summary[label] = {k: float(np.mean([v[k] for v in values])) for k in scalar_keys}
            summary[label].update({k: np.mean([v[k] for v in values], axis=0).tolist() for k in vector_keys})
        summary['lower_objective_references'] = {label: sum(r['snapshots'][label]['objective'] < r['snapshots']['baseline']['objective'] for r in cohort) for label in ['A50', 'B50']}
        source_summaries[source] = dict(references=len(cohort), summary=summary)
    preflight_differences = []
    row_map = {r['reference']: r for r in rows}
    for route in ['A','B']:
        for r in read(OUT/('preflight_'+route+'.json'))['rows']:
            value = row_map[r['reference']]['snapshots']['baseline']['objective']
            preflight_differences.append(dict(route=route, reference=r['reference'],
                saved_array_CPU_objective=value, reported_initial_VM_objective=r['terms']['objective'],
                difference=value-r['terms']['objective']))
    protected = read(ANALYSIS/'protected_before.json')['files']
    assert all(sha(ROOT/name) == digest for name,digest in protected.items())
    record = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), source_summaries=source_summaries,
        rows=rows, baseline_CPU_VM_differences=preflight_differences,
        arithmetic_source_sha256={name: sha(PARENT/name) for name in checked},
        formula='same normalized weighted training terms, RGB anchors, soft regression and per-image brightness hinge',
        saved_recognizer_embedding_batch=15, training_recognizer_embedding_batch=5,
        fresh_GPU_training_objective_replayed=False, optimizer_gradient_or_direction_claim=False,
        preserved_scientific_gate_replaced=False, acceptance_or_model_qualification=False,
        neural_calls=0, gradient_queries=0, optimizer_updates=0, local_learned_model_construction=False,
        fixed_array_loss_arithmetic_only=True, seconds=time.monotonic()-start)
    write(ANALYSIS/'saved_objective_audit.json', record)
    print(json.dumps(dict(complete=True, source_summaries=source_summaries,
                         saved_array_objective_only=True, neural_calls=0)))


if __name__ == '__main__':
    main()
