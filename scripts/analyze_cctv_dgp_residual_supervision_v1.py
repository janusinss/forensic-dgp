"""Bounded saved-TRAIN arithmetic: prospective supervision, no model or learning."""
from pathlib import Path
import hashlib
import json
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cctv_dgp_residual_supervision_v1 import (
    projected_teacher, correction_terms, TERMS, EPSILON, DELTA_LIMIT,
    LEVEL_WEIGHTS, SUPPORT_RADII)

OUT = ROOT / 'outputs/cctv_dgp_residual_supervision_v1'
BUNDLE = ROOT / 'outputs/cctv_dgp_actual_step_review_v1_vm'
GEOMETRY = ROOT / 'outputs/cctv_dgp_output_geometry_v1'
CAP = 300


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    start = time.monotonic()
    assert not OUT.exists(), 'Preserve previous evidence; use a distinct version'
    geometry = read(GEOMETRY / 'analysis.json')
    audit = read(GEOMETRY / 'independent_audit.json')
    assert audit['complete'] and audit['analysis_sha256'] == sha(GEOMETRY / 'analysis.json')
    selected = {r['id']: r for r in read(GEOMETRY / 'plan.json')['selected']}
    p = read(BUNDLE / 'protocol.json')
    assert len(p['cases']) == len(selected) == 145
    assert all(c['role'] == 'train' for c in p['cases'])
    original = read(ROOT / 'outputs/cctv_dgp_spatial_fit_vm_v40/protocol.json')
    calibration = set(original['preview_case_ids'])
    assert len(calibration) == 50 and calibration <= set(selected)
    torch.set_num_threads(1)
    OUT.mkdir()
    plan = {'format': 'own-DGP-direct-preclamp-residual-supervision-arithmetic-v1',
            'cases': 145, 'references': 29, 'calibration': sorted(calibration),
            'calibration_selection': 'Unchanged historical initial50 TRAIN IDs; only40 degraded cases set scales',
            'epsilon': EPSILON, 'delta_limit': DELTA_LIMIT,
            'level_weights': LEVEL_WEIGHTS, 'support_radii': SUPPORT_RADII,
            'normalizer_floor': 1e-3, 'terms': TERMS, 'cap_seconds': CAP,
            'fractions': [0., .25, .5, 1.], 'clear_teacher': 'zero correction to the retained DGP',
            'degraded_teacher': 'Bounded paired correction before the final image clamp; observed RGB mean removed',
            'new_VM_recipe_frozen': False, 'gradient_or_capacity_test': False,
            'native_DEV_or_reserved_final_used': False, 'optimizer_updates': 0}
    write(OUT / 'plan.json', plan)
    bindings = {(GEOMETRY.relative_to(ROOT) / name).as_posix(): sha(GEOMETRY / name)
                for name in ['analysis.json', 'plan.json', 'independent_audit.json']}
    bindings.update({str(path.relative_to(ROOT)).replace('\\', '/'): sha(path)
                     for path in [BUNDLE / 'protocol.json', Path(__file__).resolve(),
                                  ROOT / 'cctv_dgp_residual_supervision_v1.py',
                                  ROOT / 'outputs/cctv_dgp_spatial_fit_vm_v40/protocol.json',
                                  ROOT / 'outputs/cctv_dgp_spatial_fit_vm_v40/cctv_dgp_degraded_objective_v24.py']})
    rows = []
    with torch.inference_mode():
        for slot, case in enumerate(p['cases']):
            assert time.monotonic() - start < CAP, 'Saved-array arithmetic cap exceeded'
            npz = ROOT / selected[case['id']]['npz']
            name = str(npz.relative_to(ROOT)).replace('\\', '/')
            assert sha(npz) == geometry['source_sha256'][name]
            bindings[name] = sha(npz)
            with np.load(npz, allow_pickle=False) as arrays:
                baseline = arrays['original_rgb'].copy()
            assert baseline.shape == (256, 256, 3) and baseline.dtype == np.float32
            images = {}
            for key in ['target', 'observed', 'input']:
                path = BUNDLE / case[key]
                assert sha(path) == p['assets_sha256'][case[key]]
                bindings[str(path.relative_to(ROOT)).replace('\\', '/')] = sha(path)
                with Image.open(path) as im:
                    images[key] = np.asarray(im.convert('L' if key == 'observed' else 'RGB')).copy()
            mask = images['observed'] > 0
            baseline[~mask] = images['input'][~mask].astype(np.float32) / np.float32(255)
            target = images['target'].astype(np.float64) / 255
            tensor = lambda a: torch.from_numpy(a.transpose(2, 0, 1).copy())[None]
            base = tensor(baseline.astype(np.float64))
            clean = tensor(target)
            support = torch.from_numpy(mask.copy())[None, None]
            patch = np.zeros((256, 256), np.uint8)
            for x, y in np.floor(case['landmarks5_canvas_xy']).astype(int):
                patch[max(0, y-12):min(256, y+12), max(0, x-12):min(256, x+12)] = 1
            patch &= cv2.erode(mask.astype(np.uint8), np.ones((13, 13), np.uint8),
                               borderType=cv2.BORDER_CONSTANT, borderValue=0)
            feature = torch.from_numpy(patch.astype(bool))[None, None]
            clear = torch.tensor([case['profile'] == 'clear'], dtype=torch.bool)
            teacher, oracle = projected_teacher(base, clean, support, clear)
            losses = {str(fraction): {key: float(value[0]) for key, value in
                      correction_terms(teacher * fraction, teacher, support, feature).items()}
                      for fraction in plan['fractions']}
            assert all(value == 0 for value in losses['1.0'].values())
            row = {'id': case['id'], 'reference_id': case['reference_id'], 'source': case['source'],
                   'profile': case['profile'], 'saved_original_npz': name,
                   'teacher_mean_maximum_absolute': float(teacher.sum((2, 3)).abs().max() / support.sum()),
                   'teacher_maximum_absolute': float(teacher.abs().max()),
                   'target_has_nonzero_correction': bool(teacher.abs().max() > 0),
                   'teacher_f64_CHW_sha256': hashlib.sha256(teacher.numpy().tobytes()).hexdigest(),
                   'oracle_f32_CHW_sha256': hashlib.sha256(oracle.float().numpy().tobytes()).hexdigest(),
                   'padding_exact': bool(torch.equal(oracle.masked_select(~support), base.masked_select(~support))),
                   'losses': losses}
            assert row['padding_exact'] and row['teacher_mean_maximum_absolute'] < 1e-12
            rows.append(row)
            if (slot + 1) % 25 == 0:
                print({'cases_checked': slot + 1, 'of': 145}, flush=True)
    calibration_rows = [r for r in rows if r['id'] in calibration and r['profile'] != 'clear']
    assert len(calibration_rows) == 40
    normalizers = {key: max(plan['normalizer_floor'], float(np.mean([r['losses']['0.0'][key]
                   for r in calibration_rows]))) for key in TERMS}
    groups = []
    for source in ['all', *sorted({r['source'] for r in rows})]:
        for profile in ['all', 'clear', 'degraded', 'blur_lr24', 'lowlight_lr32', 'motion_lr48', 'compound_lr24']:
            subset = [r for r in rows if (source == 'all' or source == r['source']) and
                      (profile == 'all' or profile == r['profile'] or
                       (profile == 'degraded' and r['profile'] != 'clear'))]
            assert subset
            groups.append({'source': source, 'profile': profile, 'cases': len(subset),
                           'initial_raw_terms': {key: float(np.mean([r['losses']['0.0'][key] for r in subset]))
                                                 for key in TERMS}})
    result = {'complete': True, 'plan_sha256': sha(OUT / 'plan.json'), 'rows': rows,
              'groups': groups, 'normalizers_initial40_degraded': normalizers,
              'source_bindings': bindings, 'seconds': time.monotonic() - start,
              'neural_model_forwards': 0, 'gradient_queries': 0, 'backwards': 0, 'optimizer_updates': 0,
              'teacher_access_is_training_only': True, 'unavailable_as_a_deployment_output': True,
              'model_capacity_or_preservation_pass': False, 'new_training_protocol_prepared': False,
              'native_DEV_or_reserved_final_used': False, 'app_promoted': False, 'goal_complete': False}
    write(OUT / 'analysis.json', result)
    print({'complete': True, 'cases': 145, 'normalizers': normalizers,
           'seconds': result['seconds'], 'neural_or_training_calls': 0}, flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if OUT.exists() and not (OUT / 'failure.json').exists():
            write(OUT / 'failure.json', {'complete': False, 'error': repr(exc),
                  'neural_forwards': 0, 'gradient_queries': 0, 'optimizer_updates': 0})
        raise
