"""Audit24 frozen native development inputs and120 signal calculations; no model."""
import argparse
from collections import Counter
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cctv_dgp_pilot import read, write, sha
from face_workflow import decode_image, quality_signals, square_pad
from cctv_input_quality import observed_quality_signals, quality_for_crop
from dgp_face_restoration import prepare_crop

PLAN = ROOT / 'outputs/cctv_dgp_native_pilot_review_v1/frozen_plan.json'
PLAN_SHA = '475c2225a368fe035d8c02d62953e6cb32c836fdb71b7aa5a8704c6d276f3ded'
INPUT_ROOT = ROOT / 'outputs/cctv_native_development_v2'
SOURCE_FILES = ['scripts/audit_cctv_input_policy_v1.py', 'cctv_input_quality.py',
                'face_workflow.py', 'dgp_face_restoration.py', 'tests/test_cctv_input_quality.py']


def run(output):
    if output.exists():
        raise ValueError('Preserve existing input-policy audit')
    started = time.monotonic()
    if sha(PLAN) != PLAN_SHA:
        raise ValueError('Frozen native development plan changed')
    plan = read(PLAN)
    cases = plan['cases']
    if len(cases) != 24 or len({c['id'] for c in cases}) != 24 or any(c['role'] != 'development' for c in cases):
        raise ValueError('Fixed24 development inputs required')
    for name in ('outputs/cctv_native_development_v2/frozen_subset.json',
                 'outputs/cctv_native_development_v2/input_review.json'):
        if sha(ROOT / name) != plan['assets_sha256'][name]:
            raise ValueError('Input-only cohort/review changed')
    rows = []
    for case in cases:
        path = (INPUT_ROOT / case['source_file']).resolve()
        if not path.is_relative_to((INPUT_ROOT / 'development').resolve()) or sha(path) != case['source_sha256']:
            raise ValueError('Changed/escaped development input')
        with Image.open(path) as image:
            rgb = np.asarray(image.convert('RGB')).copy()
        review = plan['input_reviews'][case['id']]
        if review['source_sha256'] != case['source_sha256']:
            raise ValueError('Input review/source lineage changed')
        try:
            decoded = decode_image(path.read_bytes())
            if not np.array_equal(decoded, rgb):
                raise ValueError('Decoded native pixels differ')
            accepted, reason = True, None
        except ValueError as error:
            accepted, reason = False, str(error)
        zero = np.zeros(rgb.shape[:2], np.uint8)
        padded, square_mask, _ = square_pad(rgb, zero)
        legacy = quality_signals(padded, square_mask)
        _, observed, removal, geometry = prepare_crop(rgb, zero)
        # Both ablation canvases use the identical native square at127. Only
        # interpolation or the scoring support changes, one factor at a time.
        area = cv2.resize(padded, (256, 256), interpolation=cv2.INTER_AREA)
        bilinear = np.asarray(Image.fromarray(padded).resize((256, 256), Image.Resampling.BILINEAR)).copy()
        full = np.ones((256, 256), bool)
        variants = {
            'area_full_support': observed_quality_signals(area, full, removal),
            'area_observed_support': observed_quality_signals(area, observed, removal),
            'bilinear_full_support': observed_quality_signals(bilinear, full, removal),
            'bilinear_observed_support': observed_quality_signals(bilinear, observed, removal),
        }
        for key in ('blur_variance', 'noise_sigma_255', 'suggest_restoration', 'visible_signal_pixels'):
            if legacy[key] != variants['area_full_support'][key]:
                raise ValueError('Legacy control was not reproduced exactly')
        prepared = quality_for_crop(rgb, zero)
        rows.append({'id': case['id'], 'source_sha256': case['source_sha256'],
                     'source_file': case['source_file'], 'native_size': geometry['native_size'],
                     'input_structure_review': review['input_structure_review'], 'pose_review': review['pose_review'],
                     'active_decoder_accepted': accepted, 'decoder_rejection': reason,
                     'legacy_quality': legacy, 'controlled_variants': variants, 'dgp_prepared_quality': prepared})
    core = [r for r in rows if r['input_structure_review'] == 'coarse' and r['pose_review'] == 'frontal_or_mild_approximate']
    insufficient = [r for r in rows if r['input_structure_review'] == 'insufficient']
    variants = list(rows[0]['controlled_variants'])
    effects = {}
    for first, second in [('area_full_support', 'area_observed_support'),
                          ('bilinear_full_support', 'bilinear_observed_support'),
                          ('area_full_support', 'bilinear_full_support'),
                          ('area_observed_support', 'bilinear_observed_support')]:
        effects[first + ' -> ' + second] = {
            'restoration_suggestion_changed': [r['id'] for r in rows if r['controlled_variants'][first]['suggest_restoration'] != r['controlled_variants'][second]['suggest_restoration']],
            'mean_absolute_blur_variance_change': float(np.mean([abs(r['controlled_variants'][first]['blur_variance'] - r['controlled_variants'][second]['blur_variance']) for r in rows])),
        }
    report = {'complete': True, 'date': '2026-10-04', 'plan_sha256': PLAN_SHA,
              'source_code_sha256': {name: sha(ROOT / name) for name in SOURCE_FILES},
              'cases': 24, 'signal_calculations': 120, 'legacy_control_exact': True,
              'active_decoder_accepted': sum(r['active_decoder_accepted'] for r in rows),
              'active_decoder_rejected': sum(not r['active_decoder_accepted'] for r in rows),
              'core_coarse_frontal_mild': len(core),
              'core_rejected_by_32_pixel_floor': [r['id'] for r in core if not r['active_decoder_accepted']],
              'input_reviewed_insufficient': len(insufficient),
              'insufficient_accepted_by_decoder': [r['id'] for r in insufficient if r['active_decoder_accepted']],
              'legacy_restoration_suggestions': sum(r['legacy_quality']['suggest_restoration'] for r in rows),
              'dgp_prepared_restoration_suggestions': sum(r['dgp_prepared_quality']['suggest_restoration'] for r in rows),
              'controlled_suggestions': {name: sum(r['controlled_variants'][name]['suggest_restoration'] for r in rows) for name in variants},
              'controlled_effects': effects, 'rows': rows, 'seconds': time.monotonic() - started,
              'model_forwards': 0, 'backward_calls': 0, 'optimizer_updates': 0,
              'native_reserved_inputs_read': 0, 'application_default_changed': False,
              'structure_qualification_established': False, 'training_recipe_changed': False,
              'limitations': ['Development inputs only; no independent final/held-out qualification evidence.',
                              'Blur/noise scoring does not establish readable facial structure or valid face pose.',
                              'The helper is preparation for a reviewed DGP route; the active application still uses its historical decoder and quality policy.']}
    output.mkdir(parents=True)
    write(output / 'results.json', report)
    print({k: v for k, v in report.items() if k not in ('rows', 'source_code_sha256')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/cctv_input_policy_audit_v1')
    run(parser.parse_args().output)
