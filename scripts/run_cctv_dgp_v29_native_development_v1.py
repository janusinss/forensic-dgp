"""Frozen V29 comparison on the already input-reviewed24 native development crops."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cctv_dgp_pilot import state_hash
from cctv_input_quality import observed_quality_signals
from dgp_mean_centered_inference_v29 import load_mean_centered_dgp_v29

BASE = ROOT / 'outputs/cctv_chokepoint_native_development_v1'
CACHE = ROOT / 'outputs/cctv_chokepoint_native_comparison_v1'
OUT = ROOT / 'outputs/cctv_dgp_v29_native_development_v1'
ORIGINAL = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27/weights/dgp_v2.pth'
FINAL = ROOT / 'outputs/cctv_dgp_mean_centered_decoder_v29_return/outputs/update800/dgp_candidate_v29.pth'
ARMS = ['resize', 'phase3', 'original_dgp', 'codeformer_w1', 'v29', 'v29_auto']


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def pixels(path, mode='RGB'):
    with Image.open(path) as im: return np.asarray(im.convert(mode)).copy()


def verify_bindings(bindings):
    for name, digest in bindings.items():
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT) and sha(path) == digest, name


def prepare():
    assert not OUT.exists(), 'Preserve any previous/partial native comparison'
    audit = ROOT / 'outputs/cctv_dgp_mean_centered_decoder_v29_independent_audit.json'
    review = ROOT / 'outputs/cctv_dgp_mean_centered_decoder_v29_visual_review/visual_review.json'
    parity = ROOT / 'outputs/cctv_dgp_v29_single_input_parity_v1/results.json'
    assert read(audit)['necessary_capacity_pass'] and read(review)['cases_reviewed'] == 50 and read(parity)['complete']
    old, old_result = read(CACHE / 'plan.json'), read(CACHE / 'results.json')
    assert read(CACHE / 'saved_output_audit.json')['complete'] and old_result['complete']
    verify_bindings(old['sources_sha256'])
    verify_bindings({(CACHE / name).relative_to(ROOT).as_posix(): h for name, h in old_result['artifacts_sha256'].items()})
    subset, input_review = read(BASE / 'frozen_subset.json'), read(BASE / 'input_review.json')
    cases = [c for c in subset['cases'] if c['role'] == 'development']
    assert cases == old['cases'] and len(cases) == 24 and input_review['usable_cases'] == 24
    assert input_review['reviewed_before_model_outputs'] and input_review['subset_sha256'] == sha(BASE / 'frozen_subset.json')
    assert subset['reserved_crops_rendered'] == subset['reserved_image_pixels_decoded'] == 0
    paths = [Path(__file__), ROOT / 'scripts/verify_cctv_dgp_v29_native_development_v1.py',
             ROOT / 'dgp_mean_centered_inference_v29.py', ROOT / 'cctv_dgp_pilot.py',
             ROOT / 'cctv_input_quality.py', ROOT / 'dgp_face_restoration.py',
             ROOT / 'dgp_frozen_inference_v2.py', ROOT / 'cctv_dgp_frozen_norm.py',
             audit, review, parity, ORIGINAL, FINAL,
             CACHE / 'plan.json', CACHE / 'results.json', CACHE / 'execution.json', CACHE / 'saved_output_audit.json',
             BASE / 'selection_plan.json', BASE / 'frozen_subset.json', BASE / 'input_review.json',
             BASE / 'independent_geometry_audit.json', BASE / 'LICENSE_SOURCE.html', BASE / 'DERIVATIVE_NOTICE.txt',
             ROOT / 'CCTV_NATIVE_SOURCE_EXTENSION_V1.md', ROOT / 'CCTV_BENCHMARK_STATUS.md',
             ROOT / 'outputs/cctv_native_development_v2/input_review.json']
    paths += sorted((ROOT / 'models').glob('*.py'))
    for c, r in zip(cases, old_result['records']):
        assert c['id'] == r['id']
        paths += [BASE / c[k] for k in ['native_crop', 'input', 'observed']]
        paths += [CACHE / r['outputs'][a] for a in ['resize', 'phase3', 'identity_v2', 'codeformer_w1']]
        paths += [CACHE / ('raw/' + c['id'] + '_' + a + '.npy') for a in ['phase3', 'identity_v2', 'codeformer_w1']]
    bindings = {x.relative_to(ROOT).as_posix(): sha(x) for x in paths}
    OUT.mkdir()
    (OUT / 'LICENSE_SOURCE.html').write_bytes((BASE / 'LICENSE_SOURCE.html').read_bytes())
    (OUT / 'DERIVATIVE_NOTICE.txt').write_text(
        'Modified noncommercial thesis research crops and restoration estimates; not released originals or recovered identity. '
        'Retain LICENSE_SOURCE.html. Acknowledge NICTA and Wong et al., CVPR Workshops2011, DOI10.1109/CVPRW.2011.5981881.\n',
        encoding='utf-8', newline='\n')
    write(OUT / 'plan.json', {
        'format': 'V29-native-ChokePoint-development-v1', 'date': '2026-10-06',
        'frozen_before_new_model_outputs': True, 'cases': cases, 'arms': ARMS,
        'common_input': 'Exact previously input-reviewed256 RGB crop; NumPy float32/255 before device transfer; same observed support. No extra alignment, degradation, sharpening or colour enhancement.',
        'cached_arms': {'phase3': old['phase3'], 'original_dgp': old['primary_dgp'], 'codeformer_w1': old['pretrained_comparison']},
        'cache_policy': 'Reuse independently audited raw tensors and exact PNGs on identical inputs; fresh original DGP replay must match cached raw within1e-5. Saved-output audit does not claim a fresh Phase3/CodeFormer neural replay.',
        'v29_policy': 'Audited final800 only plus freshly evaluated frozen same-input original; the training mean-centering path is part of inference.',
        'PNG_policy': 'floor(raw*float32(255)), observed-only composition, exact original padding; no display enhancement.',
        'automatic_selection': 'Unchanged input-only blur<24 or noise>=8 suggestion; exact alias of input or V29. No detector-based usable-face claim and no threshold fitting.',
        'criteria': ['Compare eyes, nose, mouth, outline and overall visible appearance together against input and unchanged DGP.',
                     'Useful structure can be soft; contrast alone does not qualify a result. Record altered glasses, hair, gaze, expression or feature geometry.',
                     'Keep input-only usability decisions; no output-based exclusions or relabeling usable input as insufficient because the model is soft.',
                     'Review every24 case and all144 displayed cells; no cherry-picked best crop or temporal clean-reference claim.',
                     'Development impressions do not establish recovered identity, independent final review or app promotion.'],
        'budget': {'original_DGP_forwards': 24, 'candidate_DGP_forwards': 24, 'wall_seconds': 180, 'threads': 4, 'device': 'cpu'},
        'sources_sha256': bindings, 'source': 'ChokePoint/P1E_S1_C1', 'capture_country': 'Unspecified by acquired metadata',
        'native_ROI_range': [[93, 112], [189, 227]], 'release_camera_frame_size': [800, 600],
        'development_identities': 12, 'reserved_ChokePoint_identities': 13,
        'source_specific_label_overlap_with_reserved': 0,
        'cross_source_historical_pretrained_person_overlap': 'Unknown',
        'Asian_capture_priority_limit': 'QMUL includes Asian source datasets but per-image capture source is unmapped; no country or ethnicity inferred from this separate ChokePoint sequence.',
        'QMUL_original_input_rejections_retained': True,
        'paired_reference': False, 'PSNR': None, 'SSIM': None, 'identity_accuracy': None,
        'reserved58_final_cases45_namespaced_identities_used': False,
        'source_country_inferred': False, 'ethnicity_inferred': False, 'zamboanga_validation': False,
        'optimizer_updates': 0, 'local_gradient_calls': 0, 'app_changed': False, 'goal_complete': False,
    })
    print(json.dumps({'prepared': True, 'cases': 24, 'new_DGP_forwards': 48, 'plan_sha256': sha(OUT / 'plan.json')}))


def run():
    assert not (OUT / 'execution.json').exists(), 'No automatic repeat of completed/failed inference'
    p = read(OUT / 'plan.json'); verify_bindings(p['sources_sha256'])
    torch.set_num_threads(4); start = time.monotonic()
    counts = {'original_DGP': 0, 'candidate_DGP': 0}; rows = []; handles = []
    model, provenance = load_mean_centered_dgp_v29(ORIGINAL, FINAL)
    before = {n: state_hash(getattr(model, n)) for n in ['original', 'candidate']}
    for key, net in [('original_DGP', model.original), ('candidate_DGP', model.candidate)]:
        def hook(module, args, out, key=key): counts[key] += 1
        handles.append(net.register_forward_hook(hook))
    write(OUT / 'execution.json', {'plan_sha256': sha(OUT / 'plan.json'), 'states_before': before,
          'provenance': provenance, 'torch': torch.__version__, 'numpy': np.__version__,
          'models_frozen': True, 'local_gradient_calls': 0, 'optimizer_constructed': False})
    for folder in ['raw', 'images']: (OUT / folder).mkdir()
    cache_rows = {r['id']: r for r in read(CACHE / 'results.json')['records']}
    try:
        for c in p['cases']:
            assert time.monotonic() - start <= p['budget']['wall_seconds']
            image = pixels(BASE / c['input']); support = pixels(BASE / c['observed'], 'L') > 0
            x = torch.from_numpy(image.astype(np.float32) / np.float32(255)).permute(2, 0, 1)[None]
            s = torch.from_numpy(support)[None, None]
            parts = model.forward_components(x, s)
            raw = {k: v[0].permute(1, 2, 0).numpy().copy() for k, v in parts.items()}
            ref = np.load(CACHE / ('raw/' + c['id'] + '_identity_v2.npy'), allow_pickle=False)
            replay_error = float(np.abs(raw['original_raw'] - ref).max()); assert replay_error <= 1e-5
            for key, value in raw.items(): np.save(OUT / ('raw/' + c['id'] + '_' + key + '.npy'), value, allow_pickle=False)
            images = {'resize': image}
            for key, old_arm in [('phase3', 'phase3'), ('original_dgp', 'identity_v2'), ('codeformer_w1', 'codeformer_w1')]:
                images[key] = pixels(CACHE / cache_rows[c['id']]['outputs'][old_arm])
            images['v29'] = np.floor(raw['result'] * np.float32(255)).astype(np.uint8)
            images['v29'][~support] = image[~support]
            signals = observed_quality_signals(image, support, np.zeros_like(support))
            chosen = 'v29' if signals['suggest_restoration'] else 'resize'
            images['v29_auto'] = images[chosen]
            outputs = {}
            for arm, value in images.items():
                assert np.array_equal(value[~support], image[~support])
                name = 'images/' + c['id'] + '_' + arm + '.png'
                Image.fromarray(value).save(OUT / name); outputs[arm] = name
            rows.append({'id': c['id'], 'source_person_id': c['source_person_id'],
                         'native_size': [c['native_width'], c['native_height']], 'outputs': outputs,
                         'fresh_original_cache_maximum_error': replay_error, 'auto_selected': chosen,
                         'quality_signals': signals, 'input_only_usable': True})
            print(json.dumps({'native_case': len(rows), 'of': 24, 'id': c['id'], 'seconds': time.monotonic() - start}), flush=True)
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14)
        sheets = []
        for first in range(0, 24, 4):
            sheet = Image.new('RGB', (1604, 1220), '#f1f2f4'); draw = ImageDraw.Draw(sheet); cells = []
            draw.text((8, 4), 'Native unpaired ChokePoint development | V29 final800 | no clean reference', font=font, fill='black')
            for col, arm in enumerate(ARMS): draw.text((8 + col * 266, 27), arm, font=font, fill='black')
            for ri, row in enumerate(rows[first:first + 4]):
                y = 57 + ri * 290
                draw.text((8, y), row['id'] + ' / native' + str(row['native_size']) + ' / Auto=' + row['auto_selected'], font=font, fill='black')
                for col, arm in enumerate(ARMS):
                    xy = [8 + col * 266, y + 24]; value = pixels(OUT / row['outputs'][arm])
                    sheet.paste(Image.fromarray(value), tuple(xy))
                    cells.append({'id': row['id'], 'arm': arm, 'xy': xy, 'path': row['outputs'][arm],
                                  'pixel_sha256': hashlib.sha256(value.tobytes()).hexdigest()})
            name = 'comparison-' + str(first // 4 + 1).zfill(2) + '.png'; sheet.save(OUT / name)
            sheets.append({'file': name, 'cells': cells})
        after = {n: state_hash(getattr(model, n)) for n in before}
        assert after == before and counts == {'original_DGP': 24, 'candidate_DGP': 24}
        assert all(not m.training for m in model.modules()) and all(not t.requires_grad for t in model.parameters())
        assert time.monotonic() - start <= p['budget']['wall_seconds']; verify_bindings(p['sources_sha256'])
        artifacts = {x.relative_to(OUT).as_posix(): sha(x) for x in OUT.rglob('*') if x.is_file()}
        write(OUT / 'results.json', {'complete': True, 'plan_sha256': sha(OUT / 'plan.json'), 'rows': rows,
              'sheets': sheets, 'artifacts_sha256': artifacts, 'seconds': time.monotonic() - start,
              'states_before_after': after, 'fresh_neural_forwards': counts,
              'cached_phase3_codeformer_neural_replay': False,
              'native_evidence_unpaired': True, 'PSNR': None, 'SSIM': None, 'identity_accuracy': None,
              'local_gradient_calls': 0, 'optimizer_updates': 0, 'reserved_final_used': False,
              'visual_review_pending': True, 'independent_final_review': False,
              'app_changed': False, 'restoration_qualified': False, 'goal_complete': False})
        print(json.dumps({'complete': True, 'cases': 24, 'seconds': time.monotonic() - start, 'forwards': counts}), flush=True)
    except Exception as e:
        write(OUT / 'failure.json', {'error': str(e), 'type': type(e).__name__, 'cases_completed': len(rows),
                                   'forwards': counts, 'seconds': time.monotonic() - start,
                                   'local_gradient_calls': 0, 'optimizer_updates': 0})
        raise
    finally:
        for h in handles: h.remove()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--prepare', action='store_true'); a = ap.parse_args()
    prepare() if a.prepare else run()
