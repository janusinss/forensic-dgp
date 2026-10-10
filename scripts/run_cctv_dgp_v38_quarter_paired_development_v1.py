"""All520 existing paired photographic development cases; inference only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import convolve1d
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cctv_dgp_pilot import FixedObservedIdentity, grid112, exported_pixel_metrics, state_hash
from cctv_dgp_v38_quarter_inference_v1 import load_v38_quarter

BASE = ROOT / 'outputs/cctv_dgp_generalization_vm_v15'
PARENT = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'
OUT = ROOT / 'outputs/cctv_dgp_v38_quarter_paired_development_v1'
ORIGINAL = PARENT / 'weights/dgp_v2.pth'
FINAL = ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_return/projected_displacement.npy'
ARMS = ['resize', 'original_dgp', 'v38_quarter']
V15_SHA = '6c0e1de5bcb56b755b82a3289adb4905c591f49908f02365ad6ad5fbd151f9da'


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def pixels(path, mode='RGB'):
    with Image.open(path) as im: return np.asarray(im.convert(mode)).copy()


def tensor(image): return torch.from_numpy(image.astype(np.float32) / np.float32(255)).permute(2, 0, 1)[None]


def verify_bindings(bindings):
    for name, digest in bindings.items():
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT) and sha(path) == digest, name


def feature_support(mask, landmarks):
    interior = cv2.erode(mask.astype(np.uint8), np.ones((13, 13), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    out = np.zeros((256, 256), bool)
    points = np.asarray(landmarks, np.float64).reshape(5, 2)
    for x, y in np.floor(points).astype(int):
        out[max(0, y - 12):min(256, y + 12), max(0, x - 12):min(256, x + 12)] = True
    out &= interior; assert out.any()
    return out


def detail_metric(png, target, feature):
    z = np.arange(-6, 7, dtype=np.float64); k = np.exp(-.5 * (z / 2)**2); k /= k.sum()
    luma = np.asarray([.299, .587, .114])
    def high(a):
        y = (a.astype(np.float64) / 255 * luma).sum(2)
        return y - convolve1d(convolve1d(y, k, axis=0, mode='reflect'), k, axis=1, mode='reflect')
    return float(np.square(high(png) - high(target))[feature].mean())


def summarize(rows):
    result = {}
    sources, profiles = sorted({r['source'] for r in rows}), sorted({r['profile'] for r in rows})
    keys = ['all', 'clear', 'degraded'] + [s + '/' + p for s in sources for p in ['all', 'degraded', *profiles]]
    for key in keys:
        chosen = [r for r in rows if key in ['all', 'clear' if r['profile'] == 'clear' else 'degraded',
                  r['source'] + '/all', r['source'] + ('/clear' if r['profile'] == 'clear' else '/degraded'),
                  r['source'] + '/' + r['profile']]]
        assert chosen
        result[key] = {arm: {'cases': len(chosen), **{
            metric: float(np.mean([r['metrics'][arm][metric] for r in chosen]))
            for metric in ['MSE', 'SSIM', 'ArcFace_observed_fixed', 'landmark_high_frequency_MSE']}}
            for arm in ARMS}
    assert len(result) == 17
    return result


def regressions(groups):
    failed = []
    for key, arms in groups.items():
        b, a = arms['original_dgp'], arms['v38_quarter']
        for metric in ['MSE', 'SSIM', 'ArcFace_observed_fixed']:
            bad = a[metric] > b[metric] + 1e-12 if metric == 'MSE' else a[metric] < b[metric] - 1e-6
            if bad: failed.append({'group': key, 'metric': metric, 'original': b[metric], 'v38_quarter': a[metric]})
    gains = {key: 1 - arms['v38_quarter']['landmark_high_frequency_MSE'] / arms['original_dgp']['landmark_high_frequency_MSE']
             for key, arms in groups.items() if key == 'degraded' or key.endswith('/degraded')}
    return failed, gains


def prepare():
    assert not OUT.exists(), 'Preserve any previous/partial development comparison'
    assert sha(BASE / 'generalization_protocol_v15.json') == V15_SHA
    p = read(BASE / 'generalization_protocol_v15.json')
    assert len(p['references']) == 104 and len(p['cases']) == 520 and len(p['training_identity_inventory']) == 781
    for key in ['id', 'source_sha256', 'target_rgb_sha256']:
        assert not ({r[key] for r in p['references']} & {r[key] for r in p['training_identity_inventory']})
    verify_bindings({(BASE / name).relative_to(ROOT).as_posix(): digest for name, digest in p['assets_sha256'].items()})
    audit = ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_independent_audit.json'
    parity = ROOT / 'outputs/cctv_dgp_v38_quarter_single_input_parity_v1/results.json'
    native = ROOT / 'outputs/cctv_dgp_v38_quarter_native_development_v1/visual_review.json'
    assert read(audit)['finite_probe_complete'] and read(parity)['complete'] and read(native)['cases_reviewed'] == 24
    paths = [Path(__file__), ROOT / 'scripts/verify_cctv_dgp_v38_quarter_paired_development_v1.py',
             ROOT / 'dgp_mean_centered_inference_v29.py', ROOT / 'scripts/cctv_dgp_v38_quarter_inference_v1.py', ROOT / 'cctv_dgp_pilot.py',
             ROOT / 'dgp_face_restoration.py', ROOT / 'dgp_frozen_inference_v2.py', ROOT / 'cctv_dgp_frozen_norm.py',
             audit, parity, native, ORIGINAL, FINAL, ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_return/theta_before.npy', ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_return/protocol.json', ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_analysis_v1/independent_analysis_page_audit.json', PARENT / 'weights/w600k_r50.onnx',
             BASE / 'generalization_protocol_v15.json', BASE / 'protocol.sha256',
             ROOT / 'outputs/cctv_paired_regression_v1/frozen_protocol.json', ROOT / 'outputs/cctv_paired_regression_v1/input_review.json']
    paths += sorted((ROOT / 'models').glob('*.py'))
    for r in p['references']: paths += [BASE / r[k] for k in ['target', 'observed', 'identity_target']]
    paths += [BASE / c['input'] for c in p['cases']]
    bindings = {x.relative_to(ROOT).as_posix(): sha(x) for x in paths}
    before = read(PARENT / 'protocol.json')['assets_sha256']['weights/w600k_r50.onnx']
    assert sha(PARENT / 'weights/w600k_r50.onnx') == before
    OUT.mkdir()
    write(OUT / 'plan.json', {
        'format': 'V38-quarter-all520-paired-development-v1', 'date': '2026-10-08',
        'frozen_before_new_outputs': True, 'original_V15_protocol_sha256': V15_SHA,
        'cases': p['cases'], 'references': p['references'], 'arms': ARMS,
        'preview_reference_ids': p['preview_reference_ids'], 'sources_sha256': bindings,
        'original741_assets_verified': True,
        'common_input': 'Existing exact256 RGB photographic target camera proxies; canonical NumPy float32/255. One input per model call, exact shared support. No extra alignment or display processing.',
        'selection': 'Single .25 disposable state selected from TRAIN before new development outputs; all104 references xfive profiles. No development-based step or checkpoint selection; no actual training here.',
        'paired_metrics': 'DeliveredPNG agreement with each processed photographic target. Observed MSE/PSNR, SSIM on7x7 eroded support; no native CCTV PSNR/SSIM.',
        'landmark_high_frequency_metric': 'Same fixed Gaussian13 sigma2 luma highpass squared error on union24x24 target-landmark patches within13x13 eroded observed support.',
        'recognizer': 'Frozen same verified ArcFace encoder. Reference matrix and observed112 support fixed across all arms. HQ counterpart where existing V15 declares one; never output-based alignment.',
        'regression_rule': 'Retain all17 group MSE increases>1e-12 and SSIM/ArcFace drops>1e-6 relative to original; these unchanged tolerances are diagnostics and not app qualification.',
        'source_labels': ['dataset/asian_faces', 'dataset/thumbnails128x128'],
        'split_limit': p['split_limit'],
        'identity_overlap_limits': 'Exact file/target-content disjointness from conditioning training, not proven person-disjointness from original historical DGP/pretrained exposure. Previously exposed development, not independent final testing.',
        'Phase3_pretrained_comparison_scope': 'Separate already-reviewed native24 includes Phase3 and declared CodeFormer on identical inputs; this broader520 cohort compares resize and original/V38-quarter own DGP only. Do not combine cohorts or claim520 CodeFormer comparisons.',
        'budget': {'wall_seconds_including_loading': 1200, 'threads': 4, 'device': 'cpu',
                   'original_DGP_forwards': 520, 'candidate_DGP_forwards': 520,
                   'recognizer_target_forwards': 104, 'recognizer_output_forwards': 1560},
        'no_optimizer_or_gradient_work': True, 'native_pixels_used': False,
        'reserved_final_used': False, 'ethnicity_inferred': False, 'zamboanga_validation': False,
        'independent_final_review': False, 'app_changed': False, 'goal_complete': False,
    })
    print(json.dumps({'prepared': True, 'paired_cases': 520, 'plan_sha256': sha(OUT / 'plan.json')}))


def run():
    assert not (OUT / 'execution.json').exists(), 'No automatic repeat'
    p = read(OUT / 'plan.json'); verify_bindings(p['sources_sha256'])
    torch.set_num_threads(4); cv2.setNumThreads(4); start = time.monotonic()
    model, provenance = load_v38_quarter(ORIGINAL)
    recognizer = FixedObservedIdentity(PARENT / 'weights/w600k_r50.onnx', 'cpu')
    states = {n: state_hash(getattr(model, n)) for n in ['original', 'candidate']}; states['recognizer'] = state_hash(recognizer)
    counts = {'original_DGP': 0, 'candidate_DGP': 0, 'recognizer_target': 0, 'recognizer_output': 0}
    handles, rows, references = [], [], {}
    for key, net in [('original_DGP', model.original), ('candidate_DGP', model.candidate)]:
        def hook(module, args, out, key=key): counts[key] += 1
        handles.append(net.register_forward_hook(hook))
    write(OUT / 'execution.json', {'plan_sha256': sha(OUT / 'plan.json'), 'states_before': states,
          'provenance': provenance, 'torch': torch.__version__, 'numpy': np.__version__,
          'optimizer_constructed': False, 'local_gradient_calls': 0, 'device': 'cpu'})
    for f in ['raw', 'images', 'embeddings']: (OUT / f).mkdir()
    try:
        with torch.inference_mode():
            for r in p['references']:
                assert time.monotonic() - start <= 1200
                target = pixels(BASE / r['target']); support = pixels(BASE / r['observed'], 'L') > 0
                identity_target = pixels(BASE / r['identity_target'])
                s = torch.from_numpy(support)[None, None].float(); grid = torch.from_numpy(grid112(r['matrix112']))[None]
                truth = recognizer.embedding(tensor(identity_target), s, grid)[0].numpy().copy(); counts['recognizer_target'] += 1
                np.save(OUT / ('embeddings/' + r['id'] + '_target.npy'), truth, allow_pickle=False)
                references[r['id']] = {'target': target, 'mask': support, 's': s, 'grid': grid, 'truth': truth,
                                      'feature': feature_support(support, r['landmarks5'])}
            for c in p['cases']:
                assert time.monotonic() - start <= 1200
                ref = references[c['reference_id']]; image = pixels(BASE / c['input']); mask = ref['mask']
                parts = model.forward_components(tensor(image), ref['s'].bool())
                raw = {k: v[0].permute(1, 2, 0).numpy().copy() for k, v in parts.items()}
                for key, value in raw.items(): np.save(OUT / ('raw/' + c['id'] + '_' + key + '.npy'), value, allow_pickle=False)
                images = {'resize': image}
                for arm, key in [('original_dgp', 'original_raw'), ('v38_quarter', 'result')]:
                    png = np.floor(raw[key] * np.float32(255)).astype(np.uint8); png[~mask] = image[~mask]; images[arm] = png
                outputs, scores = {}, {}
                for arm in ARMS:
                    png = images[arm]; name = 'images/' + c['id'] + '_' + arm + '.png'; Image.fromarray(png).save(OUT / name); outputs[arm] = name
                    vector = recognizer.embedding(tensor(png), ref['s'], ref['grid'])[0].numpy().copy(); counts['recognizer_output'] += 1
                    np.save(OUT / ('embeddings/' + c['id'] + '_' + arm + '.npy'), vector, allow_pickle=False)
                    metric = exported_pixel_metrics(png, ref['target'], mask)
                    metric['ArcFace_observed_fixed'] = float(vector @ ref['truth'])
                    metric['landmark_high_frequency_MSE'] = detail_metric(png, ref['target'], ref['feature']); scores[arm] = metric
                shift = (raw['result'] - raw['original_raw'])[mask].astype(np.float64).mean(0)
                mean = np.clip(raw['original_raw'].astype(np.float64) + shift, 0, 1).astype(np.float32)
                mean_png = np.floor(mean * np.float32(255)).astype(np.uint8); mean_png[~mask] = image[~mask]
                rows.append({'id': c['id'], 'reference_id': c['reference_id'], 'source': c['source'], 'profile': c['profile'],
                             'outputs': outputs, 'metrics': scores, 'postclip_mean_RGB_shift': shift.tolist(),
                             'constant_mean_shift_only_MSE': exported_pixel_metrics(mean_png, ref['target'], mask)['MSE']})
                if len(rows) % 20 == 0:
                    print(json.dumps({'paired_cases': len(rows), 'of': 520, 'seconds': time.monotonic() - start}), flush=True)
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14); sheets = []
        lookup = {r['id']: r for r in rows}
        for index, rid in enumerate(p['preview_reference_ids'], 1):
            cases = [c for c in p['cases'] if c['reference_id'] == rid]; assert len(cases) == 5
            sheet = Image.new('RGB', (1072, 1516), '#f1f2f4'); draw = ImageDraw.Draw(sheet); cells = []
            draw.text((12, 5), 'Paired DEVELOPMENT | ' + rid + ' | ' + cases[0]['source'], font=font, fill='black')
            for col, arm in enumerate(ARMS + ['paired_target']): draw.text((12 + col * 264, 30), arm, font=font, fill='black')
            for ri, c in enumerate(cases):
                y = 64 + ri * 288; draw.text((12, y), c['profile'] + ' | ' + c['id'], font=font, fill='black')
                paths = [OUT / lookup[c['id']]['outputs'][a] for a in ARMS]
                target_ref = next(r for r in p['references'] if r['id'] == rid); paths.append(BASE / target_ref['target'])
                for col, path in enumerate(paths):
                    value = pixels(path); xy = [12 + col * 264, y + 24]; sheet.paste(Image.fromarray(value), tuple(xy))
                    cells.append({'id': c['id'], 'column': col, 'xy': xy, 'path': path.relative_to(ROOT).as_posix(),
                                  'pixel_sha256': hashlib.sha256(value.tobytes()).hexdigest()})
            name = 'preview_' + str(index).zfill(2) + '_' + rid + '.png'; sheet.save(OUT / name)
            sheets.append({'file': name, 'cells': cells})
        after = {n: state_hash(getattr(model, n)) for n in ['original', 'candidate']}; after['recognizer'] = state_hash(recognizer)
        assert after == states and counts == {'original_DGP': 520, 'candidate_DGP': 520, 'recognizer_target': 104, 'recognizer_output': 1560}
        assert all(not m.training for net in [model, recognizer] for m in net.modules())
        assert all(not t.requires_grad for net in [model, recognizer] for t in net.parameters())
        assert time.monotonic() - start <= 1200; verify_bindings(p['sources_sha256'])
        groups = summarize(rows); failures, gains = regressions(groups)
        b, a = groups['degraded']['original_dgp']['MSE'], groups['degraded']['v38_quarter']['MSE']
        mean_only = float(np.mean([r['constant_mean_shift_only_MSE'] for r in rows if r['profile'] != 'clear']))
        brightness = max(0, b - mean_only) / max(b - a, 1e-12)
        artifacts = {x.relative_to(OUT).as_posix(): sha(x) for x in OUT.rglob('*') if x.is_file()}
        write(OUT / 'results.json', {'complete': True, 'plan_sha256': sha(OUT / 'plan.json'), 'rows': rows, 'groups': groups,
              'diagnostic_preservation_failures': failures, 'degraded_structure_gain_fraction': gains,
              'brightness_gain_fraction': brightness, 'sheets': sheets, 'artifacts_sha256': artifacts,
              'states_before_after': after, 'model_forwards': counts, 'seconds': time.monotonic() - start,
              'paired_photographic_development_not_native': True, 'all520_cases_retained': True,
              'local_gradient_calls': 0, 'optimizer_updates': 0, 'reserved_final_used': False,
              'visual_review_pending': True, 'independent_final_review': False,
              'app_changed': False, 'restoration_qualified': False, 'goal_complete': False})
        print(json.dumps({'complete': True, 'cases': 520, 'failures': len(failures), 'gains': gains, 'seconds': time.monotonic() - start}), flush=True)
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
