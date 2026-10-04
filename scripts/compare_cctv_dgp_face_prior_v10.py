"""Execute the separate frozen V10 feasibility plan; inference only on CPU."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cctv_dgp_face_prior_v10 as v


def run():
    v.require(not (v.OUT / 'run').exists(), 'Preserve completed/partial run; no repeat')
    p = v.verify()
    import numpy as np
    from PIL import Image, ImageDraw
    import torch
    from cctv_dgp_pilot import FixedObservedIdentity, aggregate, exported_pixel_metrics, grid112, state_hash
    from dgp_face_restoration import as_tensor, prepare_crop
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from pretrained_face_restoration import load_face_restorer
    torch.set_num_threads(4); torch.manual_seed(20261004)
    dgp, dgp_provenance = load_frozen_dgp_restorer(v.ROOT / v.DGP_WEIGHT,
        expected_sha256=p['assets_sha256'][v.DGP_WEIGHT], device='cpu')
    cf, cf_provenance = load_face_restorer(v.ROOT / v.CF_WEIGHT, 'cpu')
    identity = FixedObservedIdentity(v.ROOT / v.IDENTITY_WEIGHT, 'cpu')
    models = {'dgp': dgp, 'codeformer': cf, 'recognizer': identity}
    v.require(all(not m.training and not any(x.requires_grad for x in m.parameters()) for m in models.values()),
              'Only frozen inference permitted')
    before = {name: state_hash(m) for name, m in models.items()}
    counts = {name: 0 for name in models}
    def hook(name):
        def record(module, args, result): counts[name] += 1
        return record
    dgp.register_forward_hook(hook('dgp')); cf.register_forward_hook(hook('codeformer'))
    identity.encoder.register_forward_hook(hook('recognizer'))
    out = v.OUT / 'run'; out.mkdir()
    v.write(out / 'execution.json', {'plan_sha256': v.sha(v.OUT / 'frozen_plan.json'),
        'source_sha256': v.sha(Path(__file__)), 'model_state_before': before,
        'torch': torch.__version__, 'device': 'cpu', 'cpu_threads': 4,
        'optimizer_constructed': False, 'backward_calls': 0, 'optimizer_updates': 0,
        'native_reserved_used': False, 'provenance': {'dgp': dgp_provenance, 'codeformer': cf_provenance},
        'budget': p['budget']})
    started = time.monotonic(); artifacts = {}; paired_rows = []; native_rows = []; target_embeddings = {}
    def clock():
        if time.monotonic() - started > 1200: raise TimeoutError('V10 inference cap1200s exceeded')
    def save_image(name, value):
        file = out / name; file.parent.mkdir(parents=True, exist_ok=True)
        v.require(not file.exists(), 'No artifact overwrite')
        Image.fromarray(value).save(file); artifacts[name] = v.sha(file)
        return name
    def save_float(name, value):
        file = out / name; file.parent.mkdir(parents=True, exist_ok=True)
        with file.open('xb') as stream: np.save(stream, value, allow_pickle=False)
        artifacts[name] = v.sha(file); return name
    def prior(value):
        clock()
        with torch.inference_mode():
            return cf(as_tensor(value), fidelity=1.0)[0].permute(1, 2, 0).numpy().copy()
    def embedding(value, mask, matrix):
        clock()
        with torch.inference_mode():
            return identity.embedding(as_tensor(value), torch.from_numpy(mask.astype(np.float32))[None, None],
                torch.from_numpy(grid112(matrix))[None])[0].numpy().copy()
    def observed(path):
        with Image.open(path) as im: result = np.asarray(im).copy() > 0
        v.require(result.shape == (256, 256) and result.any(), 'Wrong observed mask')
        return result
    for index, case in enumerate(p['paired_cases'], 1):
        clock(); files = case['files']; input_rgb = v.rgb(v.ROOT / files['input'])
        support = observed(v.ROOT / files['observed']); target = v.rgb(v.ROOT / files['target'])
        outputs = {arm: v.rgb(v.ROOT / files[arm]) for arm in p['paired_arms'][:3]}
        raw = {'codeformer_input_w1': prior(input_rgb),
               'codeformer_dgp_w1_diagnostic': prior(outputs['cached_v9_epoch20_diagnostic'])}
        rid = case['reference_id']
        if rid not in target_embeddings:
            target_embeddings[rid] = embedding(target, support, case['matrix112'])
            save_float('targets/' + rid + '.npy', target_embeddings[rid])
        for arm, value in raw.items(): outputs[arm] = v.png(value, input_rgb, support)
        rows = {}
        for arm, value in outputs.items():
            v.require(np.array_equal(value[~support], input_rgb[~support]), 'Padding pixels differ')
            vec = embedding(value, support, case['matrix112'])
            row = {**{key: case[key] for key in ['id', 'reference_id', 'source', 'profile']},
                **exported_pixel_metrics(value, target, support),
                'ArcFace_observed_fixed': float(np.clip(vec @ target_embeddings[rid], -1, 1)),
                'prediction': save_image('paired/' + arm + '/images/' + case['id'] + '.png', value),
                'embedding': save_float('paired/' + arm + '/embeddings/' + case['id'] + '.npy', vec)}
            if arm in raw: row['raw_float'] = save_float('paired/' + arm + '/raw/' + case['id'] + '.npy', raw[arm])
            rows[arm] = row
        paired_rows.append({'id': case['id'], 'arms': rows})
        if index == 5:
            elapsed = time.monotonic() - started
            projection = elapsed / 5 * 50 + elapsed / 10 * 24 + 90
            v.write(out / 'timing_at5.json', {'paired_cases': 5, 'measured_seconds': elapsed,
                'projected_seconds': projection, 'cap_seconds': 1200,
                'interpretation': 'First5 include10 prior calls and26 recognizer calls; native allowance uses one prior per case.'})
            if projection > 1200: raise TimeoutError('V10 timing projection exceeds1200-second cap')
        if index % 5 == 0: print(f'paired {index}/50 elapsed={time.monotonic()-started:.0f}s', flush=True)
    for index, case in enumerate(p['native_cases'], 1):
        clock()
        with Image.open(v.ROOT / case['files']['source']) as im: native = np.asarray(im.convert('RGB')).copy()
        common, support, _, geometry = prepare_crop(native)
        v.require(np.array_equal(common, v.rgb(v.ROOT / case['files']['input'])), 'Native input differs from cache')
        with torch.inference_mode(): raw_dgp = dgp(as_tensor(common))[0].permute(1, 2, 0).numpy().copy()
        dgp_png = v.png(raw_dgp, common, support)
        values = {'input': common, 'v9_epoch20_diagnostic': dgp_png}
        raws = {'v9_epoch20_diagnostic': raw_dgp,
                'codeformer_dgp_w1_diagnostic': prior(dgp_png)}
        for arm in ['cached_phase3', 'cached_codeformer_w1']:
            raw = np.load(v.ROOT / case['files'][arm], allow_pickle=False)
            values[arm] = v.png(raw, common, support)
        values['codeformer_dgp_w1_diagnostic'] = v.png(raws['codeformer_dgp_w1_diagnostic'], common, support)
        images = {arm: save_image('native/' + case['id'] + '_' + arm + '.png', value) for arm, value in values.items()}
        rawfiles = {arm: save_float('native/' + case['id'] + '_' + arm + '.npy', value) for arm, value in raws.items()}
        native_rows.append({'id': case['id'], 'geometry': geometry, 'images': images, 'raw_float': rawfiles,
            'input_review': case['input_review'], 'PSNR': None, 'SSIM': None, 'identity_accuracy': None})
        if index % 6 == 0: print(f'native {index}/24 elapsed={time.monotonic()-started:.0f}s', flush=True)
    def grid(name, ids, arms, rows, target=False):
        cols = len(arms) + int(target); sheet = Image.new('RGB', (cols * 260, 24 + len(ids) * 288), '#eeeeee')
        draw = ImageDraw.Draw(sheet)
        for j, label in enumerate(arms + (['reference'] if target else [])): draw.text((j*260+2, 3), label, fill='black')
        for i, cid in enumerate(ids):
            y = 24 + i * 288; row = rows[cid]
            names = [row['arms'][arm]['prediction'] for arm in arms] if target else [row['images'][arm] for arm in arms]
            for j, image_name in enumerate(names):
                draw.text((j*260+2, y+2), cid, fill='black'); sheet.paste(v.rgb(out/image_name), (j*260+2, y+28))
            if target:
                case = next(c for c in p['paired_cases'] if c['id'] == cid)
                draw.text((len(arms)*260+2, y+2), cid, fill='black')
                sheet.paste(v.rgb(v.ROOT/case['files']['target']), (len(arms)*260+2, y+28))
        save_image(name, np.asarray(sheet))
    paired_map = {r['id']: r for r in paired_rows}; native_map = {r['id']: r for r in native_rows}
    grids = []
    for profile in v.PROFILES:
        ids = [ref['id'] + '_' + profile for ref in p['references']]
        name = 'paired_' + profile + '_10_rows.png'; grid(name, ids, p['paired_arms'], paired_map, True); grids.append(name)
    for index in range(4):
        name = 'native_gallery_' + str(index+1) + '.png'
        grid(name, [r['id'] for r in native_rows[index*6:(index+1)*6]], p['native_arms'], native_map); grids.append(name)
    core = [r['id'] for r in native_rows if r['input_review']['pose_review'] == 'frontal_or_mild_approximate'
            and r['input_review']['input_structure_review'] == 'coarse']
    v.require(len(core) == 6, 'Native core cohort differs')
    name = 'native_preview_10_rows.png'
    grid(name, core + [r['id'] for r in native_rows if r['id'] not in core][:4], p['native_arms'], native_map); grids.append(name)
    after = {name: state_hash(m) for name, m in models.items()}
    v.require(before == after, 'Inference changed model state')
    v.require(counts == {'dgp': 24, 'codeformer': 124, 'recognizer': 260}, 'Finite forward count differs')
    clock(); v.verify()
    results = {'complete': True, 'format': 'dgp-face-prior-feasibility-v10',
        'plan_sha256': v.sha(v.OUT / 'frozen_plan.json'), 'seconds_after_loading': time.monotonic() - started,
        'model_forwards': counts, 'model_state_after': after, 'model_states_unchanged': before == after,
        'paired_rows': paired_rows, 'paired_summary': {arm: aggregate([r['arms'][arm] for r in paired_rows]) for arm in p['paired_arms']},
        'native_rows': native_rows, 'native_core_ids': core, 'grids': grids, 'artifacts_sha256': artifacts,
        'optimizer_updates': 0, 'backward_calls': 0, 'native_reserved_used': False,
        'training': False, 'production_promoted': False, 'checkpoint_selected': False,
        'independent_final_review_pending': True, 'useful_output_established': False,
        'limitation': 'Feasibility comparison uses a failed DGP snapshot as diagnostic conditioning and a declared pretrained prior. Paired photography proxies and unpaired native CCTV are separate. No new trained prior adapter, native ground-truth or identity accuracy.'}
    v.write(out/'results.json', results)
    print({'complete': True, 'seconds': results['seconds_after_loading'], 'forwards': counts, 'training': False}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true'); parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.prepare: v.prepare()
    elif args.run:
        try: run()
        except Exception as error:
            output = v.OUT/'run'
            if output.exists() and not (output/'failure.json').exists():
                v.write(output/'failure.json', {'complete': False, 'error_type': type(error).__name__, 'error': str(error),
                    'optimizer_updates': 0, 'backward_calls': 0, 'resume_permitted': False})
            raise
    else: parser.error('Choose --prepare or --run')
