"""Independent saved-array/pixel/state audit; no neural or gradient replay."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image
from scipy.special import logsumexp
from skimage.metrics import structural_similarity


def audit(root, parent, parent_results, expected_sha, out, receipt):
    sys.path.insert(0, str(parent)); sys.path.insert(0, str(root))
    from cctv_dgp_direct_codes_v14 import verify, require, sha, read, write, PARENT_RESULTS, DESIGN
    p = verify(root, parent, expected_sha); start = time.monotonic()
    require(sha(parent_results / 'results.json') == PARENT_RESULTS, 'Parent returned results differ')
    pr = read(parent_results / 'results.json'); pp = read(parent / 'face_code_fit_protocol_v12.json')
    r = read(out / 'results.json'); n = read(out / 'neural_execution_receipt.json')
    require(r['complete'] and n['complete'] and r['protocol_sha256'] == n['protocol_sha256'] == expected_sha,
            'Incomplete/different returned result')
    require(all(r[k] is False for k in ['native_used','validation_used','native_reserved_used',
                                      'production_promoted','checkpoint_selected','best_checkpoint_created']), 'Wrong scope')
    require(n['frozen_before'] == n['frozen_after'], 'Recorded frozen state changed')
    expected = {'dgp': 50, 'direct_conditioner': 1151, 'prior_encoder': 0,
                'prior_transformer_head': 0, 'prior_generator': 454,
                'unused_v11_conditioner': 0, 'recognizer': 450}
    require(r['counts'] == n['counts'] == n['expected_counts'] == expected, 'Recorded forward counts differ')
    require(r['optimizer_updates'] == n['optimizer_updates'] == 1000 and
            r['backward_calls'] == n['backward_calls'] == 1001 and r['training_exposures'] == 2000, 'Fitting counts differ')
    require(r['seconds'] <= 600 and r['peak_vram_bytes'] == n['peak_vram_bytes'] <= DESIGN['peak_vram_cap_bytes'], 'Budget differs')
    for name, pin in r['artifacts_sha256'].items():
        require((out / name).resolve().is_relative_to(out.resolve()) and sha(out / name) == pin, 'Changed/unsafe result artifact')
    for name, pin in read(out / 'used_parent_artifacts.json').items():
        require(pin == pr['artifacts_sha256'][name] == sha(parent_results / name), 'Changed inherited array/pixel')
    preflight = read(out / 'cuda_preflight.json')
    require(preflight['complete'] and preflight['backward_calls'] == 1 and preflight['optimizer_updates'] == 0 and
            preflight['head_state_unchanged'] and preflight['frozen_states_unchanged'], 'Gradient preflight differs')
    require(all(v is not None and np.isfinite(v) for v in preflight['gradient_norms'].values()) and
            all(preflight['gradient_norms'][k] > 0 for k in ['code_projection.weight','code_projection.bias',
                'stats_projection.weight','stats_projection.bias']), 'Both heads not reached')
    require(len(n['baseline_parity']) == 100 and all(v['png_equal'] and v['maximum_float_difference'] <= 2e-6
            for v in n['baseline_parity']), 'Recorded starting parity differs')
    require(read(out / 'timing.json')['projected_total_seconds'] <= 600, 'Timing gate failed')
    trace = read(out / 'update_trace.json')['steps']
    line_trace = [json.loads(s) for s in (out / 'update_trace.jsonl').read_text().splitlines()]
    schedule = read(root / 'schedule_v14.json')['steps']
    require(trace == line_trace and len(trace) == len(schedule) == 1000, 'Update trace missing')
    for i, (step, actual) in enumerate(zip(schedule, trace), 1):
        require(actual['update'] == i and actual['case_ids'] == step['case_ids'] and actual['epoch'] == step['epoch'], 'Balanced schedule differs')
        require(np.isclose(actual['loss'], actual['code_ce'] + actual['mean_mse'] + actual['logstd_mse'],
                           rtol=2e-6, atol=2e-6), 'Trace loss arithmetic differs')
        require(all(np.isfinite(actual[k]) for k in ['loss','gradient_norm_before_clip','seconds']) and
                0 <= actual['code_accuracy'] <= 1, 'Invalid update scalar')

    # Execute only the pinned arithmetic functions, not the parent neural imports.
    tree = ast.parse((parent / 'cctv_dgp_pilot.py').read_text())
    nodes = [v for v in tree.body if isinstance(v, ast.FunctionDef) and
             v.name in ['state_hash','exported_pixel_metrics']]
    require(len(nodes) == 2, 'Pinned metric/state helper differs')
    import torch
    ns = {'np': np, 'cv2': cv2, 'structural_similarity': structural_similarity, 'hashlib': hashlib}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'pinned_arithmetic_only', 'exec'), ns)
    weights = torch.load(parent / pp['weights']['prior'], map_location='cpu', weights_only=True)['params_ema']
    codebook = weights['quantize.embedding.weight'].numpy(); del weights

    def rgb(path):
        with Image.open(path) as im:
            require(im.mode == 'RGB' and im.size == (256, 256), 'Invalid RGB256')
            return np.asarray(im).copy()

    def feature_stats(array):
        v = array.astype(np.float64).reshape(1, 256, 256)
        return v.mean(-1), np.sqrt(v.var(-1, ddof=1) + 1e-5)

    def scalar(actual, expected, tolerance=2e-6):
        if expected is None or isinstance(expected, bool): require(actual == expected, 'Flag differs')
        else: require(np.isclose(actual, expected, rtol=tolerance, atol=tolerance), 'Saved scalar differs')

    refs = {v['id']: v for v in p['references']}; cases = {c['id']: c for c in p['cases']}
    target = {}; masks = {}; labels = {}; stats = {}; embeddings = {}
    for ref in p['references']:
        rid = ref['id']; target[rid] = rgb(parent / ref['target'])
        masks[rid] = np.asarray(Image.open(parent / ref['observed'])) > 0
        labels[rid] = np.load(parent_results / ('teacher/' + rid + '_codes.npy'), allow_pickle=False)
        features = np.load(parent_results / ('teacher/' + rid + '_features.npy'), allow_pickle=False)
        require(labels[rid].dtype == np.int64 and labels[rid].shape == (1, 256), 'Invalid teacher codes')
        np.testing.assert_array_equal(features, codebook[labels[rid]].reshape(1, 16, 16, 256).transpose(0, 3, 1, 2))
        stats[rid] = feature_stats(features)
        embeddings[rid] = np.load(parent_results / ('teacher/' + rid + '_embedding.npy'), allow_pickle=False)
    for c in p['cases']:
        a = np.load(out / ('dgp_cache/' + c['id'] + '.npy'), allow_pickle=False)
        require(a.dtype == np.float32 and a.shape == (1, 3, 256, 256) and np.isfinite(a).all() and
                0 <= a.min() <= a.max() <= 1, 'Invalid frozen-DGP cache')
    require([s['update'] for s in r['snapshots']] == [0, 300, 1000], 'Snapshots differ')
    summaries = {}; rows_by_stage = {}; pngs = probes = cosines = 0; initial = None; changed = []
    parent_rows = {(v['id'], v['fidelity']): v for v in read(parent_results / 'update0/metrics.json')['rows']}
    for snapshot in r['snapshots']:
        update = snapshot['update']; data = read(out / snapshot['metrics'])
        require(data['update'] == update, 'Snapshot update differs')
        state = torch.load(out / snapshot['checkpoint'], map_location='cpu', weights_only=True)
        require(sum(v.numel() for v in state.values()) == DESIGN['trainable_parameters'] and
                all(torch.isfinite(v).all() for v in state.values()), 'Head schema/nonfinite state')
        require(ns['state_hash'](state) == data['state_hash'], 'Checkpoint/state hash differs')
        if update == 0:
            initial = state
            require(data['state_hash'] == n['initial_state'] and all(torch.count_nonzero(state[k]) == 0
                for k in ['code_projection.weight','code_projection.bias','stats_projection.weight','stats_projection.bias']), 'Starting head is not zero')
        else:
            require(state.keys() == initial.keys(), 'Checkpoint schema changed')
            keys = [k for k in state if not torch.equal(state[k], initial[k])]
            require(all(any(k.startswith(prefix) for k in keys) for prefix in
                        ['image_features.','fusion.','code_projection.','stats_projection.']), 'Our heads/trunk did not change')
            changed.append({'update': update, 'changed_parameter_tensors': len(keys)})
            if update == 1000: require(data['state_hash'] == n['final_state'], 'Final checkpoint hash differs')
        expected_order = [(c['id'], mode) for c in p['cases'] for mode in DESIGN['statistics_modes']]
        require([(v['id'], v['statistics']) for v in data['rows']] == expected_order, 'Snapshot rows missing/order changed')
        rows_by_stage[update] = {(v['id'], v['statistics']): v for v in data['rows']}
        grouped = {}; seen = set()
        for row in data['rows']:
            cid = row['id']; c = cases[cid]; rid = c['reference_id']; mode = row['statistics']; mask = masks[rid]
            require(all(row[k] == c[k] for k in ['reference_id','source','profile']), 'Case metadata differs')
            raw = np.load(out / row['raw'], allow_pickle=False); image = rgb(out / row['prediction'])
            require(raw.dtype == np.float32 and raw.shape == (256, 256, 3) and np.isfinite(raw).all() and
                    0 <= raw.min() <= raw.max() <= 1, 'Invalid raw render')
            composed = (raw * 255).astype(np.uint8); composed[~mask] = rgb(parent / c['input'])[~mask]
            np.testing.assert_array_equal(image, composed)
            for key, value in ns['exported_pixel_metrics'](image, target[rid], mask).items(): scalar(row[key], value, 1e-7)
            vec = np.load(out / row['embedding'], allow_pickle=False)
            require(vec.dtype == np.float32 and vec.shape == (512,) and np.isclose(np.linalg.norm(vec), 1, atol=1e-5), 'Invalid recognition vector')
            scalar(row['ArcFace_observed_fixed'], float(np.clip(vec @ embeddings[rid], -1, 1)), 1e-7)
            pngs += 1; cosines += 1
            if cid not in seen:
                seen.add(cid); probes += 1
                arrays = {k: np.load(out / path, allow_pickle=False) for k, path in row['probes'].items()}
                l = arrays['logits']; require(l.dtype == np.float32 and l.shape == (1,256,1024) and np.isfinite(l).all(), 'Invalid logits')
                for key in ['mean','std','logstd']:
                    require(arrays[key].dtype == np.float32 and arrays[key].shape == (1,256) and np.isfinite(arrays[key]).all(), 'Invalid statistics')
                require(arrays['std'].min() > 0 and abs(arrays['logstd']).max() <= 8, 'Unsafe statistics')
                np.testing.assert_allclose(np.log(arrays['std'].astype(np.float64)), arrays['logstd'], rtol=2e-6, atol=2e-6)
                tokens = mask[::16, ::16].reshape(1,256)
                ce = logsumexp(l.astype(np.float64), axis=2) - np.take_along_axis(l.astype(np.float64), labels[rid][...,None], axis=2)[...,0]
                tm, ts = stats[rid]
                scalar(row['code_ce'], float(ce[tokens].mean()))
                scalar(row['code_accuracy'], float((l.argmax(2) == labels[rid])[tokens].mean()))
                scalar(row['mean_mse'], float(((arrays['mean'].astype(np.float64) - tm)**2).mean()))
                scalar(row['logstd_mse'], float(((arrays['logstd'].astype(np.float64) - np.log(ts))**2).mean()))
                if update == 0:
                    np.testing.assert_array_equal(l, np.load(parent_results / ('update0/code_probes/' + cid + '_logits.npy'), allow_pickle=False))
                    old_f = np.load(parent_results / ('update0/code_probes/' + cid + '_features.npy'), allow_pickle=False)
                    om, os = feature_stats(old_f)
                    np.testing.assert_allclose(arrays['mean'], om, rtol=2e-6, atol=2e-6)
                    np.testing.assert_allclose(arrays['std'], os, rtol=2e-6, atol=2e-6)
            else:
                first = rows_by_stage[update][(cid, 'observed')]
                require(row['probes'] == first['probes'], 'Rendering arms used different code/stat probes')
                for key in ['code_ce','code_accuracy','mean_mse','logstd_mse']: scalar(row[key], first[key], 1e-7)
            if update == 0 and mode != 'none':
                previous = parent_rows[(cid, 0.)]
                np.testing.assert_allclose(raw, np.load(parent_results / previous['raw'], allow_pickle=False), rtol=0, atol=2e-6)
                np.testing.assert_array_equal(image, rgb(parent_results / previous['prediction']))
            group_names = ['all', c['source']+'/all', c['source']+'/'+c['profile'], 'profile/'+c['profile']]
            if c['profile'] != 'clear': group_names += ['degraded', c['source']+'/degraded']
            for group in group_names: grouped.setdefault(mode+'/'+group, []).append(row)
        summaries[str(update)] = {}
        for group, rows in grouped.items():
            mse = float(np.mean([v['MSE'] for v in rows]))
            summaries[str(update)][group] = {'cases': len(rows), 'MSE': mse, 'PSNR': float(-10*np.log10(mse)) if mse else None,
                **{k: float(np.mean([v[k] for v in rows])) for k in ['SSIM','MAE','ArcFace_observed_fixed','code_ce','code_accuracy','mean_mse','logstd_mse']}}
    cells = 0
    for profile in p['profiles']:
        with Image.open(out / ('grids/' + profile + '_10_rows.png')) as im: sheet = np.asarray(im)
        require(sheet.shape == (2904,1820,3), 'Grid geometry differs')
        for i, ref in enumerate(p['references']):
            c = next(c for c in p['cases'] if c['reference_id'] == ref['id'] and c['profile'] == profile)
            cid = c['id']
            images = [rgb(parent / c['input']), rgb(out / rows_by_stage[0][(cid,'observed')]['prediction']),
                      rgb(out / rows_by_stage[300][(cid,'predicted')]['prediction'])]
            images.extend(rgb(out / rows_by_stage[1000][(cid, mode)]['prediction']) for mode in DESIGN['statistics_modes'])
            images.append(target[ref['id']])
            for j, image in enumerate(images):
                y = 52+i*288
                np.testing.assert_array_equal(sheet[y:y+256,2+j*260:258+j*260], image); cells += 1
    require(len(n['oracle_codebook_statistics_parity']) == 2, 'Oracle checks missing')
    for row in n['oracle_codebook_statistics_parity']:
        rid = row['reference_id']
        plain = np.load(out / ('oracle_parity/'+rid+'_none.npy'), allow_pickle=False)
        standardized = np.load(out / ('oracle_parity/'+rid+'_target_stats.npy'), allow_pickle=False)
        require(plain.shape == standardized.shape == (1,3,256,256), 'Oracle geometry differs')
        maximum = float(np.max(np.abs(plain-standardized)))
        require(maximum == row['maximum_float_difference'], 'Reported oracle decoder difference differs')
        q = np.load(out / ('oracle_parity/'+rid+'_codebook.npy'), allow_pickle=False)
        normalized = np.load(out / ('oracle_parity/'+rid+'_normalized.npy'), allow_pickle=False)
        mean = np.load(out / ('oracle_parity/'+rid+'_mean.npy'), allow_pickle=False)
        std = np.load(out / ('oracle_parity/'+rid+'_std.npy'), allow_pickle=False)
        np.testing.assert_array_equal(q, codebook[labels[rid]].reshape(1,16,16,256).transpose(0,3,1,2))
        tm, ts = feature_stats(q)
        np.testing.assert_allclose(mean, tm, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(std, ts, rtol=2e-6, atol=2e-6)
        bound = 8*np.finfo(np.float32).eps*max(1.,float(abs(q).max()))
        maximum_latent = float(abs(normalized-q).max())
        require(row['exact_codebook_statistics'] and maximum_latent == row['maximum_latent_difference'] and
                maximum_latent <= bound == row['latent_float32_bound'], 'Codebook-stat latent invariant differs')
        np.testing.assert_allclose(normalized, (q-mean[:,:,None,None])/std[:,:,None,None]*std[:,:,None,None]+mean[:,:,None,None],
                                   rtol=0, atol=bound)
    require((pngs,probes,cosines,cells) == (450,150,450,350), 'Audit case/count differs')
    result = {'complete': True, 'protocol_sha256': expected_sha, 'results_sha256': sha(out/'results.json'),
        'auditor_sha256': sha(Path(__file__)), 'pngs_checked': pngs, 'raw_renders_checked': pngs,
        'code_and_stats_probes_rebuilt': probes, 'embedding_cosines_rebuilt': cosines,
        'teacher_code_labels_checked': 2560, 'dgp_cache_arrays_checked': 50,
        'baseline_parity_renders_checked': 100, 'oracle_parity_arrays_checked': 12,
        'grid_cells_checked': cells, 'trace_updates_checked': 1000, 'exposures_checked': 2000,
        'checkpoint_changes': changed, 'summaries': summaries, 'seconds': time.monotonic()-start,
        'neural_forwards_in_audit': 0, 'local_backward_calls': 0, 'local_optimizer_updates': 0,
        'validation_used': False, 'native_used': False, 'production_promoted': False,
        'limitation': 'Rebuilds saved pixel/logit/stat/codebook arithmetic and recorded state/counts; no CUDA gradient, DGP, renderer or recognizer replay. Ten-face capacity is not generalization or CCTV usefulness.'}
    write(receipt, result)
    print(json.dumps({k:v for k,v in result.items() if k != 'summaries'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent-bundle', type=Path, required=True)
    parser.add_argument('--parent-results', type=Path)
    parser.add_argument('--expected-protocol-sha', required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    parent = args.parent_bundle.resolve()
    audit(args.root.resolve(), parent, args.parent_results.resolve() if args.parent_results else
          parent/'outputs/cctv_dgp_face_code_fit_v12', args.expected_protocol_sha,
          args.results.resolve(), args.receipt.resolve())
