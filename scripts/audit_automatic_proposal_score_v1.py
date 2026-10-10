"""Independent saved-array/pixel checks and two layout-exact fresh forward replays."""
import argparse
import math
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image
import torch
from automatic_proposal_score_v1_common import ROOT, OUT, sha, read, write, rgb, binary, bindings, state_sha


def compare(a, b):
    if isinstance(b, dict):
        assert a.keys() == b.keys()
        for key in b: compare(a[key], b[key])
    elif isinstance(b, list):
        assert len(a) == len(b)
        for x, y in zip(a, b): compare(x, y)
    elif isinstance(b, float): assert math.isfinite(a) and abs(a-b) <= 1e-12
    else: assert a == b


def independent_stats(values):
    values = values.astype(np.float64)
    if not len(values): return {'pixels': 0, 'minimum': None, 'maximum': None, 'mean': None,
        'q10': None, 'median': None, 'q90': None, 'fraction_ge_0_5': None}
    return {'pixels': len(values), 'minimum': float(np.min(values)), 'maximum': float(np.max(values)),
        'mean': float(np.sum(values)/len(values)), 'q10': float(np.percentile(values, 10)),
        'median': float(np.percentile(values, 50)), 'q90': float(np.percentile(values, 90)),
        'fraction_ge_0_5': int(np.count_nonzero(values >= .5))/len(values)}


def marked(source, selected, color):
    target = source.copy()
    target[selected] = np.floor(source[selected].astype(np.float64)*.55+np.asarray(color)*.45+.5).astype(np.uint8)
    return target


def main():
    started = time.monotonic(); parser = argparse.ArgumentParser(); parser.add_argument('--protocol-sha', required=True)
    pin = parser.parse_args().protocol_sha; assert sha(OUT/'protocol.json') == pin
    p = read(OUT/'protocol.json'); bindings(p); r = read(OUT/'results.json')
    assert r['complete'] and r['protocol_sha256'] == pin and r['state_before'] == r['state_after']
    assert r['counts'] == {'detector': 36, 'internal_completion': 0} and r['seconds'] <= 180
    for key in ['DGP_forwards', 'completion_forwards', 'gradient_queries', 'backward_calls', 'optimizer_updates', 'generation_requests']:
        assert r[key] == 0
    outer = read(OUT/'run_external_receipt.json'); assert outer['complete'] and outer['exit_code'] == 0 and not outer['timeout']
    assert outer['protocol_sha256'] == pin and outer['log_sha256'] == sha(OUT/'run.log')
    assert not r['native_or_final_used'] and not r['app_adoption'] and not r['goal_complete']
    assert [row['id'] for row in r['rows']] == [c['id'] for c in p['cases']]
    expected = {folder+'/'+c['id']+suffix for c in p['cases'] for folder, suffix in
        [('stages', '.npz'), ('stages', '.json'), ('masks', '_raw.png'), ('masks', '_proposal.png')]}
    expected |= {'pages/page_'+format(i, '02d')+'.png' for i in range(1, 10)}
    assert set(r['artifact_sha256']) == expected and len(expected) == 153
    for name, digest in r['artifact_sha256'].items(): assert sha(OUT/name) == digest
    assert sum((OUT/name).stat().st_size for name in expected) <= p['budgets']['artifact_bytes']
    rechecked = {}; logistic_error = 0.
    for case, row in zip(p['cases'], r['rows']):
        assert time.monotonic()-started <= 180
        assert read(OUT/'stages'/(case['id']+'.json')) == row
        source = rgb(ROOT/case['input']); support = {}
        if case.get('masks'):
            masks = {key: binary(ROOT/name) for key, name in case['masks'].items()}
            support = {'assisted_core': masks['core'], 'visible_face': masks['face'] & ~masks['removal'],
                'protected_appearance': masks['protected'], 'outside_face': ~masks['face']}
        with np.load(OUT/'stages'/(case['id']+'.npz'), allow_pickle=False) as z:
            assert set(z.files) == {'neural_input', 'logits', 'network_probability', 'canvas_probability'}
            x, logits, probability, canvas = [z[key].copy() for key in ['neural_input', 'logits', 'network_probability', 'canvas_probability']]
        assert all(v.dtype == np.float32 and np.isfinite(v).all() for v in [x, logits, probability, canvas])
        size = r['detector_input_size']; assert x.shape == (1, 3, size, size) and logits.shape == (1, 1, size, size)
        assert probability.shape == (size, size) and canvas.shape == (256, 256)
        assert ((probability >= 0) & (probability <= 1)).all()
        sample = cv2.resize(source, (size, size), interpolation=cv2.INTER_LINEAR)
        with torch.inference_mode():
            expected_input = torch.from_numpy(sample.copy()).permute(2, 0, 1).float()[None]/255
        assert np.array_equal(x, expected_input.numpy()) and list(expected_input.stride()) == row['input_strides']
        # Independent stable sigmoid calculation: float64 arithmetic, then float32.
        stable = np.exp(-np.logaddexp(0., -logits[0, 0].astype(np.float64))).astype(np.float32)
        logistic_error = max(logistic_error, float(np.abs(stable-probability).max())); assert logistic_error <= 2e-7
        assert np.array_equal(canvas, cv2.resize(probability, (256, 256), interpolation=cv2.INTER_LINEAR))
        raw = canvas >= .5; proposed = cv2.dilate(raw.astype(np.uint8), np.ones((7, 7), np.uint8)) != 0
        assert np.array_equal(raw, binary(OUT/'masks'/(case['id']+'_raw.png')))
        assert np.array_equal(proposed, binary(OUT/'masks'/(case['id']+'_proposal.png')))
        assert np.array_equal(proposed, binary(ROOT/case['automatic']))
        assert row['raw_pixels'] == int(raw.sum()) and row['proposal_pixels'] == int(proposed.sum())
        assert row['rejected_input_retained'] == case['rejected'] and row['input_review'] == case['input_review']
        compare(row['whole_canvas_scores'], independent_stats(canvas.reshape(-1)))
        compare(row['regions'], {name: independent_stats(canvas[mask]) for name, mask in support.items()})
        assert row['proposal_region_pixels'] == {name: int(np.count_nonzero(proposed & mask)) for name, mask in support.items()}
        rechecked[case['id']] = row
    for pair in r['pairs']:
        a, b = [rechecked[pair[key]] for key in ['original_id', 'degraded_id']]
        assert a['base_id'] == b['base_id'] == pair['base_id']
        for field in ['proposal_pixels', 'raw_pixels']:
            assert pair[field+'_original'] == a[field] and pair[field+'_degraded'] == b[field]
        assert abs(pair['maximum_score_difference']-(b['whole_canvas_scores']['maximum']-a['whole_canvas_scores']['maximum'])) <= 1e-12
        compare(pair['region_mean_score_differences'], {key: None if a['regions'][key]['mean'] is None else
            b['regions'][key]['mean']-a['regions'][key]['mean'] for key in a['regions']})
    assert len(r['pairs']) == 18 and len({v['base_id'] for v in r['pairs']}) == 18
    for index in range(9):
        page = rgb_page = np.array(Image.open(OUT/'pages'/('page_'+format(index+1, '02d')+'.png')).convert('RGB'))
        assert page.shape == (1200, 1280, 3)
        for slot, case in enumerate(p['cases'][index*4:index*4+4]):
            source = rgb(ROOT/case['input']); automatic = binary(ROOT/case['automatic'])
            with np.load(OUT/'stages'/(case['id']+'.npz'), allow_pickle=False) as z: score = z['canvas_probability']
            gray = np.floor(score*255+.5).astype(np.uint8)
            core = binary(ROOT/case['masks']['core']) if case.get('masks') else binary(ROOT/case['reviewed'])
            protected = binary(ROOT/case['masks']['protected']) if case.get('masks') else np.zeros((256, 256), bool)
            cells = [source, marked(source, automatic, [16, 185, 129]), np.repeat(gray[..., None], 3, 2),
                     marked(source, core, [255, 140, 20]), marked(source, protected, [190, 80, 255])]
            y = 52+slot*292
            for col, cell in enumerate(cells): assert np.array_equal(page[y:y+256, col*256:col*256+256], cell)
    # Exact replay of the predetermined pair with the actual production input layout.
    from completion_inference import load_completion
    torch.set_num_threads(4); model, _ = load_completion(ROOT/'outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth', 'cpu')
    model.eval().requires_grad_(False); before = state_sha(model); assert before == r['state_before']
    counts = [0]; handle = model.segmenter.register_forward_hook(lambda *_: counts.__setitem__(0, counts[0]+1))
    with torch.inference_mode():
        for cid in p['replay_ids']:
            with np.load(OUT/'stages'/(cid+'.npz'), allow_pickle=False) as z:
                x, logits = z['neural_input'].copy(), z['logits'].copy()
            row = rechecked[cid]; replay = torch.empty_strided(tuple(x.shape), tuple(row['input_strides']), dtype=torch.float32)
            replay.copy_(torch.from_numpy(x)); assert list(replay.stride()) == row['input_strides']
            fresh = model.detect(replay); assert list(fresh.stride()) == row['logit_strides']
            assert np.array_equal(fresh.numpy(), logits)
    handle.remove(); assert counts == [2] and before == state_sha(model)
    bindings(p); assert time.monotonic()-started <= 180
    write(OUT/'independent_audit.json', {'complete': True, 'protocol_sha256': pin, 'results_sha256': sha(OUT/'results.json'),
        'source_bindings_verified': len(p['sources_sha256']), 'artifacts_verified': 153, 'exact256_page_cells': 180,
        'all36_proposals_match_cached_current_app': True, 'all_score_and_region_arithmetic_verified': True,
        'paired_records_verified': 18, 'maximum_logistic_arithmetic_error': logistic_error,
        'fresh_exact_detector_replays': 2, 'fresh_replay_input_layout_preserved': True,
        'state_before': before, 'state_after': state_sha(model), 'DGP_forwards': 0, 'completion_forwards': 0,
        'gradient_queries': 0, 'optimizer_updates': 0, 'native_or_final_used': False,
        'visual_review_pending': True, 'app_adoption': False, 'goal_complete': False,
        'checker_sha256': sha(Path(__file__)), 'seconds': time.monotonic()-started})
    print({'complete': True, 'exact_detector_replays': 2, 'page_cells': 180, 'seconds': time.monotonic()-started}, flush=True)


if __name__ == '__main__': main()
