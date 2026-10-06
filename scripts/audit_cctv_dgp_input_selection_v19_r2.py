"""Independent V19 source/PNG/raw/cosine/selection audit; optional 24 CPU replays."""
import argparse
import math
from pathlib import Path
import sys
import time


def audit(root, parent, r2, mixed, baseline, pin, out, receipt, replay_head=False):
    sys.path.insert(0, str(root))
    import cctv_dgp_input_selection_v19_r2 as q
    started = time.monotonic()
    p, v, old = q.verify(root, parent, r2, mixed, baseline, pin)
    assets = root / 'original_v19'
    from dgp_input_normalization_v19_r2 import spatial_cpu_replay_input
    import numpy as np
    from PIL import Image
    import torch
    from dgp_input_selector_v19 import select_restoration
    from dgp_structure_conditioner_v18 import DGPStructureResidualHead
    from cctv_dgp_pilot import state_hash
    from cctv_dgp_structure_audit_numeric_v18_r2 import verify_psnr_summaries, verify_preservation_report
    torch.set_num_threads(4)
    r = q.read(out / 'results.json')
    q.validate_result_scope(r, pin)
    def clock():
        q.require(time.monotonic() - started <= 300, 'V19 independent audit exceeds300s')
    for name, digest in r['artifacts_sha256'].items():
        clock(); q.require(q.sha(q.safe(out, name)) == digest, 'Returned V19 artifact differs:' + name)
    cases = {c['id']: c for c in p['cases']}
    refs = {r['id']: r for r in p['references']}
    train_cases = {c['id']: c for c in p['training_parity_cases']}
    train_refs = {r['id']: r for r in p['training_parity_references']}
    before = q.read(baseline / 'results.json')
    base_rows = {r['id']: r for r in before['rows']}
    def vec(path):
        value = np.load(path, allow_pickle=False)
        q.require(value.dtype == np.float32 and value.shape == (512,) and np.isfinite(value).all()
                  and abs(float(np.linalg.norm(value)) - 1) <= 2e-6, 'Embedding schema/norm differs')
        return value
    def support_for(ref):
        with Image.open(mixed / ref['observed']) as image: return np.asarray(image).copy() > 0
    def raw_for(name):
        value = np.load(q.safe(out, name), allow_pickle=False)
        q.require(value.dtype == np.float32 and value.shape == (256, 256, 3) and np.isfinite(value).all()
                  and 0 <= value.min() <= value.max() <= 1, 'Raw RGB256 schema/range differs')
        return value
    def decision_check(actual, camera, support):
        expected = select_restoration(camera, support)
        q.require(set(actual) == set(expected), 'Input selection schema differs')
        for key, value in expected.items():
            if key in ['laplacian_MSE', 'luminance_variance', 'laplacian_over_variance']:
                q.require(math.isfinite(actual[key]) and abs(actual[key] - value) <= 1e-12, 'Input feature differs:' + key)
            else: q.require(actual[key] == value, 'Frozen input-only decision differs:' + key)
        return expected
    truth = {rid: vec(out / ('target_embeddings/' + rid + '.npy')) for rid in refs}
    # Target embeddings are fresh: compare to the separately audited baseline
    # binding rather than silently reusing its reference vectors.
    for rid in refs:
        prior = vec(baseline / ('target_embeddings/' + rid + '.npy'))
        q.require(float(np.max(np.abs(truth[rid] - prior))) <= 2e-6, 'Fresh target embedding differs from baseline')
    parity = q.read(out / 'fresh_training_parity.json')
    q.require(parity['complete'] and parity['training'] is False
              and [i['id'] for i in parity['cases']] == [c['id'] for c in p['training_parity_cases']],
              'All50 fresh training parity cases required')
    control = {c['id']: c for c in q.read(assets / 'lineage/processing_results_v19.json')['decisions']}
    for item in parity['cases']:
        clock(); cid = item['id']; c = train_cases[cid]; ref = train_refs[c['reference_id']]
        camera = v.rgb(mixed / c['input']); support = support_for(ref)
        decision = decision_check(item['decision'], camera, support)
        q.require(decision['branch'] == control[cid]['branch'], 'Original training processing branch differs')
        for key, name, folder in [('retained_dgp_v2', item['base_raw'], 'dgp_base'),
                                  ('structure_v18_update600', item['terminal_raw'], 'structure_update600')]:
            q.require(name == ('parity/base/' if folder == 'dgp_base' else 'parity/terminal/') + cid + '.npy', 'Parity path differs')
            value = raw_for(name)
            prior = np.load(assets / ('parity/' + folder + '/' + cid + '.npy'), allow_pickle=False)
            delta = float(np.max(np.abs(value - prior)))
            q.require(delta == item['maximum_float_differences'][key] and delta <= 2e-6
                      and np.array_equal(v.png(value, camera, support), v.png(prior, camera, support)), 'Fresh raw/PNG parity differs')
    q.require([row['id'] for row in r['rows']] == [c['id'] for c in p['cases']], 'All520 fixed validation cases/order required')
    rows = {}
    counts = {'raw_PNG': 0, 'PNGs_metrics': 0, 'cosines': 0, 'input_decisions': 50,
              'internal_spatial_DGP_bases': 0, 'canonical_baseline_raw_previews': 0}
    for row in r['rows']:
        clock(); cid = row['id']; c = cases[cid]; ref = refs[c['reference_id']]
        q.require(row['role'] == 'validation' and all(row[k] == c[k] for k in ['reference_id', 'source', 'profile']), 'Development identity/role differs')
        q.validate_automatic_alias(row)
        q.require(row['internal_spatial_DGP_base_raw'] == 'internal_dgp_base/' + cid + '.npy', 'Internal spatial base path differs')
        raw_for(row['internal_spatial_DGP_base_raw'])
        counts['internal_spatial_DGP_bases'] += 1
        camera = v.rgb(mixed / c['input']); target = v.rgb(mixed / ref['target']); support = support_for(ref)
        decision = decision_check(row['decision'], camera, support)
        counts['input_decisions'] += 1
        for arm in q.ARMS[:-1]:
            item = row['arms'][arm]
            q.require(item['prediction'] == 'predictions/' + arm + '/' + cid + '.png'
                      and item['embedding'] == 'embeddings/' + arm + '/' + cid + '.npy', 'Arm PNG/embedding path differs')
            image = v.rgb(q.safe(out, item['prediction']))
            if arm in ['retained_dgp_v2', 'structure_v18_update600']:
                q.require(item['raw'] == 'raw/' + arm + '/' + cid + '.npy', 'Arm raw path differs')
                q.require(np.array_equal(image, v.png(raw_for(item['raw']), camera, support)), 'Raw/PNG composition differs')
                counts['raw_PNG'] += 1
            elif arm == 'basic_resizing':
                q.require(np.array_equal(image, camera), 'Resizing arm differs from prepared input')
            else:
                old_arm = base_rows[cid]['arms']['starting_prior_none']
                q.require(np.array_equal(image, v.rgb(baseline / old_arm['prediction'])), 'Declared pretrained baseline PNG differs')
            for key, expected in v.metrics(image, target, support).items():
                q.require(expected == item[key] if isinstance(expected, bool) or expected is None
                          else abs(expected - item[key]) <= 1e-9, 'PNG metric differs:' + arm + '/' + key)
            counts['PNGs_metrics'] += 1
            embedding = vec(q.safe(out, item['embedding']))
            if arm != 'structure_v18_update600':
                original = base_rows[cid]['input_metrics'] if arm == 'basic_resizing' else base_rows[cid]['arms'][
                    'retained_dgp_v2' if arm == 'retained_dgp_v2' else 'starting_prior_none']
                q.require(np.array_equal(embedding, vec(baseline / original['embedding'])) and item['embedding_reused'] is True,
                          'Audited baseline embedding reuse differs')
            else: q.require(item['embedding_reused'] is False, 'Terminal recognizer must be fresh')
            cosine = float(np.clip(embedding @ truth[c['reference_id']], -1, 1))
            q.require(abs(cosine - item['ArcFace_observed_fixed']) <= 1e-7, 'Cosine arithmetic differs')
            counts['cosines'] += 1
        q.require(np.array_equal(v.rgb(out / row['arms']['retained_dgp_v2']['prediction']),
                                 v.rgb(baseline / base_rows[cid]['arms']['retained_dgp_v2']['prediction'])), 'Fresh DGP baseline PNG differs')
        old_raw = base_rows[cid]['arms']['retained_dgp_v2'].get('raw')
        if old_raw is not None:
            prior_raw = np.load(baseline / old_raw, allow_pickle=False)
            delta = float(np.max(np.abs(raw_for(row['arms']['retained_dgp_v2']['raw']) - prior_raw)))
            q.require(delta <= 2e-6, 'Canonical DGP raw preview differs from frozen V15')
            counts['canonical_baseline_raw_previews'] += 1
        rows[cid] = row
    summaries = {arm: q.aggregate([{**{k: row[k] for k in ['source', 'profile']}, **row['arms'][arm]} for row in r['rows']]) for arm in q.ARMS}
    checks = {arm: verify_psnr_summaries(summary, r['summaries'][arm]) for arm, summary in summaries.items()}
    guards = {arm: old.strict_preservation(summaries[arm], summaries['retained_dgp_v2'])
              for arm in ['structure_v18_update600', 'automatic_v19']}
    for arm, expected in guards.items():
        checks[arm + '/preservation'] = verify_preservation_report(expected, r['preservation'][arm], summaries[arm], summaries['retained_dgp_v2'])
    timing = q.read(out / 'inference_timing.json')
    samples = timing['steady_sample_seconds']
    expected = timing['seconds'] + 500 * float(np.mean(samples)) * 1.25 + 60
    q.require(timing['cases'] == 20 and len(samples) == 19 and timing['cap_seconds'] == 1200
              and all(math.isfinite(t) and t > 0 for t in samples)
              and abs(expected - timing['projected_seconds']) <= 1e-6
              and 0 < timing['seconds'] <= timing['projected_seconds'] <= 1200, 'Timing sample/projection/cap differs')
    neural = q.read(out / 'neural_receipt.json'); execution = q.read(out / 'execution.json')
    parent_receipt = q.read(assets / 'lineage/v18_neural_receipt.json')
    q.require(neural['complete'] and neural['counts'] == q.expected_counts()
              and neural['before'] == neural['after'] == execution['states_before']
              and {key: neural['before'][key] for key in parent_receipt['frozen_before']} == parent_receipt['frozen_before']
              and neural['before']['spatial_decoder'] == p['terminal_state_hash']
              and neural['optimizer_constructed'] is False and execution['optimizer_constructed'] is False
              and neural['backward_calls'] == neural['optimizer_updates'] == 0
              and 0 < neural['peak_allocated_vram_bytes'] <= 20 * 1024**3, 'Inference/state/neural receipt differs')
    head = DGPStructureResidualHead().eval()
    state = torch.load(assets / 'weights/structure_update600.pth', map_location='cpu', weights_only=True)
    head.load_state_dict(state, strict=True)
    q.require(state_hash(head) == p['terminal_state_hash'], 'Terminal checkpoint state differs')
    first = [p['training_parity_references'][0]['id'], p['training_parity_references'][5]['id']]
    wanted = {(c['id'], 'train') for c in p['training_parity_cases'] if c['reference_id'] in first and c['profile'] in ['clear', 'blur_lr24']}
    wanted |= {(c['id'], 'validation') for c in p['cases'] if c['reference_id'] in p['preview_reference_ids'] and c['profile'] in ['clear', 'blur_lr24']}
    cached = neural['cached_head_cases']
    q.require(len(cached) == 24 and {(c['id'], c['role']) for c in cached} == wanted, 'All24 predeclared CPU cache probes required')
    max_delta, forwards = 0., 0
    for item in cached:
        clock(); cid = item['id']; c = train_cases[cid] if item['role'] == 'train' else cases[cid]
        q.require(item['file'] == 'cache_replay/' + cid + '.npz', 'CPU cache path differs')
        expected_raw = 'parity/terminal/' + cid + '.npy' if item['role'] == 'train' else rows[cid]['arms']['structure_v18_update600']['raw']
        q.require(item['raw'] == expected_raw, 'CPU replay target path differs')
        with np.load(q.safe(out, item['file']), allow_pickle=False) as arrays:
            q.require(set(arrays.files) == {'dgp_base', 'prior64'}, 'CPU cache schema differs')
            base, prior64 = arrays['dgp_base'].copy(), arrays['prior64'].copy()
        q.require(base.shape == (3, 256, 256) and prior64.shape == (256, 64, 64)
                  and all(x.dtype == np.float32 and np.isfinite(x).all() for x in [base, prior64])
                  and 0 <= base.min() <= base.max() <= 1, 'CPU cache tensors differ')
        expected_base = 'parity/base/' + cid + '.npy' if item['role'] == 'train' else rows[cid]['internal_spatial_DGP_base_raw']
        q.require(np.array_equal(base.transpose(1, 2, 0), raw_for(expected_base)), 'CPU cache/internal spatial base binding differs')
        if replay_head:
            camera = spatial_cpu_replay_input(v.rgb(mixed / c['input']))
            with torch.inference_mode():
                prediction = head(camera, torch.from_numpy(base)[None], torch.from_numpy(prior64)[None])
            delta = float(np.max(np.abs(prediction[0].permute(1, 2, 0).numpy() - raw_for(expected_raw))))
            q.require(delta <= 5e-5, 'CPU head replay exceeds frozen5e-5')
            max_delta = max(max_delta, delta); forwards += 1
    grids = ['grids/' + profile + '.png' for profile in v.PROFILES]
    q.require(r['grids'] == grids, 'All five fixed original-cell grids required')
    for profile, name in zip(v.PROFILES, grids):
        clock()
        with Image.open(out / name) as grid:
            q.require(grid.mode == 'RGB' and grid.size == (1560, 2904), 'Original256 grid dimensions differ')
            for index, rid in enumerate(p['preview_reference_ids']):
                c = next(c for c in p['cases'] if c['reference_id'] == rid and c['profile'] == profile)
                row = rows[c['id']]
                images = [v.rgb(mixed / c['input'])] + [v.rgb(out / row['arms'][a]['prediction']) for a in q.ARMS[1:]] + [v.rgb(mixed / refs[rid]['target'])]
                for col, expected in enumerate(images):
                    x, y = col * 260 + 2, 24 + index * 288 + 28
                    q.require(np.array_equal(np.asarray(grid.crop((x, y, x + 256, y + 256))), expected), 'Original grid cell differs')
    q.require(counts['internal_spatial_DGP_bases'] == 520 and counts['canonical_baseline_raw_previews'] == 50, 'All internal bases/fixed raw previews required')
    q.require(forwards == (24 if replay_head else 0) and state_hash(head) == p['terminal_state_hash'], 'CPU replay state/scope differs')
    clock()
    q.write(receipt, {'complete': True, 'protocol_sha256': pin, 'results_sha256': q.sha(out / 'results.json'),
        'validation_cases': 520, 'source_references': {'dataset/asian_faces': 51, 'dataset/thumbnails128x128': 53},
        'fresh_training_parity_cases': 50, 'automatic_aliases': 520, 'original_grid_cells': 300,
        'numeric_log10_checks': checks, **counts, 'head_forwards': forwards, 'maximum_head_replay_difference': max_delta,
        'DGP_prior_recognizer_forwards': 0, 'backward_calls': 0, 'optimizer_updates': 0,
        'scientific_guard_pass_required_for_export': False,
        'scope_limit': 'Saved provenance/arithmetic plus optional24 cached CPU decoder replays; no independent CUDA/recognizer execution replay. Development validation only, no native/reserved/final acceptance.',
        'seconds': time.monotonic() - started})
    print({'complete': True, 'PNGs_metrics': counts['PNGs_metrics'], 'CPU_head_forwards': forwards,
           'seconds': time.monotonic() - started}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'parent', 'r2', 'mixed', 'baseline', 'results', 'receipt']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--replay-head', action='store_true')
    a = parser.parse_args()
    audit(a.root.resolve(), a.parent.resolve(), a.r2.resolve(), a.mixed.resolve(), a.baseline.resolve(),
          a.expected_sha, a.results.resolve(), a.receipt.resolve(), a.replay_head)
