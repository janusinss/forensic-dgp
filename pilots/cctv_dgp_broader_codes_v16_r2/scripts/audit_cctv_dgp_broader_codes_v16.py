"""Independent return arithmetic/count/provenance audit; no torch or neural imports."""
import argparse
import json
from pathlib import Path
import sys
import time


def audit_failure(root, parent, mixed, baseline, pin, out, supervisor_failure, receipt):
    """Audit the completed prefix of a failed run without treating it as a result."""
    sys.path.insert(0, str(root))
    import cctv_dgp_broader_codes_v16 as v
    import math
    p = v.verify(root, parent, mixed, baseline, pin)
    if (out / 'cache_timing.json').is_file():
        from cctv_dgp_cache_timing_v16_r2 import verify_cache_timing
        verify_cache_timing(p, v.read(out / 'cache_timing.json'))
    failure = v.read(supervisor_failure)
    v.require(failure['complete'] is False and failure['protocol_sha256'] == pin and
              failure['resume_permitted'] is False, 'Failure protocol/scope differs')
    partial = v.read(out / 'failure.json') if (out / 'failure.json').is_file() else None
    traces = []
    if (out / 'update_trace.jsonl').is_file():
        with (out / 'update_trace.jsonl').open(encoding='utf-8') as stream:
            # Interrupted write is reported explicitly; all completed records still get checked.
            lines = stream.readlines()
        for i, line in enumerate(lines):
            try: traces.append(json.loads(line))
            except json.JSONDecodeError:
                v.require(i == len(lines) - 1 and not line.endswith('\n'), 'Corrupt completed trace record')
    steps = v.read(root / 'schedule_v16.json')['steps']; v.require(len(traces) <= 3128, 'Failure exceeds update cap')
    for step, row in zip(steps, traces):
        v.require(all(row[k] == step[k] for k in ['update', 'epoch', 'case_ids']) and
                  all(math.isfinite(row[k]) for k in ['loss', 'code_ce', 'code_accuracy', 'gradient_norm_before_clip', 'seconds']) and
                  row['code_ce'] >= 0 and abs(row['loss'] - row['code_ce']) <= 1e-7 and
                  0 <= row['code_accuracy'] <= 1 and row['gradient_norm_before_clip'] >= 0 and row['seconds'] <= 1200,
                  'Failed-run completed trace differs')
    if partial is not None:
        v.require(partial['complete'] is False and partial['resume_permitted'] is False and
                  len(traces) <= partial['optimizer_updates'] <= min(3128, len(traces) + 1) and
                  partial['optimizer_updates'] <= partial['backward_calls'] <= partial['optimizer_updates'] + 2,
                  'Failed-run partial counters differ')
    v.write(receipt, {'complete': True, 'success': False, 'protocol_sha256': pin,
        'completed_trace_records_checked': len(traces), 'exposures_checked': len(traces) * 10,
        'training_failure': partial, 'supervisor_failure': failure, 'resume_permitted': False,
        'neural_forwards': 0, 'backward_calls': 0, 'optimizer_updates': 0,
        'limitation': 'Audits returned completed trace prefix/scope; no CUDA replay, usefulness claim or checkpoint promotion.'})


def audit(root, parent, mixed, baseline, pin, out, receipt):
    sys.path.insert(0, str(root))
    import cctv_dgp_broader_codes_v16 as v
    import numpy as np
    from PIL import Image
    start = time.monotonic(); p = v.verify(root, parent, mixed, baseline, pin); d = p['design']
    from cctv_dgp_cache_timing_v16_r2 import verify_cache_timing
    timing = verify_cache_timing(p, v.read(out / 'cache_timing.json'))
    v.require(timing['passed'] and v.read(out / 'cache_initialization.json')['seconds'] == timing['initialization_seconds'], 'Cache timing evidence differs')
    r = v.read(out / 'results.json'); n = v.read(out / 'neural_execution_receipt.json')
    v.require(r['complete'] and r['protocol_sha256'] == pin and
              all(r[k] is False for k in ['native_used', 'native_reserved_used', 'production_promoted',
                  'checkpoint_selected', 'best_checkpoint_created']) and
              r['optimizer_updates'] == 3128 and r['backward_calls'] == 3129 and r['training_exposures'] == 31280,
              'Wrong returned scope/counts')
    v.require(r['cache_seconds'] <= 900 and r['fit_seconds'] <= 1200 and r['seconds'] <= 2100,
              'Finite stage caps failed')
    for name, expected in r['artifacts_sha256'].items():
        v.require(v.sha(v.safe(out, name)) == expected, 'Returned artifact differs:' + name)
    v.require(all(name in r['artifacts_sha256'] for name in ['neural_execution_receipt.json', 'execution.json', 'cache_initialization.json',
        'cache_manifest.json', 'cuda_preflight.json', 'preflight_logits.npy', 'cache_timing.json', 'fit_timing.json',
        'epoch4_fit_stop.json', 'update_trace.jsonl', 'fresh_image_parity.json']), 'Unbound required evidence')
    v.require(not any(x.name == 'best.pth' for x in out.rglob('*.pth')), 'Forbidden automatic selection')
    expected_counts = {'dgp': 1401, 'prior_encoder': 1401, 'prior_classifier': 1401, 'head': 6240,
        'prior_generator': 1810, 'teacher_encoder': 781, 'teacher_quantizer': 781, 'unused_v11': 0, 'recognizer': 1720}
    v.require(n['complete'] and n['protocol_sha256'] == pin and n['frozen_before'] == n['frozen_after'] and
              n['counts'] == n['expected_counts'] == expected_counts and n['initial_state'] != n['final_state'] and
              n['optimizer_updates'] == 3128 and n['backward_calls'] == 3129 and
              n['peak_vram_bytes'] <= d['peak_vram_cap_bytes'], 'Recorded frozen neural proof differs')
    refs = {x['id']: x for x in p['references']}; cases = {c['id']: c for c in p['training_cases'] + p['validation_cases']}
    training_ids = {rid for rid, ref in refs.items() if ref['role'] == 'train'}
    cm = v.read(out / 'cache_manifest.json')
    v.require(cm['complete'] and cm['protocol_sha256'] == pin and cm['float_dtype'] == 'float32' and
              set(cm['cases']) == set(cases) and cm['teacher_ids'] == sorted(training_ids) and
              cm['zero_logits_parity_cases'] == 4425 and cm['seconds'] <= 900, 'Cache scope differs')
    v.require(cm['cache_bytes'] == sum(x['bytes'] for x in cm['cases'].values()), 'Cache size arithmetic differs')
    for cid, item in cm['cases'].items():
        c = cases[cid]; ref = refs[c['reference_id']]
        v.require(all(item[k] == c[k] for k in ['reference_id', 'source', 'profile']) and
                  item['role'] == ref['role'] and item['input_sha256'] == p['data_assets_sha256'][c['input']] and
                  item['zero_logits_equal'] is True and len(item['sha256']) == 64 and item['bytes'] > 0,
                  'Cache binding differs:' + cid)
    teacher_files = list((out / 'teacher').glob('*_codes.npy'))
    v.require(len(teacher_files) == 781 and {x.name[:-10] for x in teacher_files} == training_ids,
              'Teacher roles differ; no validation teachers allowed')
    labels = {}
    for rid in sorted(training_ids):
        name = 'teacher/' + rid + '_codes.npy'; v.require(name in r['artifacts_sha256'], 'Unbound teacher')
        x = np.load(out / name, allow_pickle=False)
        v.require(x.dtype == np.int64 and x.shape == (256,) and x.min() >= 0 and x.max() < 1024, 'Invalid teacher labels')
        labels[rid] = x
    exec_receipt = v.read(out / 'execution.json')
    v.require(exec_receipt['protocol_sha256'] == pin and exec_receipt['design'] == d and
              exec_receipt['teacher_roles'] == ['train'] and exec_receipt['initial_state'] == n['initial_state'] and
              exec_receipt['frozen_before'] == n['frozen_before'] and 'L4' in exec_receipt['gpu'], 'Execution binding differs')

    def support(rid):
        with Image.open(mixed / refs[rid]['observed']) as im: return np.asarray(im).copy() > 0

    def vector(path):
        x = np.load(path, allow_pickle=False)
        v.require(x.dtype == np.float32 and x.shape == (512,) and np.isfinite(x).all() and
                  abs(float(np.linalg.norm(x)) - 1) < 1e-5, 'Invalid embedding')
        return x

    def code(logits, cid):
        rid = cases[cid]['reference_id']
        return v.code_metrics(logits, labels[rid], support(rid)[::16, ::16].reshape(256))

    preflight = v.read(out / 'cuda_preflight.json'); logit = np.load(out / 'preflight_logits.npy', allow_pickle=False)
    steps = v.read(root / 'schedule_v16.json')['steps']
    v.require(preflight['complete'] and preflight['optimizer_updates'] == 0 and preflight['backward_calls'] == 1 and
              preflight['state_hash'] == n['initial_state'] and preflight['case_ids'] == steps[0]['case_ids'] and
              logit.shape == (10, 256, 1024), 'Backward preflight scope differs')
    stats = [code(logit[i], cid) for i, cid in enumerate(preflight['case_ids'])]
    weights = [int(support(cases[cid]['reference_id'])[::16, ::16].sum()) for cid in preflight['case_ids']]
    v.near(preflight['metrics'], {k: float(np.average([x[k] for x in stats], weights=weights))
                               for k in ['code_ce', 'code_accuracy']}, 5e-6)
    norms = preflight['gradient_norms']
    v.require(all(isinstance(x, (int, float)) and np.isfinite(x) and x >= 0 for x in norms.values()) and
              norms['code_projection.weight'] > 0 and norms['code_projection.bias'] > 0, 'Invalid gradient receipt')
    with (out / 'update_trace.jsonl').open(encoding='utf-8') as f: traces = [json.loads(line) for line in f]
    v.require(len(traces) == 3128, 'Missing updates')
    last_seconds = 0
    for step, row in zip(steps, traces):
        v.require(all(step[k] == row[k] for k in ['update', 'epoch', 'case_ids']) and
                  all(np.isfinite(row[k]) for k in ['loss', 'code_ce', 'code_accuracy', 'gradient_norm_before_clip', 'seconds']) and
                  abs(row['loss'] - row['code_ce']) <= 1e-7 and row['code_ce'] >= 0 and
                  0 <= row['code_accuracy'] <= 1 and row['gradient_norm_before_clip'] >= 0 and
                  last_seconds <= row['seconds'] <= 1200, 'Trace schedule/loss/time differs')
        last_seconds = row['seconds']
    for name, countkey, count, cap in [('cache_timing.json', 'references', 20, 900), ('fit_timing.json', 'update', 25, 1200)]:
        timing = v.read(out / name)
        v.require(timing[countkey] == count and timing['cap_seconds'] == cap and
                  0 < timing['seconds'] <= timing['projected_seconds'] <= cap, 'Timing stop receipt differs')

    br = v.read(baseline / 'results.json'); base_rows = {x['id']: x for x in br['rows']}
    snapshots = r['snapshots']; epoch_metrics = {}; epoch_rows = {}
    v.require([s['epoch'] for s in snapshots] == [0, 4, 8] and len({s['checkpoint'] for s in snapshots}) == 3,
              'Snapshot schedule differs')
    pngs = raws = cosines = codes_checked = 0
    expected_case_ids = {c['id'] for c in p['validation_cases']} | {
        c['id'] for c in p['training_cases'] if c['reference_id'] in p['train_preview_reference_ids']}
    for snap in snapshots:
        epoch = snap['epoch']; m = v.read(out / snap['metrics'])
        v.require(snap['metrics'] in r['artifacts_sha256'] and snap['checkpoint'] in r['artifacts_sha256'],
                  'Unbound snapshot')
        v.require(m['epoch'] == epoch and m['update'] == epoch * 391 and
                  m['checkpoint'] == snap['checkpoint'] and m['state_hash'] == snap['state_hash'] and
                  len(m['rows']) == 570 and {x['id'] for x in m['rows']} == expected_case_ids, 'Snapshot binding differs')
        if epoch == 0: v.require(m['state_hash'] == n['initial_state'], 'Initial head is not reset')
        if epoch == 8: v.require(m['state_hash'] == n['final_state'], 'Final head binding differs')
        rebuilt = []; training = []
        for row in m['rows']:
            c = cases[row['id']]; rid = c['reference_id']; ref = refs[rid]; mask = support(rid)
            v.require(all(row[key] in r['artifacts_sha256'] for key in ['prediction', 'embedding', 'raw', 'logits',
                'dgp_preview', 'dgp_raw'] if key in row), 'Unbound row artifact')
            if ref['role'] == 'train':
                v.require('target_embeddings/' + rid + '.npy' in r['artifacts_sha256'], 'Unbound training target embedding')
            v.require(all(row[k] == c[k] for k in ['reference_id', 'source', 'profile']) and row['role'] == ref['role'],
                      'Snapshot source/role differs')
            camera = v.rgb(mixed / c['input']); image = v.rgb(v.safe(out, row['prediction']))
            v.require(np.array_equal(image[~mask], camera[~mask]), 'Unsupported pixels changed')
            target_vec_path = (out if ref['role'] == 'train' else baseline) / ('target_embeddings/' + rid + '.npy')
            cosine = float(np.clip(vector(v.safe(out, row['embedding'])) @ vector(target_vec_path), -1, 1)); cosines += 1
            measured = {**v.metrics(image, v.rgb(mixed / ref['target']), mask), 'ArcFace_observed_fixed': cosine}
            v.near({k: row[k] for k in measured}, measured); pngs += 1
            preview = rid in p[('train' if ref['role'] == 'train' else 'validation') + '_preview_reference_ids']
            v.require(('raw' in row) == ('logits' in row) == preview, 'Raw/logit preview coverage differs')
            if preview:
                raw = np.load(v.safe(out, row['raw']), allow_pickle=False)
                v.require(np.array_equal(v.png(raw, camera, mask), image), 'Raw/PNG differs'); raws += 1
                x = np.load(v.safe(out, row['logits']), allow_pickle=False)
                v.require(x.shape == (256, 1024) and x.dtype == np.float32 and np.isfinite(x).all(), 'Invalid preview logits')
            if ref['role'] == 'train':
                measured_code = code(x, c['id']); v.near({k: row[k] for k in measured_code}, measured_code, 5e-6)
                training.append(measured_code); codes_checked += 1
                if epoch == 0:
                    dgp_raw = np.load(out / row['dgp_raw'], allow_pickle=False)
                    v.require(np.array_equal(v.png(dgp_raw, camera, mask), v.rgb(out / row['dgp_preview'])), 'DGP display differs')
            else:
                v.require('code_ce' not in row and 'code_accuracy' not in row, 'Validation teacher leakage')
                rebuilt.append({**{k: c[k] for k in ['id', 'source', 'profile']}, **measured})
                if epoch == 0:
                    v.require(row['baseline_png_equal'] is True and np.array_equal(image,
                              v.rgb(baseline / base_rows[c['id']]['arms']['starting_prior_none']['prediction'])),
                              'Starting validation parity differs')
        summaries = {a: br['summaries'][a] for a in ['retained_dgp_v2', 'starting_prior_none', 'input']}
        summaries['trained_conditioner_none'] = v.aggregate(rebuilt)
        v.near(m['summaries'], summaries); v.require(m['guard_report'] == v.guard_report(summaries), 'Guard changed')
        v.near(m['training_preview'], {k: float(np.mean([x[k] for x in training])) for k in ['code_ce', 'code_accuracy']}, 5e-6)
        epoch_metrics[epoch] = m; epoch_rows[epoch] = {x['id']: x for x in m['rows']}
    v.require(len({s['state_hash'] for s in snapshots}) == 3, 'Trained snapshots unchanged')
    old = epoch_metrics[0]['training_preview']['code_ce']; new = epoch_metrics[4]['training_preview']['code_ce']
    fit_stop = v.read(out / 'epoch4_fit_stop.json')
    v.near(fit_stop, {'relative_code_ce_improvement': (old - new) / old,
        'required': d['minimum_fit_ce_improvement_epoch4'], 'passed': True}, 5e-6)
    v.require((old - new) / old >= d['minimum_fit_ce_improvement_epoch4'], 'Epoch4 fitting stop failed')
    parity = v.read(out / 'fresh_image_parity.json'); expected_parity = {
        (epoch, c['id']) for epoch in [0, 8] for c in p['validation_cases']
        if c['reference_id'] in p['validation_preview_reference_ids']}
    v.require(parity['complete'] and len(parity['cases']) == 100 and
              {(x['epoch'], x['id']) for x in parity['cases']} == expected_parity, 'Fresh parity coverage differs')
    for item in parity['cases']:
        v.require(item['raw'] in r['artifacts_sha256'], 'Unbound fresh parity')
        row = epoch_rows[item['epoch']][item['id']]; c = cases[item['id']]
        raw = np.load(out / row['raw'], allow_pickle=False); fresh = np.load(out / item['raw'], allow_pickle=False)
        delta = float(np.max(np.abs(raw - fresh)))
        v.require(delta <= d['float_parity_tolerance'] and abs(delta - item['maximum_float_difference']) <= 1e-12 and
                  item['png_equal'] is True and np.array_equal(v.png(fresh, v.rgb(mixed / c['input']),
                      support(c['reference_id'])), v.rgb(out / row['prediction'])), 'Fresh raw/PNG parity differs')
    expected_grids = [f'grids/{role}_{profile}_10_rows.png' for role in ['train', 'validation'] for profile in v.PROFILES]
    v.require(r['grids'] == expected_grids, 'Fixed grids differ')
    v.require(all(name in r['artifacts_sha256'] for name in expected_grids), 'Unbound grid')
    grid_cells = 0
    for name in expected_grids:
        role = 'train' if '/train_' in name else 'validation'
        profile = name.split(role + '_')[1].split('_10_rows')[0]
        sheet = np.asarray(Image.open(out / name).convert('RGB'))
        v.require(sheet.shape == (2904, 1560, 3), 'Grid geometry differs')
        for i, rid in enumerate(p[role + '_preview_reference_ids']):
            c = next(c for c in cases.values() if c['reference_id'] == rid and c['profile'] == profile); cid = c['id']
            retained = out / epoch_rows[0][cid]['dgp_preview'] if role == 'train' else baseline / base_rows[cid]['arms']['retained_dgp_v2']['prediction']
            images = [v.rgb(mixed / c['input']), v.rgb(retained)] + [v.rgb(out / epoch_rows[e][cid]['prediction']) for e in [0, 4, 8]] + [v.rgb(mixed / refs[rid]['target'])]
            for j, image in enumerate(images):
                y = 24 + i * 288 + 28; x = j * 260 + 2
                v.require(np.array_equal(sheet[y:y + 256, x:x + 256], image), 'Grid cell differs'); grid_cells += 1
    v.require((pngs, raws, cosines, codes_checked, grid_cells) == (1710, 300, 1710, 150, 600), 'Audit coverage differs')
    result = {'complete': True, 'protocol_sha256': pin, 'results_sha256': v.sha(out / 'results.json'),
        'auditor_sha256': v.sha(Path(__file__)), 'pngs_checked': pngs, 'raw_previews_checked': raws,
        'embedding_cosines_rebuilt': cosines, 'training_code_probes_checked': codes_checked,
        'fresh_image_parity_checked': 100, 'teacher_labels_checked': 781, 'cache_bindings_checked': 4425,
        'updates_checked': 3128, 'exposures_checked': 31280, 'grid_cells_checked': grid_cells,
        'guard_reports': {str(e): m['guard_report'] for e, m in epoch_metrics.items()},
        'seconds': time.monotonic() - start, 'neural_forwards': 0, 'backward_calls': 0, 'optimizer_updates': 0,
        'native_reserved_used': False, 'production_promoted': False,
        'limitation': 'Serialized pixel/logit/embedding arithmetic and execution receipts; not a CUDA gradient/recognizer replay or final human review. Full9GB cache remains on VM, checked there by checksums; local audit checks bindings and returned probes. Frozen historical baseline was separately audited; its metrics are reused, not recomputed here.'}
    v.write(receipt, result)
    print({k: result[k] for k in ['complete', 'pngs_checked', 'training_code_probes_checked', 'seconds']}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'parent', 'mixed', 'baseline', 'results', 'receipt']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-sha', required=True); args = parser.parse_args()
    audit(args.root.resolve(), args.parent.resolve(), args.mixed.resolve(), args.baseline.resolve(),
          args.expected_sha, args.results.resolve(), args.receipt.resolve())
