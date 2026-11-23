"""Independent predictor-only CE V1 saved-output audit. No Torch, model or VM calls.

Reconstructs exact delivered PNG pixels from returned RAW arrays and frozen
reviewed source/masks, and recomputes all prospective quality endpoints.
"""
from __future__ import annotations
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import argparse, hashlib, json, os, sys, time, zipfile
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/completion_codeformer_predictor_ce_v1_return'
RETURN = BASE / 'cctv_completion_codeformer_predictor_ce_v1_results'
STAGE = ROOT / 'outputs/completion_codeformer_predictor_ce_v1_packet/draft_001/cctv_completion_codeformer_predictor_ce_v1_vm'
PROTOCOL = '15aa3e1478a67c120b2fcb6af50ecae58d0558eb4e65cef473d0339572bf084b'
KINDS = ('lower_face', 'eye_band', 'image_left_lens', 'cheek_object')
METRICS = ('raw_hole_L1', 'raw_boundary_L1', 'PNG_hole_L1', 'PNG_boundary_L1')
EVAL_SCHEMA = {'input256', 'mask256', 'result256', 'raw512', 'predicted_codes'}
PARITY_SCHEMA = {'input256', 'mask256', 'own_input512', 'app_input512', 'own_raw512', 'app_raw512', 'own_codes', 'app_codes', 'own_result256', 'app_result256'}
FIT_SCHEMA = {'logits', 'labels', 'case_index', 'cycle', 'microbatch'}
RAW_TOL, PNG_TOL = 2e-7, 1e-14
CHECKS = 0


def need(ok, label):
    global CHECKS
    CHECKS += 1
    if not ok:
        raise ValueError(label)


def physical(path):
    path = Path(path).absolute()
    return Path('\\\\?\\' + str(path)) if os.name == 'nt' and not str(path).startswith('\\\\?\\') else path


def sha(path):
    h = hashlib.sha256()
    with physical(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def pin(path):
    return {'path': str(Path(path).absolute()), 'bytes': physical(path).stat().st_size, 'sha256': sha(path)}


def read(path):
    def reject(value):
        raise ValueError('Nonfinite JSON ' + value)
    return json.loads(physical(path).read_text(encoding='utf-8-sig'), parse_constant=reject)


def pixels(path, mode='RGB'):
    with Image.open(physical(path)) as im:
        need(im.format == 'PNG' and im.mode == mode and im.size == (256, 256), 'Exact PNG256 geometry/mode ' + str(path))
        return np.array(im, copy=True)


def same(a, b, label):
    need(a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b), label)


def finite(a, shape):
    need(a.dtype == np.float32 and a.shape == shape and np.isfinite(a).all(), 'Finite exact FP32 array shape ' + str(shape))


def code(a):
    need(a.dtype == np.int64 and a.shape == (1, 256) and 0 <= a.min() and a.max() < 512, 'Exact int64[1,256] inpainting-codebook indices')


def png(a):
    return np.rint(np.clip(a[0].transpose(1, 2, 0), 0, 1) * np.float32(255)).astype(np.uint8)


def ring(support):
    padded = np.pad(support, 3, constant_values=True)
    interior = np.lib.stride_tricks.sliding_window_view(padded, (7, 7)).all(axis=(-1, -2))
    return support & ~interior


def decode(raw, source, support):
    a = np.clip((raw + np.float32(1)) / np.float32(2), 0, 1)
    # Exactly2x align_corners=False bilinear samples at four equal neighbours.
    small = ((a[:, :, 0::2, 0::2] + a[:, :, 0::2, 1::2]) +
             (a[:, :, 1::2, 0::2] + a[:, :, 1::2, 1::2])) * np.float32(.25)
    value = source.copy()
    value[:, :, support] = small[:, :, support]
    return value


def metrics(value, target_float, support, delivered, target):
    boundary = ring(support)
    need(boundary.any(), 'Nonempty inner7x7 boundary')
    raw = np.abs(value - target_float)[0].transpose(1, 2, 0).astype(np.float64)
    quant = np.abs(delivered.astype(np.float64) - target.astype(np.float64)) / 255
    return {'raw_hole_L1': float(raw[support].mean()), 'raw_boundary_L1': float(raw[boundary].mean()),
            'PNG_hole_L1': float(quant[support].mean()), 'PNG_boundary_L1': float(quant[boundary].mean())}


def aggregate(rows, condition=False):
    def avg(part):
        need(bool(part), 'Nonempty declared subgroup')
        return {key: sum(row[key] for row in part) / len(part) for key in METRICS}
    value = {'macro': avg(rows), 'geometry': {kind: avg([r for r in rows if r['mask_kind'] == kind]) for kind in KINDS}}
    if condition:
        value['condition'] = {name: avg([r for r in rows if r['condition'] == name]) for name in sorted({r['condition'] for r in rows})}
    return value


def compare(now, baseline, condition=False):
    groups = [('macro', now['macro'], baseline['macro'])]
    for key in ('geometry', 'condition') if condition else ('geometry',):
        need(set(now[key]) == set(baseline[key]), 'Identical compared subgroups')
        groups.extend((key + '/' + name, now[key][name], baseline[key][name]) for name in now[key])
    failures = [{'group': label, 'metric': key, 'candidate': a[key], 'baseline': b[key], 'delta': a[key] - b[key]}
                for label, a, b in groups for key in METRICS if a[key] > b[key] + 1e-12]
    gain = (baseline['macro']['PNG_hole_L1'] - now['macro']['PNG_hole_L1']) / max(baseline['macro']['PNG_hole_L1'], 1e-12)
    return {'PNG_macro_hole_gain': gain, 'PNG_macro_hole_gain_percent': gain * 100,
            'all_raw_PNG_hole_boundary_groups_nonworse': not failures, 'nonworse_failures': failures,
            'catastrophic_regression_stop': any(a[k] > b[k] * 1.10 for _, a, b in groups for k in METRICS),
            'prospective_numeric_screen_pass': bool(gain >= .01 and not failures), 'condition_groups_included': condition,
            'nonworse_tolerance': 1e-12, 'model_qualified': False}


def gray(rows):
    rows = [row for row in rows if row['style'] == 'GRAY']
    need(len(rows) == 8 and len({row['id'] for row in rows}) == 8, 'Eight fixed GRAY challenge cases')
    keys = ('GRAY_raw_mean_channel_spread', 'GRAY_PNG_mean_channel_spread')
    def avg(part):
        need(bool(part), 'Nonempty GRAY style subgroup')
        return {key: sum(r[key] for r in part) / len(part) for key in keys}
    geometries = {kind: [r for r in rows if r['mask_kind'] == kind] for kind in KINDS}
    need(all(len(part) == 2 for part in geometries.values()), 'Two GRAY cases each geometry')
    return {'macro': avg(rows), 'geometry': {kind: avg(part) for kind, part in geometries.items()},
            'cases': [{k: r[k] for k in ('id', 'mask_kind', *keys)} for r in rows]}


def compare_gray(now, baseline):
    a, b = gray(now), gray(baseline)
    pairs = [('macro', a['macro'], b['macro'])] + [('geometry/' + kind, a['geometry'][kind], b['geometry'][kind]) for kind in KINDS]
    failures = [{'group': label, 'metric': key, 'candidate': x[key], 'baseline': y[key], 'delta': x[key] - y[key]}
                for label, x, y in pairs for key in x if x[key] > y[key] + 1e-12]
    return {'all_GRAY_macro_geometry_chroma_nonworse': not failures, 'nonworse_failures': failures,
            'tolerance_raw': 1e-12, 'tolerance_PNG': 1e-12, 'candidate': a, 'R1_baseline': b, 'model_qualified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transport-sha', required=True)
    args = parser.parse_args()
    start = time.monotonic()
    destination = BASE / 'independent_saved_output_audit.json'
    need(not destination.exists(), 'Preserve earlier pixel audit')
    transport_path = BASE / 'transport_audit.json'
    need(sha(transport_path) == args.transport_sha, 'Exact root-admitted transport receipt')
    transport = read(transport_path)
    need(transport.get('complete') is True, 'Wait for transport admission')
    need(sha(RETURN / 'protocol.json') == sha(STAGE / 'protocol.json') == PROTOCOL, 'Exact actual and sealed protocol')
    p = read(RETURN / 'protocol.json')
    need(p['format'] == 'completion-codeformer-predictor-ce-v1', 'Exact pilot format')
    manifest = read(RETURN / 'export_manifest.json')
    need(manifest['protocol_sha256'] == PROTOCOL, 'Export protocol')
    pins = {r['path']: r for r in manifest['files']}
    need(len(pins) == len(manifest['files']), 'Unique export manifest')
    files = {r['path']: r for r in p['files']}
    need(len(files) == len(p['files']) == 670, 'Exactly670 immutable staged payloads')
    source_pins, returned_pins, array_reviews = {}, {}, []
    for relative, row in files.items():
        path = STAGE / relative
        need(physical(path).is_file() and not physical(path).is_symlink() and physical(path).stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Full staged source/data pin ' + relative)
        source_pins[relative] = row
    def evidence(path):
        relative = path.relative_to(RETURN).as_posix()
        need(relative in pins, 'Manifest-bound returned evidence ' + relative)
        row = pins[relative]
        need(physical(path).is_file() and not physical(path).is_symlink() and physical(path).stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Exact returned pin ' + relative)
        returned_pins[relative] = row
        return path
    def asset(relative, mode='RGB'):
        need(relative in files, 'Pinned source asset')
        return pixels(STAGE / relative, mode)
    def arrays(path, schema, returned=True):
        if returned:
            evidence(path)
        with zipfile.ZipFile(physical(path)) as z:
            members = z.infolist()
            need(len(members) == len(schema) and {m.filename for m in members} == {key + '.npy' for key in schema}, 'Exact NPZ schema ' + str(path))
            need(len({m.filename for m in members}) == len(members), 'Unique NPZ entries')
            need(all(not m.is_dir() and not m.flag_bits & 1 and 0 < m.file_size <= 16 * 1024 ** 2 and m.compress_type in (0, 8) and PurePosixPath(m.filename).name == m.filename for m in members), 'Safe bounded NPZ namespace')
            need(z.testzip() is None, 'Every NPZ member CRC')
        with np.load(physical(path), allow_pickle=False, max_header_size=512) as z:
            values = {key: z[key].copy() for key in z.files}
        need(all(not value.dtype.hasobject and value.dtype in (np.float32, np.int64, np.uint8) and np.isfinite(value).all() for value in values.values()), 'Numeric finite arrays only')
        if returned:
            array_reviews.append({'path': path.relative_to(RETURN).as_posix(), 'schema': sorted(values), 'CRC_pass': True, 'allow_pickle': False, 'max_header_size': 512, 'numeric_finite_only': True})
        return values
    teacher = {}
    for row in p['cached_labels']:
        z = arrays(STAGE / row['path'], {'prequant_latent', 'nearest_codes'}, returned=False)
        finite(z['prequant_latent'], (1, 256, 16, 16)); code(z['nearest_codes'])
        teacher[row['teacher_key']] = z['nearest_codes']
    need(len(teacher) == 80, 'All80 exact original teacher labels retained')
    out = RETURN / 'outputs'
    final = read(evidence(out / 'results.json'))
    need(final['complete'] and final['protocol_sha256'] == PROTOCOL, 'Actual terminal results contract')
    terminal = final['predictor_steps_completed_newphase']
    need(terminal in (8, 16, 32), 'Completed bounded endpoint supported by this actual audit')
    snapshots = [x for x in (0, 8, 16, 32) if x <= terminal]
    parity_summary = read(evidence(out / 'fresh_app_parity/summary.json'))
    parity_by_id = {row['id']: row for row in parity_summary['cases']}
    parity = []
    for case in p['parity_cases']:
        ident = case['id']; folder = out / 'fresh_app_parity'
        record = read(evidence(folder / (ident + '.json')))
        need(record == parity_by_id[ident], 'Parity summary exact row')
        z = arrays(folder / (ident + '.npz'), PARITY_SCHEMA)
        finite(z['input256'], (1, 3, 256, 256)); finite(z['mask256'], (1, 1, 256, 256))
        need(np.isin(z['mask256'], [0, 1]).all(), 'Binary parity support')
        support = z['mask256'][0, 0].astype(bool)
        same(png(z['input256']), asset(case['input256']), 'Parity exact source bytes')
        same((z['mask256'][0, 0] * 255).astype(np.uint8), asset(case['mask'], 'L'), 'Parity reviewed mask exact')
        same(z['own_input512'], z['app_input512'], 'Fresh app network input exact')
        same(z['own_codes'], z['app_codes'], 'Fresh app selected codes exact')
        code(z['own_codes']); code(z['app_codes'])
        for side in ('own', 'app'):
            finite(z[side + '_input512'], (1, 3, 512, 512)); finite(z[side + '_raw512'], (1, 3, 512, 512)); finite(z[side + '_result256'], (1, 3, 256, 256))
            same(z[side + '_result256'][:, :, ~support], z['input256'][:, :, ~support], 'Parity visible FP32 exact')
            replay = decode(z[side + '_raw512'], z['input256'], support)
            need(float(np.max(np.abs(replay - z[side + '_result256']))) <= np.finfo(np.float32).eps, 'Parity raw replay within1ulp')
        rawerr = float(np.max(np.abs(z['own_raw512'] - z['app_raw512'])))
        error = float(np.max(np.abs(z['own_result256'] - z['app_result256'])))
        need(error == record['max_raw256_error'] <= 1e-6 and rawerr == record['model_raw512_max_error'], 'Fresh parity metrics exact')
        need(record['optimizer_steps_before_parity'] == 0 and record['input512_exact'] and record['input_stride_exact'] and record['selected_codes_exact'], 'Parity precedes learning and preserves app layout')
        parity.append({'id': ident, 'raw256_max_error': error, 'raw512_max_error': rawerr, 'input_codes_outside_exact': True})
    need(len(parity) == 4 and parity_summary['training_updates'] == 0 and parity_summary['all_input_values_and_layout_exact'], 'Allfour baseline parity pairs')
    history = read(evidence(out / 'historical_R1_compatibility.json'))
    need(history['complete'] and history['pass'] and history['all48_before_update'] and history['updates_before'] == 0, 'All48 initial historical matches before optimization')
    history_rows = {(r['kind'], r['id']): r for r in history['cases']}
    evaluations, grouped, metrics_by_snapshot, compatibility = [], {}, {}, []
    outside_bytes = 0; max_replay = 0.; maximum_metric_difference = {'raw': 0., 'PNG': 0.}; empty_saved = 0
    baseline_codes = {}; codes_changed = {}; screen_decision_mismatches = []
    for cycle in snapshots:
        for kind, key in [('monitor', 'monitor_cases'), ('challenge', 'challenge_cases')] + ([('public_development', 'development_cases')] if cycle in (0, terminal) else []):
            folder = out / ('u%04d' % cycle) / kind
            summary = read(evidence(folder / 'results.json'))
            cases = p[key]
            need(summary['cycle'] == cycle and summary['kind'] == kind and [r['id'] for r in summary['records']] == [c['id'] for c in cases], 'Exact fixed case order in completed cohort')
            rows = []
            for case, reported in zip(cases, summary['records']):
                ident = case['id']; image = folder / (ident + '.png')
                evidence(image); delivered = pixels(image)
                source8 = asset(case['input256']); mask8 = asset(case['mask'], 'L')
                need(np.isin(mask8, [0, 255]).all(), 'Binary reviewed source support')
                support = mask8 == 255
                need(int(support.sum()) == case['mask_pixels'], 'Original reviewed mask area exact')
                z = arrays(folder / (ident + '.npz'), EVAL_SCHEMA)
                source, result, mask = z['input256'], z['result256'], z['mask256']
                finite(source, (1, 3, 256, 256)); finite(result, (1, 3, 256, 256)); finite(mask, (1, 1, 256, 256))
                need(np.isin(mask, [0, 1]).all() and result.min() >= 0 and result.max() <= 1, 'Output range/binary mask')
                same(png(source), source8, 'Observed source RGB bytes exact')
                need(float(np.max(np.abs(source[0].transpose(1, 2, 0) - source8.astype(np.float32) / np.float32(255)))) <= np.finfo(np.float32).eps, 'Canonical source float within1ulp')
                same((mask[0, 0] * 255).astype(np.uint8), mask8, 'Reviewed mask exact')
                same(result[:, :, ~support], source[:, :, ~support], 'Every outside FP32 pixel exact')
                same(delivered[~support], source8[~support], 'Every outside source RGB byte exact')
                same(png(result), delivered, 'Every saved FP32→PNG exact')
                outside_bytes += int((~support).sum()) * 3
                empty = not bool(support.any())
                if empty:
                    need(z['raw512'].dtype == np.float32 and z['raw512'].shape == (0,) and z['predicted_codes'].dtype == np.int64 and z['predicted_codes'].shape == (0,), 'Empty mask has no fabricated raw/codes')
                    same(source, result, 'Empty source bypass float exact'); same(source8, delivered, 'Empty source bypass RGB exact')
                    empty_saved += 1; replay_error = 0.
                else:
                    finite(z['raw512'], (1, 3, 512, 512)); code(z['predicted_codes'])
                    replay = decode(z['raw512'], source, support)
                    replay_error = float(np.max(np.abs(replay - result)))
                    max_replay = max(max_replay, replay_error)
                    need(replay_error <= np.finfo(np.float32).eps, 'Raw512→FP32256 composition within1ulp')
                    same(png(replay), delivered, 'Every raw512→PNG exact')
                    if cycle == 0:
                        baseline_codes[(kind, ident)] = z['predicted_codes'].copy()
                    else:
                        codes_changed[(cycle, kind, ident)] = int(np.count_nonzero(z['predicted_codes'] != baseline_codes[(kind, ident)]))
                need(reported['id'] == ident and reported['mask_kind'] == case.get('mask_kind') and reported['family'] == case.get('family') and reported['condition'] == case.get('condition') and reported['style'] == case.get('style') and reported['outside_exact'] and reported['empty'] == empty, 'Exact reported case metadata')
                need(reported['PNG'] == image.relative_to(out).as_posix() and reported['PNG_sha256'] == sha(image), 'Exact PNG record binding')
                row = {'id': ident, 'kind': kind, 'cycle': cycle, 'mask_kind': case.get('mask_kind'), 'family': case.get('family'), 'condition': case.get('condition'), 'style': case.get('style'), 'teacher_key': case.get('teacher_key'), 'empty': empty, 'NPZ': pin(folder / (ident + '.npz')), 'PNG': pin(image), 'outside_FP32_PNG_exact': True, 'raw512_replay_max_error': replay_error}
                if kind != 'public_development':
                    need(not empty, 'All paired reviewed masks nonempty')
                    target = asset(case['clean256'])
                    target_float = source if np.array_equal(target, source8) else target.astype(np.float32).transpose(2, 0, 1)[None] / np.float32(255)
                    row.update(metrics(result, target_float, support, delivered, target))
                    for metric in METRICS:
                        difference = abs(row[metric] - reported[metric]); typ = 'PNG' if metric.startswith('PNG') else 'raw'
                        maximum_metric_difference[typ] = max(maximum_metric_difference[typ], difference)
                        need(difference <= (PNG_TOL if typ == 'PNG' else RAW_TOL), 'Independent per-case error metric ' + metric)
                    agreement = float((z['predicted_codes'] == teacher[case['teacher_key']]).mean())
                    need(agreement == reported['same_codebook_clean_context_agreement_diagnostic'], 'Saved predicted-code/monitor-label agreement exact')
                    row['whole_grid_clean_context_code_agreement_diagnostic_not_identity_accuracy'] = agreement
                    if case['style'] == 'GRAY':
                        raw_rgb = result[0].transpose(1, 2, 0)
                        row['GRAY_raw_mean_channel_spread'] = float(np.ptp(raw_rgb, axis=-1)[support].mean())
                        row['GRAY_PNG_mean_channel_spread'] = float(np.ptp(delivered.astype(np.int16), axis=-1)[support].mean())
                        need(row['GRAY_raw_mean_channel_spread'] == reported['GRAY_raw_mean_channel_spread'] and row['GRAY_PNG_mean_channel_spread'] == reported['GRAY_PNG_mean_channel_spread'], 'Exact independent GRAY raw/PNG spread')
                        need(reported['GRAY_final32_style_gate_required'] and not reported['GRAY_chroma_diagnostic_not_selection_gate'], 'Required final GRAY gate not diagnostic-only')
                    if cycle == 0:
                        historical = next(h for h in p['historical_control']['initial_rows'][kind] if h['id'] == ident)
                        old = arrays(STAGE / historical['result256'], {'result256'}, returned=False)
                        finite(old['result256'], (1, 3, 256, 256))
                        error = float(np.max(np.abs(old['result256'] - result)))
                        vm = history_rows[(kind, ident)]
                        exact_png = sha(image) == sha(STAGE / historical['PNG'])
                        need(error <= 1e-6 and exact_png and vm['max_raw256_error'] == error and vm['PNG_bytes_exact'] == exact_png, 'All48 historical R1 raw/PNG baseline matches')
                        compatibility.append({'kind': kind, 'id': ident, 'raw256_max_error': error, 'PNG_bytes_exact': exact_png})
                rows.append(row); evaluations.append(row)
            grouped[(cycle, kind)] = rows
            if kind != 'public_development':
                value = aggregate(rows, condition=kind == 'challenge')
                metrics_by_snapshot[(cycle, kind)] = value
                for group in value:
                    pairs = [(value[group], summary['metrics'][group])] if group == 'macro' else [(value[group][name], summary['metrics'][group][name]) for name in value[group]]
                    for a, b in pairs:
                        for key_metric in METRICS:
                            tol = PNG_TOL if key_metric.startswith('PNG') else RAW_TOL
                            need(abs(a[key_metric] - b[key_metric]) <= tol, 'All subgroup metric reductions verified')
                if kind == 'challenge':
                    need(gray(rows) == summary['GRAY_style'], 'GRAY macro/pergeometry/percase report exact')
            print(json.dumps({'cycle': cycle, 'cohort': kind, 'outputs_audited': len(evaluations), 'outside_RGB_bytes_exact': outside_bytes}), flush=True)
    comparisons, gray_comparisons, percase = {}, {}, {}
    for cycle in snapshots[1:]:
        comparisons[str(cycle)] = {kind: compare(metrics_by_snapshot[(cycle, kind)], metrics_by_snapshot[(0, kind)], condition=kind == 'challenge') for kind in ('monitor', 'challenge')}
        gray_comparisons[str(cycle)] = compare_gray(grouped[(cycle, 'challenge')], grouped[(0, 'challenge')])
        reported = read(evidence(out / ('screen_u%04d.json' % cycle)))
        need(reported['early_selection_forbidden'] == (cycle in (8, 16)) and reported['only_catastrophic_preservation_stop_before32'] and reported['GRAY_gate_required_only_at32'], 'Intermediate selection contract exact')
        for kind in ('monitor', 'challenge'):
            a, b = comparisons[str(cycle)][kind], reported[kind]
            need(abs(a['PNG_macro_hole_gain'] - b['PNG_macro_hole_gain']) <= PNG_TOL, 'Reported PNG gain independently exact')
            for field in ('prospective_numeric_screen_pass', 'all_raw_PNG_hole_boundary_groups_nonworse', 'catastrophic_regression_stop'):
                if a[field] != b[field]:
                    screen_decision_mismatches.append({'cycle': cycle, 'kind': kind, 'field': field, 'VM': b[field], 'independent': a[field]})
            old_by_id = {row['id']: row for row in grouped[(0, kind)]}
            changes = []
            for row in grouped[(cycle, kind)]:
                old = old_by_id[row['id']]
                changes.append({'id': row['id'], 'mask_kind': row['mask_kind'], 'condition': row['condition'], 'PNG_hole_gain_percent': 100 * (old['PNG_hole_L1'] - row['PNG_hole_L1']) / max(old['PNG_hole_L1'], 1e-12), 'metric_deltas': {k: row[k] - old[k] for k in METRICS}, 'PNG_hole_worse': row['PNG_hole_L1'] > old['PNG_hole_L1'] + 1e-12, 'PNG_boundary_worse': row['PNG_boundary_L1'] > old['PNG_boundary_L1'] + 1e-12})
            percase[(str(cycle) + '/' + kind)] = {'cases': changes, 'PNG_hole_worse_count': sum(r['PNG_hole_worse'] for r in changes), 'PNG_boundary_worse_count': sum(r['PNG_boundary_worse'] for r in changes)}
        need(gray_comparisons[str(cycle)] == reported['GRAY_style'], 'Reported GRAY gate fully replayed')
    need(not screen_decision_mismatches, 'Reported and independent quality gate decisions differ; retain and inspect')
    final_screen = final['numeric_screen']
    final_pass = bool(terminal == 32 and not final['stop_reason'] and comparisons[str(terminal)]['monitor']['prospective_numeric_screen_pass'] and comparisons[str(terminal)]['challenge']['prospective_numeric_screen_pass'] and gray_comparisons[str(terminal)]['all_GRAY_macro_geometry_chroma_nonworse'])
    need(final_screen['acceptance_endpoint'] == 32 and final_screen['endpoint_reached'] == (terminal == 32) and final_screen['prospective_numeric_screen_pass'] == final_pass and final_screen['early_selection_forbidden'] and final_screen['GRAY_final32_required'] and final_screen['cohorts_never_pooled'] and not final_screen['model_qualified'], 'Actual final complete quality endpoint independently verified')
    for cycle in snapshots:
        receipt = read(evidence(out / ('empty_controls_u%04d.json' % cycle)))
        need(receipt['references'] == sorted(row['id'] for row in p['references']) and receipt['exact_bypasses'] == 40 and receipt['model_calls'] == 0, 'All40 separate empty/uncovered controls per snapshot')
    # Extended-path objects must be normalized back to canonical path strings.
    all_npz = {str(path).replace('\\\\?\\', '').replace(str(RETURN) + os.sep, '').replace(os.sep, '/') for path in physical(out).rglob('*.npz')}
    for relative in sorted(all_npz - {row['path'] for row in array_reviews}):
        path = RETURN / relative
        if '/FIT_logits/' in relative:
            z = arrays(path, FIT_SCHEMA)
            finite(z['logits'], (1, 256, 512)); code(z['labels'])
            for key in ('case_index', 'cycle', 'microbatch'):
                need(z[key].dtype == np.int64 and z[key].shape == (), 'FIT scalar array schema')
        elif '/teacher_labels/' in relative:
            z = arrays(path, {'prequant_latent', 'nearest_codes'})
            finite(z['prequant_latent'], (1, 256, 16, 16)); code(z['nearest_codes'])
        else:
            raise ValueError('Unexpected returned NPZ ' + relative)
    need(all_npz == {row['path'] for row in array_reviews}, 'Every retained NPZ CRC/schema/finite independently covered')
    all_png = {str(path).replace('\\\\?\\', '').replace(str(RETURN) + os.sep, '').replace(os.sep, '/') for path in physical(out).rglob('*.png')}
    need(all_png == {str(Path(row['PNG']['path'])).replace(str(RETURN) + os.sep, '').replace(os.sep, '/') for row in evaluations}, 'Every actual evaluation PNG audited')
    need(len(evaluations) == 48 * len(snapshots) + 64 and len(compatibility) == 48 and empty_saved == 8, 'Complete fixed evaluation and empty output counts')
    for relative, row in returned_pins.items():
        need(physical(RETURN / relative).stat().st_size == row['bytes'] and sha(RETURN / relative) == row['sha256'], 'Returned payload unchanged during independent audit')
    for relative, row in source_pins.items():
        need(sha(STAGE / relative) == row['sha256'], 'Immutable staged payload unchanged during audit')
    need(not any(name == 'torch' or name.startswith('torch.') or name.startswith('third_party.codeformer') for name in sys.modules), 'No Torch/model runtime imported')
    report = {'format': 'completion-codeformer-predictor-ce-v1-independent-saved-output-audit-v1', 'complete': True, 'independent': True,
              'created_UTC': datetime.now(timezone.utc).isoformat(), 'auditor': pin(__file__), 'transport_admission': pin(transport_path), 'protocol': pin(RETURN / 'protocol.json'),
              'terminal_new_predictor_updates': terminal, 'actual_snapshots': snapshots, 'actual_evaluation_PNGs': len(evaluations), 'all_retained_NPZ': len(array_reviews),
              'all_NPZ_schema_CRC_numeric_finite_pass': True, 'all_raw512_to_PNG_exact': True, 'all_saved_float256_to_PNG_exact': True,
              'raw512_to_result_max_FP32_error': max_replay, 'outside_FP32_PNG_all_exact': True, 'outside_source_RGB_bytes_exact': outside_bytes,
              'fresh_R1_parity_pairs': parity, 'historical_R1_48case_compatibility': compatibility, 'saved_empty_controls': empty_saved,
              'additional_empty_functional_controls': 40 * len(snapshots), 'independent_cohort_metrics': {str(cycle): {kind: metrics_by_snapshot[(cycle, kind)] for kind in ('monitor', 'challenge')} for cycle in snapshots},
              'independent_comparisons_vs_R1': comparisons, 'independent_GRAY_style_comparisons': gray_comparisons, 'independent_percase_comparisons': percase,
              'final_prospective_numeric_screen_pass': final_pass, 'VM_screen_decisions_independently_verified': True, 'VM_metric_maximum_reduction_differences': maximum_metric_difference,
              'actual_all_grid_code_changes_vs_R1': [{'cycle': c, 'kind': k, 'id': ident, 'changed_indices': value, 'positions': 256} for (c, k, ident), value in codes_changed.items()],
              'all_outputs': evaluations, 'array_reviews': array_reviews, 'full_staged_source_payload_pins': source_pins,
              'quality_thresholds_unchanged': {'minimum_separate_clear_challenge_PNG_hole_gain': .01, 'RAW_PNG_subgroup_nonworse': 1e-12, 'GRAY_raw_PNG_macro_geometry_nonworse': 1e-12, 'RAW_metric_recompute_absolute_tolerance': RAW_TOL, 'PNG_metric_recompute_absolute_tolerance': PNG_TOL},
              'scope': 'Paired synthetic adaptive TRAIN monitor/challenge use fixed reviewed assisted masks. Public development photos are unpaired; no clean hidden reference, native CCTV or FINAL truth is inferred. Hole-L1 reduction is not restoration structure or hidden identity recovery. R1 remains unchanged pending full visual/user review.',
              'state_Adam_RNG_full_audit_owned_separately': True, 'FIT_CE_replay_owned_separately': True, 'visual_quality_review_performed_by_auditor': False,
              'model_qualified': False, 'automatic_app_promotion': False, 'execution': {'Torch_imported': False, 'neural_calls': 0, 'gradients': 0, 'optimizer_updates': 0, 'VM_calls': 0},
              'checks': CHECKS, 'seconds': time.monotonic() - start}
    with destination.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'complete': True, 'report': pin(destination), 'checks': CHECKS, 'outputs': len(evaluations), 'NPZ': len(array_reviews), 'outside_RGB_bytes_exact': outside_bytes, 'terminal_comparison': comparisons[str(terminal)], 'GRAY': gray_comparisons[str(terminal)], 'model_qualified': False, 'seconds': report['seconds']}), flush=True)


if __name__ == '__main__':
    main()
