"""Independent clean-SFT saved-array audit and six exact256 comparison sheets.

No Torch, model construction, neural replay, browser, VM or training occurs.
Only the completed producer's frozen files and actual call journal are read.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/codeformer_clean_skip_oracle_local_20261019_v1'
INPUT = ROOT / 'outputs/codeformer_hq_code_oracle_inputs_20261019_v1'
REVIEW = OUT / 'review_v1'
SOURCES = tuple('tr_ffhq_' + x for x in ('03761', '55791', '60978', '34217', '31449', '52422'))
PROFILES = ('clear', 'moderate_existing_gaussian_2p4', 'strong_fixed_blur3p2_lr64_jpeg50')
SITES = {'32': 256, '64': 256, '128': 128, '256': 128}
UPSTREAM = {'logits': ((1, 256, 1024), np.float32), 'lq': ((1, 256, 16, 16), np.float32),
            'postAdaIN': ((1, 256, 16, 16), np.float32), 'codes': ((1, 256), np.int64)}
MANIFEST_SHA = 'cdb70da025284909c96144fd1884464cf89fea6c7f4167e9c5d99bd834dbc2ed'
R2_STATE_SHA = 'a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f'
START = time.monotonic()


def need(value, message):
    if not value:
        raise ValueError(message)


def safe(path, exists=True):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    path = path.absolute()
    need(path.resolve(strict=exists).is_relative_to(ROOT), 'Workspace evidence required')
    for parent in (path, *path.parents):
        if parent.exists() or parent.is_symlink():
            row = parent.lstat()
            need(not stat.S_ISLNK(row.st_mode) and not getattr(row, 'st_file_attributes', 0) & 1024,
                 'Linked/reparse evidence refused')
    if exists:
        row = path.stat()
        need(stat.S_ISREG(row.st_mode) and row.st_nlink == 1, 'Unique regular evidence required')
    return path


def relative(path):
    return str(Path(path).absolute().relative_to(ROOT)).replace('\\', '/')


def pin(path):
    need(time.monotonic() - START < 300, 'Finite300s offline audit cap')
    path = safe(path); before = path.stat()
    with path.open('rb') as stream:
        value = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    need((before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns),
         'Evidence changed during hash')
    return {'bytes': before.st_size, 'sha256': value}


def load(path, cap=4 * 1024**2):
    path = safe(path); need(path.stat().st_size <= cap, 'Bounded metadata required')
    def unique(pairs):
        value = {}
        for key, item in pairs:
            need(key not in value, 'Duplicate metadata field refused'); value[key] = item
        return value
    return json.loads(path.read_bytes(), object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite metadata refused')))


def output_path(name):
    need(isinstance(name, str) and ':' not in name and '\\' not in name and
         not name.startswith('/') and all(p not in ('', '.', '..') for p in name.split('/')), 'Safe portable output path required')
    return safe(OUT.joinpath(*PurePosixPath(name).parts))


def array(name, shape, dtype):
    path = output_path(name); need(path.stat().st_size <= 40 * 1024**2, 'Bounded saved array required')
    value = np.load(io.BytesIO(path.read_bytes()), allow_pickle=False)
    need(isinstance(value, np.ndarray) and value.shape == shape and value.dtype == dtype and np.isfinite(value).all(),
         'Saved finite array shape/dtype differs: ' + name)
    return value


def rgb(path):
    path = safe(path); need(path.stat().st_size <= 4 * 1024**2, 'Bounded RGB256 image required')
    with Image.open(path) as image:
        need(image.format == 'PNG' and image.mode == 'RGB' and image.size == (256, 256), 'Exact RGB256 PNG required')
        return np.asarray(image).copy()


def td(value):
    value = np.ascontiguousarray(value)
    name = {np.dtype(np.float32): 'torch.float32', np.dtype(np.int64): 'torch.int64'}.get(value.dtype)
    need(name is not None, 'Known canonical tensor dtype required')
    header = json.dumps(['value', name, list(value.shape)], separators=(',', ':')).encode()
    result = hashlib.sha256(); result.update(len(header).to_bytes(8, 'big')); result.update(header); result.update(value.tobytes())
    return result.hexdigest()


def measured(prefix, target):
    raw = array(prefix + '-raw.npy', (256, 256, 3), np.float32); pixels = rgb(OUT / (prefix + '.png'))
    need(raw.min() >= 0 and raw.max() <= 1 and np.array_equal(pixels, np.floor(raw * np.float32(255)).astype(np.uint8)),
         'FiniteRAW/floorPNG conversion differs')
    values = {}
    for view, image in (('RAW', raw.astype(np.float64)), ('PNG', pixels.astype(np.float64) / 255.)):
        residual = image - target
        values[view] = {'MAE': float(np.abs(residual).mean()),
            'SSIM': float(structural_similarity(image, target, data_range=1.0, channel_axis=2)),
            'adjacent_RGB_residual_error': float(.5 * (np.abs(residual[:, 1:] - residual[:, :-1]).mean() +
                                                    np.abs(residual[1:] - residual[:-1]).mean()))}
    upstream = {suffix: array(prefix + '-' + suffix + '.npy', *shape_dtype) for suffix, shape_dtype in UPSTREAM.items()}
    codes = upstream['codes']; need(codes.min() >= 0 and codes.max() < 1024, 'Natural codes outside1024 book')
    # A selected code must maximize the actual saved logits. Exact topk tie
    # ordering is not inferred from NumPy; full code arrays remain pair-equal.
    selected = np.take_along_axis(upstream['logits'], codes[..., None], axis=2)[..., 0]
    need(np.array_equal(selected, upstream['logits'].max(axis=2)), 'Saved codes are not actual logit maxima')
    return raw, pixels, upstream, values


def compare(actual, expected, label):
    need(type(actual) in (float, int) and np.isfinite(actual) and abs(actual - expected) <= 2e-15,
         'Saved metric readback differs: ' + label)


def write_json(name, value):
    path = safe(REVIEW / name, exists=False); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    return {'path': relative(path), **pin(path)}


def main():
    need((OUT / 'result_v1.json').is_file() and not (OUT / 'failure.json').exists(), 'Completed successful root diagnostic required')
    need(not REVIEW.exists(), 'Preserve previous/partial offline audit')
    result_path = OUT / 'result_v1.json'; result_pin = pin(result_path); result = load(result_path)
    need(result['complete'] is True and result['neural_forward_attempts'] == result['neural_forward_returns'] == 42 and
         result['clear_capture_forwards'] == 6 and result['paired_forwards'] == 36 and result['SFT_invocations'] == 168,
         'Exact42calls/168SFT completed census required')
    need(all(result[key] == 0 for key in ('optimizer_updates', 'gradient_queries', 'backward_calls', 'VM_calls')) and
         result['model_state_before'] == result['model_state_after'] == R2_STATE_SHA and result['hooks_restored'] is True and
         all(result[key] is False for key in ('training_admitted', 'model_qualified', 'app_default_changed', 'FINAL_used', 'native_CCTV_used')),
         'Frozen inference-only research boundary changed')
    sources = result['source_pins']
    producer = 'scripts/run_codeformer_clean_skip_oracle_local_v1.py'
    need(producer in sources, 'Exact producer source binding required')
    for name, value in sources.items():
        need(pin(name) == value, 'Diagnostic source/input changed: ' + name)
    launch = load(OUT / 'launch.json'); worker = load(OUT / 'worker_started.json')
    need(launch['source_pins'] == sources and launch['source_sha256'] == sources[producer]['sha256'] and
         launch['planned_model_calls'] == 42, 'Prospective launch source/census differs')
    import psutil
    if psutil.pid_exists(worker['pid']):
        try: command = psutil.Process(worker['pid']).cmdline()
        except psutil.NoSuchProcess: command = []
        need(not (safe(producer) in [Path(w).absolute() for w in command if w.endswith('.py')] and '--worker' in command),
             'Wait for exact inference worker to terminate')
    manifest_path = INPUT / 'manifest_v1.json'; need(pin(manifest_path)['sha256'] == MANIFEST_SHA, 'Frozen input manifest changed')
    manifest = load(manifest_path); labels_doc = load(INPUT / 'root_input_review_v1_r1.json')
    labels = {row['id']: row['input_only_label'] for row in labels_doc['cases']}
    need(manifest['source_order'] == list(SOURCES) and len(manifest['cases']) == 18 and
         Counter(labels.values()) == {'readable': 16, 'borderline': 2}, 'Fixed6source18case16/2 scope differs')
    expected = {'site_events.json'}
    for rid in SOURCES:
        expected |= {'capture/' + rid + suffix for suffix in ('-raw.npy', '.png', '-logits.npy', '-lq.npy', '-postAdaIN.npy', '-codes.npy')}
        expected |= {'capture/' + rid + '-encoder' + site + '.npy' for site in SITES}
    for row in manifest['cases']:
        for arm in ('natural', 'clean_skip_oracle'):
            expected |= {'cases/' + row['id'] + '/' + arm + suffix for suffix in
                         ('-raw.npy', '.png', '-logits.npy', '-lq.npy', '-postAdaIN.npy', '-codes.npy')}
    need(set(result['files']) == expected and len(expected) == 277, 'Exact277 output pins required')
    for name, value in result['files'].items():
        need(pin(output_path(name)) == value, 'Output hash differs: ' + name)
    journal_path = OUT / 'operations.jsonl'; need(pin(journal_path) == result['operations_journal'], 'Actual call journal changed')
    need(journal_path.stat().st_size < 128 * 1024, 'Bounded journal required')
    journal = [json.loads(line) for line in journal_path.read_text().splitlines()]
    clear_rows = {row['reference_id']: row for row in manifest['cases'] if row['profile'] == 'clear'}
    call_order = [(clear_rows[rid]['id'], 'capture') for rid in SOURCES]
    call_order += [(row['id'], arm) for row in manifest['cases'] for arm in ('natural', 'clean_skip_oracle')]
    need(len(journal) == 84 and len(call_order) == 42, 'Exact84attempt/return records required')
    for index, (cid, arm) in enumerate(call_order, 1):
        need(journal[2 * index - 2:2 * index] == [{'event': event, 'number': index, 'id': cid, 'arm': arm}
             for event in ('attempt', 'return')], 'Actual42-call order differs')
    site_events = load(OUT / 'site_events.json')
    need(len(site_events) == 42 and [(r['id'], r['arm']) for r in site_events] == call_order, 'Exact42SFT records required')
    events = {(row['id'], row['arm']): row['sites'] for row in site_events}
    clean_hashes = {}; captures = {}; clean_feature_pins = []
    for rid in SOURCES:
        clean_hashes[rid] = {}
        for site, channels in SITES.items():
            name = 'capture/' + rid + '-encoder' + site + '.npy'
            feature = array(name, (1, channels, int(site), int(site)), np.float32)
            clean_hashes[rid][site] = td(feature)
            clean_feature_pins.append({'reference_id': rid, 'site': site, 'path': relative(OUT / name),
                **pin(OUT / name), 'tensor_value_sha256': td(feature), 'shape': list(feature.shape)})
        row = clear_rows[rid]; target = rgb(ROOT / row['metrics_target256']['path']).astype(np.float64) / 255.
        captures[rid] = measured('capture/' + rid, target)
    result_rows = {row['id']: row for row in result['cases']}
    need(len(result_rows) == 18 and list(result_rows) == [r['id'] for r in manifest['cases']], 'Exact18result rows required')
    audited = []; event_report = []; clear_parity = []; paired_upstream = []
    for row in manifest['cases']:
        cid = row['id']; rid = row['reference_id']; target = rgb(ROOT / row['metrics_target256']['path']).astype(np.float64) / 255.
        arms = {}; data = {}
        for arm in ('natural', 'clean_skip_oracle'):
            raw, pixels, upstream, values = measured('cases/' + cid + '/' + arm, target)
            saved = result_rows[cid]['arms'][arm]
            compare(saved['RAW_MAE'], values['RAW']['MAE'], cid + '/' + arm + '/RAW')
            compare(saved['PNG_MAE'], values['PNG']['MAE'], cid + '/' + arm + '/PNG')
            need(saved['upstream'] == {name: td(value) for name, value in upstream.items()}, 'Actual four upstream hashes differ')
            need(saved['SFT_sites'] == events[cid, arm], 'Per-case SFT events differ from actual full journal')
            arms[arm] = values; data[arm] = (raw, pixels, upstream)
            if row['profile'] == 'clear':
                capture = captures[rid]
                need(np.array_equal(raw, capture[0]) and np.array_equal(pixels, capture[1]) and
                     all(np.array_equal(upstream[name], capture[2][name]) for name in UPSTREAM), 'Clean self-substitution/capture parity differs')
        need(all(np.array_equal(data['natural'][2][name], data['clean_skip_oracle'][2][name]) for name in UPSTREAM),
             'Oracle changed logits/LQ/codes/postAdaIN arrays')
        paired_upstream.append({'id': cid, 'all4_actual_upstream_arrays_exact': True})
        if row['profile'] == 'clear':
            clear_parity.append({'id': cid, 'capture_natural_oracle_RAW_PNG_upstream_exact': True})
        for arm in ('natural', 'clean_skip_oracle'):
            sites = events[cid, arm]; need(set(sites) == set(SITES), 'All4sites required')
            for site in SITES:
                entry = sites[site]
                need(entry['decoder_argument_passed_through'] is True and
                     all(isinstance(entry[key], str) and len(entry[key]) == 64 for key in
                         ('observed_encoder_sha256', 'used_encoder_sha256', 'decoder_input_sha256')), 'Valid source-bound site event required')
                need(entry['observed_encoder_sha256'] == events[cid, 'natural'][site]['observed_encoder_sha256'],
                     'Observed feature hashes changed between paired arms')
                need(entry['used_encoder_sha256'] == (entry['observed_encoder_sha256'] if arm == 'natural' else clean_hashes[rid][site]),
                     'Actual used branch features do not match natural or capturedclean teacher')
            event_report.append({'id': cid, 'arm': arm, 'sites': sites, 'actual_used_source_checked': True})
        need(events[cid, 'natural']['32']['decoder_input_sha256'] == events[cid, 'clean_skip_oracle']['32']['decoder_input_sha256'],
             'First SFT decoder input changed before oracle intervention')
        audited.append({'id': cid, 'reference_id': rid, 'profile': row['profile'], 'input_only_label': labels[cid],
            'arms': arms, 'oracle_minus_natural': {view: {field: arms['clean_skip_oracle'][view][field] - arms['natural'][view][field]
                for field in ('MAE', 'SSIM', 'adjacent_RGB_residual_error')} for view in ('RAW', 'PNG')}})
    for rid in SOURCES:
        cid = clear_rows[rid]['id']; sites = events[cid, 'capture']
        need(set(sites) == set(SITES) and all(sites[site]['observed_encoder_sha256'] ==
            sites[site]['used_encoder_sha256'] == clean_hashes[rid][site] and sites[site]['decoder_argument_passed_through'] is True
            for site in SITES), 'Capture feature arrays differ from actual hook event hashes')
        need(sites == events[cid, 'natural'] == events[cid, 'clean_skip_oracle'], 'All6clean site-event three-way parity differs')
        event_report.append({'id': cid, 'arm': 'capture', 'sites': sites, 'actual_used_source_checked': True})
    need(len(event_report) == 42 and sum(len(row['sites']) for row in event_report) == 168, 'Actual168SFT event census required')
    groups = {'all18_TRAIN': audited, 'readable16_TRAIN': [r for r in audited if r['input_only_label'] == 'readable'],
        'borderline2_stress': [r for r in audited if r['input_only_label'] == 'borderline'],
        'readable4_strong': [r for r in audited if r['input_only_label'] == 'readable' and r['profile'] == PROFILES[2]]}
    groups.update({profile: [r for r in audited if r['profile'] == profile] for profile in PROFILES})
    summaries = {}
    for name, rows in groups.items():
        summary = {'cases': len(rows), 'views': {}}
        for view in ('RAW', 'PNG'):
            fields = {}
            for field in ('MAE', 'SSIM', 'adjacent_RGB_residual_error'):
                natural = float(np.mean([r['arms']['natural'][view][field] for r in rows]))
                oracle = float(np.mean([r['arms']['clean_skip_oracle'][view][field] for r in rows]))
                fields[field] = {'natural_mean': natural, 'oracle_mean': oracle, 'oracle_minus_natural': oracle - natural,
                    'oracle_better_cases': sum((r['arms']['clean_skip_oracle'][view][field] > r['arms']['natural'][view][field])
                        if field == 'SSIM' else (r['arms']['clean_skip_oracle'][view][field] < r['arms']['natural'][view][field]) for r in rows),
                    'relative_mean_error_reduction_percent': (natural - oracle) / natural * 100 if natural and field != 'SSIM' else None}
            summary['views'][view] = fields
        summaries[name] = summary
    REVIEW.mkdir(); sheets = []; headers = ['Observed input', 'Paired photograph target', 'R2 natural', 'Clean-SFT-input oracle']
    for rid in SOURCES:
        sheet = Image.new('RGB', (1024, 970), '#f4f1e8'); draw = ImageDraw.Draw(sheet)
        draw.text((6, 6), rid + ' | TRAIN only | exact256 cells | no display enhancement', fill='#111111')
        draw.text((6, 24), 'Unavailable clean-image features replace SFT inputs only. Codes/AdaIN stay observed. Diagnostic, not deployable.', fill='#111111')
        for col, heading in enumerate(headers): draw.text((col * 256 + 4, 48), heading, fill='#111111')
        panels = []
        for index, profile in enumerate(PROFILES):
            cid = rid + '__' + profile; row = next(r for r in manifest['cases'] if r['id'] == cid); y = 98 + index * 288
            draw.text((4, y - 24), profile + ' | input-only: ' + labels[cid], fill='#111111')
            paths = [ROOT / row['input']['path'], ROOT / row['metrics_target256']['path'],
                OUT / ('cases/' + cid + '/natural.png'), OUT / ('cases/' + cid + '/clean_skip_oracle.png')]
            for col, path in enumerate(paths):
                sheet.paste(Image.fromarray(rgb(path)), (col * 256, y))
                panels.append({'case_id': cid, 'column': headers[col], 'path': relative(path), **pin(path),
                    'xyxy_half_open': [col * 256, y, col * 256 + 256, y + 256]})
        path = REVIEW / ('sheets/' + rid + '.png'); path.parent.mkdir(exist_ok=True)
        with path.open('xb') as stream: sheet.save(stream, format='PNG')
        sheets.append({'reference_id': rid, 'path': relative(path), **pin(path), 'panels': panels,
                       'root_personal_visual_review_complete': False})
    need(pin(result_path) == result_pin and all(pin(name) == value for name, value in sources.items()), 'Producer/source changed during audit')
    need(all(pin(output_path(name)) == value for name, value in result['files'].items()), 'Saved output changed during audit')
    need(not any(name == 'torch' or name.startswith('torch.') for name in sys.modules), 'Offline no-neural auditor imported Torch')
    visual_pin = write_json('visual_manifest_v1.json', {'complete': True, 'sheets': sheets, 'count': 6,
        'exact256_cells_no_enhancement': True, 'root_personal_review_pending': True, 'model_qualified': False})
    audit = {'format': 'clean-SFT-feature-oracle-independent-saved-array-audit-v1', 'complete': True, 'audit_pass': True,
        'UTC': datetime.now(timezone.utc).isoformat(), 'source_result': {'path': relative(result_path), **result_pin},
        'source_pins': sources, 'auditor': {'path': relative(__file__), **pin(__file__)},
        'saved_output_file_pins_verified': 277, 'actual_model_calls': 42, 'actual_journal_attempts_returns': [42, 42],
        'actual_SFT_invocations': 168, 'actual6capture24clean_feature_arrays': clean_feature_pins,
        'clean_feature_payload_FP32_bytes': 6 * 45 * 1024**2, 'all18_upstream4arrays_exact': paired_upstream,
        'all6_clear_capture_natural_oracle_parity': clear_parity, 'all42_source_bound_SFT_records_checked': event_report,
        'degraded_branch_full_arrays_saved': False,
        'observed_feature_evidence_limit': 'Degraded branch features are source-bound actual hook hashes, not independently recomputed full tensors; paired upstream arrays are independently equal.',
        'decoder_first32_input_pair_exact': True, 'decoder_later_sites_allowed_to_change_causally': True,
        'all42_RAW_finite_float32_and_PNG_floor_exact': True, 'all36_saved_RAW_PNG_MAE_recomputed': True,
        'independent_additional_metrics': {'SSIM': 'skimage structural_similarity float64 RGB, data_range1, channel_axis2, defaults',
            'adjacent_RGB_residual_error': '.5*mean(abs(horizontal residual differences)) + .5*mean(abs(vertical residual differences))',
            'target': 'direct1024-to-LANCZOS256 RGB8.astype(float64)/255', 'not_old_DGP_structure_gate': True},
        'cases': audited, 'profile_readability_summaries': summaries, 'visual_manifest': visual_pin,
        'root_personal_visual_review_pending': True, 'oracle_not_deployable': True,
        'model_qualified': False, 'training_admitted': False, 'app_default_changed': False, 'FINAL_used': False, 'native_CCTV_used': False,
        'execution_boundary': {'Torch_imports': 0, 'model_loads': 0, 'model_constructions': 0, 'neural_forwards': 0,
            'gradient_queries': 0, 'optimizer_updates': 0, 'browser_calls': 0, 'VM_calls': 0, 'root_document_edits': 0},
        'seconds': time.monotonic() - START}
    receipt = write_json('audit_v1.json', audit)
    print(json.dumps({'complete': True, 'audit_pass': True, 'receipt': receipt, 'visual_manifest': visual_pin,
        'sheets': 6, 'auditor_neural_calls': 0, 'summaries': summaries, 'seconds': time.monotonic() - START}))


if __name__ == '__main__':
    main()
