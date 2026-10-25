"""Independent saved-array/codebook audit and six exact256-cell sheets.

Never executes the inference producer, imports returned code, starts a model,
performs a neural forward/backward, trains, launches a browser or uses a VM.
Two weights_only CPU reads provide exact frozen codebooks for arithmetic checks.
Run only after the separately bounded diagnostic worker has finished.
"""
from collections import Counter
import argparse
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/codeformer_hq_code_oracle_local_20261019_v1'
INPUT = ROOT / 'outputs/codeformer_hq_code_oracle_inputs_20261019_v1'
REVIEW = OUT / 'review_v1'
MANIFEST_SHA = 'cdb70da025284909c96144fd1884464cf89fea6c7f4167e9c5d99bd834dbc2ed'
R2 = ROOT / 'outputs/codeformer_role_preserving_training_v1_r2_model/codeformer_role_preserving_r2.pth'
R2_SHA = 'd4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896'
HQ = ROOT / 'outputs/codeformer_teacher_pretrained_v1/vqgan_code1024.pth'
HQ_SHA = '4d1c6741b3cffcbdc2cd1a12b2c3c2442282e042d5de66909cb643d4fa31b20f'
SOURCES = tuple('tr_ffhq_' + x for x in ('03761', '55791', '60978', '34217', '31449', '52422'))
PROFILES = ('clear', 'moderate_existing_gaussian_2p4', 'strong_fixed_blur3p2_lr64_jpeg50')
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
            value = parent.lstat()
            need(not stat.S_ISLNK(value.st_mode) and not getattr(value, 'st_file_attributes', 0) & 1024,
                 'Linked/reparse evidence refused')
    if exists:
        value = path.stat()
        need(stat.S_ISREG(value.st_mode) and (value.st_nlink == 1 or path == HQ),
             'Ordinary evidence required; exact read-only HQ teacher may keep historical hardlinks')
    return path


def pin(path):
    need(time.monotonic() - START < 300, 'Finite300s offline audit cap')
    path = safe(path); before = path.stat()
    with path.open('rb') as stream:
        value = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    need((before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns),
         'Evidence changed during read')
    return {'bytes': before.st_size, 'sha256': value}


def relative(path):
    return str(Path(path).absolute().relative_to(ROOT)).replace('\\', '/')


def bounded_json(path, maximum=4 * 1024**2):
    path = safe(path); need(path.stat().st_size <= maximum, 'Bounded metadata required')
    def unique(pairs):
        result = {}
        for name, value in pairs:
            need(name not in result, 'Duplicate metadata key refused')
            result[name] = value
        return result
    return json.loads(path.read_bytes(), object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite metadata refused')))


def out_path(name):
    need(isinstance(name, str) and '\\' not in name and ':' not in name and not name.startswith('/'),
         'Portable output filename required')
    parts = PurePosixPath(name).parts
    need(parts and all(x not in ('', '.', '..') for x in name.split('/')), 'Safe relative output required')
    return safe(OUT.joinpath(*parts))


def array(name, shape=None, dtype=None):
    path = out_path(name); need(path.stat().st_size <= 4 * 1024**2, 'Bounded saved array required')
    result = np.load(io.BytesIO(path.read_bytes()), allow_pickle=False)
    need(isinstance(result, np.ndarray) and (shape is None or result.shape == shape) and
         (dtype is None or result.dtype == dtype), 'Saved array shape/dtype differs: ' + name)
    need(np.isfinite(result).all(), 'Nonfinite saved array: ' + name)
    return result


def rgb(path, size=256):
    path = safe(path); need(path.stat().st_size <= 4 * 1024**2, 'Bounded RGB image required')
    with Image.open(path) as image:
        need(image.format == 'PNG' and image.mode == 'RGB' and image.size == (size, size), 'Exact RGB PNG geometry required')
        return np.asarray(image).copy()


def tensor_digest(name, values, torch):
    tensor = torch.from_numpy(np.ascontiguousarray(values))
    header = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(',', ':')).encode()
    digest = hashlib.sha256(); digest.update(len(header).to_bytes(8, 'big')); digest.update(header)
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def new_json(name, value):
    path = safe(REVIEW / name, exists=False); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    return {'path': relative(path), **pin(path)}


def compare_float(actual, expected, message):
    need(type(actual) in (float, int) and np.isfinite(actual) and abs(float(actual) - float(expected)) <= 2e-15,
         message)


def main():
    need((OUT / 'result_v1.json').is_file() and not (OUT / 'failure.json').exists(),
         'Actual terminal successful diagnostic result required; preserve failures')
    need(not REVIEW.exists(), 'Preserve prior or partial offline audit; no overwrite')
    result_path = OUT / 'result_v1.json'; result_pin = pin(result_path); result = bounded_json(result_path)
    need(result['complete'] is True and result['evidence_integrity_pass'] is True and
         result['neural_forward_attempts'] == result['neural_forward_returns'] == 43 and
         result['teacher_forwards'] == 6 and result['CodeFormer_forwards'] == 37 and
         all(result[key] == 0 for key in ('optimizer_updates', 'gradient_queries', 'backward_calls', 'VM_calls')),
         'Exact completed43 inference-only call count required')
    need(result['model_qualified'] is False and result['app_default_changed'] is False and
         result['FINAL_used'] is False and result['native_CCTV_used'] is False and result['training_admitted'] is False,
         'Diagnostic deployment/qualification scope changed')
    source_pins = result['source_pins']
    producer_names = [name for name in source_pins if name.startswith('scripts/run_codeformer_hq_code_oracle_local_') and name.endswith('.py')]
    need(len(producer_names) == 1, 'Exact one actual producing inference-script source binding required')
    producer_path = safe(producer_names[0])
    need(pin(producer_path) == source_pins[producer_names[0]], 'Actual producing inference-script source changed')
    launch = bounded_json(OUT / 'launch.json')
    need(launch['source_sha256'] == source_pins[producer_names[0]]['sha256'] and
         launch['source_pins'] == source_pins and launch['planned_forwards'] == 43 and launch['optimizer_updates'] == 0,
         'Actual result must bind exact launch source and prospective finite census')
    # A result file is not by itself a terminal-process observation.
    import psutil
    worker = bounded_json(OUT / 'worker_started.json')
    if psutil.pid_exists(worker['pid']):
        try:
            command = psutil.Process(worker['pid']).cmdline()
        except psutil.NoSuchProcess:
            command = []
        actual_scripts = [Path(word).absolute() for word in command if word.endswith('.py')]
        need(producer_path not in actual_scripts or '--worker' not in command,
             'Wait for the exact diagnostic worker to terminate')
    manifest_path = INPUT / 'manifest_v1.json'; need(pin(manifest_path)['sha256'] == MANIFEST_SHA, 'Frozen18-input manifest differs')
    manifest = bounded_json(manifest_path); input_plan = bounded_json(INPUT / 'plan_v1.json')
    review_path = INPUT / 'root_input_review_v1_r1.json'; review = bounded_json(review_path)
    need(manifest['source_order'] == list(SOURCES) and len(manifest['cases']) == 18 and len(manifest['targets']) == 6,
         'Exact fixed six-source18-case set required')
    need(review['complete'] is True and review['manifest_sha256'] == MANIFEST_SHA and
         review['inference_only_diagnostic_admitted'] is True and review['training_admitted'] is False,
         'Exact input-only diagnostic review required')
    labels = {row['id']: row['input_only_label'] for row in review['cases']}
    need(Counter(labels.values()) == {'readable': 16, 'borderline': 2}, 'Keep declared16readable/2borderline scope')
    for name, value in source_pins.items():
        need(pin(name) == value, 'Diagnostic input/source pin changed: ' + name)
    need(source_pins[relative(R2)]['sha256'] == R2_SHA and source_pins[relative(HQ)]['sha256'] == HQ_SHA,
         'Exact R2/HQ model fingerprints required')
    need(result['state_before'] == result['state_after'] and result['state_before']['R2'] == R2_STATE_SHA,
         'Retained frozen model-state claims differ')
    expected_names = {'parity/unwrapped_R2' + suffix for suffix in ('-raw.npy', '.png', '-logits.npy', '-lq.npy')}
    for reference in SOURCES:
        expected_names |= {'teacher/' + reference + suffix for suffix in
            ('-codes.npy', '-native-HQ-codes.npy', '-HQ-encoder-features.npy', '-raw.npy', '.png')}
    for case in manifest['cases']:
        expected_names.add('cases/' + case['id'] + '/metrics.json')
        for arm in ('natural', 'HQ_code_oracle'):
            expected_names |= {'cases/' + case['id'] + '/' + arm + suffix for suffix in
                ('-raw.npy', '.png', '-logits.npy', '-lq.npy', '-natural-codes.npy', '-used-codes.npy')}
    need(set(result['files']) == expected_names, 'Exact planned saved-array/image/metric output file census required')
    for name, value in result['files'].items():
        need(pin(out_path(name)) == value, 'Saved output pin differs: ' + name)
    journal_path = OUT / 'operations.jsonl'
    need(pin(journal_path) == result['operations_journal'] and journal_path.stat().st_size < 128 * 1024,
         'Exact bounded43-call operation journal required')
    journal = [json.loads(line) for line in journal_path.read_text().splitlines()]
    expected_labels = ['unwrapped_R2_parity/' + manifest['cases'][0]['id']] + ['HQ/' + r for r in SOURCES]
    expected_labels += [row['id'] + '/' + arm for row in manifest['cases'] for arm in ('natural', 'HQ_code_oracle')]
    need(len(journal) == 86 and len(expected_labels) == 43, 'Exact86 attempt/return records required')
    for number, label in enumerate(expected_labels, 1):
        need(journal[2 * (number - 1):2 * number] == [
            {'event': 'attempt', 'number': number, 'label': label}, {'event': 'return', 'number': number, 'label': label}],
            'Actual operation journal order or outcome differs')
    # Deserialization and arithmetic only: construct no neural modules.
    import torch
    torch.set_num_threads(4)
    r2_saved = torch.load(R2, weights_only=True, map_location='cpu', mmap=True)
    hq_saved = torch.load(HQ, weights_only=True, map_location='cpu', mmap=True)
    need(set(r2_saved) == set(hq_saved) == {'params_ema'}, 'Exact inference container key required')
    r2_book = r2_saved['params_ema']['quantize.embedding.weight'].detach().cpu().contiguous()
    hq_book = hq_saved['params_ema']['quantize.embedding.weight'].detach().cpu().contiguous()
    need(r2_book.shape == hq_book.shape == (1024, 256) and r2_book.dtype == hq_book.dtype == torch.float32,
         'Exact finite FP32 codebook geometry required')
    need(torch.isfinite(r2_book).all().item() and torch.isfinite(hq_book).all().item(), 'Nonfinite codebook')
    rb64, hb64 = r2_book.double(), hq_book.double()
    distances = ((hb64 ** 2).sum(1)[:, None] + (rb64 ** 2).sum(1)[None, :] - 2 * hb64 @ rb64.T).clamp_min(0)
    nearest = distances.topk(2, largest=False, dim=1)
    compatibility = {'byte_exact_codebook': torch.equal(rb64, hb64),
        'maximum_codebook_absolute_error': float((rb64 - hb64).abs().max().item()),
        'nearest_rows_identity': int((nearest.indices[:, 0] == torch.arange(1024)).sum().item()),
        'minimum_second_minus_first_squared_distance': float((nearest.values[:, 1] - nearest.values[:, 0]).min().item()),
        'HQ_generator_is_R2_generator_byte_exact': all(torch.equal(r2_saved['params_ema'][name], value)
            for name, value in hq_saved['params_ema'].items() if name.startswith('generator.')),
        'teacher_reconstruction_is_separate_reference_not_exact_R2_prior_ceiling': True}
    need(compatibility == result['codebook_compatibility'], 'Actual saved codebook/generator compatibility differs')
    source_originals = {row['reference_id']: row['source1024'] for row in input_plan['frozen_sources']}
    teacher_metrics = {}; teacher_comparisons = []; teacher_input_pins = []
    teacher_arrays = {}; all_pairs = []; case_reports = []
    for target in manifest['targets']:
        rid = target['reference_id']; need(rid in SOURCES, 'Unknown teacher reference')
        hq_input = rgb(ROOT / target['HQ512']['path'], 512)
        need(hashlib.sha256(hq_input.tobytes()).hexdigest() == target['HQ512']['RGB8_sha256'], 'Exact teacher input pixels differ')
        original_pin = source_originals[rid]
        need(pin(original_pin['path']) == {key: original_pin[key] for key in ('bytes', 'sha256')}, 'Original1024 source changed')
        source1024 = rgb(ROOT / original_pin['path'], 1024)
        source_image = Image.fromarray(source1024)
        need(np.array_equal(hq_input, np.asarray(source_image.resize((512, 512), Image.Resampling.LANCZOS))) and
             np.array_equal(rgb(ROOT / target['metrics256']['path']), np.asarray(source_image.resize((256, 256), Image.Resampling.LANCZOS))),
             'Exact direct1024 HQ512/metrics256 source derivation differs')
        teacher_input_pins.append({'reference_id': rid, 'input_HQ512': target['HQ512'],
            'original1024': original_pin, 'actual_direct1024_HQ512_and_metrics256_exact': True})
        features = array('teacher/' + rid + '-HQ-encoder-features.npy', (1, 256, 16, 16), np.float32)
        z = torch.from_numpy(features).permute(0, 2, 3, 1).contiguous().view(-1, 256)
        projected = ((z ** 2).sum(1, keepdim=True) + (r2_book ** 2).sum(1)[None, :] -
                     2 * torch.matmul(z, r2_book.T)).argmin(1).reshape(1, 256).numpy()
        native = ((z ** 2).sum(1, keepdim=True) + (hq_book ** 2).sum(1) -
                  2 * torch.matmul(z, hq_book.t())).argmin(1).reshape(1, 256).numpy()
        saved_projected = array('teacher/' + rid + '-codes.npy', (1, 256), np.int64)
        saved_native = array('teacher/' + rid + '-native-HQ-codes.npy', (1, 256), np.int64)
        need(np.array_equal(projected, saved_projected) and np.array_equal(native, saved_native),
             'Actual HQ feature projection or native HQ label differs')
        teacher_arrays[rid] = saved_projected
        teacher_comparisons.append({'reference_id': rid, 'native_HQ_vs_R2_projected_changed_positions': int(np.count_nonzero(saved_native != saved_projected)),
            'positions': 256, 'HQ_native_indices_recomputed_exact': True, 'exact_R2_projected_indices_recomputed_exact': True})
        raw = array('teacher/' + rid + '-raw.npy', (256, 256, 3), np.float32)
        png = rgb(OUT / ('teacher/' + rid + '.png'))
        need(raw.min() >= 0 and raw.max() <= 1 and np.array_equal(png, np.floor(raw * np.float32(255)).astype(np.uint8)),
             'HQ teacher delivered RAW/PNG differs')
        target256 = rgb(ROOT / target['metrics256']['path']).astype(np.float64) / 255.
        teacher_metrics[rid] = {'RAW_MAE': float(np.abs(raw.astype(np.float64) - target256).mean()),
            'PNG_MAE': float(np.abs(png.astype(np.float64) / 255. - target256).mean()),
            'separate_teacher_generator_not_exact_R2_prior_ceiling': True}
    cases_by_id = {row['id']: row for row in result['cases']}
    need(len(cases_by_id) == 18 and [row['id'] for row in result['cases']] == [row['id'] for row in manifest['cases']],
         'Exact18 output case order required')
    for row in manifest['cases']:
        cid = row['id']; rid = row['reference_id']; target = rgb(ROOT / row['metrics_target256']['path']).astype(np.float64) / 255.
        measured = {}; pair = {}
        saved_metric = bounded_json(OUT / ('cases/' + cid + '/metrics.json'))
        need(saved_metric == cases_by_id[cid], 'Saved standalone case metrics differ from completed result')
        for arm in ('natural', 'HQ_code_oracle'):
            prefix = 'cases/' + cid + '/' + arm
            raw = array(prefix + '-raw.npy', (256, 256, 3), np.float32)
            pixels = rgb(OUT / (prefix + '.png'))
            need(raw.min() >= 0 and raw.max() <= 1 and np.array_equal(pixels, np.floor(raw * np.float32(255)).astype(np.uint8)),
                 'Actual finite RAW/floor PNG conversion differs')
            logits = array(prefix + '-logits.npy', (1, 256, 1024), np.float32)
            lq = array(prefix + '-lq.npy', (1, 256, 16, 16), np.float32)
            natural_codes = array(prefix + '-natural-codes.npy', (1, 256), np.int64)
            used_codes = array(prefix + '-used-codes.npy', (1, 256), np.int64)
            need(natural_codes.min() >= 0 and natural_codes.max() < 1024 and used_codes.min() >= 0 and used_codes.max() < 1024,
                 'Saved code indices outside exact1024 book')
            predicted = torch.topk(torch.softmax(torch.from_numpy(logits), dim=2), 1, dim=2).indices.squeeze(2).numpy()
            need(np.array_equal(predicted, natural_codes), 'Natural code selection does not match actual saved logits')
            need(np.array_equal(used_codes, natural_codes if arm == 'natural' else teacher_arrays[rid]), 'Used intervention codes differ')
            values = {'RAW_MAE': float(np.abs(raw.astype(np.float64) - target).mean()),
                'PNG_MAE': float(np.abs(pixels.astype(np.float64) / 255. - target).mean()),
                'natural_code_mismatch_to_HQ': float(torch.from_numpy(natural_codes != teacher_arrays[rid]).float().mean().item()),
                'used_code_mismatch_to_HQ': float(torch.from_numpy(used_codes != teacher_arrays[rid]).float().mean().item())}
            for name, value in values.items():
                compare_float(saved_metric['arms'][arm][name], value, 'Actual saved metric differs: ' + cid + '/' + arm + '/' + name)
            need(saved_metric['arms'][arm]['logits_tensor_sha256'] == tensor_digest('logits', logits, torch) and
                 saved_metric['arms'][arm]['lq_tensor_sha256'] == tensor_digest('lq', lq, torch), 'Upstream canonical tensor fingerprint differs')
            measured[arm] = values
            pair[arm] = {'logits': logits, 'lq': lq, 'codes': natural_codes}
        need(all(np.array_equal(pair['natural'][name], pair['HQ_code_oracle'][name]) for name in ('logits', 'lq', 'codes')),
             'Intervention changed actual upstream features/logits/natural codes')
        all_pairs.append({'id': cid, 'actual_logits_LQ_natural_codes_equal': True})
        case_reports.append({'id': cid, 'reference_id': rid, 'profile': row['profile'], 'input_only_label': labels[cid],
            'arms': measured, 'oracle_minus_natural_MAE': {view: measured['HQ_code_oracle'][view] - measured['natural'][view]
                    for view in ('RAW_MAE', 'PNG_MAE')}, 'separate_HQ_teacher': teacher_metrics[rid]})
    first = manifest['cases'][0]['id']; natural_prefix = 'cases/' + first + '/natural'
    for suffix in ('-raw.npy', '-logits.npy', '-lq.npy'):
        need(np.array_equal(array('parity/unwrapped_R2' + suffix), array(natural_prefix + suffix)), 'Unwrapped/natural parity changed')
    need(np.array_equal(rgb(OUT / 'parity/unwrapped_R2.png'), rgb(OUT / (natural_prefix + '.png'))), 'Unwrapped/natural PNG parity changed')
    summaries = {}
    groups = {'all18_TRAIN': case_reports, 'readable16_TRAIN': [r for r in case_reports if r['input_only_label'] == 'readable'],
              'borderline2_stress': [r for r in case_reports if r['input_only_label'] == 'borderline']}
    groups.update({profile: [r for r in case_reports if r['profile'] == profile] for profile in PROFILES})
    for name, rows in groups.items():
        summaries[name] = {'cases': len(rows), 'views': {}}
        for view in ('RAW_MAE', 'PNG_MAE'):
            natural = float(np.mean([r['arms']['natural'][view] for r in rows]))
            oracle = float(np.mean([r['arms']['HQ_code_oracle'][view] for r in rows]))
            summaries[name]['views'][view] = {'natural_mean': natural, 'oracle_mean': oracle,
                'oracle_minus_natural': oracle - natural,
                'relative_mean_MAE_reduction_percent': (natural - oracle) / natural * 100 if natural else None,
                'oracle_better_cases': sum(r['arms']['HQ_code_oracle'][view] < r['arms']['natural'][view] for r in rows)}
    REVIEW.mkdir()
    sheets = []
    headers = ['Observed input', 'Paired photo target', 'R2 natural', 'HQ-code oracle', 'HQ teacher (separate)']
    by_case = {row['id']: row for row in manifest['cases']}
    for rid in SOURCES:
        sheet = Image.new('RGB', (1280, 970), '#f4f1e8'); draw = ImageDraw.Draw(sheet)
        draw.text((6, 6), rid + ' | TRAIN diagnostic | exact256 cells | no display enhancement', fill='#111111')
        draw.text((6, 24), 'Oracle uses unavailable clean-reference codes. HQ teacher has a separate generator; it is not the R2-prior ceiling.', fill='#111111')
        for column, header in enumerate(headers):
            draw.text((column * 256 + 4, 48), header, fill='#111111')
        panels = []
        for index, profile in enumerate(PROFILES):
            cid = rid + '__' + profile; row = by_case[cid]; y = 98 + index * 288
            draw.text((4, y - 24), profile + ' | input-only: ' + labels[cid], fill='#111111')
            paths = [ROOT / row['input']['path'], ROOT / row['metrics_target256']['path'],
                OUT / ('cases/' + cid + '/natural.png'), OUT / ('cases/' + cid + '/HQ_code_oracle.png'),
                OUT / ('teacher/' + rid + '.png')]
            for column, path in enumerate(paths):
                pixels = rgb(path); sheet.paste(Image.fromarray(pixels), (column * 256, y))
                panels.append({'case_id': cid, 'column': headers[column], 'path': relative(path), **pin(path),
                               'xyxy_half_open': [column * 256, y, column * 256 + 256, y + 256]})
        destination = REVIEW / ('sheets/' + rid + '.png'); destination.parent.mkdir(exist_ok=True)
        with destination.open('xb') as stream:
            sheet.save(stream, format='PNG')
        sheets.append({'reference_id': rid, 'path': relative(destination), **pin(destination),
            'panels': panels, 'root_personal_visual_review_complete': False})
    need(pin(result_path) == result_pin and all(pin(name) == value for name, value in source_pins.items()), 'Producer evidence changed during audit')
    need(all(pin(out_path(name)) == value for name, value in result['files'].items()), 'Output pixels changed during audit')
    visual_pin = new_json('visual_manifest_v1.json', {'complete': True, 'sheets': sheets, 'count': 6,
        'exact256_pixels_no_enhancement': True, 'root_personal_review_pending': True,
        'columns': headers, 'model_qualified': False, 'oracle_not_deployable': True})
    result_audit = {'format': 'codeformer-HQ-code-oracle-independent-array-audit-v1', 'complete': True, 'audit_pass': True,
        'UTC': datetime.now(timezone.utc).isoformat(), 'source_result': {'path': relative(result_path), **result_pin},
        'source_pins': source_pins, 'audited_output_file_count': len(result['files']),
        'actual_result_root': relative(OUT), 'actual_producing_source': {'path': producer_names[0], **pin(producer_path)},
        'source_auditor': {'path': relative(__file__), **pin(__file__)},
        'actual_operation_journal_attempts': 43, 'actual_operation_journal_returns': 43,
        'actual_top_level_model_calls': 43, 'auditor_neural_forwards': 0,
        'actual_unwrapped_natural_RAW_PNG_logits_LQ_parity': True,
        'all18_upstream_pair_equalities': all_pairs, 'all18_natural_logits_to_codes_exact': True,
        'all18_oracle_used_codes_equal_exact_HQ_features_projected_R2_codes': True,
        'all6_HQ_features_native_and_exact_R2_projection_recomputed': teacher_comparisons,
        'all6_exact_teacher_input_files': teacher_input_pins,
        'codebook_generator_compatibility_recomputed': compatibility,
        'all_RAW_finite_float32_bounded_and_PNG_floor_exact': True,
        'all36_RAW_PNG_paired_metrics_recomputed': True, 'case_metrics': case_reports,
        'profile_readability_summaries': summaries, 'separate_HQ_teacher_metrics': teacher_metrics,
        'visual_manifest': visual_pin, 'root_personal_visual_review_pending': True,
        'clean_reference_oracle_unavailable_at_deployment': True,
        'teacher_reconstruction_is_not_R2_prior_ceiling': True,
        'oracle_gain_is_not_learnability_or_identity_or_deployment_proof': True,
        'source_model_state_before_after_claims_exact': result['state_before'] == result['state_after'],
        'model_qualified': False, 'app_default_changed': False, 'training_admitted': False,
        'FINAL_used': False, 'native_CCTV_used': False,
        'execution_boundary': {'weights_only_CPU_reads': 2, 'model_constructions': 0, 'neural_forwards': 0,
            'gradient_queries': 0, 'optimizer_updates': 0, 'VM_calls': 0, 'browser_calls': 0, 'root_document_edits': 0},
        'seconds': time.monotonic() - START}
    audit_pin = new_json('audit_v1.json', result_audit)
    print(json.dumps({'complete': True, 'audit_pass': True, 'receipt': audit_pin, 'visual_manifest': visual_pin,
        'sheets': 6, 'actual_producer_model_calls': 43, 'audit_neural_calls': 0,
        'profile_readability_summaries': summaries, 'seconds': time.monotonic() - START}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-root', type=Path, default=OUT,
                        help='Exact completed diagnostic namespace, including the preserved-v1 R1 replacement.')
    args = parser.parse_args()
    chosen = args.result_root if args.result_root.is_absolute() else ROOT / args.result_root
    chosen = chosen.absolute()
    need(chosen.resolve(strict=False).parent == (ROOT / 'outputs').resolve() and
         chosen.name in ('codeformer_hq_code_oracle_local_20261019_v1', 'codeformer_hq_code_oracle_local_20261019_v1_r1'),
         'Choose the exact known v1 or R1 result namespace; do not redirect to arbitrary evidence')
    OUT = chosen
    REVIEW = OUT / 'review_v1'
    main()
