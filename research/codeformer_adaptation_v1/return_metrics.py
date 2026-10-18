"""Independent saved-pixel audit of CodeFormer adaptation returns.

CPU image/array work only: no neural framework imports, gradients or VM calls.
Native CCTV has no clean-reference metrics. VM-generated input pixels are
authoritative after their saved hashes are verified; cross-platform JPEG pixel
identity is deliberately not required. Pair parameters/roles/source and clean
target hashes remain exact.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import time

import numpy as np
from PIL import Image

from . import data

PROTOCOL_SHA = 'e463222441c0930a01a85ee0711a1ea31b0d80ed3487a4a5459c71e65f517d3e'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def checked_path(root, relative):
    posix = PurePosixPath(relative)
    require(not posix.is_absolute() and posix.as_posix() == relative
            and '..' not in posix.parts and ':' not in relative and '\\' not in relative,
            'Invalid relative artifact path')
    path = Path(root).joinpath(*posix.parts)
    require(path.resolve(strict=True).is_relative_to(Path(root).resolve(strict=True)), 'Artifact path escape')
    require(path.is_file() and not path.is_symlink(), 'Artifact missing or nonregular')
    return path


def rgb8(path):
    with Image.open(path) as image:
        require(image.mode == 'RGB' and image.size == (256, 256), 'Expected RGB 256 PNG')
        return np.asarray(image).copy()


def mask256(path):
    with Image.open(path) as image:
        require(image.size == (256, 256), 'Observed mask geometry differs')
        pixels = np.asarray(image.convert('L')).copy()
    require(np.isin(pixels, [0, 255]).all() and pixels.any(), 'Observed mask must be binary/nonempty')
    return pixels > 0


def load_raw(path, expected_sha):
    require(sha(path) == expected_sha, 'RAW file hash differs')
    with Path(path).open('rb') as stream:
        raw = np.load(stream, allow_pickle=False)
        require(stream.read(1) == b'', 'Unexpected bytes follow RAW array')
    require(raw.dtype == np.float32 and raw.shape == (256, 256, 3), 'RAW dtype/geometry differs')
    require(raw.flags.c_contiguous and np.isfinite(raw).all(), 'RAW is nonfinite/noncontiguous')
    require(raw.min() >= 0 and raw.max() <= 1, 'RAW range differs')
    return raw


def expected_native_png(raw, input_pixels, observed):
    require(raw.shape == input_pixels.shape == (256, 256, 3)
            and input_pixels.dtype == np.uint8 and observed.shape == (256, 256)
            and observed.dtype == np.bool_, 'PNG reconstruction input geometry/dtype differs')
    result = np.floor(np.clip(raw, 0, 1) * 255).astype(np.uint8)
    # The VM restores exact original pixels outside observed support.
    result[~observed] = input_pixels[~observed]
    return result


def paired_metrics(prediction, target, observed, *, evidence):
    require(prediction.shape == (256, 256, 3) and observed.shape == (256, 256)
            and observed.dtype == np.bool_ and observed.any(), 'Metric geometry/observed support differs')
    require(np.isfinite(prediction).all(), 'Nonfinite metric prediction')
    if evidence == 'native_unpaired':
        require(target is None, 'Native CCTV cannot have an invented clean target')
        return {'mae': None, 'mse': None, 'psnr': None}
    require(evidence == 'paired_synthetic' and target is not None and target.shape == prediction.shape,
            'Paired target/evidence differs')
    require(np.isfinite(target).all(), 'Nonfinite metric target')
    # Select observed scalar samples explicitly; no padding contributes.
    error = prediction.astype(np.float64)[observed].reshape(-1) - target.astype(np.float64)[observed].reshape(-1)
    absolute = np.abs(error)
    mse = float(np.dot(error, error) / error.size)
    return {'mae': float(absolute.sum(dtype=np.float64) / absolute.size),
            'mse': mse, 'psnr': None if mse == 0 else float(-10 * np.log10(mse))}


def compare_metric_dict(rebuilt, saved, label):
    require(set(saved) == {'mae', 'mse', 'psnr'}, 'Metric schema differs: ' + label)
    for key, expected in rebuilt.items():
        actual = saved[key]
        if expected is None:
            require(actual is None, 'Native/perfect metric null differs: ' + label + '/' + key)
        else:
            require(isinstance(actual, (int, float)) and math.isfinite(actual)
                    and math.isclose(actual, expected, rel_tol=2e-11, abs_tol=2e-12),
                    'Saved metric differs: ' + label + '/' + key)


def independently_clear_gate(candidate, baseline, evaluation):
    require(set(candidate) == set(baseline) and len(candidate) == 61, 'Clear namespace must contain exact61 controls')
    report = {'pass': True, 'views': {}, 'qualification_not_implied': True}
    for view in ('raw', 'native_png'):
        deltas = {identifier: candidate[identifier][view] - baseline[identifier][view] for identifier in sorted(candidate)}
        mean, worst = sum(deltas.values()) / len(deltas), max(deltas.values())
        passed = (mean <= evaluation['mean_clear_mae_delta_stop']
                  and worst <= evaluation['worst_per_case_clear_mae_delta_stop'])
        report['views'][view] = {'cases': len(deltas), 'mean_mae_delta': mean,
            'worst_per_case_mae_delta': worst, 'deltas': deltas, 'pass': passed}
        report['pass'] = report['pass'] and passed
    return report


def _audit_vm_inputs(packet, returned, protocol, out):
    references = data.load_reference_manifest(packet, 'validation', packet / protocol['data_manifest'],
        expected_manifest_sha256=protocol['files_sha256'][protocol['data_manifest']])
    profiles = protocol['evaluation']['profiles']
    manifest = read(returned / protocol['outputs'] / 'validation_inputs.json')
    require(manifest['complete'] and manifest['cases_count'] == 318 and len(manifest['cases']) == 318,
            'Validation input manifest incomplete')
    require(manifest['saved_VM_pixels_reused_for_all_snapshots'] is True
            and manifest['cross_platform_degraded_pixels_not_assumed_equal'] is True, 'VM input reuse policy differs')
    specs, receipts, differences = {}, [], []
    rows = {r['id']: r for r in read(packet / protocol['data_manifest'])['references']}
    for ref_index, reference in enumerate(references):
        clean = data.derive_clean_target(reference)
        clean_rgb8 = np.rint(clean * 255).astype(np.uint8)
        target_path = checked_path(packet, rows[reference.id]['existing_target_path'])
        require(sha(target_path) == rows[reference.id]['existing_target_sha256'], 'Validation target hash differs')
        target = rgb8(target_path).astype(np.float32) / np.float32(255)
        for profile_index, profile in enumerate(profiles):
            identifier = 'validation_' + reference.id + '_' + profile
            seed = protocol['seed'] + 1000000 + ref_index * 100 + profile_index
            entry = manifest['cases'][identifier]
            input_path = checked_path(returned, entry['path'])
            record_path = checked_path(returned, entry['pair_record_path'])
            require(sha(input_path) == entry['sha256'], 'Saved VM input file hash differs')
            record = read(record_path)
            require(record['format'] == 'codeformer-adaptation-photographic-pair-v1'
                    and record['reference_id'] == reference.id and record['role'] == 'validation', 'Validation source/role differs')
            require(record['source_path'] == reference.source_relative_path
                    and record['source_sha256'] == reference.source_sha256
                    and record['reference_manifest_sha256'] == reference.manifest_sha256,
                    'Validation source/manifest binding differs')
            require(record['profile'] == profile and record['seed'] == seed, 'Validation seed/profile differs')
            parameters = data.profile_parameters(profile, seed)
            require(record['parameters'] == parameters, 'Recorded degradation parameters differ')
            require(record['geometry'] == {'source': [1024, 1024], 'input': [256, 256], 'target': [512, 512],
                    'crop': None, 'warp': None, 'covering_simulation': False}, 'Validation geometry differs')
            pixels = rgb8(input_path)
            floats = pixels.astype(np.float32) / np.float32(255)
            require(data.pixel_sha256(pixels) == record['input256_rgb8_sha256']
                    and data.pixel_sha256(floats) == record['input256_float32_sha256'], 'VM input pixel hash differs')
            require(data.pixel_sha256(clean) == record['clean512_float32_sha256']
                    and data.pixel_sha256(clean_rgb8) == record['clean512_rgb8_sha256']
                    == reference.clean512_rgb8_sha256, 'Clean512 target hash differs')
            # Rebuild locally only to measure platform differences. Require clean
            # and no-JPEG clear controls exact; never demand LinuxJPEG==Windows.
            local = data.make_pair(reference, profile, seed)
            local_pixels = np.rint(local.input256 * 255).astype(np.uint8)
            maximum = int(np.max(np.abs(local_pixels.astype(np.int16) - pixels.astype(np.int16))))
            differing = int(np.count_nonzero(local_pixels != pixels))
            if profile == 'clear':
                require(maximum == 0, 'Clear input differs from exact source geometry')
            if differing:
                differences.append({'id': identifier, 'profile': profile, 'differing_RGB_scalars': differing,
                    'maximum_abs_255': maximum, 'VM_runtime': record['runtime'], 'local_runtime': local.record['runtime']})
            specs[identifier] = {'id': identifier, 'evidence': 'paired_synthetic', 'profile': profile,
                'source': 'FFHQ validation', 'input_path': str(input_path), 'input_sha256': entry['sha256'],
                'target_path': str(target_path), 'input_pixels': pixels, 'target': target,
                'observed': np.ones((256, 256), dtype=bool)}
            receipts.append({'id': identifier, 'input_file_sha256': entry['sha256'], 'pair_record_sha256': sha(record_path),
                'source_sha256': reference.source_sha256, 'parameters': parameters,
                'clean512_rgb8_sha256': reference.clean512_rgb8_sha256, 'VM_pixel_hashes_exact': True})
    require(set(specs) == set(manifest['cases']), 'Validation input ID namespace differs')
    write(out / 'validation_input_audit.json', {'complete': True, 'cases': receipts,
        'cross_platform_input_differences': differences, 'VM_pixels_authoritative': True,
        'all_roles_seeds_parameters_and_clean512_exact': True, 'native_or_final_as_GT': False})
    return specs, {'cases': len(receipts), 'cross_platform_differing_cases': len(differences),
                   'differing_profiles': dict(Counter(r['profile'] for r in differences))}


def audit_return_metrics(packet, returned, output):
    packet, returned, output = Path(packet).resolve(), Path(returned).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    require(not (output / 'metrics_audit.json').exists(), 'Retain prior metrics audit')
    started = time.monotonic()
    require(sha(packet / 'protocol.json') == sha(returned / 'protocol.json') == PROTOCOL_SHA, 'Protocol differs')
    protocol = read(packet / 'protocol.json')
    run = returned / protocol['outputs']
    specs, input_summary = _audit_vm_inputs(packet, returned, protocol, output)
    comparisons = read(packet / protocol['comparison_cases'])['cases']
    for row in comparisons:
        identifier = 'comparison_' + row['id']
        source = checked_path(packet, row['prepared_path'])
        observed_path = checked_path(packet, row['observed_path'])
        require(sha(source) == row['prepared_sha256'] and sha(observed_path) == row['observed_sha256'], 'Original comparison input/mask differs')
        target_path = checked_path(packet, row['target_path']) if row.get('target_path') else None
        if target_path:
            require(sha(target_path) == row['target_sha256'], 'Original paired target differs')
        specs[identifier] = {'id': identifier, 'evidence': row['evidence'], 'profile': row['profile'],
            'source': row['source'], 'input_path': str(source), 'input_sha256': row['prepared_sha256'],
            'target_path': str(target_path) if target_path else None,
            'input_pixels': rgb8(source), 'target': rgb8(target_path).astype(np.float32) / np.float32(255) if target_path else None,
            'observed': mask256(observed_path)}
    require(len(specs) == 406, 'Full snapshot input namespace differs')
    all_rows, summaries, visual, baseline_clear = [], [], [], None
    for snapshot in sorted(run.glob('snapshot_[0-9][0-9][0-9][0-9]')):
        update = int(snapshot.name.rsplit('_', 1)[1])
        require(update in protocol['training']['snapshots'], 'Undeclared snapshot update')
        complete_path, partial_path = snapshot / 'metrics.json', snapshot / 'partial_metrics.json'
        require(complete_path.exists() != partial_path.exists(), 'Snapshot metric receipt ambiguous/missing')
        receipt = read(complete_path if complete_path.exists() else partial_path)
        require(receipt['update'] == update and receipt['complete'] == complete_path.exists(), 'Snapshot receipt status differs')
        saved_rows = receipt['rows']
        require(len({r['id'] for r in saved_rows}) == len(saved_rows), 'Duplicate snapshot rows')
        if complete_path.exists():
            require({r['id'] for r in saved_rows} == set(specs), 'Full snapshot cohort differs')
        else:
            require({r['id'] for r in saved_rows} <= set(specs), 'Unknown partial snapshot IDs')
        output_ids = {p.parent.name for p in snapshot.glob('*/raw.npy')} | {
            p.parent.name for p in snapshot.glob('*/native.png')}
        require(output_ids <= set(specs), 'Unknown output directory in snapshot')
        orphan_ids = sorted(output_ids - {r['id'] for r in saved_rows})
        require(not orphan_ids or not complete_path.exists(), 'Unrecorded image outputs in complete snapshot')
        clear = {}
        for saved in saved_rows:
            identifier = saved['id']; spec = specs[identifier]
            require(all(saved[k] == spec[k] for k in ('evidence', 'profile', 'source')), 'Saved metric role/profile/source differs')
            case_dir = snapshot / identifier
            raw_path, png_path = case_dir / 'raw.npy', case_dir / 'native.png'
            raw = load_raw(raw_path, saved['raw_sha256'])
            require(sha(png_path) == saved['native_png_sha256'], 'Saved PNG file hash differs')
            png = rgb8(png_path)
            require(np.array_equal(png, expected_native_png(raw, spec['input_pixels'], spec['observed'])), 'Native PNG floor/padding conversion differs')
            require(saved['observed_pixels'] == int(spec['observed'].sum()), 'Observed pixel count differs')
            if identifier.startswith('validation_'):
                require(saved['input_sha256'] == spec['input_sha256'], 'Snapshot validation input hash differs')
            rebuilt = {}
            for view, prediction in [('raw', raw), ('native_png', png.astype(np.float32) / np.float32(255))]:
                values = paired_metrics(prediction, spec['target'], spec['observed'], evidence=spec['evidence'])
                compare_metric_dict(values, saved[view], f'{snapshot.name}/{identifier}/{view}')
                rebuilt[view] = values
                all_rows.append({'snapshot': update, 'snapshot_complete': complete_path.exists(), 'id': identifier,
                    'source': spec['source'], 'evidence': spec['evidence'], 'profile': spec['profile'], 'view': view,
                    'observed_pixels': saved['observed_pixels'], **values,
                    'artifact_path': str(raw_path if view == 'raw' else png_path),
                    'artifact_sha256': saved['raw_sha256'] if view == 'raw' else saved['native_png_sha256']})
            if spec['evidence'] == 'paired_synthetic' and spec['profile'] == 'clear':
                clear[identifier] = {view: rebuilt[view]['mae'] for view in ('raw', 'native_png')}
            visual.append({'snapshot': update, 'snapshot_complete': complete_path.exists(), 'id': identifier,
                'source': spec['source'], 'evidence': spec['evidence'], 'profile': spec['profile'],
                'input': spec['input_path'], 'input_sha256': spec['input_sha256'],
                'target': spec['target_path'],
                'target_sha256': sha(spec['target_path']) if spec['target_path'] else None,
                'raw': str(raw_path), 'raw_sha256': saved['raw_sha256'],
                'native_png': str(png_path), 'native_png_sha256': saved['native_png_sha256']})
        require(set(clear) == set(receipt['clear']), 'Saved clear control namespace differs')
        for identifier in clear:
            for view in clear[identifier]:
                require(math.isclose(clear[identifier][view], receipt['clear'][identifier][view], rel_tol=2e-11, abs_tol=2e-12), 'Saved clear control metric differs')
        gate = None
        if update == 0:
            require(complete_path.exists() and len(clear) == 61 and receipt['quality_guard'] is None, 'Baseline not complete61clear')
            baseline_clear = clear
        elif complete_path.exists():
            require(baseline_clear is not None, 'No full update0 baseline')
            gate = independently_clear_gate(clear, baseline_clear, protocol['evaluation'])
            saved_gate = receipt['quality_guard']
            require(gate['pass'] == saved_gate['pass'], 'Clear guard decision differs')
            for view in ('raw', 'native_png'):
                for key in ('mean_mae_delta', 'worst_per_case_mae_delta'):
                    require(math.isclose(gate['views'][view][key], saved_gate['views'][view][key], abs_tol=2e-12, rel_tol=2e-11), 'Clear guard summary differs')
        summaries.append({'snapshot': update, 'complete': complete_path.exists(), 'cases': len(saved_rows),
            'clear_controls': len(clear), 'independent_quality_guard': gate,
            'unrecorded_partial_output_ids': orphan_ids,
            'receipt_sha256': sha(complete_path if complete_path.exists() else partial_path)})
    require(summaries and summaries[0]['snapshot'] == 0, 'No snapshot evidence')
    groups = defaultdict(list)
    for row in all_rows:
        groups[(row['snapshot'], row['snapshot_complete'], row['view'], row['source'], row['evidence'], row['profile'])].append(row)
    grouped = []
    for (update, complete, view, source, evidence, profile), rows in sorted(groups.items()):
        group = {'snapshot': update, 'snapshot_complete': complete, 'view': view, 'source': source,
                 'evidence': evidence, 'profile': profile, 'cases': len(rows)}
        for key in ('mae', 'mse', 'psnr'):
            values = [r[key] for r in rows if r[key] is not None]
            group['mean_' + key] = float(sum(values) / len(values)) if values else None
        grouped.append(group)
    baseline_rows = {(r['id'], r['view']): r for r in all_rows if r['snapshot'] == 0}
    changes = []
    for row in all_rows:
        if row['snapshot'] == 0:
            continue
        baseline = baseline_rows[(row['id'], row['view'])]
        change = {k: row[k] for k in ('snapshot', 'snapshot_complete', 'id', 'source', 'evidence', 'profile', 'view')}
        for key in ('mae', 'mse', 'psnr'):
            change['candidate_minus_update0_' + key] = row[key] - baseline[key] if row[key] is not None and baseline[key] is not None else None
        changes.append(change)
    change_groups = defaultdict(list)
    for row in changes:
        for profile in {row['profile'], 'all_clear' if row['profile'] == 'clear' else 'all_degraded'}:
            change_groups[(row['snapshot'], row['snapshot_complete'], row['view'], row['source'], row['evidence'], profile)].append(row)
    group_changes = []
    for (update, complete, view, source, evidence, profile), rows in sorted(change_groups.items()):
        group = {'snapshot': update, 'snapshot_complete': complete, 'view': view, 'source': source,
                 'evidence': evidence, 'profile': profile, 'cases': len(rows)}
        for key in ('mae', 'mse', 'psnr'):
            values = [r['candidate_minus_update0_' + key] for r in rows if r['candidate_minus_update0_' + key] is not None]
            group['mean_candidate_minus_update0_' + key] = float(sum(values) / len(values)) if values else None
            group['cases_lower_' + key] = sum(v < 0 for v in values) if values else None
            group['cases_higher_' + key] = sum(v > 0 for v in values) if values else None
        group_changes.append(group)
    write(output / 'candidate_vs_update0.json', {'complete': True, 'case_deltas': changes,
        'source_profile_deltas': group_changes, 'partial_snapshots_are_incomplete': True,
        'native_reference_metrics_null': True, 'model_quality_not_implied': True})
    write(output / 'metrics.json', {'complete': True, 'rows': all_rows, 'groups': grouped,
        'synthetic_paired_and_native_unpaired_separate': True, 'no_native_reference_metrics': True})
    with (output / 'metrics.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    write(output / 'visual_manifest.json', {'complete': True, 'cases': visual,
        'views': ['raw', 'native_png'], 'all_declared_snapshot_rows_included': True,
        'candidate_qualified': False, 'visual_review_pending': True})
    audit = {'complete': True, 'protocol_sha256': PROTOCOL_SHA, 'VM_validation_input_audit': input_summary,
        'snapshot_summaries': summaries, 'metric_rows': len(all_rows), 'groups': len(grouped),
        'saved_RAW_nativePNG_metrics_recomputed': True, 'native_PNG_floor_and_padding_exact': True,
        'paired_and_native_separate': True, 'all_native_reference_metrics_null': True,
        'clear_guard_thresholds': {k: protocol['evaluation'][k] for k in ('mean_clear_mae_delta_stop', 'worst_per_case_clear_mae_delta_stop')},
        'neural_forwards': 0, 'gradient_queries': 0, 'optimizer_updates': 0,
        'final_pixels_opened': 0, 'torch_imported': 'torch' in sys.modules,
        'review_source_sha256': sha(Path(__file__)), 'seconds': time.monotonic() - started,
        'candidate_qualified': False, 'state_and_trajectory_audit_separate': True}
    write(output / 'metrics_audit.json', audit)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--returned', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit_return_metrics(args.packet, args.returned, args.output)
    print(json.dumps({k: report[k] for k in ('complete', 'metric_rows', 'groups', 'snapshot_summaries', 'seconds')}, indent=2))
