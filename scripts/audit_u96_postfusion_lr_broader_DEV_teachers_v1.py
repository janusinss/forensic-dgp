"""Data-only historical teacher readback. No framework/model/network imports."""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/codeformer_u96_postfusion_lr_broader_DEV_preparation'
R2ROOT = ROOT / 'outputs/codeformer_role_preserving_training_v1_r2_return'
R2OUT = R2ROOT / 'outputs/role_preserving_training_v1_r2'
PREP = ROOT / 'outputs/codeformer_role_preserving_training_v1_r2_preparation'
PIX = ROOT / 'outputs/codeformer_role_preserving_training_v1_r2_return_review/pixels_v1'
IMPORT = ROOT / 'outputs/codeformer_role_preserving_training_v1_r2_return_review/return_import_v1.json'
METRIC_SOURCE = ROOT / 'research/codeformer_filter_fusion_training_v1/metrics.py'
ORIGINAL = '6cb93cee937f67814edcf4625a68bae4bf8f1c2351f7306864ceefa63d4803bd'
R2 = 'a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f'


def need(value, message):
    if not value:
        raise ValueError(message)


def pin(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), 'Regular local file required: ' + str(path))
    data = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def ordered(values):
    result = 0.0
    for value in values:
        result += float(value)
    return result


def main():
    started = time.monotonic()
    target_receipt = OUT / 'independent_teacher_reference_audit_v1.json'
    need(not target_receipt.exists(), 'Preserve previous receipt')
    # Hard block any accidental framework import, including imports by dependencies.
    class BlockFramework:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in {'torch', 'torchvision', 'tensorflow', 'jax', 'third_party'}:
                raise ImportError('Teacher audit forbids model/framework imports: ' + fullname)
    sys.meta_path.insert(0, BlockFramework())
    import numpy as np
    from PIL import Image, __version__ as pillow_version
    import scipy
    import skimage
    spec = importlib.util.spec_from_file_location('historical_saved_metrics_only', METRIC_SOURCE)
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)

    bundle_path = ROOT / 'outputs/codeformer_filter_fusion_training_v1_preparation/evaluation_bundle_v1.json'
    bundle = read(bundle_path)
    cohort = {r['id']: r for r in bundle['cases'] if r['cohort'] == 'historical_DEV406'}
    paired = {k: r for k, r in cohort.items() if r['evidence'] == 'paired_synthetic'}
    native = {k: r for k, r in cohort.items() if r['evidence'] == 'native_unpaired'}
    need(len(cohort) == 406 and len(paired) == 358 and len(native) == 48, 'Frozen historical counts')
    stage = read(PREP / 'stage_v1.json')
    imported = read(IMPORT)
    manifest = read(R2OUT / 'export_manifest.json')
    original = read(R2OUT / 'retained_original_metrics.json')['cases']
    r2 = read(R2OUT / 'snapshot_0050/metrics.json')['cases']
    prior = read(PIX / 'audit.json')
    prior_metrics = read(PIX / 'metrics.json')['cases']
    conversion = read(PIX / 'conversion_and_parity_checks.json')
    conversions = {(r['bank'], r['id']): r for r in conversion['checks']}
    export_path = ROOT / 'outputs/codeformer_role_preserving_training_v1_r2_model/model_export_v1.json'
    export = read(export_path)
    state_path = ROOT / 'outputs/codeformer_role_preserving_training_v1_r2_return_review/state_v1/state_audit_v1.json'
    state = read(state_path)
    need(prior['complete'] and prior['audit_pass'] and prior['full_worker_scalar_metrics_reproduced'] and
         not prior['saved_metric_mismatches'] and not prior['saved_gate_mismatches'], 'Retained independent pixel proof')
    need(prior['import_receipt_sha256'] == pin(IMPORT)['sha256'] == export['import_receipt_sha256'], 'Import chain')
    need(export['state_audit_sha256'] == pin(state_path)['sha256'] and
         state['checkpoint_readback']['update50']['model_tensor_sha256'] == R2 == export['canonical_model_tensor_sha256'], 'R2 final canonical chain')
    need(read(R2OUT / 'factory.json')['canonical_tensor_state_sha256']['original'] == ORIGINAL, 'Original canonical chain')
    need(prior['archive_sha256'] == imported['archive_sha256'] == export['archive_sha256'] and
         prior['protocol_sha256'] == imported['protocol_sha256'] == manifest['protocol_sha256'], 'Exact archive/protocol chain')
    sources = [bundle_path, PREP / 'stage_v1.json', IMPORT, R2OUT / 'export_manifest.json',
               R2OUT / 'retained_original_metrics.json', R2OUT / 'snapshot_0050/metrics.json',
               R2OUT / 'factory.json', R2OUT / 'runtime.json', PIX / 'audit.json', PIX / 'metrics.json',
               PIX / 'conversion_and_parity_checks.json', state_path, export_path, METRIC_SOURCE,
               ROOT / 'research/codeformer_role_preserving_training_v1/worker.py',
               ROOT / 'scripts/audit_codeformer_role_preserving_training_r2_pixels_v1.py', Path(__file__)]
    source_pins = [pin(p) for p in sources]
    need(pin(sources[-2])['sha256'] == prior['audit_source_sha256'], 'Retained audit source exact')
    for name, expected in prior['outputs_sha256'].items():
        if name in {'metrics.json', 'conversion_and_parity_checks.json'}:
            need(pin(PIX / name)['sha256'] == expected, 'Prior result companion hash')
    for path in [R2OUT / 'retained_original_metrics.json', R2OUT / 'snapshot_0050/metrics.json', R2OUT / 'factory.json']:
        relative = path.relative_to(R2ROOT).as_posix()
        need(pin(path)['sha256'] == imported['files_sha256'][relative] == manifest['files_sha256'][relative], 'Saved metadata export binding')

    rows = []; missing_raw = []; file_pins = {}; differences = []
    for number, (identifier, case) in enumerate(paired.items(), 1):
        def image_from(p, mode):
            file_pins[p.relative_to(ROOT).as_posix()] = pin(p)
            with Image.open(p) as image:
                need(image.mode == mode and image.size == (256, 256), 'Declared image256 schema')
                return np.asarray(image).copy()
        target = image_from(ROOT / case['target']['path'], 'RGB')
        need(file_pins[case['target']['path']]['sha256'] == case['target']['sha256'], 'Photographic target pin')
        target32 = target.astype(np.float32) / np.float32(255)
        need(hashlib.sha256(target.tobytes()).hexdigest() == case['target_pixels']['RGB8_sha256'] and
             hashlib.sha256(target32.tobytes()).hexdigest() == case['target_pixels']['float32_sha256'], 'Target RGB/FP32 pins')
        support = np.ones((256, 256), dtype=bool)
        if case['support'] is not None:
            support_path = ROOT / case['support']['path']
            support = image_from(support_path, 'L') == 255
            need(file_pins[support_path.relative_to(ROOT).as_posix()]['sha256'] == case['support']['sha256'], 'Support pin')
        need(support.any(), 'Nonempty exact historical observed support')
        answer = {'id': identifier, 'source': case['source'], 'profile': case['profile'], 'evidence': case['evidence'],
                  'observed_pixels': int(support.sum()), 'input': case['input'], 'target': case['target'],
                  'support': case['support'], 'teachers': {}}
        for label, bank, old_label in [('original', original, 'original0'), ('r2', r2, 'candidate50')]:
            record = bank[identifier]
            need(all(record[k] == case[k] for k in ('id', 'source', 'profile', 'evidence')), 'Teacher case labels')
            if label == 'original':
                logical = 'outputs/adain_gradient_isolation_v1/snapshot_0000/' + identifier
                raw_pin = stage['retained_output_dependencies'][logical + '/raw.npy']
                png_pin = stage['retained_output_dependencies'][logical + '/native.png']
                raw_path = Path(stage['local_dependency_resolution'][logical + '/raw.npy'])
                png_path = Path(stage['local_dependency_resolution'][logical + '/native.png'])
                need(prior['parent_bindings_checked'][logical + '/raw.npy']['sha256'] == raw_pin['sha256'] and
                     prior['parent_bindings_checked'][logical + '/native.png']['sha256'] == png_pin['sha256'], 'Original audited physical bindings')
            else:
                raw_pin, png_pin = record['raw'], record['png']
                raw_path, png_path = R2ROOT / raw_pin['path'], R2ROOT / png_pin['path']
                for p, expected in [(raw_path, raw_pin), (png_path, png_pin)]:
                    rel = p.relative_to(R2ROOT).as_posix()
                    need(imported['files_sha256'][rel] == manifest['files_sha256'][rel] == expected['sha256'], 'R2 output pin export chain')
            png = image_from(png_path, 'RGB')
            actual_pin = file_pins[png_path.relative_to(ROOT).as_posix()]
            need(actual_pin['sha256'] == png_pin['sha256'] and actual_pin['bytes'] == png_pin['bytes'], 'Teacher PNG exact bytes')
            check = conversions[(old_label, identifier)]
            need(check['RAW_PNG_exact'] and check['PNG_padding_exact'] and
                 hashlib.sha256(png.tobytes()).hexdigest() == check['PNG_RGB8_SHA256'], 'Prior floor/padding and current RGB digest')
            # metrics()'s native_png branch does not depend on the raw argument.
            # Supply PNG/255 solely to satisfy its shape checks; discard raw branch.
            measured = metrics.metrics(png.astype(np.float32) / np.float32(255), png, target32, support,
                                       case['evidence'])['native_png']
            numeric = {}
            for field, expected in record['metrics']['native_png'].items():
                if expected is not None:
                    numeric[field] = measured[field] - expected
                    if measured[field] != expected:
                        differences.append({'teacher': label, 'id': identifier, 'field': field,
                                            'difference': numeric[field], 'recorded': expected, 'current': measured[field]})
            prior_raw = prior_metrics[old_label][identifier]['metrics']['raw']
            raw_differences = {k: prior_raw[k] - v for k, v in record['metrics']['raw'].items() if v is not None}
            need(all(math.isfinite(x) for x in raw_differences.values()), 'Finite retained RAW metrics')
            if not raw_path.exists():
                missing_raw.append({'teacher': label, 'id': identifier, 'path': str(raw_path), 'recorded_pin': raw_pin})
            else:
                need(pin(raw_path)['sha256'] == raw_pin['sha256'], 'Unexpected present RAW hash')
            answer['teachers'][label] = {'png': actual_pin, 'png_RGB8_sha256': check['PNG_RGB8_SHA256'],
                'raw_file_present': raw_path.exists(), 'raw_recorded_pin': raw_pin,
                'raw_array_sha256_from_prior_exact_conversion': check['RAW_SHA256'],
                'retained_recorded_metrics': record['metrics'], 'current_recomputed_png_metrics': measured,
                'current_minus_recorded_png_fields': numeric,
                'prior_independent_RAW_minus_recorded_fields': raw_differences}
        rows.append(answer)
        if number % 100 == 0:
            print(json.dumps({'paired_teacher_cases_checked': number, 'of': 358}), flush=True)

    groups = {'clear61': sorted(k for k, r in paired.items() if r['profile'] == 'clear'),
              'normal_blur53': sorted(k for k, r in paired.items() if r['source'] == 'FFHQ validation' and r['profile'] == 'blur'),
              'lowlight53': sorted(k for k, r in paired.items() if r['source'] == 'FFHQ validation' and r['profile'] == 'lowlight'),
              'compound53': sorted(k for k, r in paired.items() if r['source'] == 'FFHQ validation' and r['profile'] == 'compound')}
    need([len(v) for v in groups.values()] == [61, 53, 53, 53], 'All unchanged paired gate members')
    dark = {}
    for profile in ('lowlight', 'compound'):
        ids = groups[profile + '53']
        dark[profile] = {field: {'recorded_R2_ordered_mean': ordered(r2[k]['metrics']['native_png'][field] for k in ids) / 53,
            'recorded_R2_python313_sum_mean': sum(r2[k]['metrics']['native_png'][field] for k in ids) / 53}
            for field in ('mae', 'ssim')}
    # Native members can remain genuine metadata-only slots. Poison metric access
    # proves these unchanged gates never consume a native score or output.
    class NoNativeMetrics(dict):
        def __getitem__(self, key):
            if key == 'metrics':
                raise ValueError('Native clean metrics accessed')
            return super().__getitem__(key)
    structural_bank = {k: (NoNativeMetrics(r) if k in native else r) for k, r in original.items()}
    need(metrics.preservation(structural_bank, structural_bank)['pass'], 'Paired gate API structural membership proof')
    need(metrics.dark_retention(structural_bank, structural_bank)['pass'], 'Dark gate API structural membership proof')
    archive_files = [ROOT / 'outputs/cctv-codeformer-adain-gradient-isolation-v1-results.tar.gz',
                     ROOT / 'outputs/cctv-codeformer-role-preserving-training-v1-r2-results.tar.gz',
                     ROOT / 'outputs/cctv-codeformer-filter-fusion-training-v1-results.tar.gz']
    report = {'format': 'u96-postfusion-lr-broader-DEV-independent-teacher-reference-audit-v1',
        'complete': True, 'independent_teacher_reference_audit_pass': True,
        'UTC': datetime.now(timezone.utc).isoformat(), 'sources': source_pins, 'current_file_pins': file_pins,
        'historical_membership': {'paired': 358, 'native_metadata_only': 48, 'total': 406,
            'new_candidate_forward_calls_in_this_audit': 0, 'new_teacher_forward_calls': 0},
        'teacher_canonical515': {'original': ORIGINAL, 'r2': R2,
            'old_AdaIN0050_is_distinct_parent': '92ac81750b86ef1da6f1f7c0480b9e64773fafe07b094e4a5ad7e88b3f7a0c8b'},
        'current_teacher_PNGs_rehashed_and_metrics_recomputed': 716, 'rows': rows,
        'current_PNG_metric_scalar_differences': differences,
        'current_PNG_metric_maximum_absolute_difference': max((abs(d['difference']) for d in differences), default=0.),
        'PNG_metric_comparison_new_tolerance_added': False,
        'RAW_reference_evidence': {'classification': 'retained_recorded_RAW_references_bound_to_prior_independent_pixel_audit',
            'current_RAW_pixel_recomputation': False, 'missing_files': missing_raw,
            'prior_independent_scalar_comparison_atol': conversion['scalar_comparison_atol'],
            'prior_image_array_equality_atol': conversion['image_array_equality_atol'],
            'prior_scalar_policy_is_not_a_new_quality_gate_tolerance': True,
            'prior_audit_RAW_PNG_pairs_checked': prior['total_RAW_PNG_pairs_checked'],
            'use_worker_recorded_RAW_values_as_sealed_references_not_CPU_audit_rounded_replacements': True},
        'archives_local_availability': [{'path': p.relative_to(ROOT).as_posix(), 'present': p.is_file()} for p in archive_files],
        'gate_case_ids': groups, 'R2_dark_recorded_means': dark,
        'gate_reference_policy': {'raw': 'retained_recorded_metrics.raw; exact sealed worker values, not current pixel recomputation',
            'native_png': 'retained_recorded_metrics.native_png; exact sealed worker values; current_recomputed_png_metrics are independent measured corroboration with every difference disclosed, not replacement gate references',
            'no_reference_rebaselining_or_rounding': True},
        'gate_representation': {'unchanged_metrics_py_API_can_receive_406_genuine_metadata_members': True,
            'native_metrics_access_poison_self_reference_fixture_passed': True,
            'fixture_is_API_demonstration_not_candidate_quality_pass': True,
            'native_entries': '48 original native rows with original labels; outputs=null; metrics=null; execution_status=NOT_RUN_IN_BROADER_PAIRED_CHECK',
            'separate_prior_fixed68_native_evidence': '32 saved native cases may be cited separately only after exact input/model/policy joins; 16 excluded QMUL remain zero-call',
            'must_report_358_paired_outputs_per_arm_not_406_new_outputs': True,
            'preservation': '61 clear MAE and53 normal-blur luminance_bias/coarse_rgb_mae, RAW and native_png separately; ordered mean<=.01 and worst<=.03',
            'dark': '53 lowlight and53 compound PNG mean MAE<=exact R2 control and SSIM>=exact R2 control',
            'arithmetic': 'sorted IDs, explicit left-to-right Python float64 accumulation matching historical CPython3.10; exact comparisons; no epsilon',
            'implementation_note': 'metrics.py currently uses builtin sum, whose CPython3.13 compensated result can differ. Use an explicitly reviewed ordered-sum binding or extracted gate adapter; do not mutate historical source or relax gates.'},
        'conventions': {'target': 'RGB8.astype(float32)/float32(255); paired256 same frozen photographic target',
            'metric': 'actual existing metrics.py native_png branch; float32 PNG/255 promoted to float64; mean luminance [.2126,.7152,.0722];16x16 coarse blocks; .5*(H-edge mean+V-edge mean)',
            'SSIM': 'skimage.metrics.structural_similarity(data_range=1,channel_axis=2), default uniform7 window/sample covariance; differs fixed68 custom Gaussian11 diagnostic SSIM',
            'delivery': 'historical model clamp native512 RGB before bilinear256 align_corners=False; saved FP32 RAW then floor(RAW*float32(255)); input bytes outside support',
            'old_runtime': {k: v for k, v in read(R2OUT / 'runtime.json').items() if k in ('python','torch','numpy','Pillow','device','TF32','autocast','deterministic_algorithms_enabled','cudnn_deterministic')},
            'current_runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'Pillow': pillow_version,
                'scipy': scipy.__version__, 'scikit_image': skimage.__version__},
            'CPU_L4_bitwise_forward_equality_not_assumed': True},
        'limits': ['RAW arrays and local result archives are unavailable; no new RAW pixel/floor readback occurred.',
            'RAW references use exact retained worker metric tables supported by the separately pinned prior independent audit; that audit used1e-10 scalar verification and zero image tolerance.',
            'Current PNG scalar differences are recorded without silently rounding or adding a gate tolerance.',
            'Native unpaired rows receive no clean metrics, identity accuracy or new inference-completion claim.',
            'No candidate716-forward check has been run by this audit; no FINAL use, model qualification or default change.'],
        'framework_model_gradient_optimizer_training_VM_browser_calls': 0, 'model_qualified': False,
        'seconds': time.monotonic() - started}
    OUT.mkdir(exist_ok=True, parents=True)
    with target_receipt.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'receipt': pin(target_receipt), 'paired': 358, 'PNGs': 716,
                      'missing_RAW': len(missing_raw), 'PNG_scalar_maxdiff': report['current_PNG_metric_maximum_absolute_difference'],
                      'seconds': report['seconds']}), flush=True)


if __name__ == '__main__':
    main()
