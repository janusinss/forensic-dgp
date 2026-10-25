"""Resolve exact478 evaluation slots from historical read-only pixels; no NN.

406 historical DEV +72 mixed-role filter probes are retained distinctly. The
24 insufficient filter inputs remain refusal controls without model forwards.
Native CCTV has no aligned clean target. No checkpoint deserialization occurs.
"""
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_filter_fusion_training_v1 import data as d

OUT = ROOT / 'outputs/codeformer_filter_fusion_training_v1_preparation'
STAGE = 'outputs/codeformer_role_preserving_training_v1_r2_preparation/stage_v1.json'
R2 = 'outputs/cctv_codeformer_role_preserving_training_v1_r2_vm'
FILTER = 'outputs/codeformer_filter_pairs_v1_data'
PINS = {
    STAGE: 'a2b909cbd42c8ec2efcea289dcdeea9535fd2b4cdc894f067434aa3058547e7d',
    R2 + '/protocol.json': '82fcd34cada1eed238ef65a341c22580d12146e3ee30c2ff3094312d3ba1c85a',
    R2 + '/data_bundle.json': '9982957a244b1e94b0b5b87240d1842df05eec88a2c021574e3fa7728f464bd6',
    'outputs/codeformer_filter_comparison_v1/manifest.json':
        '7b476134e4f0f7975eb4dd41385e1197b2b08f7094b970a25165b196c0dd7be8',
    FILTER + '/manifest.json': '7823847ce7a709308fe1f4d35d35e2c14f752b2d5c303f5fa1cb4b7c8de8c3a8',
    'outputs/codeformer_filter_pairs_v1_preparation/root_input_only_review_v1.json':
        '51bff1593cd33ad2b369a7c20d2b5282ed67521df4b62e17a53f082941fc3b50',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pin(name, expected=None):
    path = ROOT / d.relative(name)
    p = {'path': name, 'bytes': path.stat().st_size, 'sha256': sha(path.read_bytes())}
    d.require(expected is None or p['sha256'] == expected, 'Historical metadata differs: ' + name)
    d.verified_bytes(ROOT, p)
    return p


def decoded(pin_, mask=False):
    import numpy as np
    from PIL import Image
    with Image.open(io.BytesIO(d.verified_bytes(ROOT, pin_))) as im:
        d.require(im.size == (256, 256) and im.mode == ('L' if mask else 'RGB'), 'Exact saved256 RGB/L required')
        im.load()
        arr = np.asarray(im).copy()
    if mask:
        d.require(np.isin(arr, [0, 255]).all() and arr.any(), 'Binary nonempty support required')
        return {'RGB8_or_L8_sha256': sha(arr.tobytes()), 'observed_pixels': int((arr > 0).sum()),
                'shape': list(arr.shape)}
    floats = np.ascontiguousarray(arr.astype(np.float32) / np.float32(255))
    return {'RGB8_sha256': sha(arr.tobytes()), 'float32_sha256': sha(floats.tobytes()), 'shape': list(arr.shape)}


def collect():
    start = time.monotonic()
    import numpy as np
    deps = {name: pin(name, expected) for name, expected in PINS.items()}
    docs = {name: json.loads(d.verified_bytes(ROOT, p)) for name, p in deps.items()}
    stage = docs[STAGE]
    protocol = docs[R2 + '/protocol.json']
    oldbundle = docs[R2 + '/data_bundle.json']
    d.require(stage['protocol_sha256'] == PINS[R2 + '/protocol.json'] and
              stage['complete'] is True, 'Exact completed historical local resolver required')
    logical_pins = {}
    for group in ('dependencies', 'retained_output_dependencies', 'extra_returned_dependencies'):
        d.require(stage[group] == protocol[group], 'Oldstage/protocol mapping differs')
        logical_pins.update(stage[group])
    cache, logical = {}, {}
    def oldfile(name):
        d.require(name in logical_pins and name in stage['local_dependency_resolution'], 'Unbound oldlogical file')
        path = Path(stage['local_dependency_resolution'][name]).absolute()
        d.require(path.is_relative_to(ROOT), 'Historical resolver escaped workspace')
        relative = path.relative_to(ROOT).as_posix()
        p = {'path': relative, **logical_pins[name]}
        d.verified_bytes(ROOT, p)
        logical[name] = {'logical_path': name, 'local': p, 'historical_pin': logical_pins[name]}
        return p
    def pixels(p, mask=False):
        key = (p['path'], mask)
        if key not in cache:
            cache[key] = decoded(p, mask)
        return cache[key]
    cases = []
    d.require(len(oldbundle['development']) == 406, 'Historical406DEV missing')
    references = json.loads(d.verified_bytes(ROOT, oldfile('references.json')))
    d.require(Counter(r['historical_role'] for r in references['references']) ==
              {'train': 391, 'validation': 53}, 'Historical source split changed')
    for row in oldbundle['development']:
        d.require(time.monotonic() - start < 300, 'Data-only elapsedcheckpoint cap300s exceeded')
        inp = oldfile(row['input_path'])
        target = oldfile(row['target_path']) if row['target_path'] else None
        support = oldfile(row['observed_path']) if row['observed_path'] else None
        native = row['evidence'] == 'native_unpaired'
        d.require((native and target is None) or
                  (row['evidence'] == 'paired_synthetic' and target is not None),
                  'Native/paired target boundary differs')
        if 'pair_record' in row['sourcebindings']:
            pair = row['sourcebindings']['pair_record']
            refs = {r['id']: r for r in references['references']}
            d.require(pair['role'] == refs[pair['reference_id']]['historical_role'] == 'validation',
                      'DEV pair crossed historical TRAIN role')
        cases.append({'id': row['id'], 'original_id': row['id'], 'cohort': 'historical_DEV406',
            'input': inp, 'input_pixels': pixels(inp), 'target': target,
            'target_pixels': pixels(target) if target else None, 'support': support,
            'support_pixels': pixels(support, True) if support else None,
            'null_support_policy': 'all256pixels_observed' if support is None else None,
            'source': row['source'], 'profile': row['profile'], 'evidence': row['evidence'],
            'split': 'DEV', 'sourcebindings': row['sourcebindings'], 'optimization_permitted_now': False,
            'target_float32_policy': 'RGB8.astype(float32)/float32(255)' if target else None,
            'native_clean_reference': None if native else 'paired_declared_synthetic_only',
            'reserved_FINAL': False})
    d.require(Counter(r['evidence'] for r in cases) == {'paired_synthetic': 358, 'native_unpaired': 48}
              and sum(r['profile'] == 'clear' and r['evidence'] == 'paired_synthetic' for r in cases) == 61,
              'HistoricalDEV406/358paired/48native/61clear census differs')
    filtermanifest = docs[FILTER + '/manifest.json']
    comparison = docs['outputs/codeformer_filter_comparison_v1/manifest.json']
    review = docs['outputs/codeformer_filter_pairs_v1_preparation/root_input_only_review_v1.json']
    reviews = {r['reference_id'] + '__' + r['profile']: r for r in review['cases']}
    d.require(len(filtermanifest['cases']) == len(comparison['cases']) == len(reviews) == 96,
              'All96 filter observations/refusals retained')
    comp = {r['id']: r for r in comparison['cases']}
    refused = []
    for row in filtermanifest['cases']:
        d.require(comp[row['id']]['pair_record'] == row['pair_record'] and
                  comp[row['id']]['input'] == row['input'], 'Priorfilter/comparison bytes differ')
        recordpin = {**row['pair_record'], 'path': FILTER + '/' + row['pair_record']['path']}
        record = json.loads(d.verified_bytes(ROOT, recordpin))
        inp = {**record['observed_input'], 'path': FILTER + '/' + record['observed_input']['path']}
        target = {**record['targets']['clean256'], 'path': FILTER + '/' + record['targets']['clean256']['path']}
        support = {**record['observed_support'], 'path': FILTER + '/' + record['observed_support']['path']}
        d.require(row['split'] in ('TRAIN', 'DEV') and record['split'] == row['split'] and
                  pixels(inp)['RGB8_sha256'] == record['input256_pixels']['RGB8_sha256'] and
                  pixels(target)['RGB8_sha256'] == record['target256_pixels']['RGB8_sha256'],
                  'Filter savedinput/target/split differs')
        entry = {'id': 'filter_probe__' + row['id'], 'original_id': row['id'],
            'reference_id': row['reference_id'], 'cohort': 'retained_filter72_mixed_roles',
            'input': inp, 'input_pixels': pixels(inp), 'target': target, 'target_pixels': pixels(target),
            'support': support, 'support_pixels': pixels(support, True),
            'source': 'FFHQ retained filter ' + row['split'], 'profile': row['profile'],
            'evidence': 'paired_synthetic', 'split': row['split'],
            'input_review': reviews[row['id']], 'sourcebindings': record,
            'flagged_reference': row['reference_id'] == 'va_ffhq_08025',
            'target_float32_policy': 'RGB8.astype(float32)/float32(255)',
            'optimization_permitted_now': False, 'reserved_FINAL': False}
        if comp[row['id']]['inference_planned']:
            d.require(reviews[row['id']]['decision'] in ('readable', 'borderline'), 'Refused case inferred')
            cases.append(entry)
        else:
            d.require(reviews[row['id']]['decision'] == 'insufficient', 'Noninsufficient refusal differs')
            refused.append({**entry, 'cohort': 'retained_filter24_refused_controls',
                            'new_model_forward_permitted': False, 'restored_output': None})
    d.require(len(cases) == len({r['id'] for r in cases}) == 478 and len(refused) == 24,
              'Exact478evaluated/24refused namespace differs')
    probes = [r for r in cases if r['cohort'] == 'retained_filter72_mixed_roles']
    d.require(Counter(r['split'] for r in probes) == {'TRAIN': 36, 'DEV': 36} and
              sum(r['flagged_reference'] for r in probes) == 9 and
              Counter(r['split'] for r in refused) == {'TRAIN': 12, 'DEV': 12},
              'Mixedrole36/36,flag9,refused12/12 scope changed')
    allpins = {}
    for row in cases + refused:
        for key in ('input', 'target', 'support'):
            if row[key] is not None:
                p = row[key]
                old = allpins.setdefault(p['path'], p)
                d.require(old == p, 'Samephysical dependency has conflicting pins')
    for p in [*deps.values(), *allpins.values(), *(v['local'] for v in logical.values())]:
        d.verified_bytes(ROOT, p)
    return {'format': 'codeformer-filter-fusion-evaluation-bundle-v1', 'complete': True,
            'metadata_dependencies': deps, 'cases': cases, 'refused_controls': refused,
            'output_slots_per_full_snapshot': 478, 'historical_DEV_slots': 406,
            'retained_mixed_filter_probe_slots': 72, 'mixed_probe_roles': {'TRAIN': 36, 'DEV': 36},
            'historical_DEV_native_unpaired': 48, 'historical_DEV_paired_synthetic': 358,
            'retained_filter_core63_flagged9': {'core': 63, 'flagged': 9},
            'refused_controls_no_model_calls': 24, 'all_optimization_false': True,
            'all_original_cohort_membership_and_metadata_retained': True,
            'source_ids_not_proven_unique_people': True, 'person_pretraining_overlap_unknown': True,
            'native_or_Zamboanga_identity_accuracy_claim': False, 'reserved_FINAL_used': False,
            'actual_neural_gradient_optimizer_calls': 0, 'seconds': time.monotonic() - start,
            'local_source_map': {'logical_oldR2_dependencies': logical,
                                 'evaluation_pixel_files': allpins,
                                 'unique_pixel_files': len(allpins),
                                 'unique_pixel_bytes': sum(p['bytes'] for p in allpins.values())}}


def main():
    result = collect()
    mapping = result.pop('local_source_map')
    mapping.update(format='codeformer-filter-fusion-evaluation-source-map-v1', complete=True,
                   metadata_dependencies=result['metadata_dependencies'], checkpoint_files_loaded=0)
    for filename in ('evaluation_bundle_v1.json', 'evaluation_source_map_v1.json'):
        d.require(not (OUT / filename).exists(), 'Exclusive fresh eval namespace required')
    for filename, value in [('evaluation_bundle_v1.json', result), ('evaluation_source_map_v1.json', mapping)]:
        payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode('utf-8') + b'\n'
        with (OUT / filename).open('xb') as handle:
            handle.write(payload)
        print(json.dumps({'path': filename, 'bytes': len(payload), 'sha256': sha(payload)}))
    print(json.dumps({'complete': True, 'output_slots': 478, 'refused_controls': 24,
                      'unique_pixel_files': mapping['unique_pixel_files'], 'neural_calls': 0}))


if __name__ == '__main__':
    main()
