"""Independent CPU data/cache/evaluation review; no model/framework imports."""
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
FP32_SHA = '8acff86e31c3d135994fdac6ed781374e036150ce1ae2110db82ca0dc5a3aff1'
V2_SHA = '029291a2d58d7bc8b33fb99b86774e13db4e5c098bb8af5f508fdc5e24b20183'
MIB, GIB = 1024**2, 1024**3


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * MIB), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON field')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def bound(root, relative):
    p = PurePosixPath(relative)
    require(p.as_posix() == relative and not p.is_absolute() and '..' not in p.parts
        and '\\' not in relative and ':' not in relative, 'Unsafe relative evidence path')
    path = Path(root).joinpath(*p.parts)
    require(path.is_file() and not path.is_symlink() and path.resolve(strict=True).is_relative_to(Path(root).resolve(strict=True)),
        'Evidence missing, linked or escaping root')
    return path


def pixel_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def rgb(path, size):
    with Image.open(path) as im:
        require(im.mode == 'RGB' and im.size == (size, size), 'Expected exact RGB geometry')
        return np.asarray(im).copy()


def array(path, shape, dtype):
    with Path(path).open('rb') as stream:
        result = np.load(stream, allow_pickle=False)
        require(stream.read(1) == b'', 'Trailing NPY data')
    require(result.shape == shape and result.dtype == np.dtype(dtype) and result.flags.c_contiguous
        and np.isfinite(result).all(), 'Array geometry/dtype/finite differs')
    return result


def first50_counts():
    edges = (24, 28, 32, 40)
    eligible = sum(2 * (edges[(update - 1) % 4] >= 32) + 2 for update in range(1, 51))
    return {'optimizer_step_attempts': 50, 'backward_calls': 200,
        'gradient_queries': 9, 'perceptual_prerequisite_forwards': 1,
        'candidate_training_forwards': 200, 'perceptual_training_forwards': eligible,
        'behavior_teacher_forwards': 50, 'teacher_forwards': 391,
        'snapshot_forwards': 812, 'baseline_forwards': 2,
        'initializer_parity_candidate_forwards': 2, 'training_input_writes': 200,
        'gradient_witness_candidate_forwards': 4, 'gradient_witness_LPIPS_forwards': 4}


def npy_file_bytes(shape, dtype):
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {'descr': np.dtype(dtype).str,
        'fortran_order': False, 'shape': tuple(shape)})
    return header.tell() + math.prod(shape) * np.dtype(dtype).itemsize


def resource_proposal():
    old = read(ROOT / 'outputs/codeformer_fp32_comparison_v1_preparation/final_resource_ledger.json')['ledger']
    witness_arrays = 4 * (npy_file_bytes((512, 512, 3), np.float32) +
        npy_file_bytes((256, 256, 3), np.float32) + npy_file_bytes((256,), np.int64))
    witness_png_ceiling = 4 * 198656
    return {'status': 'PROVISIONAL UNTIL NEW WORKER/FILE SCHEMA IS FROZEN',
        'original_FP32_payload_bound': old['conservative_payload_bytes'],
        'witness4_native512_raw256_token_arrays': witness_arrays,
        'optional4_witness_PNG_file_ceiling': witness_png_ceiling,
        'combined_payload_bound_if_all_other_components_unchanged': old['conservative_payload_bytes'] + witness_arrays + witness_png_ceiling,
        'payload_cap': int(3.5 * GIB),
        'remaining_payload_margin': int(3.5 * GIB) - old['conservative_payload_bytes'] - witness_arrays - witness_png_ceiling,
        'metadata_remains_128MiB_runtime_enforced': True,
        'checkpoint_writers_require_384MiB_initial_832MiB_nonempty_Adam_caps': True,
        'only3_full_checkpoints_no_fourth_safe_copy': True,
        'full_call_counts_pending_literal_final_worker_key_names': first50_counts(),
        'guarded_reservations_sum': sum(first50_counts().values()),
        'actual_neural_forward_sum': sum(value for key, value in first50_counts().items() if key.endswith('_forwards')),
        'no_cache_read_replacement': True, 'HQ_recomputed391_before_Adam_and_compared_to_retained_arrays': True}


def audit_parent(output):
    started = time.monotonic()
    output = Path(output).resolve()
    require(not output.exists(), 'Preserve earlier parent review')
    packet = ROOT / 'outputs/cctv_codeformer_fp32_comparison_v1_vm'
    parent = ROOT / 'outputs/cctv_codeformer_adaptation_v2_vm'
    returned = ROOT / 'outputs/codeformer_fp32_comparison_v1_return'
    require(sha(packet / 'protocol.json') == FP32_SHA and sha(parent / 'protocol.json') == V2_SHA, 'Frozen parent protocol changed')
    p = read(packet / 'protocol.json')
    transport_path = ROOT / 'outputs/codeformer_fp32_comparison_v1_return_review/transport_v1.json'
    transport = read(transport_path)
    require(transport['complete'], 'Prior FP32 return must be closed')
    manifest_path = bound(returned, transport['manifest_member'])
    require(sha(manifest_path) == transport['manifest_sha256'], 'Prior closed export manifest changed')
    returned_map = read(manifest_path)['files_sha256']
    files = {}
    def resolve(relative):
        require(relative in p['files_sha256'], 'Unbound parent dependency')
        root = packet if (packet / relative).exists() else parent
        require(root == packet or p['dependency_files_sha256'].get(relative) == p['files_sha256'][relative], 'Unapproved static fallback')
        path = bound(root, relative)
        require(sha(path) == p['files_sha256'][relative], 'Parent static file changed: ' + relative)
        files[relative] = {'root': str(root), 'sha256': p['files_sha256'][relative], 'bytes': path.stat().st_size}
        return path
    for relative in p['files_sha256']:
        resolve(relative)
    require(len(files) == 1925 and len(p['dependency_files_sha256']) == 1399, 'Parent static census changed')
    refs = read(resolve(p['data_manifest']))
    by_id = {r['id']: r for r in refs['references']}
    train = {k: r for k, r in by_id.items() if r['historical_role'] == 'train'}
    val = {k: r for k, r in by_id.items() if r['historical_role'] == 'validation'}
    require(len(train) == 391 and len(val) == 53 and not set(train) & set(val), 'TRAIN/validation namespace changed')
    require(not {r['source_sha256'] for r in train.values()} & {r['source_sha256'] for r in val.values()}, 'Exact TRAIN/validation source-file overlap')
    clean512 = {}
    for rid, ref in by_id.items():
        require(ref['decision'] == 'accept_clean_restoration' and all(ref['attribution'].get(k) for k in ('photo_url', 'license', 'license_url')), 'Genuine source provenance/terms absent')
        source = resolve(ref['source_path'])
        with Image.open(source) as image:
            require(image.mode == 'RGB' and image.size == (1024, 1024), 'Genuine1024 source geometry differs')
            pixels = np.asarray(image.resize((512, 512), Image.Resampling.LANCZOS)).copy()
            require(pixel_sha(pixels) == ref['proposed_512_rgb_sha256'], 'Direct clean512 target pixels differ')
            clean512[rid] = {'RGB8_sha256': pixel_sha(pixels), 'float32_sha256': pixel_sha(pixels.astype(np.float32) / np.float32(255))}
            if rid in val:
                target = rgb(resolve(ref['existing_target_path']), 256)
                require(np.array_equal(target, np.asarray(image.resize((256, 256), Image.Resampling.LANCZOS))), 'Direct validation256 target differs')
    cache_root = returned / p['outputs'] / 'HQ_cache'
    require({path.name for path in cache_root.iterdir()} == set(train), 'HQ cache must cover exactly391 TRAIN sources')
    caches, array_bytes, record_bytes = {}, 0, 0
    for rid in sorted(train):
        directory = cache_root / rid
        require({path.name for path in directory.iterdir()} == {'indices.npy', 'features.npy', 'record.json'}, 'HQ cache file namespace differs')
        record_path = directory / 'record.json'
        relative = record_path.relative_to(returned).as_posix()
        require(sha(record_path) == returned_map[relative], 'HQ provenance record no longer bound to closed return')
        record = read(record_path)
        require(record['reference_id'] == rid and record['role'] == 'train' and record['source_path'] == train[rid]['source_path'] and
            record['source_sha256'] == train[rid]['source_sha256'] and record['clean512_float32_sha256'] == clean512[rid]['float32_sha256'], 'HQ genuine source/clean target binding differs')
        indices_path = bound(returned, record['indices_path']); features_path = bound(returned, record['features_path'])
        for path, digest in ((indices_path, record['indices_sha256']), (features_path, record['features_sha256'])):
            require(sha(path) == digest == returned_map[path.relative_to(returned).as_posix()], 'HQ saved array fingerprint differs')
        indices = array(indices_path, (1, 256), 'int64')
        features = array(features_path, (1, 256, 16, 16), 'float32')
        require(((indices >= 0) & (indices < 1024)).all(), 'HQ index outside frozen dictionary')
        array_bytes += indices_path.stat().st_size + features_path.stat().st_size
        record_bytes += record_path.stat().st_size
        caches[rid] = {'record_sha256': sha(record_path), 'source_path': record['source_path'], 'source_sha256': record['source_sha256'],
            'clean512_float32_sha256': record['clean512_float32_sha256'], 'indices_sha256': record['indices_sha256'],
            'features_sha256': record['features_sha256'], 'indices_array_sha256': pixel_sha(indices), 'features_array_sha256': pixel_sha(features),
            'teacher_state_sha256': record['teacher_state_sha256'], 'restoration_codebook_state_sha256': record['restoration_codebook_state_sha256']}
    require(array_bytes == 103399168 and len({r['teacher_state_sha256'] for r in caches.values()}) ==
        len({r['restoration_codebook_state_sha256'] for r in caches.values()}) == 1, 'HQ exact array byte or component state census changed')
    parity = read(resolve(p['training_parity']['path']))
    require(sha(resolve(p['training_parity']['path'])) == p['training_parity']['sha256'], 'Training parity manifest changed')
    require(len(parity['cases']) == 200 and len(parity['behavior_anchors']) == 50 and len(parity['updates']) == 50, 'Training/anchor trajectory census changed')
    require([(r['update'], r['slot']) for r in parity['cases']] == [(u, s) for u in range(1, 51) for s in range(1, 5)], 'TRAIN exposure order differs')
    selected = []
    for case in parity['cases']:
        require(case['reference_id'] in train, 'NonTRAIN training reference')
        pixels = rgb(resolve(case['input_path']), 256)
        require(pixel_sha(pixels) == case['input256_rgb8_sha256'] and pixel_sha(pixels.astype(np.float32) / np.float32(255)) == case['input256_float32_sha256'], 'Exact retained TRAIN pixels differ')
        require(case['clean512_float32_sha256'] == clean512[case['reference_id']]['float32_sha256'], 'TRAIN paired clean512 binding differs')
        record = read(resolve(case['pair_record_path']))
        require(record['role'] == 'train' and record['reference_id'] == case['reference_id'] and
            record['source_sha256'] == case['source_sha256'] and record['profile'] == case['profile'], 'TRAIN pair provenance differs')
        if case['slot'] in (1, 3):
            selected.append(case['reference_id'])
    require(len(selected) == len(set(selected)) == 100 and sum(bool(r['image_eligible']) for r in parity['cases']) == 148, 'Reference count or LPIPS eligibility differs')
    for anchor in parity['behavior_anchors']:
        require(anchor['slot'] == 3 and anchor['teacher_FP32_eval_frozen'] is True, 'Original FP32 behavior anchor boundary differs')
        for key, shape in (('delivered', (1, 3, 256, 256)), ('logits', (1, 256, 1024))):
            path = resolve(anchor[key + '_path'])
            require(sha(path) == anchor[key + '_sha256'], 'Retained behavior array differs')
            value = array(path, shape, 'float32')
            if key == 'delivered':
                require(value.min() >= 0 and value.max() <= 1, 'Retained delivered target outside[0,1]')
    validation = read(resolve(p['validation_saved']['manifest']))
    comparisons = read(resolve(p['comparison_cases']))['cases']
    require(len(validation['items']) == 318 and len(comparisons) == 88, 'Full406 comparison namespace differs')
    validation_profiles = Counter()
    for case in validation['items']:
        for key in ('input', 'pair_record', 'target'):
            require(sha(resolve(case[key + '_path'])) == case[key + '_sha256'], 'Saved DEV companion changed')
        record = read(resolve(case['pair_record_path']))
        require(record['role'] == 'validation' and record['reference_id'] in val, 'Nonvalidation reference in saved DEV')
        validation_profiles[record['profile']] += 1
    require(validation_profiles == {name: 53 for name in p['evaluation']['profiles']}, 'All53×6 validation profiles required')
    qmul = read(resolve('provenance/native/qmul_frozen_subset.json'))
    choke = read(resolve('provenance/native/chokepoint_selection_plan.json'))
    qdev = {str(r['global_person_id']) for r in qmul['cases'] if r['role'] == 'development'}
    qfinal = {str(r['global_person_id']) for r in qmul['cases'] if r['role'] == 'reserved_evaluation'}
    cdev, cfinal = set(choke['development_person_ids']), set(choke['reserved_person_ids'])
    require(len(qdev) == 24 and len(qfinal) == 32 and not qdev & qfinal and len(cdev) == 12 and len(cfinal) == 13 and not cdev & cfinal, 'Native labeled DEV/FINAL separation changed')
    for case in comparisons:
        resolve(case['prepared_path'])
        if case['evidence'] == 'native_unpaired':
            require(not case.get('target_path') and not case.get('clean_path'), 'Invented paired native target')
            label = case['identity'].split('/')[-1]
            require((case['source'] == 'QMUL-SurvFace' and label in qdev and label not in qfinal) or
                (case['source'] == 'ChokePoint' and label in cdev and label not in cfinal), 'Reserved or unbound native label entered DEV')
    evidence = Counter(r['evidence'] for r in comparisons)
    require(evidence == {'native_unpaired': 48, 'paired_synthetic': 40}, 'Paired/native source accounting differs')
    witness = [r for r in parity['cases'] if r['update'] == 3 and r['slot'] in (1, 2)]
    require(len(witness) == 2 and all(r['image_eligible'] and r['profile'] in ('severe_normal', 'severe_dark') for r in witness), 'Prospective u3A witness is not eligible matched pair')
    for row in witness:
        record = read(resolve(row['pair_record_path']))
        require(record['parameters']['long_edge'] == 32, 'Prospective witness edge differs')
    result = {'complete': True, 'approved_for_new_training_or_seal': False,
        'status': 'PARENT DATA VERIFIED; NEW DESIGN/WORKER/STAGING STILL REQUIRE REVIEW',
        'parent_protocol_sha256': FP32_SHA, 'older_V2_protocol_sha256': V2_SHA,
        'parent_static_files': len(files), 'older_V2_dependencies': 1399,
        'parent_static_payload_bytes': sum(r['bytes'] for r in files.values()),
        'all_parent_static_files_freshly_rehashed': True,
        'TRAIN_references': 391, 'validation_references': 53, 'source_provenance_terms_and_direct_targets_verified': True,
        'TRAIN_exposures': 200, 'selected_photographic_reference_IDs': 100, 'LPIPS_eligible_exposures': 148,
        'retained_behavior_pairs': 50, 'HQ_cache': {'cases': 391, 'files': 1173, 'array_bytes': array_bytes,
            'record_bytes': record_bytes, 'teacher_state_sha256': next(iter(caches.values()))['teacher_state_sha256'],
            'restoration_codebook_state_sha256': next(iter(caches.values()))['restoration_codebook_state_sha256'],
            'cache_read_replacement_authorized': False, 'new_VM_HQ_forwards_required': 391,
            'fresh_generated_cache_must_match_these_exact_saved_fingerprints_before_Adam': True, 'cases_map': caches},
        'evaluation': {'cases_per_snapshot': 406, 'snapshots': [0, 50], 'snapshot_outputs': 812,
            'paired_HQ': 318, 'old_paired': 40, 'native_unpaired': 48,
            'native_sources': dict(Counter(r['source'] for r in comparisons if r['evidence'] == 'native_unpaired')),
            'quality_guard_values_unchanged': p['evaluation'], 'clear_controls': 61, 'normal_blur_controls': 53,
            'QMUL_labeled_DEV_FINAL': [24, 32], 'ChokePoint_labeled_DEV_FINAL': [12, 13],
            'reserved_FINAL_pixels_read': 0, 'unpaired_native_has_no_clean_target': True},
        'witness_input_bindings': witness, 'first50_recipe_and_data_horizon_preserved': p['training'],
        'provisional_resource_only': resource_proposal(),
        'identity_and_inference_limits': ['Photographic reference IDs are not independently proven unique people.',
            'Exact TRAIN/validation source files and native within-source labeled DEV/FINAL splits are disjoint; cross-source/pretraining people overlap remains unestablished.',
            'No new model, witness gradient, training update, visual qualification, ethnicity or Zamboanga performance claim.'],
        'source_sha256': sha(__file__), 'closed_parent_transport_sha256': sha(transport_path),
        'closed_parent_manifest_sha256': sha(manifest_path), 'files': files,
        'seconds': time.monotonic() - started, 'neural_model_gradient_optimizer_VM_network_calls': 0}
    write(output, result)
    print(json.dumps({key: result[key] for key in ('complete', 'status', 'parent_static_files', 'TRAIN_exposures', 'retained_behavior_pairs', 'seconds')}), flush=True)
    return result
