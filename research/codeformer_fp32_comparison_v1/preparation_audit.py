"""Independent CPU data, parity, role and finite-resource preparation audit.

No restorer import, neural forward, gradient, optimizer, checkpoint load or VM
operation. Static frozen dependencies may be resolved from the original V2
packet while the new transfer overlay is assembled. This is not a sealer.
"""
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import time

import numpy as np
from PIL import Image


PARENT_SHA = '029291a2d58d7bc8b33fb99b86774e13db4e5c098bb8af5f508fdc5e24b20183'
REQUIRED_DATA_SHA = '3fcc25c68c7df44aa9f3f5d9dcaa8dea837fea684a597082170750d31f4dc815'
REQUIRED_MODEL_SHA = '8cb64d7a2c3fac46fe38805ac54c00287a050b0473608a87dfe650b5d9e508e3'
MAX_UPDATES, SCHEDULE_HORIZON = 50, 600
MIB, GIB = 1024**2, 1024**3


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * MIB), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, 'Duplicate JSON key: ' + key)
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
    require(isinstance(relative, str) and relative and '\\' not in relative and ':' not in relative,
            'Unsafe relative path')
    p = PurePosixPath(relative)
    require(not p.is_absolute() and p.as_posix() == relative and '..' not in p.parts,
            'Noncanonical relative path')
    path = Path(root).joinpath(*p.parts)
    require(path.resolve(strict=True).is_relative_to(Path(root).resolve(strict=True)) and
            path.is_file() and not path.is_symlink(), 'File escapes root or is not regular')
    return path


def pixel_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def rgb(path, size):
    with Image.open(path) as image:
        require(image.mode == 'RGB' and image.size == size, 'RGB geometry differs: ' + str(path))
        return np.asarray(image).copy()


def array(path, shape):
    with Path(path).open('rb') as stream:
        value = np.load(stream, allow_pickle=False)
        require(stream.read(1) == b'', 'Trailing NPY data')
    require(value.shape == shape and value.dtype == np.float32 and value.flags.c_contiguous and
            np.isfinite(value).all(), 'Float32 anchor shape/dtype/finiteness differs')
    return value


def first50_counts():
    edges, variants = (24, 28, 32, 40), ('clear256', 'clear128', 'clear64', 'light256', 'light128', 'light64')
    eligibility = []
    cycles = []
    for update in range(1, 51):
        u = update - 1
        cycles.append({'update': update, 'A_edge': edges[u % 4], 'B_variant': variants[(u // 4) % 6]})
        eligibility.extend([edges[u % 4] >= 32] * 2 + [True, True])
    return {'optimizer_step_attempts': 50, 'optimizer_updates': 50, 'backward_calls': 200,
            'candidate_training_forwards': 200, 'training_input_writes': 200,
            'behavior_teacher_forwards': 50, 'teacher_forwards': 391,
            'snapshot_forwards': 812, 'baseline_forwards': 2,
            'initializer_parity_candidate_forwards': 2, 'perceptual_training_forwards': sum(eligibility),
            'perceptual_prerequisite_forwards': 1, 'gradient_queries': 1, 'cycles': cycles,
            'eligibility_by_order': eligibility}


def first50_lr(update, group):
    require(type(update) is int and 1 <= update <= 50 and group in ('encoder_token', 'fusion'),
            'Invalid first50 learning-rate request')
    base = 1e-5 if group == 'encoder_token' else 2e-5
    return base * update / 50


def npy_file_bytes(shape, dtype):
    # NPY serialization header is measured on an empty small fixture with the
    # same header shape, without allocating a research cache or model tensor.
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {'descr': np.dtype(dtype).str,
        'fortran_order': False, 'shape': tuple(shape)})
    return header.tell() + math.prod(shape) * np.dtype(dtype).itemsize


def resource_ledger():
    """Conservative final namespace bound; temporary checkpoint is separate."""
    raw_file = npy_file_bytes((256, 256, 3), np.float32)
    anchor_file = npy_file_bytes((1, 3, 256, 256), np.float32) + npy_file_bytes((1, 256, 1024), np.float32)
    cache_file = npy_file_bytes((1, 256), np.int64) + npy_file_bytes((1, 256, 16, 16), np.float32)
    components = {'snapshot_RAW_812': 812 * raw_file, 'snapshot_PNG_812': 812 * 198656,
        'TRAIN_saved_PNG_200': 200 * 198656,
        'TRAIN_actual_captured_tokens_200': 200 * npy_file_bytes((256,), np.int64),
        'teacher_anchor_arrays_50pairs': 50 * anchor_file,
        'exact_retained_behavior_target_arrays_50pairs': 50 * anchor_file,
        'HQ_cache_391_index_feature_pairs': 391 * cache_file,
        'checkpoint_update0_conservative': 384 * MIB,
        'checkpoint_update50_conservative': 832 * MIB,
        'checkpoint_stopped_conservative': 832 * MIB,
        'first_training_example_bound': 4 * MIB,
        'all_metadata_records_logs_manifests': 128 * MIB,
        'source_license_companions_allowance': 8 * MIB}
    payload = sum(components.values())
    file_count_plan = {'per_call_trace_records': 2057, 'TRAIN_inputs_records_tokens_dtype_parity': 1000,
        'snapshot_images_states_and_metadata_allowance': 1642,
        'behavior_online_and_used_arrays_records_parity': 350, 'HQ_cache_arrays_and_records': 1173,
        'first_example_files_allowance': 10, 'operational_metadata_sources_companions_allowance': 512}
    # Each regular file may need a PAX long-path header plus alignment padding;
    # this matches/exceeds the runtime export's4096bytes+511padding per file.
    tar_padding_headers = 7000 * 4608 + 10240
    tar_upper = payload + tar_padding_headers
    gzip_upper = tar_upper + math.ceil(tar_upper / 16383) * 5 + 18
    cap = int(3.5 * GIB)
    archive_cap = math.floor(3.6 * GIB)
    return {'components': components, 'conservative_payload_bytes': payload,
        'payload_cap_bytes': cap, 'payload_margin_bytes': cap - payload,
        'regular_file_count_plan': file_count_plan, 'planned_file_count_upper': sum(file_count_plan.values()),
        'regular_file_count_cap': 7000, 'tar_header_padding_allowance': tar_padding_headers,
        'conservative_tar_gzip_upper_bytes': gzip_upper, 'archive_cap_bytes': archive_cap,
        'archive_margin_bytes': archive_cap - gzip_upper,
        'startup_free_bytes_after_install': 16 * GIB, 'reserve_bytes': 4 * GIB,
        'atomic_checkpoint_temporary_bytes': GIB,
        'combined_caps_plus_temporary_and_reserve': cap + archive_cap + GIB + 4 * GIB,
        'startup_margin_bytes': 16 * GIB - (cap + archive_cap + GIB + 4 * GIB),
        'extra_safe_checkpoint_copy_allowed': False,
        'last_safe_checkpoint_must_be_a_reference_not_fourth_serialized_file': True,
        'metadata_and_logs_bound_must_be_enforced_at_runtime': True,
        'long_path_or_export_namespace_must_stay_inside_declared_allowance': True}


def audit(packet_root, parent_root, v2_return_root, output_dir, native_qmul, native_chokepoint,
          protocol_file=None):
    started = time.monotonic()
    packet = Path(packet_root).resolve(strict=True)
    parent = Path(parent_root).resolve(strict=True)
    returned = Path(v2_return_root).resolve(strict=True)
    output = Path(output_dir).resolve()
    require(not output.exists() and not output.is_relative_to(packet) and not output.is_relative_to(parent)
            and not output.is_relative_to(returned), 'Preserve packet/dependencies/return/audit')
    protocol_path = Path(protocol_file).resolve(strict=True) if protocol_file else packet / 'protocol.json'
    p = read(protocol_path)
    old = read(parent / 'protocol.json')
    require(sha(parent / 'protocol.json') == PARENT_SHA, 'Original V2 protocol changed')
    files, dependency = p['files_sha256'], p['dependency_files_sha256']
    require(all(relative in old['files_sha256'] and digest == old['files_sha256'][relative]
                for relative, digest in dependency.items()), 'Inherited static dependency differs from original V2')
    file_checks = {}

    def checked(relative, digest=None):
        require(relative in files and (digest is None or digest == files[relative]), 'Unbound packet asset')
        if (packet / relative).exists():
            path = bound(packet, relative)
        else:
            require(relative in dependency and dependency[relative] == files[relative], 'Missing overlay asset')
            path = bound(parent, relative)
        require(sha(path) == files[relative], 'Packet/dependency hash differs: ' + relative)
        file_checks[relative] = files[relative]
        return path

    for rel in ('research/codeformer_adaptation_v2/data.py', 'research/codeformer_adaptation_v2/model.py'):
        expected = REQUIRED_DATA_SHA if rel.endswith('data.py') else REQUIRED_MODEL_SHA
        checked(rel, expected)
    parity_path = p['training_parity']['path']
    parity = read(checked(parity_path, p['training_parity']['sha256']))
    require(parity['format'] == 'codeformer-v2-first50-training-parity-v1', 'Unexpected training parity format')
    require(p['training_parity']['source_protocol_sha256'] == PARENT_SHA and
            p['training_parity']['cases'] == 200 and p['training_parity']['updates'] == 50 and
            p['training_parity']['behavior_anchors'] == 50, 'Parity declared counts differ')
    training = p['training']
    require(training['max_updates'] == 50 and training['schedule_horizon_updates'] == 600 and
            training['data_horizon_updates'] == 600 and training['warmup_updates'] == 50,
            'Execution/schedule/data horizons changed')
    require(training['optimizer'] == old['training']['optimizer'] == 'Adam' and
            all(training[name] == old['training'][name] for name in ('microbatch', 'accumulate', 'betas', 'eps',
                'weight_decay', 'encoder_token_lr', 'fusion_lr', 'cosine_end_lr', 'loss_weights', 'gradient_clipping',
                'grad_scaler', 'GAN', 'EMA', 'resume', 'pair_policy', 'severe_edges', 'control_variants',
                'cycle', 'behavior_slot', 'distillation_temperature')), 'V2 optimization recipe changed')
    require(training['snapshots'] == [0, 50] and p['evaluation'] == old['evaluation'], 'Snapshots/quality comparison changed')
    require(p['seed'] == old['seed'] == 20261015, 'Training seed changed')
    require(training['autocast'] == 'disabled-FP32' and training['max_backward_calls'] == 200 and
            training['max_gradient_queries'] == 1, 'Candidate precision or differentiation bound differs')
    expected_limits = {'initialization_seconds': 600, 'benchmark_seconds': 600,
        'projected_training_seconds': 900, 'training_seconds': 900, 'snapshot_seconds': 600,
        'worker_seconds': 3000, 'export_seconds': 600, 'initial_free_bytes': 16 * GIB,
        'reserve_free_bytes': 4 * GIB, 'result_payload_bytes': int(3.5 * GIB),
        'archive_bytes': math.floor(3.6 * GIB), 'checkpoint_temporary_bytes': GIB,
        'PNG_file_bytes': 198656, 'metadata_bytes': 128 * MIB, 'CUDA_reserved_bytes': 22 * GIB}
    require(all(p['limits'][name] == expected for name, expected in expected_limits.items()),
            'Prospective finite resource cap differs from reviewed design')
    references = read(checked(p['data_manifest']))
    refs = references['references']
    train = {row['id']: row for row in refs if row['historical_role'] == 'train' and
             row['selected_for_this_finite_pilot'] and row['decision'] == 'accept_clean_restoration'}
    validation = {row['id']: row for row in refs if row['historical_role'] == 'validation'}
    require(len(refs) == 444 and len(train) == 391 and len(validation) == 53 and not set(train) & set(validation),
            'Historical TRAIN/validation reference namespace differs')
    require(references['historical_roles_preserved'] and references['no_pretrained_identity_nonoverlap_claim'],
            'Overlap caveats absent')
    train_hashes, val_hashes = {x['source_sha256'] for x in train.values()}, {x['source_sha256'] for x in validation.values()}
    require(not train_hashes & val_hashes and len(train_hashes) == 391 and len(val_hashes) == 53,
            'TRAIN/validation exact photographic overlap')
    require(all(str(row['id']).startswith('tr_ffhq_') for row in train.values()), 'Nonhistorical TRAIN reference')
    source_rows, clean_cache = [], {}
    for ref in refs:
        path = checked(ref['source_path'], ref['source_sha256'])
        source_pixels = rgb(path, (1024, 1024))
        attribution = ref['attribution']
        require(all(attribution.get(key) for key in ('photo_url', 'license', 'license_url')),
                'Missing photographic provenance/terms')
        require(ref['decision'] == 'accept_clean_restoration' and
                type(ref['source_reviewed_at_original_resolution']) is bool,
                'Historical source decision/review flag differs')
        image = Image.fromarray(source_pixels)
        derived = np.asarray(image.resize((512, 512), Image.Resampling.LANCZOS)).copy()
        require(pixel_sha(derived) == ref['proposed_512_rgb_sha256'], 'Genuine direct512 target derivation differs')
        source_rows.append({'id': ref['id'], 'role': ref['historical_role'], 'source_sha256': ref['source_sha256'],
            'clean512_rgb8_sha256': pixel_sha(derived), 'clean512_float32_sha256': pixel_sha(derived.astype(np.float32) / np.float32(255)),
            'attribution': attribution, 'genuine_original1024_geometry': True,
            'historical_source_reviewed_at_original_resolution': ref['source_reviewed_at_original_resolution'],
            'source_visual_review_not_repeated': True})
        if ref['id'] in train:
            clean_cache[ref['id']] = source_rows[-1]
    history_path = returned / 'outputs/adaptation_v2/training.jsonl'
    history_bytes = history_path.read_bytes()
    history = [json.loads(line) for line in history_bytes.decode('utf-8').splitlines()]
    require(len(history) == 50 and [row['update'] for row in history] == list(range(1, 51)), 'Retained V2 history not50')
    # Independent PCG64 permutation confirms distinct IDs and order, not people.
    permutation = np.random.Generator(np.random.PCG64(20261015)).permutation(sorted(train)).tolist()[:100]
    selected = [row['TRAIN_selection'][name] for row in history for name in ('id_a', 'id_b')]
    require(selected == permutation and len(set(selected)) == 100, 'First100 TRAIN reference order differs')
    old_manifest_path = returned / 'outputs/adaptation_v2/export_manifest_20261015T140245Z-40fec745.json'
    old_manifest = read(old_manifest_path)
    require(old_manifest['protocol_sha256'] == PARENT_SHA, 'Retained return manifest differs')
    source_export = old_manifest['files_sha256']
    require(sha(history_path) == source_export['outputs/adaptation_v2/training.jsonl'], 'Retained trajectory hash differs')
    require(parity['source_export_manifest_sha256'] == sha(old_manifest_path), 'Parity original export binding differs')
    source_history = checked(parity['source_history'], parity['source_history_sha256'])
    require(source_history.read_bytes() == history_bytes, 'Staged complete50 history is not byte-identical')
    runtime_path = returned / 'outputs/adaptation_v2/runtime.json'
    runtime = read(checked(parity['source_runtime'], parity['source_runtime_sha256']))
    require(sha(runtime_path) == source_export['outputs/adaptation_v2/runtime.json'] == sha(checked(parity['source_runtime'])) and
            runtime == read(runtime_path), 'Staged original runtime differs')
    require(parity['runtime_expected'] == runtime,
            'Expected run software/runtime policy changed')
    cases = parity['cases']; anchors = parity['behavior_anchors']; updates = parity['updates']
    require(len(cases) == 200 and [(x['update'], x['slot']) for x in cases] ==
            [(u, s) for u in range(1, 51) for s in range(1, 5)], 'All200 parity cases must be ordered')
    counts = first50_counts()
    require(sum(bool(row['image_eligible']) for row in cases) == counts['perceptual_training_forwards'] == 148,
            'LPIPS eligibility count differs')
    audited_cases = []
    for index, case in enumerate(cases):
        update, slot = case['update'], case['slot']
        record = read(checked(case['pair_record_path'], case['pair_record_sha256']))
        source_record_path = 'outputs/adaptation_v2/training_inputs/update_%04d/slot_%s/pair_record.json' % (update, slot)
        source_record = read(bound(returned, source_record_path))
        require(sha(bound(returned, source_record_path)) == source_export[source_record_path] and
                record == source_record == history[update - 1]['microbatches'][slot - 1]['pair_record'],
                'Saved TRAIN record differs from immutable V2 history')
        ref = case['reference_id']
        require(ref in train and ref == selected[2 * (update - 1) + (0 if slot <= 2 else 1)] and
                record['role'] == 'train' and record['reference_id'] == ref and
                record['source_sha256'] == train[ref]['source_sha256'] == case['source_sha256'] and
                case['clean_source_path'] == train[ref]['source_path'], 'TRAIN reference/role/source differs')
        require(record['update'] == update and record['slot'] == slot and case['profile'] == record['profile'] and
                bool(case['image_eligible']) == counts['eligibility_by_order'][index] == bool(record['parameters']['image_eligible']),
                'Camera cycle/slot/eligibility differs')
        cycle = counts['cycles'][update - 1]
        require(record['parameters']['long_edge'] == (cycle['A_edge'] if slot <= 2 else int(re.sub('[^0-9]', '', cycle['B_variant']))) and
                record['member'] == ('normal' if slot % 2 else 'dark') and
                record['pair_role'] == ('A' if slot <= 2 else 'B'), 'Matched camera order differs')
        input_path = checked(case['input_path'], case['input_sha256'])
        source_input = 'outputs/adaptation_v2/training_inputs/update_%04d/slot_%s/input.png' % (update, slot)
        require(sha(input_path) == source_export[source_input] == sha(bound(returned, source_input)) == record['saved_input_sha256'],
                'Saved V2 TRAIN PNG equality differs')
        pixels = rgb(input_path, (256, 256)); input_float = pixels.astype(np.float32) / np.float32(255)
        require(pixel_sha(pixels) == case['input256_rgb8_sha256'] == record['input256_rgb8_sha256'] and
                pixel_sha(input_float) == case['input256_float32_sha256'] == record['input256_float32_sha256'],
                'Exact TRAIN input tensor binding differs')
        require(case['clean512_float32_sha256'] == record['clean512_float32_sha256'] == clean_cache[ref]['clean512_float32_sha256'] and
                case['clean512_rgb8_sha256'] == record['clean512_rgb8_sha256'] == clean_cache[ref]['clean512_rgb8_sha256'],
                'Exact clean512 target binding differs')
        audited_cases.append({'update': update, 'slot': slot, 'reference_id': ref,
            'input_sha256': case['input_sha256'], 'float32_input_sha256': case['input256_float32_sha256'],
            'clean512_float32_sha256': case['clean512_float32_sha256'], 'image_eligible': case['image_eligible'], 'profile': case['profile']})
    require(len(anchors) == 50 and [x['update'] for x in anchors] == list(range(1, 51)), '50 behavior anchors required')
    anchor_rows = []
    for row in anchors:
        update = row['update']; case = cases[(update - 1) * 4 + 2]
        require(row['slot'] == 3 and row['reference_id'] == case['reference_id'] and
                row['input256_float32_sha256'] == case['input256_float32_sha256'] and
                row['teacher_FP32_eval_frozen'] is True, 'Behavior input/role/frozen binding differs')
        original_record_rel = 'outputs/adaptation_v2/behavior_anchors/update_%04d/record.json' % update
        original_record_path = bound(returned, original_record_rel)
        require(sha(original_record_path) == source_export[original_record_rel], 'Original behavior record hash differs')
        original_record = read(original_record_path)
        require(row['teacher_state_sha256'] == original_record['teacher_state_sha256'] and
                original_record['teacher_FP32_eval_frozen'] is True and
                original_record['input256_float32_sha256'] == row['input256_float32_sha256'],
                'Retained original teacher state/input provenance differs')
        record = read(checked(row['pair_record_path'], row['pair_record_sha256']))
        require(record == read(checked(case['pair_record_path'])), 'Behavior pair record differs')
        for label, shape in [('delivered', (1, 3, 256, 256)), ('logits', (1, 256, 1024))]:
            path = checked(row[label + '_path'], row[label + '_sha256'])
            old_rel = 'outputs/adaptation_v2/behavior_anchors/update_%04d/%s.npy' % (update, 'delivered256' if label == 'delivered' else 'logits')
            require(sha(path) == source_export[old_rel] == sha(bound(returned, old_rel)), 'Exact original FP32 anchor differs')
            value = array(path, shape)
            if label == 'delivered':
                require(value.min() >= 0 and value.max() <= 1, 'Behavior delivered anchor outside[0,1]')
        anchor_rows.append({'update': update, 'reference_id': row['reference_id'], 'delivered_sha256': row['delivered_sha256'],
                            'logits_sha256': row['logits_sha256'], 'ordinary_FP32_saved_targets_only': True})
    require(len(updates) == 50 and [x['update'] for x in updates] == list(range(1, 51)), '50 committed LR updates required')
    for row, original in zip(updates, history):
        require(row['group_lrs'] == original['group_lrs'] and row['TRAIN_selection'] == original['TRAIN_selection'],
                'Original LR/sampler record changed')
        for group in ('encoder_token', 'fusion'):
            require(row['group_lrs'][group] == first50_lr(row['update'], group), 'Independent first50 LR sequence differs')
    validation_manifest = read(checked(p['validation_saved']['manifest'], p['validation_saved']['sha256']))
    require(validation_manifest['case_count'] == 318 and validation_manifest['locally_regenerated_development_pixels'] is False,
            'DEV validation inputs regenerated or count differs')
    comparison = read(checked(p['comparison_cases']))['cases']
    require(len(comparison) == 88 and len({x['id'] for x in comparison}) == 88, 'Comparison namespace differs')
    dev_rows = []
    for row in validation_manifest['items']:
        require(row['id'].startswith('validation_va_ffhq_'), 'Nonvalidation reference in saved DEV')
        for label in ('input', 'pair_record', 'target'):
            checked(row[label + '_path'], row[label + '_sha256'])
        dev_rows.append({'id': row['id'], 'role': 'development', 'evidence': 'paired_synthetic', 'input_sha256': row['input_sha256']})
    qmul = read(native_qmul); choke = read(native_chokepoint)
    require(sha(checked('provenance/native/qmul_frozen_subset.json')) == sha(native_qmul) and
            sha(checked('provenance/native/chokepoint_selection_plan.json')) == sha(native_chokepoint),
            'New native split provenance is not byte-identical')
    qdev = {str(x['global_person_id']) for x in qmul['cases'] if x['role'] == 'development'}
    qfinal = {str(x['global_person_id']) for x in qmul['cases'] if x['role'] == 'reserved_evaluation'}
    cdev, cfinal = set(choke['development_person_ids']), set(choke['reserved_person_ids'])
    require(len(qdev) == 24 and len(qfinal) == 32 and not qdev & qfinal and
            len(cdev) == 12 and len(cfinal) == 13 and not cdev & cfinal, 'Native DEV/FINAL label split differs')
    evidence_counts = Counter()
    for row in comparison:
        checked(row['prepared_path'], row['prepared_sha256'])
        if row['evidence'] == 'native_unpaired':
            label = row['identity'].split('/')[-1]
            require((row['source'] == 'QMUL-SurvFace' and label in qdev and label not in qfinal) or
                    (row['source'] == 'ChokePoint' and label in cdev and label not in cfinal),
                    'Native reserved or unbound identity entered DEV')
            require(not row.get('target_path') and not row.get('clean_path'), 'Native evidence must stay unpaired')
        else:
            for field in ('target_path', 'mask_path'):
                if field in row:
                    checked(row[field], row[field.replace('_path', '_sha256')])
        evidence_counts[row['evidence']] += 1
        dev_rows.append({'id': row['id'], 'role': 'development', 'source': row['source'], 'identity': row.get('identity'),
                         'evidence': row['evidence'], 'input_sha256': row['prepared_sha256']})
    require(evidence_counts['native_unpaired'] == 48 and len(comparison) - evidence_counts['native_unpaired'] == 40,
            'Paired/unpaired evidence counts differ')
    require(len(dev_rows) == 406 and len({x['id'] for x in dev_rows}) == 406, 'All406 DEV cases required')
    for counter, expected in counts.items():
        if counter in ('optimizer_updates', 'cycles', 'eligibility_by_order'):
            continue
        require(training['call_caps'][counter] == expected, 'Prospective neural/call cap differs: ' + counter)
    ledger = resource_ledger()
    require(ledger['payload_margin_bytes'] > 0 and ledger['archive_margin_bytes'] > 0 and
            ledger['startup_margin_bytes'] > 0, 'Finite disk envelope does not fit')
    output.mkdir(parents=True)
    write(output / 'TRAIN_reference_bindings.json', {'references': source_rows,
        'identity_person_nonoverlap_not_claimed': True, 'source_visual_review_repeated': False})
    write(output / 'TRAIN_input_parity.json', {'cases': audited_cases, 'updates': updates, 'behavior_anchors': anchor_rows})
    write(output / 'DEV_FINAL_separation.json', {'cases': dev_rows, 'native_sources': dict(evidence_counts),
        'native_split_files_sha256': {str(native_qmul): sha(native_qmul), str(native_chokepoint): sha(native_chokepoint)},
        'reserved_pixels_read_or_decoded': False, 'ethnicity_or_Zamboanga_claim': False,
        'cross_source_and_pretrained_person_overlap_unestablished': True})
    write(output / 'finite_resource_ledger.json', ledger)
    result = {'complete': True, 'approved_data_and_prospective_resource_contract': True,
        'protocol_or_unsealed_template_path': str(protocol_path), 'protocol_sha256': sha(protocol_path),
        'unsealed_template_audit': protocol_file is not None,
        'parity_manifest_sha256': sha(checked(parity_path)),
        'original_V2_protocol_sha256': PARENT_SHA, 'TRAIN_sources': 391, 'validation_sources': 53,
        'first50_reference_IDs': 100, 'saved_TRAIN_PNG_and_float32_hashes': 200,
        'exact_clean512_tensor_bindings': 200, 'FP32_behavior_anchor_pairs': 50,
        'full_DEV_inputs': 406, 'native_unpaired': 48, 'optimizer_updates_prepared_not_run': 50,
        'LPIPS_training_eligibility': 148, 'HQ_cache_export_pairs_planned': 391,
        'all_first50_LRs_identical_to_V2': True, 'no_reserved_native_labels_in_DEV': True,
        'hashed_bound_files': len(file_checks), 'source_sha256': sha(__file__),
        'runtime_model_quality_unverified': True, 'all_NN_gradient_optimizer_VM_calls': 0,
        'archive_or_protocol_sealed_by_this_audit': False, 'seconds': time.monotonic() - started}
    write(output / 'preparation_audit.json', result)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--packet-root', required=True)
    parser.add_argument('--parent-root', required=True)
    parser.add_argument('--v2-return-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--protocol-file', help='Explicit unsealed external template; no protocol is written or sealed')
    parser.add_argument('--native-qmul', default='outputs/cctv_native_development_v2/frozen_subset.json')
    parser.add_argument('--native-chokepoint', default='outputs/cctv_chokepoint_native_development_v1/selection_plan.json')
    args = parser.parse_args()
    print(json.dumps(audit(args.packet_root, args.parent_root, args.v2_return_root, args.output_dir,
                          args.native_qmul, args.native_chokepoint, args.protocol_file), indent=2))
