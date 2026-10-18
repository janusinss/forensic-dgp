"""Prospective source-derived byte/file/disk ledger; stdlib only, no measurements."""
import re
from . import contract as c

MIB = 1024 ** 2
GIB = 1024 ** 3
NPY_HEADER_BOUND = 256
RGB256_BYTES = 256 * 256 * 3 * 4
RGB256_NPY_BOUND = RGB256_BYTES + NPY_HEADER_BOUND
FIXED_METADATA_FILES = 128
TRAIN_METADATA_FILES_PER_OBSERVATION = 1
HISTORY_RECORD_FILES = 100
FIXED_RUNTIME_ARRAY_PAYLOAD_BOUND = 128 * 1024


def _row(count, per_file_bytes, scope):
    return {'files': count, 'per_file_bytes': per_file_bytes,
            'bytes': count * per_file_bytes, 'scope': scope}


def gzip_upper_bound(uncompressed_bytes):
    """Default zlib deflateBound envelope plus64bytes for gzip framing/rounding.

    Applies to stdlib gzip's default window/memory/strategy, as used by transport.
    It assumes no optional filename/comment header. Actual archive writes remain
    separately capped; this is not a compressed-size measurement.
    """
    c.require(type(uncompressed_bytes) is int and uncompressed_bytes >= 0, 'Nonnegative integer TAR bytes required')
    return (uncompressed_bytes + (uncompressed_bytes >> 12) + (uncompressed_bytes >> 14) +
            (uncompressed_bytes >> 25) + 64)


def checkpoint_slot_accounting(files_bytes):
    """Validate actual/proposed checkpoint file sizes without loading any state."""
    slots, temporary = {}, 0
    cap = c.DESIGN['state_and_return']['checkpoint_write_cap_bytes_each']
    for name, size in files_bytes.items():
        if not (name.endswith('.pt') or '.pt.tmp-' in name):
            continue
        matches = [slot for slot in c.CHECKPOINT_PATHS
                   if name == slot or re.fullmatch(re.escape(slot) + r'\.tmp-[0-9a-f]{32}', name)]
        c.require(len(matches) == 1 and type(size) is int and 0 <= size <= cap,
                  'Unknown or oversized checkpoint slot artifact')
        slot = matches[0]
        c.require(slot not in slots, 'Checkpoint slot cannot retain both final and temporary copies')
        slots[slot] = {'path': name, 'bytes': size, 'temporary': name != slot}
        if name != slot:
            temporary += size
    c.require(len(slots) <= 3 and temporary <= c.LIMITS['checkpoint_temporary_bytes'],
              'Aggregate checkpoint slot/temporary cap exceeded')
    return {'physical_slot_files': len(slots), 'slot_bytes': sum(r['bytes'] for r in slots.values()),
            'temporary_bytes': temporary, 'slots': slots}


def resource_ledger(protocol=None):
    """Derive the finite upper ledger; fail if any frozen prospective cap cannot fit.

    A checkpoint slot is write-once: final OR its retained partial temporary.
    Atomic promotion uses rename, never a second full copy. An unsuccessful
    artifact is retained and consumes its slot. The worker must enforce these
    namespace, aggregate metadata, file and every-write limits at runtime.
    """
    if protocol is not None:
        c.verify_protocol(protocol)
    caps, calls, census = c.LIMITS, c.COUNTS, c.CENSUS
    c.require(sum(calls.values()) == 3090 and census['all_neural_forward_calls'] == 2638 and
              census['full_DEV_output_pairs'] == 1218 and census['frozen_TRAIN_RAW_arrays'] == 400,
              'Frozen call/output census changed')
    checkpoint_cap = c.DESIGN['state_and_return']['checkpoint_write_cap_bytes_each']
    prerequisite_gradient_bytes = 2 * c.DESIGN['controller']['parameter_count'] * (4 + 8)
    prerequisite_array_bytes = prerequisite_gradient_bytes + 16 * NPY_HEADER_BOUND + \
        4 * (FIXED_RUNTIME_ARRAY_PAYLOAD_BOUND + NPY_HEADER_BOUND)
    prerequisite_json_bytes = 5 * 256 * 1024
    prerequisite_upper = prerequisite_array_bytes + prerequisite_json_bytes
    c.require(prerequisite_upper <= 2 * MIB, 'Actual shared prerequisite source exceeds2MiB envelope')
    payload = {
        'checkpoint_slots': _row(3, checkpoint_cap, 'Each slot final OR retained partial, no duplicate full copies'),
        'DEV_corrected_or_original_RAW': _row(1218, RGB256_NPY_BOUND, 'Three406 banks'),
        'controller_DEV_base_RAW': _row(812, RGB256_NPY_BOUND, 'Two406 controller banks retain unprotected base'),
        'frozen_TRAIN_RAW': _row(400, RGB256_NPY_BOUND, '200observations times original/base'),
        'TRAIN_input_PNG': _row(200, caps['one256PNG_bytes'], 'One exact retained input per observation'),
        'DEV_output_PNG': _row(1218, caps['one256PNG_bytes'], 'Three406 delivered PNG banks'),
        'known_stress_RAW': _row(2, RGB256_NPY_BOUND, 'Existing TRAIN base caches reused by reference'),
        'known_stress_PNG': _row(2, caps['one256PNG_bytes'], 'Two final head calls'),
        'prerequisite_all_artifacts': _row(1, 2 * MIB, '25files; includes own5JSON records conservatively'),
        'all_nonarray_metadata': _row(1, caps['metadata_bytes'],
            'Includes own source/protocol, operation events, records, history, logs, supervision and export metadata')}
    payload_upper = sum(row['bytes'] for row in payload.values())
    # Counts for the aggregate byte envelopes above must not be confused with one physical file.
    file_counts = {'checkpoint_slots_final_or_partial': 3, 'DEV_RAW': 1218, 'controller_DEV_base_RAW': 812,
        'frozen_TRAIN_RAW': 400, 'TRAIN_input_PNG': 200, 'DEV_output_PNG': 1218,
        'known_stress_RAW_PNG': 4, 'prerequisite_artifacts': 25,
        'operation_event_records': sum(calls.values()), 'DEV_evidence_records': 1218,
        'TRAIN_metadata_records': 200 * TRAIN_METADATA_FILES_PER_OBSERVATION,
        'history_attempt_and_committed_records': HISTORY_RECORD_FILES,
        'all_other_metadata_including_sources': FIXED_METADATA_FILES,
        'one_transient_atomic_noncheckpoint_file': 1}
    files_upper = sum(file_counts.values())
    # USTAR regular members:512header plus at most511payload padding. End marker
    # and one RECORDSIZE alignment block are reserved separately and generously.
    tar_overhead = caps['regular_files'] * (512 + 511) + 1024 + 10240
    tar_upper = payload_upper + tar_overhead
    gzip_upper = gzip_upper_bound(tar_upper)
    # Required free bytes are measured after source/dependency installation.
    # Reserve declared maxima, one archive only, one aggregate checkpoint-temp
    # allowance, one atomic image/metadata write, and HOME sidecar/closure space.
    temporary_bytes = caps['checkpoint_temporary_bytes']
    atomic_noncheckpoint_allowance = 8 * MIB
    home_sidecar_allowance = 16 * MIB
    simultaneous_disk = (caps['result_payload_bytes'] + temporary_bytes + caps['archive_bytes'] +
                         caps['reserve_free_bytes'] + atomic_noncheckpoint_allowance + home_sidecar_allowance)
    c.require(payload_upper <= caps['result_payload_bytes'], 'Prospective payload does not fit frozen3.5GiB cap')
    c.require(files_upper <= caps['regular_files'], 'Prospective physical file census exceeds9000')
    c.require(gzip_upper <= caps['archive_bytes'], 'Prospective TAR/default-gzip upper bound exceeds archive cap')
    c.require(simultaneous_disk <= caps['initial_free_bytes_after_install'],
              'Post-install free-space admission cannot contain declared simultaneous maxima and reserve')
    known_index_numpy = (200 + 406) * (256 * 256 * (3 + 12 + 1)) + 358 * RGB256_BYTES
    known_TRAIN_cache_numpy = 400 * RGB256_BYTES
    return {'format': 'codeformer-tone-control-source-resource-ledger-v1', 'complete': True,
        'prospective_only': True, 'measured_peak_resources': False, 'design_sha256': c.DESIGN_SHA256,
        'operation_reservations': sum(calls.values()), 'full_neural_forwards': 2638,
        'units': {'RGB256_float32_payload_bytes': RGB256_BYTES, 'NPY_header_bound_bytes': NPY_HEADER_BOUND,
                  'RGB256_NPY_bound_bytes': RGB256_NPY_BOUND, 'checkpoint_slot_bytes': checkpoint_cap},
        'payload_categories': payload, 'payload_upper_bytes': payload_upper,
        'payload_cap_bytes': caps['result_payload_bytes'],
        'payload_margin_bytes': caps['result_payload_bytes'] - payload_upper,
        'file_categories': file_counts, 'regular_files_upper': files_upper,
        'regular_file_cap': caps['regular_files'], 'regular_file_margin': caps['regular_files'] - files_upper,
        'prerequisite_source_detail': {'gradient_array_files': 16, 'gradient_payload_bytes': prerequisite_gradient_bytes,
            'RNG_array_files': 4, 'JSON_files': 5, 'all25files_upper_bytes': prerequisite_upper,
            'reserved_bytes': 2 * MIB},
        'archive': {'USTAR_regular_only': True, 'optional_gzip_filename_comment_forbidden': True,
            'TAR_header_padding_EOA_upper_bytes': tar_overhead, 'uncompressed_TAR_upper_bytes': tar_upper,
            'default_zlib_gzip_upper_bytes': gzip_upper, 'archive_cap_bytes': caps['archive_bytes'],
            'archive_margin_bytes': caps['archive_bytes'] - gzip_upper},
        'disk': {'source_and_parent_already_installed': True, 'one_archive_only': True,
            'aggregate_checkpoint_temporary_cap_bytes': temporary_bytes,
            'atomic_noncheckpoint_allowance_bytes': atomic_noncheckpoint_allowance,
            'HOME_sidecar_allowance_bytes': home_sidecar_allowance,
            'simultaneous_declared_maxima_plus_reserve_bytes': simultaneous_disk,
            'required_initial_free_bytes': caps['initial_free_bytes_after_install'],
            'admission_margin_bytes': caps['initial_free_bytes_after_install'] - simultaneous_disk},
        'known_CPU_numpy_residency_bytes': {'index_arrays': known_index_numpy,
            '400TRAIN_arrays_if_all_resident_upper_bound': known_TRAIN_cache_numpy,
            'sum_excludes_Python_Torch_models_and_transient_allocations': known_index_numpy + known_TRAIN_cache_numpy},
        'runtime_enforcement_required': ['Every write/promotion and aggregate bytes/files checked before mutation',
            'Checkpoint slots write-once; rename promotion; no second full checkpoint or silent deletion',
            'One event file per reservation, oneTRAIN cacheJSON per observation, twohistory files per update',
            'All source/protocol/history/log/export metadata share the128MiB aggregate cap',
            'Actual RSS/CUDA/time and root/HOME free space measured independently; projections are not peaks'],
        'neural_framework_or_gradient_optimizer_calls': 0}
