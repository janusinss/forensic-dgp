"""Independent CPU-only finite payload arithmetic for the new AdaIN packet.

This module does not import a neural framework or execute a worker.  Staged
input hashes are reviewed separately; this proves the source-derived storage
and first-fifty call ceilings, including retained incomplete writes.
"""
from copy import deepcopy

MIB = 1024**2
GIB = 1024**3


def require(value, message):
    if not value:
        raise ValueError(message)


def call_census():
    eligible = sum(2 + 2 * ((u - 1) % 4 >= 2) for u in range(1, 51))
    return {'optimizer_step_attempts': 50, 'backward_calls': 200,
        'gradient_queries': 9, 'route_witness_forwards': 4,
        'route_witness_perceptual_forwards': 4,
        'perceptual_prerequisite_forwards': 1,
        'candidate_training_forwards': 200, 'perceptual_training_forwards': eligible,
        'behavior_teacher_forwards': 50, 'teacher_forwards': 391,
        'snapshot_forwards': 812, 'baseline_forwards': 2,
        'initializer_parity_candidate_forwards': 2, 'training_input_writes': 200}


def witness_file_caps():
    # The writer uses nbytes +256 (not an assumed NumPy header length).
    arrays = {'native512': 512 * 512 * 3 * 4 + 256,
              'raw256': 256 * 256 * 3 * 4 + 256,
              'actual_tokens': 256 * 8 + 256}
    return {'four_native_raw_token_sets': 4 * sum(arrays.values()),
            'four_PNGs': 4 * 198656,
            'four_CPU_CUDA_RNG_NPYs': 4 * MIB}


def derive_ledger(parent, limits):
    """Use the reviewed parent components, adding only new witness artifacts."""
    require(parent['conservative_payload_bytes'] == 3421371648,
            'Reviewed FP32 resource baseline changed')
    require(parent['regular_file_count_plan']['per_call_trace_records'] == 2057,
            'Reviewed parent call trace count differs')
    components = deepcopy(parent['components'])
    components.update({'route_witness_' + key: value
                       for key, value in witness_file_caps().items()})
    total = sum(components.values())
    require(total == 3442100480, 'Witness payload arithmetic differs')
    files = deepcopy(parent['regular_file_count_plan'])
    files['per_call_trace_records'] = sum(call_census().values())
    files['route_witness_arrays_PNG_records_RNG_and_summary'] = 27
    count = sum(files.values())
    require(count == 6787 and count <= 7000, 'Finite output-file count differs')
    # Bound the exact worker TAR formula for all files, including padding,
    # extended headers, end records and incompressible DEFLATE overhead.
    tar_padding = count * (4096 + 511) + MIB
    tar_upper = total + tar_padding
    archive_upper = tar_upper + tar_upper // 1000 + 1 + MIB
    require(total <= limits['result_payload_bytes'] and
            archive_upper <= limits['archive_bytes'], 'Result/archive caps insufficient')
    combined = (limits['result_payload_bytes'] + limits['archive_bytes'] +
                limits['checkpoint_temporary_bytes'] + limits['reserve_free_bytes'])
    require(combined <= limits['initial_free_bytes'], 'Initial disk reserve insufficient')
    return {'components': components, 'conservative_payload_bytes': total,
        'payload_cap_bytes': limits['result_payload_bytes'],
        'payload_margin_bytes': limits['result_payload_bytes'] - total,
        'regular_file_count_plan': files, 'planned_file_count_upper': count,
        'regular_file_count_cap': 7000, 'tar_header_padding_allowance': tar_padding,
        'conservative_tar_gzip_upper_bytes': archive_upper,
        'archive_cap_bytes': limits['archive_bytes'],
        'archive_margin_bytes': limits['archive_bytes'] - archive_upper,
        'startup_free_bytes_after_install': limits['initial_free_bytes'],
        'reserve_bytes': limits['reserve_free_bytes'],
        'atomic_checkpoint_temporary_bytes': limits['checkpoint_temporary_bytes'],
        'combined_caps_plus_temporary_and_reserve': combined,
        'startup_margin_bytes': limits['initial_free_bytes'] - combined,
        'extra_safe_checkpoint_copy_allowed': False,
        'last_safe_checkpoint_must_be_a_reference_not_fourth_serialized_file': True,
        'metadata_and_logs_bound_must_be_enforced_at_runtime': True,
        'partial_checkpoint_replaces_one_of_three_state_slots': True,
        'other_partial_binary_replaces_the_same_planned_artifact': True,
        'four_RNG_files_bound_requires_exactly_one_CUDA_device': True,
        'metadata_allowance_includes_witness_reports_and_duplicated_initialization_copy': True}
