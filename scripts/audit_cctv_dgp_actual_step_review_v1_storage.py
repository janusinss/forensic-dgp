"""Measure a stopped return and prove possible lossless original-RGB deduplication."""
from pathlib import Path
import hashlib
import time
import zipfile
import numpy as np
from cctv_dgp_actual_step_review_v1_contract import NAME, read, write, sha, output_prefix, ROLES, role_ids

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs' / NAME
RETURN = ROOT / 'outputs' / (NAME + '_return')
OUT = ROOT / 'outputs/cctv_dgp_actual_step_review_v1_storage_analysis'


def main():
    start = time.monotonic()
    assert not OUT.exists()
    imported = read(ROOT / 'outputs/cctv_dgp_actual_step_review_v1_return_import.json')
    assert imported['complete']
    p = read(BUNDLE / 'protocol.json')
    original = {}
    checked = []
    removable_compressed_payload = 0
    total_original_compressed_payload = 0
    for probe in p['probes']:
        for role in ROLES:
            for cid in role_ids(p, probe, role):
                assert time.monotonic() - start < 180
                path = RETURN / output_prefix(probe['update'], 'zero') / role / (cid + '.npz')
                assert sha(path) == imported['files_sha256'][path.relative_to(RETURN).as_posix()]
                with zipfile.ZipFile(path) as archive:
                    info = archive.getinfo('original_rgb.npy')
                    encoded_bytes = info.compress_size
                with np.load(path, allow_pickle=False) as data:
                    value = data['original_rgb'].copy()
                assert value.dtype == np.float32 and value.shape == (256, 256, 3) and np.isfinite(value).all()
                digest = hashlib.sha256(value.tobytes()).hexdigest()
                if cid in original:
                    assert original[cid]['raw_sha256'] == digest
                    removable_compressed_payload += encoded_bytes
                    duplicate = True
                else:
                    original[cid] = {'raw_sha256': digest, 'first_path': path.relative_to(ROOT).as_posix(), 'compressed_payload_bytes': encoded_bytes}
                    duplicate = False
                total_original_compressed_payload += encoded_bytes
                checked.append({'update': probe['update'], 'role': role, 'id': cid, 'raw_sha256': digest,
                    'compressed_payload_bytes': encoded_bytes, 'duplicate': duplicate})
    assert len(checked) == 1050 and len(original) == 145
    retained_files = [RETURN / name for name in imported['files_sha256']]
    total = sum(path.stat().st_size for path in retained_files)
    unfinished_folder = RETURN / output_prefix(45, 'cone')
    unfinished = [path for path in retained_files if path.is_relative_to(unfinished_folder)]
    counts = {suffix: sum(path.suffix == suffix for path in unfinished) for suffix in ['.npz', '.png', '.json']}
    assert counts == {'.npz': 66, '.png': 130, '.json': 0}
    # A float NPZ here has one RGB, four512-element vectors and seven terms.
    # Use a generous1MiB/NPZ and256KiB/PNG; no compression benefit is assumed.
    missing_npz, missing_png = 105 - counts['.npz'], 210 - counts['.png']
    finite_tail_storage_upper_bound = missing_npz * 1024**2 + missing_png * 256 * 1024 + 16 * 1024**2
    prospective_full_deduplicated_upper_bound = total - removable_compressed_payload + finite_tail_storage_upper_bound
    assert prospective_full_deduplicated_upper_bound < p['budgets']['maximum_output_bytes'] - 32 * 1024**2
    OUT.mkdir()
    write(OUT / 'lossless_storage_evidence.json', {
        'complete': True, 'archive_sha256': imported['archive_sha256'], 'protocol_sha256': sha(BUNDLE / 'protocol.json'),
        'original_failure_sha256': sha(RETURN / 'outputs/failure.json'), 'original_failure_retained': True,
        'uncompressed_return_bytes': total, 'original_maximum_output_bytes': p['budgets']['maximum_output_bytes'],
        'worker_margin_bytes': 16 * 1024**2, 'export_margin_bytes': 8 * 1024**2,
        'last_cone_condition_files': counts, 'last_condition_fully_checked_slots': 0,
        'all1050_original_RGB_copies_checked': True, 'unique_original_case_RGB_arrays': 145,
        'bitwise_identical_duplicate_copies': 905, 'total_original_RGB_compressed_payload_bytes': total_original_compressed_payload,
        'minimum_duplicate_payload_saving_bytes': removable_compressed_payload,
        'retained_unique_original_RGB': original, 'all1050_checks': checked,
        'missing_NPZ_files': missing_npz, 'missing_PNG_files': missing_png,
        'prospective_missing_tail_storage_upper_bound': finite_tail_storage_upper_bound,
        'prospective_full_deduplicated_storage_upper_bound': prospective_full_deduplicated_upper_bound,
        'bound_is_serialization_estimate_not_completion_receipt': True,
        'no_deduplication_or_deletion_performed': True, 'lossless_references_would_need_a_new_frozen_format_and_checker': True,
        'scientific_image_and_preservation_gates_unchanged': True,
        'model_forwards': 0, 'optimizer_updates': 0, 'gradient_queries': 0, 'VM_calls': 0,
        'new_trained_checkpoint': False, 'app_promotion': False, 'goal_complete': False,
        'seconds': time.monotonic() - start, 'cap_seconds': 180, 'checker_sha256': sha(Path(__file__)),
    })
    print({'complete': True, 'original_RGB_copies_checked': 1050, 'bitwise_duplicate_copies': 905,
        'potential_lossless_saving_MiB': removable_compressed_payload / 1024**2,
        'full_output_serialization_upper_bound_GiB': prospective_full_deduplicated_upper_bound / 1024**3,
        'files_deleted': 0, 'seconds': time.monotonic() - start}, flush=True)


if __name__ == '__main__':
    main()
