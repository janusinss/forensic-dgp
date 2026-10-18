"""Derive metadata-only HQ target parity from a closed return; no neural imports."""
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / 'outputs/codeformer_fp32_comparison_v1_return'
OLD_REVIEW = ROOT / 'outputs/codeformer_fp32_comparison_v1_return_review'
OUT = ROOT / 'outputs/codeformer_adain_gradient_isolation_v1_preparation'
PROTOCOL_SHA = '8acff86e31c3d135994fdac6ed781374e036150ce1ae2110db82ca0dc5a3aff1'
TRAJECTORY_SHA = '66fc747b0de090d398058d9a7504c12e9847afb5124277caa6e2aaf58759c231'


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def main():
    trajectory_path = OLD_REVIEW / 'trajectory_v1/trajectory_audit.json'
    assert sha(trajectory_path) == TRAJECTORY_SHA
    trajectory = json.loads(trajectory_path.read_text(encoding='utf-8'))
    assert trajectory['evidence_consistency_pass'] and trajectory['sections']['HQ_cache']['frozen_codebook_lookup_exact']
    transport_path = OLD_REVIEW / 'transport_v1.json'
    transport = json.loads(transport_path.read_text(encoding='utf-8'))
    assert transport['complete'] and transport['protocol_sha256'] == PROTOCOL_SHA
    assert transport['archive_sha256'] == '8b2361147a1d12b7d2bad8a7d3a8965e0e90c077bb1cdbb4ad7378ad8703ef78'
    assert transport['all_members_hash_verified'] and transport['exact_namespace']
    manifest_path = PRIOR / transport['manifest_member']
    assert sha(manifest_path) == transport['manifest_sha256'] == '4562d4961c7862828d2164b96dd6df675f7b09171298109aba29cce44fb8395b'
    files = json.loads(manifest_path.read_text(encoding='utf-8'))['files_sha256']
    prefix = 'outputs/fp32_comparison_v1/HQ_cache/'
    names = sorted(name for name in files if name.startswith(prefix) and name.endswith('/record.json'))
    assert len(names) == 391
    rows, bindings, roles = [], {}, {}
    for name in names:
        record_path = PRIOR / name
        assert sha(record_path) == files[name]
        row = json.loads(record_path.read_text(encoding='utf-8'))
        result = {key: row[key] for key in ('reference_id', 'role', 'source_path', 'source_sha256',
            'clean512_float32_sha256', 'teacher_state_sha256', 'restoration_codebook_state_sha256')}
        assert result['clean512_float32_sha256'] == trajectory['sections']['HQ_cache']['clean512_float32_sha256'][row['reference_id']]
        assert result['teacher_state_sha256'] == 'dd2d202f42d5a7f4ea1bec676c50efb716fcfd28b544161bc26dba6491f9dfd8'
        assert result['restoration_codebook_state_sha256'] == '8daf2cc2b996c9ff1ef9bed3a7ac39446c701509341a0b0e1dc5974404c0d12d'
        bindings[name] = files[name]
        for label, shape, dtype in [('indices', (1, 256), 'int64'), ('features', (1, 256, 16, 16), 'float32')]:
            rel = row[label + '_path']
            assert rel == prefix + row['reference_id'] + '/' + label + '.npy'
            path = PRIOR / rel
            assert sha(path) == files[rel] == row[label + '_sha256']
            with path.open('rb') as stream:
                array = np.load(stream, allow_pickle=False)
                assert stream.read(1) == b''
            assert array.shape == shape and array.dtype == np.dtype(dtype) and array.flags.c_contiguous
            assert np.isfinite(array).all()
            if label == 'indices':
                assert ((array >= 0) & (array < 1024)).all()
            result[label + '_array_sha256'] = hashlib.sha256(array.tobytes()).hexdigest()
            bindings[rel] = files[rel]
        rows.append(result)
        roles[row['role']] = roles.get(row['role'], 0) + 1
    assert len({row['reference_id'] for row in rows}) == 391
    value = {'format': 'codeformer-fp32-HQ-cache-parity-v1', 'source_protocol_sha256': PROTOCOL_SHA,
        'cases': rows, 'case_count': 391, 'roles': roles,
        'original_teacher_nearest_indices_not_rerun_locally': True,
        'prior_frozen_dictionary_feature_lookup_verified': True,
        'metadata_only_for_fresh_VM_recomputation_before_Adam': True,
        'no_new_training_or_qualification': True}
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'hq_cache_parity_v1.json'
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    receipt = {'complete': True, 'metadata_path': str(target.relative_to(ROOT)), 'metadata_sha256': sha(target),
        'source_sha256': sha(__file__), 'source_protocol_sha256': PROTOCOL_SHA,
        'prior_trajectory_sha256': TRAJECTORY_SHA, 'prior_transport_sha256': sha(transport_path),
        'prior_manifest_sha256': sha(manifest_path),
        'freshly_verified_old_cache_files': len(bindings), 'files_sha256': bindings,
        'case_count': 391, 'roles': roles, 'local_neural_gradient_optimizer_VM_calls': 0}
    with (OUT / 'hq_cache_parity_derivation_v1.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: receipt[key] for key in ('complete', 'metadata_path', 'metadata_sha256', 'case_count', 'roles')}))


if __name__ == '__main__':
    main()
