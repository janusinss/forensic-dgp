"""Independent stdlib readback of finite native exclusion (opaque hashes only)."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_hinge_native_exclusion_v1_analysis/r1'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def resolve(base, rel):
    value = PurePosixPath(rel)
    assert not value.is_absolute() and '..' not in value.parts and ':' not in rel and '\\' not in rel
    path = base.joinpath(*value.parts)
    assert path.resolve().is_relative_to(base.resolve()) and path.is_file() and not path.is_symlink()
    return path


def main():
    started = time.monotonic()
    destination = OUT / 'independent_readback.json'
    assert not destination.exists(), 'Exclusive receipt already exists; preserve it'
    audit = read(OUT / 'audit.json')
    manifest = read(OUT / 'manifest.json')
    assert audit['complete'] and manifest['complete']
    assert manifest['audit_sha256'] == digest(OUT / 'audit.json')
    assert manifest['script_sha256'] == digest(ROOT / 'scripts/audit_cctv_dgp_hinge_native_exclusion_v1.py')
    for rel, binding in audit['all_hashed_files'].items():
        path = resolve(ROOT, rel)
        assert path.stat().st_size == binding['bytes'] and digest(path) == binding['sha256'], rel
        assert time.monotonic() - started < 120, 'Finite independent readback stop'
    hroot = ROOT / 'outputs/cctv_dgp_decoder_hinge_repair_v1_vm'
    source = ROOT / 'outputs/cctv_dgp_spatial_readout_v1_return'
    parent = ROOT / 'outputs/cctv_dgp_bank_photometric_guard_v1_return'
    replay = ROOT / 'outputs/cctv_dgp_spatial_path_replay_v1_return'
    hinge = read(hroot / 'protocol.json')
    parent_p = read(parent / 'protocol.json')
    expected_transitive = {(label, (base / rel).relative_to(ROOT).as_posix(), pin)
        for label, base, mapping in [('hinge', hroot, hinge['assets_sha256']),
                                     ('source', source, hinge['source_required_sha256']),
                                     ('parent', parent, hinge['parent_required_sha256']),
                                     ('replay', replay, hinge['replay_required_sha256'])]
        for rel, pin in mapping.items()}
    assert expected_transitive == {(r['root'], r['path'], r['sha256']) for r in audit['transitive_assets']}
    assert len(expected_transitive) == 767
    qroot = ROOT / 'outputs/cctv_native_development_v2'
    eroot = ROOT / 'outputs/cctv_native_structure_extension_v1'
    croot = ROOT / 'outputs/cctv_chokepoint_native_development_v1'
    qmul, extension, choke = (read(base / 'frozen_subset.json') for base in (qroot, eroot, croot))
    expected_native = set()
    for row in qmul['cases']:
        expected_native.add(('QMUL-SurvFace', row['role'], row['id'], 'native_source_encoded',
            (qroot / row['source_file']).relative_to(ROOT).as_posix(), row['source_sha256']))
    for row in extension['cases']:
        for kind, key in [('native_source_encoded', 'source'), ('prepared_input_encoded', 'input')]:
            expected_native.add(('QMUL-SurvFace extension', 'excluded_development', row['id'], kind,
                (eroot / row['source_file' if key == 'source' else 'input']).relative_to(ROOT).as_posix(), row[key + '_sha256']))
    for row in choke['cases']:
        if row['role'] == 'development':
            for kind, key in [('native_source_encoded', 'native_crop'), ('prepared_input_encoded', 'input')]:
                expected_native.add(('ChokePoint/P1E_S1_C1', row['role'], row['id'], kind,
                    (croot / row[key]).relative_to(ROOT).as_posix(), row[key + '_sha256']))
        else:
            assert not any(row.get(k) for k in ('native_crop', 'input', 'observed'))
        expected_native.add(('ChokePoint/P1E_S1_C1', row['role'], row['id'], 'published_frame_hash_metadata', None, row['source_frame_sha256']))
    for folder, rowskey, arm in [('cctv_native_comparison_v1', 'rows', 'input'),
                                 ('cctv_dgp_current_qmul_native_comparison_v1', 'records', 'resize')]:
        base = ROOT / 'outputs' / folder
        result = read(base / 'results.json')
        for row in result[rowskey]:
            rel = row['outputs'][arm]
            expected_native.add(('QMUL-SurvFace', 'development', row['id'], 'prepared_input_encoded_' + arm,
                (base / rel).relative_to(ROOT).as_posix(), result['artifacts_sha256'][rel]))
    actual_native = {(r['source'], r['role'], r['case_id'], r['kind'], r['path'], r['sha256'])
                     for r in audit['native_face_or_frame_hash_entries']}
    assert expected_native == actual_native and len(expected_native) == 208
    cases = [r for r in parent_p['cases'] if r['id'] in set(hinge['case_ids'])]
    assert len(cases) == 100 and all(r['role'] == 'train' for r in cases)
    assert len({r['reference_id'] for r in cases}) == 20
    expected_train = {(r['id'], r['reference_id'], r['source'], kind,
        (parent / r[kind]).relative_to(ROOT).as_posix(), parent_p['assets_sha256'][r[kind]])
        for r in cases for kind in ('input', 'target', 'observed')}
    actual_train = {(r['case_id'], r['reference_id'], r['source'], r['kind'], r['path'], r['sha256'])
                    for r in audit['TRAIN_asset_entries']}
    assert actual_train == expected_train and len(expected_train) == 300
    native_hashes = {r[-1] for r in expected_native}
    assert not {r[-1] for r in expected_train if r[3] != 'observed'} & native_hashes
    refs = [r for r in parent_p['references'] if r['id'] in {c['reference_id'] for c in cases}]
    ancestry = {(r['id'], r['source'], kind, r[kind]) for r in refs
                for kind in ('source_sha256', 'hq_source_sha256') if r.get(kind)}
    assert ancestry == {(r['reference_id'], r['source'], r['kind'], r['sha256']) for r in audit['TRAIN_ancestral_metadata']}
    assert not {r[-1] for r in ancestry} & native_hashes
    for field in ('new_pixel_decodes', 'model_forwards', 'gradient_queries', 'optimizer_updates', 'VM_connections'):
        assert audit[field] == 0
    result = {'complete': True, 'audit_sha256': digest(OUT / 'audit.json'),
              'manifest_sha256': digest(OUT / 'manifest.json'), 'plan_sha256': digest(OUT / 'plan.json'),
              'auditor_sha256': digest(Path(__file__)),
              'all_file_bindings_rehashed': len(audit['all_hashed_files']),
              'transitive_bindings_rederived': len(expected_transitive),
              'native_face_frame_entries_rederived': len(expected_native),
              'TRAIN_asset_entries_rederived': len(expected_train),
              'fresh_encoded_equality_matches': 0, 'ancestral_metadata_matches': 0,
              'pixel_decodes': 0, 'neural_forwards': 0, 'gradient_queries': 0, 'optimizer_updates': 0,
              'reserved_frame_archive_members_read': 0, 'scientific_quality_or_person_disjointness_proved': False,
              'failed_v1_source_preserved': digest(OUT.parent / 'failed_audit_source.py'),
              'failed_v1_receipt_preserved': digest(OUT.parent / 'failure.json'),
              'seconds': time.monotonic() - started}
    with destination.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
