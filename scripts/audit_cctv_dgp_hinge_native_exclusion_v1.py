"""Read-only encoded-byte/metadata exclusion audit; never decode any image.

This deliberately imports only Python's standard library. Reserved QMUL source
copies are hashed as opaque bytes under the frozen native contract. Reserved
ChokePoint records stay metadata-only: no archive member is extracted or read.
No identity-disjointness, useful-output, or unseen-data claim follows from hashes.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_hinge_native_exclusion_v1_analysis/r1'
HINGE = ROOT / 'outputs/cctv_dgp_decoder_hinge_repair_v1_vm'
SOURCE = ROOT / 'outputs/cctv_dgp_spatial_readout_v1_return'
PARENT = ROOT / 'outputs/cctv_dgp_bank_photometric_guard_v1_return'
REPLAY = ROOT / 'outputs/cctv_dgp_spatial_path_replay_v1_return'
PINS = {
    'hinge': 'a4d57c933214345a6d1cab5639007d15599f85e0db0450a0aad11aea5d715518',
    'source': '3bfbaf4b4a519f2926f0503731cd9d22390a7fc0c09da20e2f022e6a1b93c97e',
    'parent': 'f64154bd602070edaea61ccbf960ec3028e545c6540a4421ec41f9af21128a12',
    'replay': '6f9a8757466578ef57457994ded7fe9b6b170657f2ee7dd3b8b23f0bd2cc1bc7',
}
BUDGET = {'wall_seconds': 180, 'hashed_bytes': 16 * 1024**3,
          'unique_regular_files': 1500, 'metadata_file_bytes': 32 * 1024**2}


def safe(base, rel):
    value = PurePosixPath(rel)
    assert isinstance(rel, str) and not value.is_absolute()
    assert '\\' not in rel and ':' not in rel and '..' not in value.parts
    path = base.joinpath(*value.parts)
    assert path.resolve().is_relative_to(base.resolve())
    assert path.is_file() and not path.is_symlink(), 'Require exact regular file: ' + str(path)
    return path


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


class Reader:
    def __init__(self):
        self.started = time.monotonic()
        self.bindings = {}
        self.bytes = 0

    def clock(self):
        assert time.monotonic() - self.started <= BUDGET['wall_seconds'], 'Finite wall stop'
        assert self.bytes <= BUDGET['hashed_bytes'], 'Finite hash byte stop'
        assert len(self.bindings) <= BUDGET['unique_regular_files'], 'Finite file count stop'

    def digest(self, path, expected=None):
        path = path.resolve()
        assert path.is_relative_to(ROOT.resolve()) and path.is_file() and not path.is_symlink()
        key = path.relative_to(ROOT).as_posix()
        if key not in self.bindings:
            before = path.stat()
            assert before.st_size <= BUDGET['hashed_bytes']
            h = hashlib.sha256()
            with path.open('rb') as stream:
                while chunk := stream.read(1024 * 1024):
                    h.update(chunk)
                    self.bytes += len(chunk)
                    self.clock()
            after = path.stat()
            assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'File changed during hash'
            self.bindings[key] = {'sha256': h.hexdigest(), 'bytes': after.st_size}
        value = self.bindings[key]['sha256']
        assert expected is None or value == expected, 'Hash differs: ' + key
        return value

    def read(self, path, expected=None):
        assert path.stat().st_size <= BUDGET['metadata_file_bytes']
        self.digest(path, expected)
        return json.loads(path.read_text(encoding='utf-8'))


def run(reader):
    protocol_roots = {'hinge': HINGE, 'source': SOURCE, 'parent': PARENT, 'replay': REPLAY}
    protocols = {name: reader.read(root / 'protocol.json', PINS[name])
                 for name, root in protocol_roots.items()}
    hinge, source, parent = (protocols[x] for x in ('hinge', 'source', 'parent'))
    assert hinge['source_protocol_sha256'] == PINS['source']
    assert hinge['parent_protocol_sha256'] == PINS['parent']
    assert hinge['replay_protocol_sha256'] == PINS['replay']
    transitive = []
    for label, base, mapping in [('hinge', HINGE, hinge['assets_sha256']),
                                 ('source', SOURCE, hinge['source_required_sha256']),
                                 ('parent', PARENT, hinge['parent_required_sha256']),
                                 ('replay', REPLAY, hinge['replay_required_sha256'])]:
        for rel, pin in sorted(mapping.items()):
            path = safe(base, rel)
            reader.digest(path, pin)
            transitive.append({'root': label, 'path': path.relative_to(ROOT).as_posix(), 'sha256': pin})
    assert [sum(row['root'] == x for row in transitive) for x in ('hinge', 'source', 'parent', 'replay')] == [18, 410, 203, 136]

    old = reader.read(ROOT / 'outputs/cctv_dgp_native_contract_audit_20261009_v1/audit.json',
                      'a88bbff1374f0bf4ab696d451d75ecb63e738392a4f38028ce67b7646a7995f4')
    metadata = old['metadata_bindings_sha256']
    roots = {'qmul': ROOT / 'outputs/cctv_native_development_v2',
             'extension': ROOT / 'outputs/cctv_native_structure_extension_v1',
             'choke': ROOT / 'outputs/cctv_chokepoint_native_development_v1'}
    frozen = {label: reader.read(base / 'frozen_subset.json', metadata[(base / 'frozen_subset.json').relative_to(ROOT).as_posix()])
              for label, base in roots.items()}
    reviews = {label: reader.read(base / 'input_review.json', metadata[(base / 'input_review.json').relative_to(ROOT).as_posix()])
               for label, base in roots.items()}
    for label, review in reviews.items():
        assert review['subset_sha256'] == reader.digest(roots[label] / 'frozen_subset.json')
        assert review['reviewed_before_model_outputs']
    assert len(frozen['qmul']['cases']) == 56 and len(reviews['qmul']['rows']) == 24
    assert len(frozen['extension']['cases']) == 3 and len(reviews['extension']['rows']) == 3
    assert reviews['extension']['usable_cases'] == 0
    assert len(frozen['choke']['cases']) == 50 and len(reviews['choke']['rows']) == 24

    # Verify all opaque source/prepared/support files retained in the last native
    # contract. Supports are not face content and are classified separately.
    for rel, pin in old['encoded_asset_hashes_verified'].items():
        reader.digest(safe(ROOT, rel), pin)
    native = []
    def add_native(source_name, role, case_id, kind, path=None, pin=None):
        if path is not None:
            assert pin is not None
            reader.digest(path, pin)
        native.append({'source': source_name, 'role': role, 'case_id': case_id,
                       'kind': kind, 'path': path.relative_to(ROOT).as_posix() if path else None,
                       'sha256': pin, 'encoded_bytes_rehashed': path is not None})
    for c in frozen['qmul']['cases']:
        add_native('QMUL-SurvFace', c['role'], c['id'], 'native_source_encoded',
                   safe(roots['qmul'], c['source_file']), c['source_sha256'])
    for c in frozen['extension']['cases']:
        for kind, pathkey, hashkey in [('native_source_encoded', 'source_file', 'source_sha256'),
                                      ('prepared_input_encoded', 'input', 'input_sha256')]:
            add_native('QMUL-SurvFace extension', 'excluded_development', c['id'], kind,
                       safe(roots['extension'], c[pathkey]), c[hashkey])
    for c in frozen['choke']['cases']:
        assert c['role'] in ('development', 'reserved_evaluation')
        if c['role'] == 'development':
            for kind, pathkey, hashkey in [('native_source_encoded', 'native_crop', 'native_crop_sha256'),
                                          ('prepared_input_encoded', 'input', 'input_sha256')]:
                add_native('ChokePoint/P1E_S1_C1', c['role'], c['id'], kind,
                           safe(roots['choke'], c[pathkey]), c[hashkey])
        else:
            assert all(c.get(key) is None for key in ('native_crop', 'input', 'observed'))
        add_native('ChokePoint/P1E_S1_C1', c['role'], c['id'], 'published_frame_hash_metadata',
                   pin=c['source_frame_sha256'])

    # The original 24 prepared QMUL controls were omitted by the prior V42
    # exclusion audit. Include BOTH retained common-input encodings, without
    # opening pixels, manufacturing a new final input, or using model outputs.
    for folder, result_name, audit_name, rowskey, arm in [
            ('cctv_native_comparison_v1', 'results.json', 'independent_verification.json', 'rows', 'input'),
            ('cctv_dgp_current_qmul_native_comparison_v1', 'results.json', 'independent_saved_output_audit.json', 'records', 'resize')]:
        base = ROOT / 'outputs' / folder
        audit = reader.read(base / audit_name)
        result = reader.read(base / result_name, audit['results_sha256'])
        assert audit.get('complete', audit.get('verified'))
        assert len(result[rowskey]) == 24
        assert {r['id'] for r in result[rowskey]} == {c['id'] for c in frozen['qmul']['cases'] if c['role'] == 'development'}
        for row in result[rowskey]:
            rel = row['outputs'][arm]
            add_native('QMUL-SurvFace', 'development', row['id'], 'prepared_input_encoded_' + arm,
                       safe(base, rel), result['artifacts_sha256'][rel])

    selected_ids = hinge['case_ids']
    assert len(selected_ids) == len(set(selected_ids)) == 100
    assert selected_ids == source['case_ids'] and hinge['reference_blocks'] == source['reference_blocks']
    cases = [c for c in parent['cases'] if c['id'] in set(selected_ids)]
    references = [r for r in parent['references'] if r['id'] in {c['reference_id'] for c in cases}]
    assert len(cases) == 100 and len(references) == 20
    assert all(c['role'] == 'train' and c['source'] in ('dataset/asian_faces', 'dataset/thumbnails128x128') for c in cases)
    assert all(r['role'] == 'train' for r in references)
    train_assets = []
    def add_train(c, kind, rel, pin=None):
        expected = parent['assets_sha256'][rel]
        assert rel in hinge['parent_required_sha256'] and hinge['parent_required_sha256'][rel] == expected
        path = safe(PARENT, rel)
        reader.digest(path, expected)
        train_assets.append({'case_id': c['id'], 'reference_id': c['reference_id'], 'source': c['source'],
                             'kind': kind, 'path': path.relative_to(ROOT).as_posix(), 'sha256': expected,
                             'ancestral_local_source_path_metadata': parent['source_mapping'].get(rel)})
    for c in cases:
        for kind in ('input', 'target', 'observed'):
            add_train(c, kind, c[kind])
    # Source/HQ digests are declared, bound lineage metadata. Some source files
    # are intentionally absent from this finite packet; do not invent rehashes.
    ancestral = []
    for r in references:
        for key in ('source_sha256', 'hq_source_sha256'):
            if r.get(key):
                ancestral.append({'reference_id': r['id'], 'source': r['source'], 'kind': key,
                                  'source_file_metadata': r.get('source_file'), 'sha256': r[key],
                                  'current_file_rehashed': False})

    face_native = defaultdict(list)
    for row in native:
        face_native[row['sha256']].append(row)
    equality = []
    for row in train_assets:
        if row['kind'] != 'observed':
            for match in face_native.get(row['sha256'], []):
                equality.append({'train': row, 'native': match})
    ancestor_matches = [{'train_ancestry': r, 'native': m} for r in ancestral
                        for m in face_native.get(r['sha256'], [])]
    native_case_ids = {c['id'] for subset in frozen.values() for c in subset['cases']}
    assert not (set(selected_ids) & native_case_ids), 'Native case ID admitted to current TRAIN cases'
    assert not equality and not ancestor_matches, 'Exact encoded native/TRAIN equality found; retain stop'
    support_hashes = {v for k, v in old['encoded_asset_hashes_verified'].items() if '/observed/' in k}
    support_equalities = [r for r in train_assets if r['kind'] == 'observed' and r['sha256'] in support_hashes]
    # Source-specific namespaced release labels only; no cross-source matching.
    qmul_dev = {c['global_person_id'] for c in frozen['qmul']['cases'] if c['role'] == 'development'}
    qmul_final = {c['global_person_id'] for c in frozen['qmul']['cases'] if c['role'] == 'reserved_evaluation'}
    extension_ids = {c['global_person_id'] for c in frozen['extension']['cases']}
    choke_dev = {c['source_person_id'] for c in frozen['choke']['cases'] if c['role'] == 'development'}
    choke_final = {c['source_person_id'] for c in frozen['choke']['cases'] if c['role'] == 'reserved_evaluation'}
    assert len(qmul_dev) == 24 and len(qmul_final) == 32 and not qmul_dev & qmul_final
    assert len(extension_ids) == 3 and not extension_ids & (qmul_dev | qmul_final)
    assert len(choke_dev) == 12 and len(choke_final) == 13 and not choke_dev & choke_final
    source_counts = Counter(r['source'] for r in references)
    native_summary = []
    for name, role in sorted({(r['source'], r['role']) for r in native}):
        rows = [r for r in native if r['source'] == name and r['role'] == role]
        native_summary.append({'source': name, 'role': role, 'cases': len({r['case_id'] for r in rows}),
                               'entries': len(rows), 'entry_kinds': dict(Counter(r['kind'] for r in rows)),
                               'unique_encoded_files_rehashed': len({r['path'] for r in rows if r['path']}),
                               'exact_TRAIN_face_asset_matches': 0, 'ancestry_digest_matches': 0})
    evidence = {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
        'scope': 'Current finite hinge 20-reference/100-paired-TRAIN case exclusion from ALL currently frozen native cohorts and both retained QMUL prepared-input versions; opaque hashes and metadata only',
        'protocol_hashes': PINS, 'current_case_ids': selected_ids, 'current_reference_ids': [r['id'] for r in references],
        'TRAIN_references_per_source': dict(source_counts),
        'TRAIN_cases_per_source': dict(Counter(c['source'] for c in cases)),
        'native_cohorts': native_summary, 'within_source_label_overlap': {'QMUL_DEV_reserved': 0, 'QMUL_extension_original': 0, 'ChokePoint_DEV_reserved': 0},
        'transitive_asset_bindings_verified': dict(Counter(r['root'] for r in transitive)),
        'transitive_assets': transitive, 'native_face_or_frame_hash_entries': native,
        'TRAIN_asset_entries': train_assets, 'TRAIN_ancestral_metadata': ancestral,
        'TRAIN_native_encoded_face_equality': equality, 'TRAIN_ancestral_native_digest_equality': ancestor_matches,
        'source_metadata_native_case_ID_overlap': [],
        'generic_support_digest_matches': len(support_equalities),
        'generic_support_matches_are_not_face_content_or_identity_evidence': True,
        'all58_reserved_records_held_out_from_actual_TRAIN_case_selection': True,
        'QMUL_reserved_opaque_files_rehashed': 32, 'ChokePoint_reserved_metadata_records': 26,
        'ChokePoint_reserved_crop_prepared_pixels_created_or_hashed': 0,
        'new_pixel_decodes': 0, 'model_forwards': 0, 'gradient_queries': 0, 'optimizer_updates': 0,
        'VM_connections': 0, 'browser_calls': 0, 'data_acquisition': 0,
        'no_native_or_final_labels_thresholds_or_selection_changed': True,
        'canonical_docs_or_model_app_changed': False, 'scientific_quality_or_final_review_qualified': False,
        'limitations': [
            'Encoded equality detects identical bytes only, not re-encodings, alternative frames, look-alikes or complete subject identity overlap.',
            'Namespaced source labels prove within-release split separation only; cross-source/historical/restorer/prior/recognizer pretraining person overlap remains unknown.',
            'Ancestral raw/HQ hashes are verified bound metadata, not fresh rehashes of all historical image bytes. No new claim about historical training exposure is made.',
            'No reserved ChokePoint crop exists. Its bound source-frame hashes are metadata comparisons, not new hashes of reserved frame archive members or pixel equality of nonexistent crops.',
            'Hash verification of saved source artifacts is not static proof of every possible code path or adversarial read. Current explicit case selection and required input bindings are audited.',
            'All current native inputs are unpaired. No native PSNR/SSIM, ethnicity, capture-country or Zamboanga performance is established.',
            'This verifies current finite packet inputs, not exclusion from every historical training corpus, independent final review, useful restoration or seven-family completion.'
        ],
        'budgets': BUDGET, 'opaque_bytes_hashed': reader.bytes,
        'unique_regular_files_hashed': len(reader.bindings), 'seconds': time.monotonic() - reader.started,
        'all_hashed_files': reader.bindings}
    write(OUT / 'audit.json', evidence)
    write(OUT / 'manifest.json', {'complete': True, 'script_sha256': reader.digest(Path(__file__)),
                                 'audit_sha256': reader.digest(OUT / 'audit.json'), 'audited_scope_not_quality': True})
    print(json.dumps({'complete': True, 'native_hash_entries': len(native), 'TRAIN_cases': len(cases),
                      'TRAIN_references': len(references), 'transitive_bindings': len(transitive),
                      'face_encoded_matches': len(equality), 'ancestry_matches': len(ancestor_matches),
                      'seconds': time.monotonic() - reader.started, 'model_forwards': 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    assert not OUT.exists(), 'Exclusive evidence already exists; preserve it rather than overwriting'
    OUT.mkdir()
    reader = Reader()
    write(OUT / 'plan.json', {'UTC': datetime.now(timezone.utc).isoformat(), 'protocol_hashes': PINS,
                             'budgets': BUDGET, 'stdlib_hash_and_metadata_only': True,
                             'reserved_encoded_bytes_allowed_never_decode': True,
                             'no_inference_or_training_or_selection_change': True})
    try:
        run(reader)
    except BaseException:
        write(OUT / 'failure.json', {'complete': False, 'traceback': traceback.format_exc(),
                                    'seconds': time.monotonic() - reader.started,
                                    'bytes_hashed': reader.bytes, 'model_forwards': 0})
        raise


if __name__ == '__main__':
    main()
