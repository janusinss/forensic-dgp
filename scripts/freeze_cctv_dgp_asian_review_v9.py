"""Freeze source-only replay suitability and unchanged cohort roles; no models."""
import collections
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'outputs/cctv_dgp_asian_source_review_v9'
PARENT = ROOT / 'outputs/cctv_dgp_vm_bundle_v1'
V6 = ROOT / 'outputs/cctv_dgp_targets_vm_v6'
PINS = {
    'catalog.json': '8f37f3dd995197839ad7443102ce37a27d3455ffdf57f905dd4506b17932872d',
    'initial_review_observations.json': '79081d4a0568b65383aaa68b7e7d1bd1f4218a32dfec3130e3d2ced96881f5f4',
    'source_review_rubric.json': '1e0aa441917280168a87f73509e1846b15ada15ee4d8b0e62ec35bf7cc6bad3e',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_once(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    started = time.monotonic()
    names = ['final_decisions.json', 'eligible_references.json', 'mixed_cohort.json', 'source_review_audit.json']
    assert not any((REVIEW / n).exists() for n in names), 'Preserve the frozen source review'
    for name, pin in PINS.items():
        assert sha(REVIEW / name) == pin, name
    assert sha(PARENT / 'protocol.json') == 'b652914335e33375f8108ba427e4b5be99fd86e811f455b307ab72ae7cf152f6'
    assert sha(V6 / 'targets_protocol_v6.json') == '0fe7d036bf8567bf0860790df553afd627ac50bd5096cfda29cc7d09d71d320c'
    assert sha(V6 / 'lineage/eligible_references.json') == 'cadeedd2906fe169b7cc9741aac9fc29db8aedc1ba21d4438f97443490c1281e'
    catalog = read(REVIEW / 'catalog.json')
    initial = read(REVIEW / 'initial_review_observations.json')
    inspection = read(REVIEW / 'original_resolution_checks_v1.json')
    parent = read(PARENT / 'protocol.json')
    v6 = read(V6 / 'targets_protocol_v6.json')
    assert inspection['catalog_sha256'] == PINS['catalog.json']
    assert inspection['initial_observation_sha256'] == PINS['initial_review_observations.json']
    for review in (initial, inspection):
        assert review['all_29_source_pages_inspected'] is True
        assert review['model_outputs_used_for_decisions'] is False
        assert review['validation_cases_reviewed'] is False
        assert review['native_reserved_used'] is False
        assert all(review[n] == 0 for n in ('local_model_forwards', 'local_backward_calls', 'local_optimizer_updates'))
    refs = {r['id']: r for r in parent['references'] if r['role'] == 'train' and r['source'] == 'dataset/asian_faces'}
    rows = {r['reference_id']: r for r in catalog['rows']}
    assert len(rows) == len(refs) == 451 and set(rows) == set(refs)
    corrections = {'tr_asian_03086': 'tr_asian_03080', 'tr_asian_08688': 'tr_asian_08668'}
    assert {r['invalid_initial_reference_id']: r['actual_reference_id'] for r in inspection['transcription_corrections']} == corrections
    assert set(initial['exceptions']) - set(rows) == set(corrections)
    exceptions = {corrections.get(k, k): v for k, v in initial['exceptions'].items()}
    checks = {r['reference_id']: r for r in inspection['observations']}
    assert len(checks) == len(inspection['observations']) == 49
    uncertain = {rid for rid, value in exceptions.items() if value[0] == 'uncertain_review'}
    assert set(checks) == uncertain | {'tr_asian_01280', 'tr_asian_09853', 'tr_asian_03080'}
    assert len(uncertain) == 46
    page_ids = []
    assert len(catalog['pages']) == 29
    for page in catalog['pages']:
        assert time.monotonic() - started < 120
        path = REVIEW / page['path']
        assert sha(path) == page['sha256']
        with Image.open(path) as image:
            assert image.size == (1040, 1192)
            pixels = np.asarray(image.convert('RGB'))
        for i, rid in enumerate(page['reference_ids']):
            assert rows[rid]['review_page'] == page['path']
            x, y = (i % 4) * 260 + 2, (i // 4) * 292 + 58
            with Image.open(ROOT / rows[rid]['target']) as target:
                np.testing.assert_array_equal(pixels[y:y + 256, x:x + 256], np.asarray(target.convert('RGB')))
            page_ids.append(rid)
    assert page_ids == sorted(refs) and len(set(page_ids)) == 451
    entries = []
    decisions = {'accept_clean_restoration', 'exclude_covering', 'exclude_reference_quality'}
    for rid in page_ids:
        row, ref = rows[rid], refs[rid]
        assert row['original_role'] == ref['role'] == 'train'
        assert row['source_sha256'] == ref['source_sha256']
        assert sha(ROOT / row['native']) == ref['source_sha256']
        for key in ('native', 'target', 'observed'):
            assert sha(PARENT / ref[key]) == parent['assets_sha256'][ref[key]], (rid, key)
        with Image.open(ROOT / row['native']) as native:
            assert list(native.size) == ref['native_size'] == row['native_size']
            assert min(native.size) < 256
        with Image.open(ROOT / row['target']) as target:
            assert target.size == (256, 256) and target.mode == 'RGB'
            assert hashlib.sha256(np.asarray(target).tobytes()).hexdigest() == ref['target_rgb_sha256']
        first = exceptions.get(rid, [initial['default_decision'], initial['default_reason']])
        decision, reason = first
        if rid in checks:
            check = checks[rid]
            assert check['inspected_native'] is True and check['sha256'] == row['source_sha256']
            assert check['native_path_from_workspace'] == (ROOT / row['native']).relative_to(ROOT).as_posix()
            assert check['size'] == row['native_size']
            decision, reason = check['decision'], check['reason']
        assert decision in decisions and reason.strip()
        entries.append({**row, 'source': 'dataset/asian_faces', 'decision': decision, 'reason': reason,
                        'initial_decision': first[0], 'initial_reason': first[1],
                        'additional_original_inspection': rid in checks})
    accepted = [r for r in entries if r['decision'] == 'accept_clean_restoration']
    counts = dict(collections.Counter(r['decision'] for r in entries))
    hq_train = [r for r in v6['references'] if r['role'] == 'train']
    validation = [r for r in v6['references'] if r['role'] == 'validation']
    train = hq_train + [refs[r['reference_id']] for r in accepted]
    assert len(hq_train) == 391 and len(validation) == 104
    assert not ({r['id'] for r in train} & {r['id'] for r in validation})
    source_overlap = sorted({r['source_sha256'] for r in train} & {r['source_sha256'] for r in validation})
    assert not source_overlap
    counts_by_source = dict(collections.Counter(r['source'] for r in train))
    base = {'version': 9, 'date': '2026-10-04', 'reviewer': 'Assistant source-only development review',
            'source_review_complete': True, 'original_roles_preserved': True,
            'model_outputs_used_for_decisions': False, 'native_reserved_used': False,
            'local_model_forwards': 0, 'local_backward_calls': 0, 'local_optimizer_updates': 0,
            'training_ready': False, 'model_improvement_established': False,
            'interpretation': 'All Asian originals have a minimum edge below 256; this is low-resolution replay, not newly captured HQ truth. Source review is not independent final human review; identity overlap with historical data is unestablished.',
            'review_input_sha256': {n: sha(REVIEW / n) for n in [*PINS, 'original_resolution_checks_v1.json']}}
    write_once(REVIEW / names[0], {**base, 'decisions': counts, 'entries': entries,
                                'transcription_corrections': inspection['transcription_corrections'],
                                'additional_clear_exclusions': inspection['additional_clear_exclusions']})
    write_once(REVIEW / names[1], {**base, 'eligible_sources': len(accepted),
                                'eligible_by_role': {'train': len(accepted)}, 'references': accepted})
    mixed = {**base, 'training_sources': counts_by_source, 'training_references': len(train),
             'training_reference_ids': [r['id'] for r in train],
             'validation_references': 104, 'validation_cases': 520,
             'validation_reference_ids': [r['id'] for r in validation],
             'validation_sources': dict(collections.Counter(r['source'] for r in validation)),
             'unchanged_validation_cases_sha256': canonical_sha(v6['validation_cases']),
             'unchanged_validation_references_sha256': canonical_sha(validation),
             'parent_protocol_sha256': sha(PARENT / 'protocol.json'),
             'v6_protocol_sha256': sha(V6 / 'targets_protocol_v6.json'),
             'hq_source_ledger_sha256': sha(V6 / 'lineage/eligible_references.json'),
             'source_hash_overlap_train_validation': source_overlap,
             'execution_prepared': False, 'training_started': False}
    write_once(REVIEW / names[2], mixed)
    receipt = {**base, 'complete': True, 'reviewed_sources': 451, 'reviewed_pages': 29,
               'original_resolution_inspections': 49, 'exact_page_cells_checked': 451,
               'original_assets_checked': 1353, 'source_and_target_hash_pairs_checked': 451,
               'decisions': counts, 'training_sources': counts_by_source,
               'unchanged_validation_references': 104, 'unchanged_validation_cases': 520,
               'review_output_sha256': {n: sha(REVIEW / n) for n in names[:3]},
               'execution_source_sha256': sha(Path(__file__)),
               'seconds': time.monotonic() - started, 'runtime_cap_seconds': 120}
    write_once(REVIEW / names[3], receipt)
    print({k: v for k, v in receipt.items() if k not in ('interpretation', 'review_input_sha256', 'review_output_sha256')}, flush=True)


if __name__ == '__main__':
    main()
