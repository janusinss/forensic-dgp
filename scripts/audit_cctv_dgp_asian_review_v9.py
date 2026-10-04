"""Independently check frozen source-ledger files/roles; no model operations."""
import collections
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/cctv_dgp_asian_source_review_v9'
PINS = {
    'final_decisions.json': '40daf9e2b1656c073df1a134d765030466066c4b2fcc854ae3fcf7244fe1474f',
    'eligible_references.json': '3d9cf3ffd06adad3c8fb639b2fcd7719e777618b4f58b01c235c3305c5ad3917',
    'mixed_cohort.json': '01b89afc21231c4159063fc9149a7892b8ed7529e471e2c11df79cf3d7219ac9',
    'source_review_audit.json': 'e35269ddf04ab9c5d0d9e471107fe0f98c1e5caf383e2305c7deec59d1fb1816',
    'original_resolution_checks_v1.json': 'acab65f83f7aedaa54c5e99ffc454b8ff696af65e6d380028b131d2732a4f982',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main():
    started = time.monotonic()
    output = BASE / 'local_integrity_audit.json'
    assert not output.exists(), 'Preserve the existing independent receipt'
    for name, pin in PINS.items():
        assert sha(BASE / name) == pin, name
    final = read(BASE / 'final_decisions.json')
    eligible = read(BASE / 'eligible_references.json')
    mixed = read(BASE / 'mixed_cohort.json')
    audit = read(BASE / 'source_review_audit.json')
    catalog = read(BASE / 'catalog.json')
    checks = read(BASE / 'original_resolution_checks_v1.json')
    for item in (final, eligible, mixed, audit):
        assert item['source_review_complete'] is True and item['original_roles_preserved'] is True
        assert item['model_outputs_used_for_decisions'] is False and item['native_reserved_used'] is False
        assert item['training_ready'] is False and item['model_improvement_established'] is False
        assert all(item[k] == 0 for k in ('local_model_forwards', 'local_backward_calls', 'local_optimizer_updates'))
        for name, pin in item['review_input_sha256'].items():
            assert sha(BASE / name) == pin
    parent_path = ROOT / 'outputs/cctv_dgp_vm_bundle_v1/protocol.json'
    v6_path = ROOT / 'outputs/cctv_dgp_targets_vm_v6/targets_protocol_v6.json'
    assert sha(parent_path) == mixed['parent_protocol_sha256']
    assert sha(v6_path) == mixed['v6_protocol_sha256']
    parent, v6 = read(parent_path), read(v6_path)
    parent_refs = {r['id']: r for r in parent['references']}
    originals = {r['reference_id']: r for r in catalog['rows']}
    entries = {r['reference_id']: r for r in final['entries']}
    assert len(entries) == len(final['entries']) == len(originals) == 451
    assert set(entries) == set(originals)
    counts = dict(collections.Counter(r['decision'] for r in entries.values()))
    assert counts == final['decisions'] == audit['decisions'] == {
        'accept_clean_restoration': 390, 'exclude_covering': 28, 'exclude_reference_quality': 33}
    accepted = [r for r in final['entries'] if r['decision'] == 'accept_clean_restoration']
    assert accepted == eligible['references'] and eligible['eligible_sources'] == 390
    for rid, entry in entries.items():
        assert time.monotonic() - started < 120
        original, ref = originals[rid], parent_refs[rid]
        assert entry['original_role'] == ref['role'] == 'train' and entry['reason'].strip()
        assert all(entry[k] == original[k] for k in original)
        assert ref['source'] == entry['source'] == 'dataset/asian_faces'
        assert sha(ROOT / entry['native']) == entry['source_sha256'] == ref['source_sha256']
        assert sha(ROOT / entry['target']) == entry['target_sha256']
        assert min(entry['native_size']) < 256
        with Image.open(ROOT / entry['target']) as image:
            assert image.size == (256, 256) and image.mode == 'RGB'
            assert hashlib.sha256(np.asarray(image).tobytes()).hexdigest() == ref['target_rgb_sha256']
    inspected = {r['reference_id']: r for r in checks['observations']}
    assert len(inspected) == sum(r['additional_original_inspection'] for r in entries.values()) == 49
    for rid, check in inspected.items():
        assert check['decision'] == entries[rid]['decision'] and check['reason'] == entries[rid]['reason']
        assert check['sha256'] == entries[rid]['source_sha256'] and check['inspected_native'] is True
    for page in catalog['pages']:
        assert sha(BASE / page['path']) == page['sha256']
    hq = [r for r in v6['references'] if r['role'] == 'train']
    validation = [r for r in v6['references'] if r['role'] == 'validation']
    training = hq + [parent_refs[r['reference_id']] for r in accepted]
    assert len(training) == len(set(mixed['training_reference_ids'])) == 781
    assert mixed['training_reference_ids'] == [r['id'] for r in training]
    assert mixed['validation_reference_ids'] == [r['id'] for r in validation]
    assert not set(mixed['training_reference_ids']) & set(mixed['validation_reference_ids'])
    assert mixed['unchanged_validation_cases_sha256'] == fingerprint(v6['validation_cases'])
    assert mixed['unchanged_validation_references_sha256'] == fingerprint(validation)
    assert len(validation) == 104 and len(v6['validation_cases']) == 520
    assert not {r['source_sha256'] for r in training} & {r['source_sha256'] for r in validation}
    assert dict(collections.Counter(r['source'] for r in training)) == mixed['training_sources']
    assert all(r['decision'] == 'exclude_covering' for r in (entries['tr_asian_01280'], entries['tr_asian_09853'], entries['tr_asian_03080']))
    receipt = {'complete': True, 'version': 9, 'source_only': True,
        'reviewed_sources_checked': 451, 'reviewed_pages_pinned': 29, 'original_inspections_checked': 49,
        'decisions': counts, 'training_sources': mixed['training_sources'],
        'unchanged_validation_references': 104, 'unchanged_validation_cases': 520,
        'training_ready': False, 'native_reserved_used': False,
        'local_model_forwards': 0, 'local_backward_calls': 0, 'local_optimizer_updates': 0,
        'files_sha256': PINS, 'execution_source_sha256': sha(Path(__file__)),
        'seconds': time.monotonic() - started, 'runtime_cap_seconds': 120,
        'limitation': 'Independent file/cohort arithmetic checks, not an independent human source review or evidence of model improvement. Historical identity overlap remains unestablished.'}
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(receipt, flush=True)


if __name__ == '__main__':
    main()
