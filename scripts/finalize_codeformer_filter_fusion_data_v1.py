"""Data-only explicit pilot admission, gated by root's complete readable review.

Preserves every historical/provisional file. No model, optimizer, gradient, VM,
automatic launch, checkpoint loading or quality-based sampling is performed.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_filter_fusion_training_v1 import data as d

PREP = 'outputs/codeformer_filter_fusion_training_v1_preparation'
OUT = ROOT / PREP
PINS = {
    'data_bundle_provisional_v1.json': '15556caeff532a826b51a50a6c024b726ab4370e914ddf38372d03abbafd5acd',
    'data_schedule_provisional_v1.json': '18cfb3918cf57627b1907f8dad57c905c7b44def0b70a334e998a5fbec9f604d',
    'data_dark_linear_v1/manifest.json': '2dbbc701768ddc1ffacf8420520977c372cdc73c959ea63195eb799bc4d7ab68',
    'data_dark_linear_contract_v1.json': '4a707b30615ebcfe2a6dcfd1472384ab8b25f463af9cf300840f87dd92c3fed5',
}


def pin(name, expected=None):
    path = ROOT / d.relative(name)
    p = {'path': name, 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    d.require(expected is None or p['sha256'] == expected, 'Frozen metadata bytes differ')
    d.verified_bytes(ROOT, p)
    return p


def read(name, expected=None):
    p = pin(name, expected)
    return json.loads(d.verified_bytes(ROOT, p)), p


def write(name, document):
    payload = json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False).encode('utf-8') + b'\n'
    with (OUT / name).open('xb') as handle:
        handle.write(payload)
    return pin(PREP + '/' + name)


def validate_review(review, manifest, manifest_pin):
    d.require(review.get('complete') is True and review.get('reviewer') == 'root'
              and review.get('observed_panels') == 57,
              'Root must personally complete all57 input-only panels')
    d.require(review.get('manifest', {}).get('sha256') == manifest_pin['sha256'],
              'Root review must bind exact newlinear manifest')
    criteria = manifest['contract']
    contract = json.loads(d.verified_bytes(ROOT, criteria))
    d.require(review.get('criteria') == contract['criteria'], 'Root sufficiency criteria changed')
    rows = review['rows']
    d.require(len(rows) == len({r['id'] for r in rows}) == 57 and
              Counter(r['decision'] for r in rows) == {'readable': 57},
              'Any dark gap must remain held; all57readable required for this exact pilot')
    expected = {r['id']: r for r in manifest['rows']}
    d.require(set(expected) == {r['id'] for r in rows}, 'Dark review case namespace differs')
    for row in rows:
        source = expected[row['id']]
        d.require(row['reference_id'] == source['reference_id'] and row['input'] == source['input']
                  and row['RGB8_sha256'] == source['input_pixels']['RGB8_sha256'],
                  'Dark review source/input pixel join differs')


def prepare(dark_review_name, dark_review_sha256, root_admission_approved):
    d.require(root_admission_approved is True, 'Explicit root data admission instruction required')
    docs, pins = {}, {}
    for name, expected in PINS.items():
        docs[name], pins[name] = read(PREP + '/' + name, expected)
    review, reviewpin = read(dark_review_name, dark_review_sha256)
    linear = docs['data_dark_linear_v1/manifest.json']
    validate_review(review, linear, pins['data_dark_linear_v1/manifest.json'])
    original = docs['data_bundle_provisional_v1.json']
    holds = json.loads(d.verified_bytes(ROOT, original['candidate_holds']))
    candidates = d.validate_holds(holds)
    d.require(set(original['candidate_clear_filter_cases']) == {r['id'] for r in candidates},
              'Current313filter/57clear candidate set differs')
    old_schedule = docs['data_schedule_provisional_v1.json']
    expected_old = d.build_schedule(holds, list(original['dark_cases'].values()))
    d.require(old_schedule == expected_old, 'Provisional seeded schedule changed')
    schedule = copy.deepcopy(old_schedule)
    newdark = {r['reference_id']: r for r in linear['rows']}
    for update in schedule['updates']:
        dark = update['slots'][3]
        new = newdark[dark['reference_id']]
        dark['id'], dark['profile'] = new['id'], new['profile']
    schedule.update(dark_anchor_profile=d.DARK_LINEAR_PROFILE,
                    historical_old_dark157slots_replaced_by_new_samephoto_readable_linear_proxies=True,
                    old_schedule_canonical_sha256=d.digest(old_schedule))
    d.require([u['slots'][:3] for u in schedule['updates']] ==
              [u['slots'][:3] for u in old_schedule['updates']],
              'Filter/clear order or padding changed during dark substitution')
    # Verify every retained current candidate byte and exact newly reviewed RGB.
    for row in original['candidate_clear_filter_cases'].values():
        d.rgb8(ROOT, row['input'], row['input_pixels']['RGB8_sha256'])
        d.rgb8(ROOT, row['target256'], row['target256_pixels']['RGB8_sha256'])
        d.verified_bytes(ROOT, row['corruption_footprint'])
        d.verified_bytes(ROOT, row['pair_record'])
    for row in linear['rows']:
        d.rgb8(ROOT, row['input'], row['input_pixels']['RGB8_sha256'])
        d.rgb8(ROOT, row['source_clear_input'], row['source_clear_pixels']['RGB8_sha256'])
    role_cases = sorted({(s['role'], s['id']) for u in schedule['updates'] for s in u['slots']})
    admission = {'format': 'codeformer-filter-fusion-explicit-admission-v1', 'complete': True,
        'root_approved': True, 'root_approval_scope': 'finite fusion-focused157update manual pilot data only',
        'candidate_holds_sha256': original['candidate_holds']['sha256'],
        'dark_input_review_sha256': reviewpin['sha256'], 'dark_input_review': reviewpin,
        'linear_dark_manifest': pins['data_dark_linear_v1/manifest.json'],
        'schedule_sha256': d.digest(schedule),
        'admitted_role_cases': [{'role': role, 'id': case} for role, case in role_cases],
        'admitted_filter_unique': 313, 'admitted_clear_unique': 57, 'admitted_dark_unique': 57,
        'admitted_readable_repeated_row_variants_explicitly': 57,
        'strong_row_initial_tags_retained_without_changing_original_eligibility': True,
        'all_original512_cases_and_labels_retained': True, 'held512_ledger_unchanged': True,
        'historical_DEV_FINAL_native_excluded': True, 'automatic_launch_authorized': False,
        'data_admission_is_not_model_qualification_or_training_success': True}
    d.require_admission(admission, original['candidate_holds']['sha256'], reviewpin['sha256'], schedule)
    sampler = d.FiniteSampler(schedule)
    state_ledger = []
    for _ in range(157):
        preview = sampler.preview()
        state_ledger.append(preview)
        sampler.commit(preview)
    # Freeze24 TRAIN probes from declared order, before any new model outputs.
    clear_by_ref = {r['reference_id']: r['id'] for r in original['candidate_clear_filter_cases'].values()
                    if r['profile'] == 'clear'}
    anchor_refs = schedule['anchor_source_order'][:8]
    filter_order = []
    for update in schedule['updates']:
        for slot in update['slots'][:2]:
            if slot['id'] not in filter_order:
                filter_order.append(slot['id'])
    guards = ([{'role': 'clear', 'id': clear_by_ref[rid], 'reference_id': rid} for rid in anchor_refs] +
              [{'role': 'dark', 'id': newdark[rid]['id'], 'reference_id': rid} for rid in anchor_refs] +
              [{'role': 'filter', 'id': identifier,
                'reference_id': original['candidate_clear_filter_cases'][identifier]['reference_id']}
               for identifier in filter_order[:8]])
    d.require(len(guards) == 24 and Counter(r['role'] for r in guards) == {'clear': 8, 'dark': 8, 'filter': 8},
              'Fixed24TRAIN guard census differs')
    names = ('data_schedule_v1.json', 'data_admission_v1.json', 'data_sampler_ledger_v1.json',
             'data_guard_probes_v1.json', 'data_bundle_v1.json', 'data_finalization_v1.json')
    d.require(all(not (OUT / name).exists() for name in names), 'Exclusive data-finalization namespace required')
    schedulepin = write(names[0], schedule)
    admissionpin = write(names[1], admission)
    ledgerpin = write(names[2], {'format': 'codeformer-filter-fusion-admitted-prospective-ledger-v1',
        'schedule': schedulepin, 'schedule_canonical_sha256': d.digest(schedule),
        'states': state_ledger, 'final_prospective_state': sampler.state(),
        'actual_updates': 0, 'simulated_metadata_commits_only': True})
    guardpin = write(names[3], {'format': 'codeformer-filter-fusion-training-guards-v1',
        'guards': guards, 'schedule': schedulepin, 'snapshots': [0, 20],
        'selection': 'first8 fixed hash anchors and first8 distinct balanced filter IDs',
        'all_TRAIN_not_independent_DEV': True, 'new_model_outputs_used_for_selection': False})
    cases = copy.deepcopy(original['candidate_clear_filter_cases'])
    for row in linear['rows']:
        cases[row['id']] = copy.deepcopy(row)
    d.require(len(cases) == 427, 'Exact427 unique admitted role inputs required')
    bundle = {'format': 'codeformer-filter-fusion-data-bundle-v1', 'complete': True,
        'case_by_id': cases, 'candidate_holds': original['candidate_holds'],
        'dark_input_review': reviewpin, 'schedule': schedulepin,
        'schedule_canonical_sha256': d.digest(schedule), 'admission': admissionpin,
        'sampler_ledger': ledgerpin, 'guard_probes': guardpin,
        'original_provisional_bundle': pins['data_bundle_provisional_v1.json'],
        'old57lowlight_stress_not_admitted': original['dark_coverage'],
        'all512_original_hold_ledger_and_391_53_DEV406_filter72_refused24_FINAL_retained': True,
        'historical_case_optimization_flags_unchanged_final_admission_is_separate': True,
        'actual_training_performed': False, 'neural_calls': 0, 'automatic_launch_authorized': False}
    bundlepin = write(names[4], bundle)
    close = {'format': 'codeformer-filter-fusion-data-finalization-v1', 'complete': True,
        'source': pin('scripts/finalize_codeformer_filter_fusion_data_v1.py'),
        'inputs': {**pins, 'root_dark_input_review': reviewpin},
        'schedule': schedulepin, 'admission': admissionpin, 'ledger': ledgerpin,
        'guard_probes': guardpin, 'bundle': bundlepin, 'actual_updates': 0,
        'new_model_created': False, 'no_launch_or_quality_qualification': True}
    closurepin = write(names[5], close)
    print(json.dumps({'complete_data_only': True, 'bundle': bundlepin, 'admission': admissionpin,
                      'schedule_canonical_sha256': d.digest(schedule), 'closure': closurepin,
                      'actual_updates': 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dark-review', required=True)
    parser.add_argument('--dark-review-sha256', required=True)
    parser.add_argument('--root-admission-approved', action='store_true')
    args = parser.parse_args()
    prepare(args.dark_review, args.dark_review_sha256, args.root_admission_approved)


if __name__ == '__main__':
    main()
