"""Admit revised151-update data after root's55-readable/2-source-hold amendment.

Preserves every old label, source, generated pixel and157-update proposal. The
new source hold removes no research evidence and substitutes no photograph.
No neural, optimizer, gradient, VM, checkpoint load or automatic launch occurs.
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
from research.codeformer_filter_fusion_training_v1 import data as old_data
from research.codeformer_filter_fusion_training_v1 import data_v2 as d

PREP = 'outputs/codeformer_filter_fusion_training_v1_preparation'
OUT = ROOT / PREP
PINS = {
    'data_bundle_provisional_v1.json': '15556caeff532a826b51a50a6c024b726ab4370e914ddf38372d03abbafd5acd',
    'data_schedule_provisional_v1.json': '18cfb3918cf57627b1907f8dad57c905c7b44def0b70a334e998a5fbec9f604d',
    'data_dark_linear_v1/manifest.json': '2dbbc701768ddc1ffacf8420520977c372cdc73c959ea63195eb799bc4d7ab68',
    'data_dark_linear_contract_v1.json': '4a707b30615ebcfe2a6dcfd1472384ab8b25f463af9cf300840f87dd92c3fed5',
    'data_independent_scope_pass_v2.json': 'c647e7c912afc9b2665eb8a1c333066b4e3151b07f1142a68629119ad3c9f0f5',
}


def pin(name, expected=None):
    path = ROOT / old_data.relative(name)
    p = {'path': name, 'bytes': path.stat().st_size,
         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    d.require(expected is None or p['sha256'] == expected, 'Frozen metadata differs')
    old_data.verified_bytes(ROOT, p)
    return p


def read(name, expected=None):
    p = pin(name, expected)
    return json.loads(old_data.verified_bytes(ROOT, p)), p


def write(name, document):
    payload = json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False).encode('utf-8') + b'\n'
    with (OUT / name).open('xb') as handle:
        handle.write(payload)
    return pin(PREP + '/' + name)


def validate_review(review, manifest, manifestpin, contract):
    d.require(review.get('complete') is True and review.get('reviewer') == 'root'
              and review.get('observed_panels') == 57, 'Root must complete all57 exact input panels')
    d.require(review.get('manifest', {}).get('sha256') == manifestpin['sha256'] and
              review.get('criteria') == contract['criteria'], 'Manifest/criteria binding differs')
    rows = review['rows']
    d.require(len(rows) == len({r['id'] for r in rows}) == 57 and
              Counter(r['decision'] for r in rows) == {'readable': 55, 'out_of_scope': 2},
              'Exact55readable/2scopehold amendment required; retain every extra gap')
    expected = {r['id']: r for r in manifest['rows']}
    d.require(set(expected) == {r['id'] for r in rows}, 'Root review case namespace differs')
    for row in rows:
        source = expected[row['id']]
        d.require(row['reference_id'] == source['reference_id'] and row['input'] == source['input']
                  and row['RGB8_sha256'] == source['input_pixels']['RGB8_sha256'], 'Review pixel/source join differs')
        d.require(row['decision'] == ('out_of_scope' if row['reference_id'] in d.HELD_SOURCES else 'readable'),
                  'Only declared whole-source scope hold may be excluded')


def prepare(review_name, review_sha256, root_approved):
    d.require(root_approved is True, 'Explicit root prospective data admission required')
    docs, pins = {}, {}
    for name, expected in PINS.items():
        docs[name], pins[name] = read(PREP + '/' + name, expected)
    review, reviewpin = read(review_name, review_sha256)
    linear = docs['data_dark_linear_v1/manifest.json']
    validate_review(review, linear, pins['data_dark_linear_v1/manifest.json'],
                    docs['data_dark_linear_contract_v1.json'])
    original = docs['data_bundle_provisional_v1.json']
    holds = json.loads(old_data.verified_bytes(ROOT, original['candidate_holds']))
    all_candidates = old_data.validate_holds(holds)
    d.require(set(original['candidate_clear_filter_cases']) == {r['id'] for r in all_candidates},
              'Historical370 candidates changed')
    old_schedule = docs['data_schedule_provisional_v1.json']
    d.require(old_schedule == old_data.build_schedule(holds, list(original['dark_cases'].values())),
              'Original157 seeded schedule differs')
    kept = {identifier: copy.deepcopy(row) for identifier, row in
            original['candidate_clear_filter_cases'].items() if row['reference_id'] not in d.HELD_SOURCES}
    d.require(len(kept) == 356 and Counter(r['profile'] for r in kept.values()) ==
              {'clear': 55, **d.PROFILE_COUNTS}, 'Revised356 candidate census differs')
    dark = [copy.deepcopy(r) for r in linear['rows'] if r['reference_id'] not in d.HELD_SOURCES]
    d.require(len(dark) == 55, 'Exact55 samephoto dark inputs required')
    schedule = d.revised_schedule(old_schedule, linear['rows'])
    sampler = d.FiniteSampler(schedule)
    ledger = []
    for _ in range(151):
        preview = sampler.preview()
        ledger.append(preview)
        sampler.commit(preview)  # Prospective metadata simulation; actual optimizer updates=0.
    for row in kept.values():
        old_data.rgb8(ROOT, row['input'], row['input_pixels']['RGB8_sha256'])
        old_data.rgb8(ROOT, row['target256'], row['target256_pixels']['RGB8_sha256'])
        old_data.verified_bytes(ROOT, row['corruption_footprint'])
        old_data.verified_bytes(ROOT, row['pair_record'])
    for row in dark:
        old_data.rgb8(ROOT, row['input'], row['input_pixels']['RGB8_sha256'])
        old_data.rgb8(ROOT, row['source_clear_input'], row['source_clear_pixels']['RGB8_sha256'])
    held_rows = [r for r in holds['all512_cases'] if r['reference_id'] in d.HELD_SOURCES]
    removed = [r for r in all_candidates if r['reference_id'] in d.HELD_SOURCES]
    d.require(len(held_rows) == 16 and len(removed) == 14 and
              sum(r['profile'] != 'clear' for r in removed) == 12, 'Whole-source hold census differs')
    source_hold = {'format': 'codeformer-filter-fusion-prospective-whole-source-hold-v2', 'complete': True,
        'held_sources': list(d.HELD_SOURCES), 'root_input_review': reviewpin,
        'independent_input_scope_pass': pins['data_independent_scope_pass_v2.json'],
        'existing_criterion': 'Multiple faces are out_of_scope; one cropped frontal/mildly turned face required',
        'observation': 'Recognizable partial second faces at left in21374 and45780 exact clear and brightness-only inputs',
        'original_candidate_holds': original['candidate_holds'], 'all16_original_rows': held_rows,
        'former14_candidates_held': removed, 'all16_variants_and_generated2_dark_anchors_retained': True,
        'former_readable_judgments_preserved_as_historical_evidence': True,
        'no_original_label_or_pixel_or_optimization_flag_rewritten': True,
        'no_replacement_photograph_or_new_crop': True,
        'new_prospective_candidate_count': 356, 'new_filter_unique': 301,
        'new_clear_unique': 55, 'new_dark_unique': 55,
        'new_max_updates': 151, 'new_microbatches': 604, 'padding_duplicate_count': 1,
        'padding_duplicate_id': old_data.PADDING_ID}
    role_cases = sorted({(s['role'], s['id']) for u in schedule['updates'] for s in u['slots']})
    admission = {'format': 'codeformer-filter-fusion-explicit-admission-v2', 'complete': True,
        'root_approved': True, 'held_sources': list(d.HELD_SOURCES),
        'candidate_holds_sha256': original['candidate_holds']['sha256'],
        'dark_input_review_sha256': reviewpin['sha256'], 'dark_input_review': reviewpin,
        'linear_dark_manifest': pins['data_dark_linear_v1/manifest.json'],
        'schedule_sha256': d.digest(schedule),
        'admitted_role_cases': [{'role': role, 'id': identifier} for role, identifier in role_cases],
        'admitted_filter_unique': 301, 'admitted_clear_unique': 55, 'admitted_dark_unique': 55,
        'admitted_readable_repeated_row_variants_explicitly': 55,
        'held512_ledger_unchanged': True, 'historical_DEV_FINAL_native_excluded': True,
        'all_original_source_strata_and_hold_evidence_retained': True,
        'automatic_launch_authorized': False, 'quality_or_model_qualification': False}
    d.require_admission(admission, original['candidate_holds']['sha256'], reviewpin['sha256'], schedule)
    anchor_refs = schedule['anchor_source_order'][:8]
    dark_by_ref = {r['reference_id']: r for r in dark}
    ordered_filters = [s for u in schedule['updates'] for s in u['slots'][:2]]
    guards = ([{'role': 'clear', 'id': rid + '__clear', 'reference_id': rid} for rid in anchor_refs] +
              [{'role': 'dark', 'id': dark_by_ref[rid]['id'], 'reference_id': rid} for rid in anchor_refs] +
              [{'role': 'filter', 'id': r['id'], 'reference_id': r['reference_id']} for r in ordered_filters[:8]])
    d.require(len({r['id'] for r in guards}) == 24 and
              Counter(r['role'] for r in guards) == {'clear': 8, 'dark': 8, 'filter': 8}, '24TRAIN guards differ')
    cases = {**kept, **{r['id']: r for r in dark}}
    d.require(len(cases) == 411, 'Exact411 unique admitted inputs required')
    names = ('data_source_hold_v2.json', 'data_schedule_v2.json', 'data_admission_v2.json',
             'data_sampler_ledger_v2.json', 'data_guard_probes_v2.json', 'data_bundle_v2.json',
             'data_finalization_v2.json')
    d.require(all(not (OUT / name).exists() for name in names), 'Exclusive fresh v2 metadata namespace required')
    holdpin = write(names[0], source_hold)
    schedulepin = write(names[1], schedule)
    admission['source_hold'] = holdpin
    admissionpin = write(names[2], admission)
    ledgerpin = write(names[3], {'format': 'codeformer-filter-fusion-admitted-prospective-ledger-v2',
        'schedule': schedulepin, 'schedule_canonical_sha256': d.digest(schedule),
        'states': ledger, 'final_prospective_state': sampler.state(),
        'actual_updates': 0, 'simulated_metadata_commits_only': True})
    guardpin = write(names[4], {'format': 'codeformer-filter-fusion-training-guards-v2',
        'guards': guards, 'schedule': schedulepin, 'snapshots': [0, 20],
        'selection': 'first8 unchanged hash anchors after source hold; first8 distinct retained filter sequence',
        'all_TRAIN_not_independent_DEV': True, 'new_model_outputs_used_for_selection': False})
    bundle = {'format': 'codeformer-filter-fusion-data-bundle-v2', 'complete': True,
        'case_by_id': cases, 'candidate_holds': original['candidate_holds'], 'source_hold': holdpin,
        'dark_input_review': reviewpin, 'schedule': schedulepin,
        'schedule_canonical_sha256': d.digest(schedule), 'admission': admissionpin,
        'sampler_ledger': ledgerpin, 'guard_probes': guardpin,
        'original_provisional_bundle': pins['data_bundle_provisional_v1.json'],
        'original157_schedule': pins['data_schedule_provisional_v1.json'],
        'old57lowlight_stress_not_admitted': original['dark_coverage'],
        'all512_historical_labels_391_53_DEV406_filter72_refused24_FINAL_retained': True,
        'case_optimization_flags_unchanged_separate_final_prospective_admission': True,
        'actual_training_performed': False, 'neural_calls': 0, 'automatic_launch_authorized': False}
    bundlepin = write(names[5], bundle)
    closure = {'format': 'codeformer-filter-fusion-data-finalization-v2', 'complete': True,
        'source': pin('scripts/finalize_codeformer_filter_fusion_data_v2.py'),
        'sampler_source': pin('research/codeformer_filter_fusion_training_v1/data_v2.py'),
        'inputs': {**pins, 'root_dark_input_review': reviewpin}, 'source_hold': holdpin,
        'schedule': schedulepin, 'admission': admissionpin, 'ledger': ledgerpin,
        'guard_probes': guardpin, 'bundle': bundlepin, 'actual_updates': 0,
        'new_model_created': False, 'no_launch_or_quality_qualification': True}
    closurepin = write(names[6], closure)
    print(json.dumps({'complete_data_only': True, 'bundle': bundlepin, 'admission': admissionpin,
                      'schedule_canonical_sha256': d.digest(schedule), 'closure': closurepin,
                      'max_updates': 151, 'microbatches': 604, 'actual_updates': 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dark-review', required=True)
    parser.add_argument('--dark-review-sha256', required=True)
    parser.add_argument('--root-admission-approved', action='store_true')
    args = parser.parse_args()
    prepare(args.dark_review, args.dark_review_sha256, args.root_admission_approved)


if __name__ == '__main__':
    main()
