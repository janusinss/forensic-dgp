"""Record sequential root input-only observations; never optimize or alter data."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'outputs/codeformer_filter_train_v2_data'
PREP = ROOT / 'outputs/codeformer_filter_train_v2_preparation'
NOTES = PREP / 'root_input_only_pages_v1'
LABELS = {'R': 'readable', 'B': 'borderline', 'I': 'insufficient', 'O': 'out_of_scope'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def pin(path):
    data = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(data), 'sha256': sha(data)}


def write(path, value):
    with path.open('xb') as stream:
        stream.write((json.dumps(value, indent=2, allow_nan=False) + '\n').encode())
    return pin(path)


def record(page, decisions, note):
    assert type(page) is int and 1 <= page <= 32
    assert len(decisions) == 16 and set(decisions) <= set(LABELS) and len(note.strip()) >= 20
    manifest = load(DATA / 'manifest.json')
    visual = load(DATA / 'input_only_visual_manifest.json')
    assert manifest['complete'] and manifest['pairs'] == 512
    assert len(visual['sheets']) == 32 and len(visual['cells']) == 512
    NOTES.mkdir(exist_ok=True)
    prior = sorted(NOTES.glob('page_*.json'))
    assert len(prior) == page - 1 and [p.name for p in prior] == ['page_%02d.json' % n for n in range(1, page)]
    for old in prior:
        row = load(old)
        assert row['data_manifest'] == pin(DATA / 'manifest.json')
        assert row['criteria'] == pin(PREP / 'input_only_review_criteria_v1.json')
    sheet = visual['sheets'][page - 1]
    assert sheet['review_index'] == page
    assert pin(DATA / sheet['path'])['sha256'] == sheet['sha256']
    cells = [c for c in visual['cells'] if c['sheet'] == sheet['path']]
    assert len(cells) == 16 and all(c['profile'] == sheet['profile'] for c in cells)
    if page < 29:
        assert sheet['profile'] != 'clear'
    else:
        assert sheet['profile'] == 'clear'
    rows = [{'id': c['reference_id'] + '__' + c['profile'], 'reference_id': c['reference_id'],
             'profile': c['profile'], 'decision': LABELS[d], 'sheet_cell_RGB8_sha256': c['RGB8_sha256']}
            for c, d in zip(cells, decisions)]
    return write(NOTES / ('page_%02d.json' % page), {
        'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(), 'reviewer': 'root',
        'page': page, 'sheet': sheet, 'data_manifest': pin(DATA / 'manifest.json'),
        'criteria': pin(PREP / 'input_only_review_criteria_v1.json'), 'decisions': rows,
        'observations': note, 'viewed_saved256_inputs_without_targets_or_model_panels': True,
        'not_blinded_to_prior_historical_sources_or_other_corruptions': True,
        'optimization_authorized': False, 'quality_or_FINAL_qualification': False})


def finalize():
    pages = sorted(NOTES.glob('page_*.json'))
    assert [p.name for p in pages] == ['page_%02d.json' % n for n in range(1, 33)]
    rows = []
    for index, path in enumerate(pages, 1):
        doc = load(path)
        assert doc['page'] == index and doc['data_manifest'] == pin(DATA / 'manifest.json')
        assert doc['criteria'] == pin(PREP / 'input_only_review_criteria_v1.json')
        assert pin(DATA / doc['sheet']['path'])['sha256'] == doc['sheet']['sha256']
        rows.extend(doc['decisions'])
    manifest = load(DATA / 'manifest.json')
    assert len(rows) == len({r['id'] for r in rows}) == 512
    assert {r['id'] for r in rows} == {r['id'] for r in manifest['cases']}
    clear = {r['reference_id']: r['decision'] for r in rows if r['profile'] == 'clear'}
    for row in rows:
        row['source_clear_input_decision'] = clear[row['reference_id']]
        row['candidate_for_explicit_finite_pilot'] = row['decision'] == 'readable' and clear[row['reference_id']] == 'readable'
        row['optimization_permitted_now'] = False
        row['row_profile_requires_explicit_new_pilot_admission'] = row['profile'] == 'repeated_rows_4_blend0p75'
    counts = Counter(r['decision'] for r in rows)
    by_profile = {p: dict(Counter(r['decision'] for r in rows if r['profile'] == p))
                  for p in sorted({r['profile'] for r in rows})}
    candidates = [r['id'] for r in rows if r['candidate_for_explicit_finite_pilot']]
    nonclear = [r['id'] for r in rows if r['candidate_for_explicit_finite_pilot'] and r['profile'] != 'clear']
    return write(PREP / 'root_input_only_review_v1.json', {
        'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
        'data_manifest': pin(DATA / 'manifest.json'), 'criteria': pin(PREP / 'input_only_review_criteria_v1.json'),
        'reviewer': 'root', 'sheets_viewed': 32, 'exact256_panels_viewed': 512,
        'all_decisions_locked_before_later_pages': True, 'all28_nonclear_pages_locked_before_clear_pages': True,
        'counts': dict(counts), 'by_profile': by_profile, 'pages': [pin(p) for p in pages], 'cases': rows,
        'readable_source_and_input_candidates': candidates, 'nonclear_candidates': nonclear,
        'all_optimization_still_false': True, 'explicit_finite_protocol_and_source_target_review_still_required': True,
        'no_replacement_of_excluded_cases_with_easier_sources': True,
        'borderline_insufficient_out_of_scope_retained_as_challenge_or_refusal_evidence': True,
        'new_model_training_or_automatic_threshold': False, 'independent_FINAL_qualification': False,
        'existing_gates_and_prior_controls_unchanged': True, 'goal_complete': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--page', type=int)
    parser.add_argument('--decisions')
    parser.add_argument('--note')
    parser.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    print(json.dumps(finalize() if args.finalize else record(args.page, args.decisions, args.note)))
