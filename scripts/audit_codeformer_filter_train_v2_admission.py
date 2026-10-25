"""Independently check root's saved input-review metadata; no image/model calls.

This verifies joins/counts/declarations and internally recorded chronology. It
does not independently observe root's visual review or establish FINAL quality.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/codeformer_filter_train_v2_preparation'
DATA = ROOT / 'outputs/codeformer_filter_train_v2_data'
REVIEW_SHA = 'f306abc619944a85f29f946d2a0bb7afb79420ebbb5125835d17d05fb0e42d06'
EXPECTED_SOURCE_EXCLUSIONS = {'tr_ffhq_' + x for x in ('04454', '20546', '45314', '49284', '61989')}
ORDER = ('global_mosaic_8', 'local_mosaic_8', 'local_gaussian_3p0',
    'repeated_rows_4_blend0p75', 'gaussian_2p4', 'anisotropic_h5_v0p7', 'anisotropic_h0p7_v5', 'clear')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def strict(raw):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    value = json.loads(raw.decode('utf-8'), object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    def finite(v):
        if isinstance(v, float):
            require(math.isfinite(v), 'JSON overflow')
        elif isinstance(v, dict):
            for item in v.values(): finite(item)
        elif isinstance(v, list):
            for item in v: finite(item)
    finite(value)
    return value


def bound(path, sha=None, size=None):
    path = Path(path).absolute()
    require(path.resolve(strict=True).is_relative_to(ROOT), 'Workspace evidence required')
    for part in (path, *path.parents):
        if part == ROOT.parent: break
        info = part.lstat()
        require(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024,
                'Linked/reparse evidence')
    before = path.stat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size < 8 * 1024**2,
            'Bounded ordinary single-link file required')
    raw = path.read_bytes(); after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns), 'Evidence changed while reading')
    require(sha is None or hashlib.sha256(raw).hexdigest() == sha, 'Evidence SHA differs')
    require(size is None or len(raw) == size, 'Evidence bytes differ')
    return raw


def from_pin(pin, base=ROOT):
    name = pin['path']; p = PurePosixPath(name)
    require(name and ':' not in name and '\\' not in name and not p.is_absolute() and
            p.as_posix() == name and all(v not in ('.', '..') for v in p.parts), 'Canonical pin required')
    return bound(base / name, pin['sha256'], pin['bytes'])


def fingerprint(path):
    raw = bound(path)
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}


def audit():
    require(not any(x in sys.modules for x in ('torch', 'scipy', 'numpy', 'PIL', 'tensorflow', 'jax')),
            'Standard library metadata audit only')
    raw = bound(PREP / 'root_input_only_review_v1.json', REVIEW_SHA, 280681)
    review = strict(raw)
    require(review['complete'] is True and review['reviewer'] == 'root' and
            review['sheets_viewed'] == 32 and review['exact256_panels_viewed'] == 512,
            'Expected root32/512 review declarations required')
    doc = strict(from_pin(review['data_manifest']))
    criteria = strict(from_pin(review['criteria']))
    require(criteria['frozen_before_new_inputs_viewed'] is True and
            criteria['optimization_permitted_by_data_generation'] is False and
            criteria['existing_quality_gate_or_scope_changed'] is False,
            'Unchanged frozen criteria/admission scope required')
    require(doc['complete'] is True and doc['pairs'] == 512 and doc['TRAIN_pairs'] == 512 and
            doc['DEV_pairs'] == 0 and doc['optimization_permitted_now'] is False,
            'Full512 no-optimization source set required')
    require(from_pin(doc['input_only_criteria']) == from_pin(review['criteria']), 'Criteria current/published copy differs')
    cases = {x['id']: x for x in doc['cases']}
    require(len(cases) == 512, 'Unique512 source cases required')
    visual = strict(from_pin(doc['files']['input_only_visual_manifest.json'], DATA))
    cells = {x['reference_id'] + '__' + x['profile']: x for x in visual['cells']}
    require(len(cells) == 512 and visual['sheets'] == doc['input_only_sheets'], 'All exact panel joins required')
    selected = strict(from_pin(doc['files']['cohort_selection.json'], DATA))
    cohort = {x['reference_id']: x for x in selected['cohort']}
    require(len(cohort) == 64 and all(x['split'] == 'TRAIN' and x['historical_role'] == 'train'
                                    for x in cohort.values()), 'All64 historical TRAIN joins required')
    pixel_path = ROOT / 'outputs/codeformer_filter_train_v2_independent_audit/saved_pixel_audit_v1.json'
    pixel_raw = bound(pixel_path); pixels = strict(pixel_raw)
    require(pixels['pass'] is True and pixels['complete'] is True and pixels['pairs'] == 512 and
            pixels['channel_mismatches'] == 0 and pixels['manifest_sha256'] == review['data_manifest']['sha256'],
            'Prior exact independent pixel audit required')
    pixel_index = {x['id']: x for x in pixels['cases']}
    require(set(pixel_index) == set(cases) == set(cells), 'Pixel/source/sheet coverage differs')
    require(len(review['pages']) == 32, 'All32 page receipts required')
    flattened, page_rows, page_times = [], [], []
    for index, pin in enumerate(review['pages'], 1):
        require(pin['path'] == 'outputs/codeformer_filter_train_v2_preparation/root_input_only_pages_v1/page_%02d.json' % index,
                'Sequential page path required')
        page = strict(from_pin(pin)); sheet = visual['sheets'][index - 1]
        require(page['complete'] is True and page['reviewer'] == 'root' and page['page'] == index and
                page['sheet'] == sheet and page['data_manifest'] == review['data_manifest'] and
                page['criteria'] == review['criteria'] and sheet['profile'] == ORDER[(index - 1) // 4],
                'Page/sheet/profile/criteria/source joins differ')
        require(page['viewed_saved256_inputs_without_targets_or_model_panels'] is True and
                page['not_blinded_to_prior_historical_sources_or_other_corruptions'] is True and
                page['optimization_authorized'] is False and page['quality_or_FINAL_qualification'] is False,
                'Page review/optimization declarations differ')
        from_pin({k: sheet[k] for k in ('path', 'sha256', 'bytes')}, DATA)
        current_cells = [c for c in visual['cells'] if c['sheet'] == sheet['path']]
        require(len(page['decisions']) == len(current_cells) == 16, 'Sixteen saved decisions/page required')
        for d, cell in zip(page['decisions'], current_cells):
            case_id = cell['reference_id'] + '__' + cell['profile']
            require(d['id'] == case_id and d['reference_id'] == cell['reference_id'] and
                    d['profile'] == cell['profile'] and d['sheet_cell_RGB8_sha256'] ==
                    cell['RGB8_sha256'] == pixel_index[case_id]['input_RGB8_sha256'] and
                    d['decision'] in criteria['decisions'], 'Root decision/exact source panel join differs')
            ci = cases[case_id]; record = strict(from_pin(ci['pair_record'], DATA))
            require(ci['reference_id'] == d['reference_id'] and ci['split'] == record['split'] == 'TRAIN' and
                    record['input_only_review']['criteria_sha256'] == review['criteria']['sha256'] and
                    record['attribution'] == cohort[d['reference_id']]['attribution'] and
                    record['source1024'] == cohort[d['reference_id']]['source1024'] and
                    record['optimization_permitted_now'] is ci['optimization_permitted_now'] is False,
                    'Pair input/criteria/attribution/admission join differs')
            from_pin(ci['input'], DATA)
            flattened.append(d)
        when = datetime.fromisoformat(page['UTC'])
        require(when.tzinfo is not None, 'Timezone-aware page receipt required')
        page_times.append(when)
        page_rows.append({'page': index, 'profile': sheet['profile'], 'UTC': page['UTC'],
                          'receipt_sha256': pin['sha256'], 'sheet_sha256': sheet['sha256'], 'decisions': 16})
    require(all(a < b for a, b in zip(page_times, page_times[1:])) and
            max(page_times[:28]) < min(page_times[28:]) and
            page_times[-1] <= datetime.fromisoformat(review['UTC']),
            'Recorded chronology must place28 nonclear pages before four clear pages')
    require(len(flattened) == len({x['id'] for x in flattened}) == len(review['cases']) == 512,
            'Complete unique512 decisions required')
    clear_decisions = {x['reference_id']: x['decision'] for x in flattened if x['profile'] == 'clear'}
    exclusions = {rid for rid, decision in clear_decisions.items() if decision != 'readable'}
    require(exclusions == EXPECTED_SOURCE_EXCLUSIONS and len(clear_decisions) == 64, 'Exact five clear-source exclusions')
    candidates, nonclear = [], []
    per_profile = {p: Counter() for p in ORDER}
    for saved, decision in zip(review['cases'], flattened):
        require(all(saved[k] == v for k, v in decision.items()), 'Original per-input decision was changed')
        rid = saved['reference_id']; is_candidate = saved['decision'] == clear_decisions[rid] == 'readable'
        require(saved['source_clear_input_decision'] == clear_decisions[rid] and
                saved['candidate_for_explicit_finite_pilot'] is is_candidate and
                saved['optimization_permitted_now'] is False and
                saved['row_profile_requires_explicit_new_pilot_admission'] is
                (saved['profile'] == 'repeated_rows_4_blend0p75'), 'Candidate/source-override calculation differs')
        per_profile[saved['profile']][saved['decision']] += 1
        if is_candidate:
            candidates.append(saved['id'])
            if saved['profile'] != 'clear': nonclear.append(saved['id'])
    counts = Counter(x['decision'] for x in flattened)
    require(dict(counts) == review['counts'] == {'readable': 390, 'borderline': 93, 'insufficient': 1, 'out_of_scope': 28},
            'Original label census differs')
    require({p: dict(c) for p, c in per_profile.items()} == review['by_profile'], 'Original profile census differs')
    require(candidates == review['readable_source_and_input_candidates'] and len(candidates) == 386 and
            nonclear == review['nonclear_candidates'] and len(nonclear) == 327, 'Exact386/327 candidates differ')
    require(review['all_optimization_still_false'] is True and
            review['all_decisions_locked_before_later_pages'] is True and
            review['all28_nonclear_pages_locked_before_clear_pages'] is True and
            review['no_replacement_of_excluded_cases_with_easier_sources'] is True and
            review['existing_gates_and_prior_controls_unchanged'] is True and
            review['goal_complete'] is False, 'Review research boundaries differ')
    for row in cohort.values():
        from_pin(row['source1024']); from_pin(row['target256'])
    candidate_rows = [x for x in review['cases'] if x['id'] in set(candidates)]
    candidate_sources = sorted({x['reference_id'] for x in candidate_rows})
    def strata_count(rows, unique=False):
        values = {x['reference_id'] for x in rows} if unique else [x['reference_id'] for x in rows]
        counter = Counter((cohort[rid]['attribution']['license_url'], cohort[rid]['official_category']) for rid in values)
        return [{'license_url': key[0], 'official_FFHQ_category': key[1], 'count': count}
                for key, count in sorted(counter.items())]
    profile_counts = dict(Counter(x['profile'] for x in candidate_rows))
    # Freeze padding independently of model outputs: first lexical admitted mosaic.
    padding = sorted(x for x in nonclear if x.endswith('__global_mosaic_8'))[0]
    receipt = {'format': 'codeformer-filter-train-v2-admission-metadata-independent-audit-v1',
        'complete': True, 'pass': True, 'UTC': datetime.now(timezone.utc).isoformat(),
        'auditor': fingerprint(Path(__file__)), 'root_review_sha256': REVIEW_SHA,
        'root_review_bytes': len(raw), 'criteria': review['criteria'], 'data_manifest': review['data_manifest'],
        'independent_pixel_audit': fingerprint(pixel_path), 'pages': page_rows,
        'all512_case_source_criteria_page_hash_joins': True, 'source_files_rehashed': 64,
        'original_decisions_exact_not_rewritten_by_clear_overrides': True,
        'recorded32_page_times_strictly_increasing': True,
        'recorded28_nonclear_before4clear': True,
        'chronology_independently_observed_or_signed': False,
        'visual_review_independently_reperformed': False,
        'root_visual_review_declarations_verified_as_metadata_only': True,
        'input_counts': dict(counts), 'input_by_profile': {p: dict(c) for p, c in per_profile.items()},
        'excluded_sources': sorted(exclusions), 'source_override_excluded_variants': 40,
        'source_override_original_labels': dict(Counter(x['decision'] for x in review['cases'] if x['reference_id'] in exclusions)),
        'candidate_cases': 386, 'candidate_clear': 59, 'candidate_nonclear': 327, 'candidate_sources': 59,
        'candidate_counts_by_profile': profile_counts,
        'candidate_source_strata': strata_count(candidate_rows, unique=True),
        'candidate_case_strata': strata_count(candidate_rows),
        'candidate_nonclear_strata': strata_count([x for x in candidate_rows if x['profile'] != 'clear']),
        'all9_selected_strata_still_represented': len(strata_count(candidate_rows, unique=True)) == 9,
        'no_replacement_sources': True, 'source_scope_gap': 'Five original selected multi-face photos and all40 variants retained but excluded; no recrop/replacement.',
        'official_category_is_not_historical_DEV_role': True,
        'proposed_training': {'maximum_updates': 164, 'microbatches': 656, 'filter_slots': 328,
            'distinct_nonclear_candidates': 327, 'padding_duplicate': padding,
            'padding_rule': 'Lexicographically first readable-source/readable-input global_mosaic_8 candidate; fixed before new model outputs',
            'clear_slots': 164, 'dark_slots': 164, 'sampler_sealed': False, 'dark_joins_sealed': False},
        'optimization_authorized_by_audit': False, 'row_admission_requires_explicit_pilot': True,
        'new_neural_gradient_optimizer_VM_browser_calls': 0,
        'model_or_FINAL_qualification': False, 'goal_complete': False}
    target = PREP / 'admission_metadata_audit_v1.json'
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False); stream.write('\n')
    return {'complete': True, 'pass': True, 'candidate_clear': 59, 'candidate_nonclear': 327,
            'receipt': str(target), 'padding_duplicate': padding}


if __name__ == '__main__':
    print(json.dumps(audit()), flush=True)
