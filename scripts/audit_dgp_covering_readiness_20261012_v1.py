"""Recount retained covering evidence; no neural inference, edits or training."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/dgp_covering_readiness_20261012_v1'
CAP_SECONDS = 180
START = time.monotonic()
HASHES = {}
STATS = {}
BINDINGS = 0
HISTORY = {}
DOCUMENTS = {'PRACTICAL_OUTPUT_SCOPE.md', 'SYSTEM_WORKFLOW_AND_GOAL.md'}

RUNS = {
    'current_app': ('dgp_app_covering_review_v3', 'plan.json', 'saved_output_audit.json'),
    'union_context': ('completion_conditioning_union_v1', 'protocol.json', 'independent_saved_output_audit.json'),
    'fusion_off': ('completion_feature_fusion_off_v1', 'protocol.json', 'independent_saved_output_audit_r1.json'),
    'margin_blend': ('completion_margin_feather_v1', 'protocol.json', 'independent_saved_output_audit.json'),
    'converted_MAT': ('completion_mat_mirror_comparison_v1', 'protocol.json', 'independent_saved_output_audit.json'),
}
FAMILIES = {
    'masks': {'face_mask', 'hand_over_mask'},
    'sunglasses': {'sunglasses', 'sunglasses_and_glare'},
    'strong_glare': {'strong_lens_glare', 'sunglasses_and_glare'},
    'hands': {'hand', 'hand_over_mask', 'visibility_rejection'},
    'obstructing_hair': {'obstructing_hair'},
    'scarves': {'scarf'},
    'objects': {'other_object'},
}


def check_time():
    assert time.monotonic() - START < CAP_SECONDS, 'Saved-evidence audit time cap'


def local(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    assert path.is_relative_to(ROOT), 'Outside workspace: ' + str(path)
    assert path.is_file() and not path.is_symlink(), str(path)
    return path


def sha(path):
    path = local(path)
    key = path.relative_to(ROOT).as_posix()
    if key not in HASHES:
        before = path.stat()
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
                check_time()
                h.update(block)
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        HASHES[key] = h.hexdigest()
        STATS[key] = [after.st_size, after.st_mtime_ns]
    return HASHES[key]


def bind(path, digest):
    global BINDINGS
    path = local(path)
    actual = sha(path)
    if actual != digest:
        rel = path.relative_to(ROOT).as_posix()
        assert rel in DOCUMENTS, 'Hash differs: ' + str(path)
        key = rel + ':' + digest
        if key not in HISTORY:
            # Only two instruction documents may have a preserved historical
            # suffix. Current executable/model/mask/output bytes remain exact.
            published = read(ROOT/'outputs/dgp_app_flow_audit_20261012/publication.json')
            assert published['documents'][rel]['after_sha256'] == actual
            data = path.read_bytes()
            offsets = [0] + [i + 1 for i in range(len(data)-2) if data[i:i+3] == b'\n# ']
            matches = [offset for offset in offsets
                       if hashlib.sha256(data[offset:]).hexdigest() == digest]
            assert len(matches) == 1, 'Historical instruction bytes absent: ' + rel
            offset = matches[0]
            HISTORY[key] = dict(path=rel, historical_sha256=digest,
                current_sha256=actual, suffix_offset_bytes=offset,
                historical_bytes=len(data)-offset,
                verified_as_exact_preserved_suffix=True)
    BINDINGS += 1


def read(path):
    path = local(path)
    sha(path)
    return json.loads(path.read_text(encoding='utf-8'))


def rgb(path):
    sha(path)
    with Image.open(local(path)) as im:
        assert im.size == (256, 256)
        return np.asarray(im.convert('RGB')).copy()


def mask(path):
    sha(path)
    with Image.open(local(path)) as im:
        assert im.size == (256, 256)
        a = np.asarray(im.convert('L')).copy()
    assert set(np.unique(a)).issubset({0, 255}), str(path)
    return a != 0


def overlap(proposal, operator):
    intersection = int((proposal & operator).sum())
    union = int((proposal | operator).sum())
    return {
        'iou_against_operator_footprint': intersection / union if union else 1.0,
        'missed_operator_pixels': int((operator & ~proposal).sum()),
        'extra_pixels': int((proposal & ~operator).sum()),
        'proposal_pixels': int(proposal.sum()),
        'operator_pixels': int(operator.sum()),
    }


def main():
    assert not OUT.exists(), 'Immutable receipt already present'
    runs = {}
    for label, (name, protocol_file, audit_file) in RUNS.items():
        folder = ROOT / 'outputs' / name
        p, r, a, v = [read(folder / f) for f in
                      [protocol_file, 'results.json', audit_file, 'visual_review.json']]
        assert r['complete'] and a['complete'] and v['complete']
        bind(folder / protocol_file, r.get('protocol_sha256', r.get('plan_sha256')))
        bind(folder / 'results.json', a['results_sha256'])
        assert v['independent_final_review'] is False if 'independent_final_review' in v else True
        for rel, digest in p.get('sources_sha256', {}).items():
            bind(ROOT / rel, digest)
        for rel, digest in p.get('app_preservation_sha256', {}).items():
            bind(ROOT / rel, digest)
        for rel, digest in r.get('artifacts_sha256', {}).items():
            bind(folder / rel, digest)
        assert r.get('optimizer_updates', 0) == r.get('gradient_calls', 0) == 0
        assert not r.get('family_qualification', False)
        assert not r.get('automatic_quality_qualification', False)
        assert not r.get('assisted_quality_qualification', False)
        assert not r.get('app_adoption', False)
        cases = {c['id']: c for c in p['cases']}
        rows = {c['id']: c for c in r['rows']}
        assert len(cases) == len(p['cases']) and set(cases) == set(rows)
        runs[label] = dict(folder=folder, protocol=p, results=r, audit=a,
                           visual=v, cases=cases, rows=rows, audit_file=audit_file)
    current = runs['current_app']
    cases = current['cases']
    eligible = {key for key, c in cases.items() if c['input_review'] == 'usable'}
    assert len(cases) == 36 and len(eligible) == 32
    for label, run in runs.items():
        expected = eligible if label == 'converted_MAT' else set(cases)
        assert set(run['cases']) == expected
        for key, c in run['cases'].items():
            old = cases[key]
            assert all(c[k] == old[k] for k in ['input', 'family', 'condition', 'reviewed'])

    recount = []
    preservation = Counter()
    for key, c in cases.items():
        check_time()
        row = current['rows'][key]
        source, operator = rgb(c['input']), mask(c['reviewed'])
        automatic = mask(current['folder'] / 'images' / (key + '_automatic.png'))
        counts = overlap(automatic, operator)
        for field, value in counts.items():
            assert row['automatic'][field] == value, (key, field)
        for name, rel in c['cached_comparison_proposals'].items():
            cached = overlap(mask(rel), operator)
            assert all(row['cached_unqualified_comparisons'][name][k] == value
                       for k, value in cached.items()), (key, name)
        base = key.removesuffix('_native').removesuffix('_degraded')
        recount.append(dict(id=key, base=base, family=c['family'], condition=c['condition'],
            decision=c['input_review'], automatic=counts,
            operator_is_approximate_footprint_not_expert_truth=True,
            original_photo_assistance_reused_for_synthetic_pair=True))
        if key not in eligible:
            assert all(result['rejected'] for result in row['assisted'].values())
            for label in ['union_context', 'fusion_off', 'margin_blend']:
                trial = runs[label]['rows'][key]
                assert trial.get('rejected_before_neural', trial.get('rejected_before_processing'))
            preservation['retained_exclusions'] += 1
            continue
        off = rgb(current['folder'] / row['assisted']['off']['output'])
        np.testing.assert_array_equal(off[~operator], source[~operator])
        if not operator.any():
            np.testing.assert_array_equal(off, source)
            preservation['current_exact_bypasses'] += 1
        preservation['current_exact_visible_cases'] += 1
        for label in ['union_context', 'fusion_off', 'margin_blend']:
            run = runs[label]
            trial = run['rows'][key]
            config = run['cases'][key]
            final, protected = mask(config['masks']['removal']), mask(config['masks']['protected'])
            assert not (final & protected).any()
            output = rgb(run['folder'] / 'images' / (key + '.png'))
            np.testing.assert_array_equal(output[~final], source[~final])
            np.testing.assert_array_equal(output[protected], source[protected])
            if not final.any():
                np.testing.assert_array_equal(output, source)
                preservation[label + '_exact_bypasses'] += 1
            elif label in ['union_context', 'fusion_off']:
                stage = run['folder'] / trial['stages']
                sha(stage)
                with np.load(stage, allow_pickle=False) as data:
                    raw = data['completion']
                    assert raw.dtype == np.float32 and raw.shape == (1, 3, 256, 256)
                    assert np.isfinite(raw).all() and raw.min() >= 0 and raw.max() <= 1
                    expected = np.floor(raw[0].transpose(1, 2, 0) * np.float32(255)).astype(np.uint8)
                meta = read(run['folder'] / trial['metadata'])
                if meta['display_processing']['colour_policy']['applied']:
                    expected = np.repeat(cv2.cvtColor(expected, cv2.COLOR_RGB2GRAY)[..., None], 3, -1)
                np.testing.assert_array_equal(output[final], expected[final])
                preservation[label + '_exact_saved_raw_compositions'] += 1
            preservation[label + '_exact_visible_and_protected_cases'] += 1
        mat = runs['converted_MAT']
        trial = mat['rows'][key]
        for field in ['output', 'completed256', 'raw512']:
            if trial.get(field):
                bind(mat['folder'] / trial[field], trial[field + '_sha256'])
        image = rgb(mat['folder'] / trial['output'])
        np.testing.assert_array_equal(image[~operator], source[~operator])
        if not operator.any():
            np.testing.assert_array_equal(image, source)
            preservation['MAT_exact_bypasses'] += 1
        preservation['MAT_exact_visible_cases'] += 1

    pairs = defaultdict(list)
    for row in recount:
        pairs[row['base']].append(row)
    assert len(pairs) == 18 and all(len(rows) == 2 for rows in pairs.values())
    for rows in pairs.values():
        assert {r['condition'] for r in rows} == {'original_photo', 'synthetic_degraded_photo'}
        assert rows[0]['decision'] == rows[1]['decision']
        np.testing.assert_array_equal(mask(cases[rows[0]['id']]['reviewed']), mask(cases[rows[1]['id']]['reviewed']))

    family_counts = {}
    for family, tags in FAMILIES.items():
        members = [row for row in recount if row['family'] in tags]
        counts = {}
        for condition in ['original_photo', 'synthetic_degraded_photo']:
            subset = [row for row in members if row['condition'] == condition]
            usable = [row for row in subset if row['decision'] == 'usable']
            counts[condition] = dict(exposed_photo_cases=len(subset), eligible_cases=len(usable),
                rejected_cases=len(subset)-len(usable),
                empty_automatic_proposals_on_eligible_coverings=sum(r['automatic']['proposal_pixels'] == 0 for r in usable),
                automatic_operator_pixels_missed=sum(r['automatic']['missed_operator_pixels'] for r in usable),
                automatic_extra_pixels=sum(r['automatic']['extra_pixels'] for r in usable))
        assert counts['original_photo']['eligible_cases'] > 0
        family_counts[family] = dict(condition_counts=counts,
            source_photo_ids=sorted(set(r['base'] for r in members)),
            automatic_qualified=False, assisted_qualified=False,
            independent_final_review=False)

    comparisons = {}
    for label, run in runs.items():
        folder = run['folder']
        comparisons[label] = dict(protocol_sha256=sha(folder / ('plan.json' if label == 'current_app' else 'protocol.json')),
            results_sha256=sha(folder/'results.json'), saved_output_audit_sha256=sha(folder/run['audit_file']),
            historical_visual_review_sha256=sha(folder/'visual_review.json'),
            historical_review_reverified_by_hash_only=True, new_visual_review=False,
            automatic_qualified=False, assisted_qualified=False)
    condition_totals = {}
    for condition in ['original_photo', 'synthetic_degraded_photo']:
        subset = [row for row in recount if row['condition'] == condition]
        controls = [row for row in subset if row['family'] in ['uncovered_control', 'clear_glasses_control']]
        covering = [row for row in subset if row['decision'] == 'usable' and row not in controls]
        condition_totals[condition] = dict(all_cases=len(subset), eligible_covering_cases=len(covering),
            empty_proposals_all_cases=sum(r['automatic']['proposal_pixels'] == 0 for r in subset),
            empty_proposals_eligible_coverings=sum(r['automatic']['proposal_pixels'] == 0 for r in covering),
            controls=[dict(id=r['id'], automatic_pixels=r['automatic']['proposal_pixels']) for r in controls])
    script_digest = sha(Path(__file__))
    for rel, stamp in STATS.items():
        stat = (ROOT / rel).stat()
        assert [stat.st_size, stat.st_mtime_ns] == stamp, 'Evidence changed during recount: ' + rel
    result = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), seconds=time.monotonic()-START,
        cap_seconds=CAP_SECONDS, checked_bindings=BINDINGS, unique_files=len(HASHES),
        source_and_output_sha256=HASHES, comparisons=comparisons, cases=recount,
        historical_instruction_bindings=list(HISTORY.values()),
        photo_pairs=18, conditions_not_independent_subjects=True,
        family_membership_overlaps_for_combined_coverings=True,
        all_seven_families_in_scope=True, family_counts=family_counts,
        condition_totals=condition_totals, preservation=dict(preservation),
        current_app_hashes_match_retained_comparisons=True,
        scope='Exposed development photographs and matched synthetic degradations; no native CCTV or final evaluation',
        approximate_operator_footprint_overlap_not_segmentation_accuracy=True,
        hidden_reference_metrics=None, unknown_pretraining_overlap=True,
        automatic_generated_outputs_in_retained_comparisons=0,
        automatic_qualified=False, assisted_qualified=False, independent_final_review=False,
        new_neural_calls=0, gradients=0, optimizer_updates=0, new_checkpoint=False,
        app_edits=False, mask_edits=False, VM_calls=0, goal_complete=False,
        script_sha256=script_digest)
    assert result['seconds'] < CAP_SECONDS
    OUT.mkdir()
    for proof in HISTORY.values():
        saved = OUT / ('historical_' + proof['path'])
        suffix = (ROOT / proof['path']).read_bytes()[proof['suffix_offset_bytes']:]
        assert hashlib.sha256(suffix).hexdigest() == proof['historical_sha256']
        with saved.open('xb') as stream:
            stream.write(suffix)
        proof['retained_snapshot'] = saved.relative_to(ROOT).as_posix()
    with (OUT/'audit.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in
        ['complete','seconds','checked_bindings','unique_files','condition_totals','preservation','automatic_qualified','assisted_qualified','new_neural_calls']}))


if __name__ == '__main__':
    main()
