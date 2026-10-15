"""Close the actual sheet review with independent pixel/namespace checks."""
from datetime import datetime, timezone
import hashlib

import numpy as np
from PIL import Image

from import_cctv_dgp_bank_path_ablation_v1_return import ROOT, RETURN, PARENT, ANALYSIS, PIN, read, sha, write


def main():
    receipt = ANALYSIS/'visual_review.json'
    assert not receipt.exists()
    manifest_path = ANALYSIS/'visual_review/sheet_manifest.json'
    plan = read(manifest_path); p = read(RETURN/'protocol.json')
    assert read(ANALYSIS/'independent_audit.json')['complete']
    assert read(ANALYSIS/'limited_frozen_replay.json')['complete']
    assert plan['paired_sheets'] == 20 and plan['native_sheets'] == 6 and plan['cells'] == 1216
    assert len(plan['sheets']) == 26
    observations = ANALYSIS/'visual_observations'
    assert {f.name for f in observations.iterdir()} == {s['name']+'.json' for s in plan['sheets']}
    bindings, cells, paired_ids, native_ids = {}, 0, [], []
    variant_cells = {v: dict(paired=0, native=0) for v in p['variants']}
    for sheet in plan['sheets']:
        path = ANALYSIS/'visual_review'/sheet['name']
        assert sha(path) == sheet['sha256']
        observation = observations/(sheet['name']+'.json')
        row = read(observation)
        assert row['complete'] and row['sheet_sha256'] == sha(path)
        assert row['sheet_manifest_sha256'] == sha(manifest_path)
        assert row['displayed_through'] == 'view_image at original resolution'
        assert not row['independent_final_review'] and len(row['notes']) >= 40
        bindings[observation.relative_to(ROOT).as_posix()] = sha(observation)
        paired = sheet['role'] == 'paired photographic TRAIN'
        columns, rows = (10, 5) if paired else (9, 4)
        assert len(sheet['cells']) == columns*rows
        assert {(c['row'], c['column']) for c in sheet['cells']} == {(r, c) for r in range(rows) for c in range(columns)}
        with Image.open(path) as im:
            assert im.size == (columns*256, rows*278)
            pixels = np.asarray(im.convert('RGB')).copy()
        for cell in sheet['cells']:
            source = ROOT/cell['path']; assert sha(source) == cell['sha256']
            assert cell['x'] == cell['column']*256 and cell['y'] == cell['row']*278+22
            with Image.open(source) as im:
                assert im.size == (256, 256)
                expected = np.asarray(im.convert('RGB')).copy()
            x, y = cell['x'], cell['y']
            assert np.array_equal(pixels[y:y+256, x:x+256], expected)
            if source.is_relative_to(RETURN/'outputs/path_ablation_v1'):
                variant = source.relative_to(RETURN/'outputs/path_ablation_v1').parts[0]
                variant_cells[variant]['paired' if paired else 'native'] += 1
            cells += 1
        (paired_ids if paired else native_ids).extend(sheet['case_ids'])
    assert len(paired_ids) == len(set(paired_ids)) == 100 and set(paired_ids) == set(p['case_ids'])
    assert native_ids == p['native_case_ids']
    assert all(v == dict(paired=100, native=24) for v in variant_cells.values())
    for name, expected in read(ANALYSIS/'local_protected_before.json')['files_sha256'].items():
        assert sha(ROOT/name) == expected
    write(receipt, dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), checker_sha256=sha(__file__),
        protocol_sha256=PIN, independent_audit_sha256=sha(ANALYSIS/'independent_audit.json'),
        frozen_replay_sha256=sha(ANALYSIS/'limited_frozen_replay.json'),
        sheet_manifest_sha256=sha(manifest_path), viewed_sheets=26,
        sheet_cells_independently_verified=cells, per_variant_cells=variant_cells, observation_sha256=bindings,
        implementing_assistant_review=True, independent_final_review=False, native_cases_unpaired=24,
        no_single_readout_convincing_whole_face_clarity_improvement=True,
        clear_glasses_and_ordinary_visible_hair_retained_in_review=True,
        input_only_usability_labels_not_changed=True, model_softness_not_a_reason_to_exclude_usable_DEV=True,
        automatic_vs_assisted_completion_not_tested=True, all_seven_completion_families_pending=True,
        final_identities_unused=True, local_gradient_or_optimizer_calls=0,
        model_qualified=False, app_promotion=False, goal_complete=False))
    print(dict(complete=True, viewed_sheets=26, exact_cells=cells, independent_final_review=False))


if __name__ == '__main__':
    main()
