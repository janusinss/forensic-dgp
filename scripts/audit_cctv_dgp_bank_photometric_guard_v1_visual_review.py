"""Independently check every sheet cell, exact observations and preserved assets.

Observations cannot prove that viewing occurred; the actual view_image calls
in the chat provide that separate evidence. This is not final human review.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT/'outputs/cctv_dgp_bank_photometric_guard_v1_analysis'
CURRENT = ROOT/'outputs/cctv_dgp_bank_photometric_guard_v1_return'
DEST = ANALYSIS/'visual_review'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    started = time.monotonic()
    receipt = ANALYSIS/'visual_review.json'
    assert not receipt.exists()
    manifest_path = DEST/'sheet_manifest.json'
    manifest = read(manifest_path)
    assert manifest['complete'] and manifest['missing_tiles'] == []
    assert manifest['paired_sheets'] == 20 and manifest['native_sheets'] == 6
    assert manifest['cells'] == 844 and len(manifest['sheets']) == 26
    bindings = {}

    def bind(path):
        bindings[path.relative_to(ROOT).as_posix()] = sha(path)

    for name in ['import.json', 'route_import.json', 'execution_audit.json', 'independent_audit.json',
                 'preview_integrity_audit.json', 'matched_guard_comparison.json', 'visual_input_match.json']:
        path = ANALYSIS/name
        assert read(path)['complete']
        bind(path)
    comparison = read(ANALYSIS/'matched_guard_comparison.json')
    assert comparison['unchanged_baseline_RGB_embedding_packs'] == 781
    assert sha(ANALYSIS/'visual_input_match.json') == manifest['visual_input_match_sha256']
    audit = read(ANALYSIS/'independent_audit.json')
    assert audit['saved_paired_raw_PNG_cases_checked'] == 7910 and audit['native_compositions_checked'] == 72
    assert not all(r['pass'] for r in audit['metric_and_gate_decisions'].values())
    assert len(audit['limited_CPU_replays']) == 2 and audit['optimizer_updates_in_VM'] == 50
    p = read(CURRENT/'protocol.json')
    expected_preview_ids = set(p['preview_case_ids'])
    expected_native_ids = {r['id'] for r in p['native_development']}
    paired_ids, native_ids = set(), set()
    notes = ANALYSIS/'visual_observations'
    assert {f.name for f in notes.iterdir()} == {r['name']+'.json' for r in manifest['sheets']}
    bind(manifest_path)
    cells_checked = 0
    for row in manifest['sheets']:
        path = DEST/row['name']
        assert path.name == row['name'] and sha(path) == row['sha256']
        with Image.open(path) as image:
            sheet = np.asarray(image.convert('RGB')).copy()
        is_paired = row['evidence_role'] == 'paired photographic TRAIN'
        if is_paired:
            assert sheet.shape == (1390, 1792, 3) and len(row['case_ids']) == 5 and len(row['cells']) == 35
            assert not paired_ids.intersection(row['case_ids'])
            paired_ids.update(row['case_ids'])
        else:
            assert row['evidence_role'] == 'native CCTV DEV unpaired'
            assert sheet.shape == (1112, 1536, 3) and len(row['case_ids']) == 4 and len(row['cells']) == 24
            assert not native_ids.intersection(row['case_ids'])
            native_ids.update(row['case_ids'])
        occupied = set()
        for cell in row['cells']:
            source = Path(cell['path'])
            assert source.resolve().is_relative_to(ROOT/'outputs') and source.is_file() and not source.is_symlink()
            assert sha(source) == cell['sha256'] == manifest['source_tile_bindings'][str(source)]
            r, c = cell['row'], cell['column']
            assert (r, c) not in occupied
            occupied.add((r, c))
            assert cell['x'] == c*256 and cell['y'] == r*278+22
            assert cell['width'] == cell['height'] == 256
            with Image.open(source) as image:
                assert image.size == (256, 256)
                pixels = np.asarray(image.convert('RGB'))
            assert np.array_equal(sheet[cell['y']:cell['y']+256, cell['x']:cell['x']+256], pixels)
            cells_checked += 1
        assert occupied == {(r, c) for r in range(5 if is_paired else 4) for c in range(7 if is_paired else 6)}
        observation_path = notes/(row['name']+'.json')
        observation = read(observation_path)
        assert observation['complete'] and observation['sheet'] == row['name']
        assert observation['sheet_sha256'] == row['sha256']
        assert observation['sheet_manifest_sha256'] == sha(manifest_path)
        assert observation['displayed_through'] == 'view_image at original resolution'
        assert observation['reviewer'] == 'implementing assistant' and not observation['independent_final_review']
        assert len(observation['notes']) >= 30
        bind(path)
        bind(observation_path)
    assert paired_ids == expected_preview_ids and len(paired_ids) == 100
    assert native_ids == expected_native_ids and len(native_ids) == 24
    assert cells_checked == 844
    protected_path = ANALYSIS/'local_protected_before.json'
    for name, expected in read(protected_path)['files_sha256'].items():
        assert sha(ROOT/name) == expected
    bind(protected_path)
    result = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), checker_sha256=sha(__file__),
        evidence_files_sha256=bindings, viewed_sheets=26, all_planned_sheets_viewed=True,
        sheet_cells_independently_verified=cells_checked, paired_TRAIN_cases=100, native_DEV_unpaired_cases=24,
        reviewer='implementing assistant', independent_final_review=False,
        numerical_audit_and_visual_observations_separate=True,
        old_A_B_and_current_quality_stops_retained=True, candidate_qualified=False,
        app_checkpoint_and_example_inputs_unchanged=True, final_identities_used=False,
        all_seven_completion_families_pending=True, local_training_or_gradient_calls=0,
        app_promotion=False, goal_complete=False, seconds=time.monotonic()-started)
    with receipt.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(dict(complete=True, sheets=26, cells=cells_checked, independent_final_review=False), flush=True)


if __name__ == '__main__':
    main()
