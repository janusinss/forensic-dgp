"""Independent hash, saved-pixel and history readback of the closed V27 milestone."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/dgp_feature_skips_v27_audit_milestone'
PIN = '22f76701e317b38d945838a6767239c5636cff534b61ddfbf248cc42ba08a82e'
ARCHIVE = '1bd4aec2bdc10cd6aca4455f24875b006bf4bca2d1a4f1cae8f4932400b386ab'
DOCS = ('PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
        'CCTV_DGP_FEATURE_SKIPS_V27_PLAN.md', 'CCTV_DGP_FEATURE_SKIPS_V27_VM.md')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify(bindings, mapping=None):
    for name, digest in bindings.items():
        path = (ROOT / (mapping or {}).get(name, name)).resolve()
        assert path.is_relative_to(ROOT) and path.is_file() and sha(path) == digest, name
    return len(bindings)


def main():
    started = time.monotonic()
    receipt = OUT / 'independent_readback.json'
    assert not receipt.exists(), 'Preserve completed independent readback'
    milestone_path = OUT / 'milestone.json'
    m = read(milestone_path)
    current = verify(m['new_evidence_sha256'])
    path = ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261006_v2/closure_manifest.json'
    assert sha(path) == m['cleanup_closure_sha256'] == 'c6834df1627d2a93467391dd9c883a7df7a612659fa907888aead9866bd8d567'
    cleanup = read(path)
    counts = [verify(cleanup['new_evidence_sha256'], m['cleanup60_original_locations'])]
    assert cleanup['complete'] and cleanup['archives_removed'] == 14 and not cleanup['goal_complete']
    path = ROOT / 'outputs/dgp_v26_gradient_return_and_feature_skips_v27_milestone/milestone.json'
    assert sha(path) == m['previous_milestone_sha256'] == '896204c16b02aa46e5b8613df2597bcf6d9bcf78a64fcd57ab4f6d6f922d832c'
    previous = read(path)
    counts.append(verify(previous['new_evidence_sha256'], m['previous309_original_locations']))
    for name, key in [
        ('outputs/dgp_v26_corrected_objective_and_gradient_diagnostic_milestone/milestone.json', 'previous66_original_locations'),
        ('outputs/dgp_batchmatched_identity_v26_audit_milestone/milestone.json', 'previous692_original_locations'),
        ('outputs/dgp_v25_gradient_audit_and_batchmatched_identity_v26_milestone/milestone.json', 'previous299_original_locations'),
        ('outputs/dgp_spatial_features_v25_audit_and_gradient_diagnostic_milestone/milestone.json', 'previous697_original_locations'),
        ('outputs/dgp_spatial_features_v25_preparation_milestone/milestone.json', 'previous513_original_locations')]:
        path = ROOT / name
        assert sha(path) == previous['previous_milestone_sha256']
        old = read(path)
        counts.append(verify(old['new_evidence_sha256'], previous[key]))
        previous = old
    assert counts == [60, 309, 66, 692, 299, 697, 513], counts
    for name in DOCS:
        original = (OUT / 'before_docs' / name).read_bytes()
        updated = (ROOT / name).read_bytes()
        split = original.index(b'\n') + 1
        assert updated.startswith(original[:split]) and updated.endswith(original[split:]), name
        prefix = updated[split:len(updated) - len(original[split:])].decode('utf-8')
        for text in ('V27 return independently audited/reviewed', '0.2210885780%',
                     'separate candidate of the original DGP reconstruction decoder',
                     '14 backed-up archive duplicates', 'Goal active/incomplete'):
            assert text in prefix, (name, text)
    app_path = ROOT / 'outputs/dgp_app_v3_integration_record.json'
    assert sha(app_path) == m['app_record_sha256'] == 'e744f3708f248012469a111e204222f66261af4ca28bbaf9521ee4c5464b5e7a'
    app = read(app_path)
    assert verify({**app['sources_sha256'], **app['evidence_sha256']}) == 22
    imported = read(ROOT / 'outputs/cctv_dgp_feature_skips_v27_return_import.json')
    assert imported['complete'] and imported['archive_sha256'] == ARCHIVE
    assert imported['archive_bytes'] == 325777145 and imported['members'] == 631
    assert imported['protocol_sha256'] == PIN
    returned = ROOT / 'outputs/cctv_dgp_feature_skips_v27_return'
    assert verify({(returned / name).relative_to(ROOT).as_posix(): digest
                   for name, digest in imported['files_sha256'].items()}) == 631
    audit_path = ROOT / 'outputs/cctv_dgp_feature_skips_v27_independent_audit.json'
    audit = read(audit_path)
    assert audit['complete'] and audit['protocol_sha256'] == PIN
    assert audit['source_assets_verified'] == 246 and audit['returned_files_verified'] == 631
    assert audit['counts'] == {'head_forwards': 100, 'recognizer_forwards': 110,
                              'raw_PNG_pairs': 100, 'metric_rows': 100, 'complete_snapshots': 2}
    assert audit['complete_snapshot_updates'] == [0, 50] and not audit['incomplete_snapshot_folders']
    assert audit['VM_failure_present'] and not audit['VM_training_result_present']
    assert audit['failure_cause'] == 'No one-percent early structural gain; retain stop'
    stop = read(returned / 'outputs/early_structure_stop.json')
    assert stop == audit['early_structure_stop'] and stop['update'] == 50 and not stop['pass']
    assert stop['minimum'] == .01 and 0 < stop['relative_feature_error_gain'] < .01
    assert m['V27_structure_gain_percent'] == 100 * stop['relative_feature_error_gain']
    assert audit['execution_audit']['updates'] == 50 and audit['execution_audit']['backwards'] == 51
    assert audit['execution_audit']['five_direct_feature_skip_preoptimizer_gradients_verified']
    assert audit['execution_audit']['five_projection_update2_gradients_verified']
    assert audit['identity_preflight_audit']['all36_exact_zero_gradient_assertions_source_bound']
    assert audit['frozen_features_audit']['arrays'] == 250
    assert audit['backward_calls'] == audit['optimizer_updates'] == 0
    assert not audit['actual_local_derivative_replay'] and not audit['necessary_capacity_pass']
    folder = ROOT / 'outputs/cctv_dgp_feature_skips_v27_saved_output_review'
    prepared = read(folder / 'preparation.json')
    review = read(folder / 'visual_review.json')
    arithmetic = read(folder / 'stopped_capacity_arithmetic.json')
    verify(prepared['source_bindings_sha256'])
    assert prepared['audit_sha256'] == arithmetic['audit_sha256'] == sha(audit_path)
    assert prepared['protocol_sha256'] == PIN and prepared['exact_source_cells_verified'] == 200
    assert review['complete'] and review['cases_reviewed'] == 50 and review['exact_source_cells'] == 200
    assert review['preparation_sha256'] == sha(folder / 'preparation.json')
    assert review['sheet_sha256'] == prepared['sheet_sha256'] and len(prepared['sheets']) == 10
    assert {r['id'] for r in review['rows']} == {r['id'] for r in prepared['rows']}
    assert all(r['regions_reviewed'] == ['eyes', 'nose', 'mouth', 'face outline', 'overall visible appearance']
               and not r['convincing_whole_face_structure_gain'] for r in review['rows'])
    cells = 0
    for sheet in prepared['sheets']:
        path = folder / sheet['file']
        assert sha(path) == prepared['sheet_sha256'][sheet['file']]
        with Image.open(path) as img:
            assert img.size == (1072, 1516)
            array = np.array(img.convert('RGB'))
        for cell in sheet['cells']:
            with Image.open(ROOT / cell['source']) as image:
                original = np.array(image.convert('RGB'))
            assert original.shape == (256, 256, 3)
            x, y = cell['xy']
            assert np.array_equal(array[y:y+256, x:x+256], original)
            assert hashlib.sha256(original.tobytes()).hexdigest() == cell['pixel_sha256']
            cells += 1
    assert cells == 200
    assert prepared['groups']['all']['identical_PNGs'] == 0
    assert abs(prepared['groups']['degraded']['median_raw_correction_byte_units'] - 1.023144488410973) < 1e-12
    capacity = arithmetic['capacity_arithmetic_at_stopped50']
    assert capacity['preservation_failures'] == [] and not capacity['necessary_capacity_pass']
    assert capacity['degraded_feature_MSE_relative_gain'] == stop['relative_feature_error_gain']
    decision = read(ROOT / 'outputs/cctv_dgp_post_v27_architecture_review_v1/user_architecture_decision.json')
    assert m['architecture_decision'] in json.dumps(decision)
    assert not m['architecture_decision_pending'] and not m['next_training_recipe_created']
    assert m['V27_early_stop_failed'] and m['V27_updates'] == 50 and not m['unchanged_rerun_permitted']
    assert not m['VM_training_launched_by_assistant'] and m['VM_storage_maintenance_performed']
    assert m['goal_status'] == 'active' and not m['goal_complete']
    for item in (m, imported, audit, prepared, review, arithmetic):
        assert not item['app_promotion'] and not item['goal_complete']
    assert not m['reserved_final_viewed'] and not m['independent_final_review_complete']
    assert not review['native_or_reserved_used'] and not review['independent_final_review']
    result = {'complete': True, 'date': '2026-10-06', 'checker_sha256': sha(Path(__file__)),
              'milestone_sha256': sha(milestone_path), 'new_bindings_verified': current,
              'cleanup60_prior309_deeper66_692_299_697_513_bindings_verified': counts,
              'original_document_bodies_preserved': 5, 'app22_bindings_verified': 22,
              'return_members_verified': 631, 'reviewed_cases_verified': 50, 'exact_sheet_cells_verified': cells,
              'quality_failure_preserved': True, 'architecture_decision_preserved': m['architecture_decision'],
              'local_neural_or_gradient_calls': 0, 'optimizer_updates': 0, 'VM_actions': False,
              'app_promotion': False, 'goal_status': 'active', 'goal_complete': False,
              'seconds': time.monotonic() - started}
    with receipt.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
