"""Independent milestone closure: immutable return, visual ledger, docs and assets."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/cctv_dgp_multiscale_return_milestone_v1'
PACKET = ROOT/'outputs/cctv_dgp_multiscale_calibration_vm_v1'
RETURNED = ROOT/'outputs/cctv_dgp_multiscale_calibration_vm_v1_return'
REVIEW = ROOT/'outputs/cctv_dgp_multiscale_calibration_v1_return_review_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_bindings(root, bindings):
    for name,digest in bindings.items():
        path = (root/name).resolve()
        assert path.is_relative_to(root.resolve()) and path.is_file() and not (root/name).is_symlink()
        assert sha(path) == digest, name


def main():
    start = time.monotonic()
    publication = read(OUT/'publication.json')
    plan = read(OUT/'publication_plan.json')
    assert publication['complete'] and publication['publication_plan_sha256'] == sha(OUT/'publication_plan.json')
    assert not (OUT/'independent_closure_audit.json').exists()
    verify_bindings(ROOT, plan['evidence_bindings'])
    verify_bindings(ROOT, publication['documents'])
    for name,binding in publication['backups'].items():
        current = (ROOT/name).read_bytes()
        original = (ROOT/binding['backup']).read_bytes()
        assert hashlib.sha256(original).hexdigest() == binding['original_sha256']
        assert hashlib.sha256(current).hexdigest() == binding['updated_sha256']
        assert current[binding['prefix_bytes']:] == original
        if name.endswith('_VM.md'):
            assert current.startswith(b'**CLOSED') and b'Do not rerun or resume' in current[:binding['prefix_bytes']]
        else:
            assert b'all12 treatments rejected' in current[:binding['prefix_bytes']]
            assert b'Full goal incomplete' in current[:binding['prefix_bytes']]
    protected = read(ROOT/'outputs/cctv_dgp_multiscale_archive_cleanup_v1/local_protected_sha256.json')
    assert len(protected) == 346
    verify_bindings(ROOT, protected)
    assert sha(ROOT/'scripts/audit_cctv_dgp_multiscale_calibration_return_v1.py') == 'a2e92da64ae224b1ec2f5b0d3bfaec0582e16e300c85f22b0d1c8f3124bbe7c9'
    p = read(PACKET/'protocol.json')
    assert sha(PACKET/'protocol.json') == sha(RETURNED/'protocol.json') == 'c0239c2eebab3891cd4f42d34b93a5c7c2b07c8597bf93247df2cc16239dd491'
    assert len(p['assets_sha256']) == 251
    verify_bindings(PACKET, p['assets_sha256'])
    exported = read(RETURNED/'export_manifest.json')
    assert exported['complete'] and len(exported['files_sha256']) == 2276
    returned_sources = {name:digest for name,digest in p['assets_sha256'].items() if name.endswith('.py')}
    assert len(returned_sources) == 21
    assert {name for name in p['assets_sha256'] if name in exported['files_sha256']} == set(returned_sources)
    verify_bindings(RETURNED, returned_sources)
    verify_bindings(RETURNED, exported['files_sha256'])
    assert sha(ROOT/'outputs/cctv-dgp-multiscale-calibration-v1-results.tar.gz') == '61cfd317f18b0fd99b6a05a48c03b67c5fbf42e751c6b31cef9c3762de511769'
    assert (ROOT/'outputs/cctv-dgp-multiscale-calibration-v1-results.tar.gz').stat().st_size == 1639351615
    for arm in p['arms']:
        quality = read(RETURNED/'outputs'/arm['id']/'quality_gate.json')
        assert not quality['sampled_capacity_pass'] and not quality['failed_state_resume_allowed']
        assert all(not gate['pass'] for gate in quality['comparisons'].values())
    gallery = read(REVIEW/'gallery/gallery_manifest.json')
    ledger = read(REVIEW/'visual_review.json')
    assert len(gallery['pages']) == ledger['planned_sheets'] == ledger['viewed_sheets'] == 64
    assert len(gallery['cells']) == ledger['exact_saved_PNG_cells'] == 2280
    assert ledger['gallery_manifest_sha256'] == sha(REVIEW/'gallery/gallery_manifest.json')
    assert len(ledger['batch_bindings']) == 16
    verify_bindings(ROOT, ledger['batch_bindings'])
    pages = {row['name']:row for row in gallery['pages']}
    seen = set()
    for observation in ledger['observations']:
        page = pages[observation['page']]
        assert observation['page'] not in seen and observation['explicitly_viewed']
        assert observation['page_sha256'] == page['sha256'] == sha(ROOT/page['file'])
        assert observation['kind'] == page['kind'] and observation['case_ids'] == page['case_ids']
        assert observation['all_visible_features_considered'] and not observation['model_qualification']
        seen.add(observation['page'])
    assert seen == set(pages)
    assert sum(page['kind']=='paired_TRAIN' for page in pages.values()) == 40
    assert sum(page['kind']=='unpaired_native_DEV' for page in pages.values()) == 24
    assert set(ledger['paired_TRAIN_case_ids']) == {c['id'] for c in p['cases']}
    assert set(ledger['native_unpaired_DEV_case_ids']) == {c['id'] for c in p['native_development']}
    assert not ledger['full_goal_independent_final_review_complete']
    color = read(ROOT/'outputs/cctv_dgp_multiscale_color_detail_v1_r2/results.json')
    assert len(color['controls']) == 4 and color['corrected_PNG_detail_records'] == 400
    assert color['all_decisions_and_failure_locations_unchanged'] and color['exact_raw_and_other_PNG_values_preserved']
    for control in color['controls']:
        assert sha(ROOT/control['metrics_file']) == control['metrics_sha256']
        assert all(not comparison['pixel_only_requirements_pass'] for comparison in control['comparisons'].values())
    anchors = read(ROOT/'outputs/cctv_dgp_multiscale_active_anchor_v1/results.json')
    anchors_audit = read(ROOT/'outputs/cctv_dgp_multiscale_active_anchor_v1/independent_audit.json')
    assert len(anchors['contrasts']) == 6 and anchors_audit['complete'] and anchors_audit['signed_products_replayed'] == 1680
    assert anchors_audit['results_sha256'] == sha(ROOT/'outputs/cctv_dgp_multiscale_active_anchor_v1/results.json')
    for contrast in anchors['contrasts']:
        assert sha(ROOT/contrast['direction']) == contrast['direction_sha256']
        assert contrast['positive_clear_MSE_reference_slopes'] > 0 and contrast['positive_HF_reference_slopes'] > 0
    for field in ['local_optimizer_updates','local_gradient_queries']:
        assert publication[field] == 0
    assert not publication['new_VM_connection_or_launch']
    assert publication['next_recipe_is_design_not_executable_packet']
    assert publication['returned_checkpoint_app_selection_unchanged']
    assert publication['goal_status'] == 'active' and not publication['goal_complete'] and not publication['model_qualification']
    receipt = dict(complete=True, publication_sha256=sha(OUT/'publication.json'),
        checker_sha256=sha(Path(__file__)), protected_local_bindings_verified=346,
        local_packet_assets_verified=251, exact_exported_Python_source_assets_verified=21,
        returned_manifest_files_verified=2276,
        original_return_archive_verified=True, original_R1_checker_preserved=True,
        all12_failed_states_and_gate_decisions_preserved=True,
        visual_sheets_with_explicit_hash_bound_observations=64, exact_gallery_cells_previously_independently_checked=2280,
        docs_with_exact_prior_suffix_backup=4, new_report_hashes_verified=3,
        color_control_decisions_reverified=4, fixed_saved_anchor_contrasts_reverified=6,
        primary_visual_review_not_independent_final_review=True,
        next_design_not_executable=True, checkpoint_app_selection_unchanged=True,
        new_VM_training_or_diagnostic=False, local_optimizer_updates=0, local_gradient_queries=0,
        model_qualification=False, goal_complete=False, seconds=time.monotonic()-start)
    with (OUT/'independent_closure_audit.json').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(dict(complete=True, protected_bindings=346, immutable_return_files=2276,
               reviewed_sheets=64, new_report_hashes=3, seconds=time.monotonic()-start),flush=True)


if __name__=='__main__':
    main()
