"""Audit current return claims, retained histories and the unrun manual R4 guide."""
import ast
from collections import Counter
from datetime import datetime, timezone
import math
from pathlib import Path
import re
import tarfile

from import_cctv_dgp_conditioning_learning_v1_r3_return import (
    ROOT, OUT, RETURN, ANALYSIS, PIN, read, sha, write,
)
from audit_cctv_dgp_conditioning_learning_v1_r4_loss_forward_repair import audit


def constant(path, name):
    matches = [n.value.value for n in ast.parse(path.read_text(encoding='utf-8')).body
               if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
               and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert len(matches) == 1, (path, name)
    return matches[0]


def check_forward_rows(red, green):
    assert not red['complete'] and green['complete']
    assert red['real_saved_loss_checks'] == green['real_saved_loss_checks'] == 600
    assert len(red['rows']) == len(green['rows']) == 600
    assert len(red['failures']) == 1 and not green['failures']
    expected = Counter((stage, term) for stage in ['raw', 'png']
                       for term in ['MSE', 'SSIM_loss', 'landmark_structure'])
    by_case = {}
    for old, new in zip(red['rows'], green['rows']):
        for key in ['id', 'source', 'profile', 'reference', 'begin', 'stage', 'term',
                    'scientific_saved_value', 'retained_alignment_bound']:
            assert old[key] == new[key], (key, old['id'])
        by_case.setdefault(new['id'], Counter())[(new['stage'], new['term'])] += 1
        for row in [old, new]:
            values = [row[k] for k in ['utility_value', 'scientific_saved_value',
                                      'absolute_difference', 'retained_alignment_bound']]
            assert all(math.isfinite(v) for v in values)
            delta = abs(row['utility_value'] - row['scientific_saved_value'])
            assert delta == row['absolute_difference']
            assert row['pass_alignment'] == (delta <= row['retained_alignment_bound'])
            assert row['retained_alignment_bound'] == (1e-6 if row['term'] == 'SSIM_loss' else 1e-13)
        assert new['pass_alignment']
    assert len(by_case) == 100 and all(counts == expected for counts in by_case.values())
    failed = [row for row in red['rows'] if not row['pass_alignment']]
    assert failed == red['failures']
    assert failed[0]['id'] == 'v9_tr_asian_00180_blur_lr24'
    assert failed[0]['stage'] == 'raw' and failed[0]['term'] == 'SSIM_loss'
    assert failed[0]['absolute_difference'] > 1e-6
    assert red['checker_sha256'] == green['checker_sha256'] == sha(
        ROOT / 'scripts/check_cctv_dgp_conditioning_loss_forward_v1.py')
    assert red['local_neural_gradient_optimizer_parameter_displacement_calls'] == 0
    assert green['local_neural_gradient_optimizer_parameter_displacement_calls'] == 0


def main():
    folder = ANALYSIS / 'publication'
    record = read(folder / 'publication.json')
    publisher = ROOT / 'scripts/publish_cctv_dgp_conditioning_learning_v1_r3_results.py'
    assert record['complete'] and record['publisher_sha256'] == sha(publisher)
    assert len(record['documents']) == 6 and len(record['new_documents_sha256']) == 2
    for name, row in record['documents'].items():
        backup = Path(row['backup'])
        assert backup.resolve().is_relative_to(folder.resolve())
        assert sha(backup) == row['before_sha256'] and sha(ROOT / name) == row['after_sha256']
        assert (ROOT / name).read_bytes()[row['prefix_bytes']:] == backup.read_bytes()
    for name, expected_sha in {**record['new_documents_sha256'], **record['evidence_sha256']}.items():
        assert sha(ROOT / name) == expected_sha, name

    review = read(ANALYSIS / 'loss_alignment_review.json')
    failure = read(ANALYSIS / 'failure_audit.json')
    visual = read(ANALYSIS / 'visual_review.json')
    assert review['complete'] and failure['complete'] and visual['complete']
    assert review['reported_counts'] == dict(B=34, recognizer=67, gradient_queries=420,
                                           backwards=0, accepted_updates=0, parameter_proposals=0)
    assert failure['actual_update_count_reported'] == record['returned_training_updates'] == 0
    assert failure['completed_snapshots_audited'] == ['initial']
    assert not failure['gradient_audits'] and failure['completed_trials'] == 0
    assert review['complete_gradient_rounds'] == 0
    assert review['actual_failed_GPU_scalar_not_exported']
    assert review['inferred_query'] == 4 and review['inferred_stopped_stage'] == 'RAW'
    assert review['inferred_stopped_batch']['begin'] == 1970
    assert review['all420_neural_queries_not_independently_reexecuted']
    for row in review['retained_partial_gradient_samples']:
        assert sha(RETURN / row['file']) == row['sha256']
        assert row['elements'] == 2580864 and math.isfinite(row['norm']) and row['norm'] > 0
    assert len(review['retained_partial_gradient_samples']) == 4
    assert visual['all20_sheets_and300_cells_directly_viewed']
    assert visual['original_DGP_initial_controls_only'] and visual['new_trained_outputs'] == 0
    assert not visual['native_CCTV_used'] and not visual['independent_final_review']
    assert len(visual['sheets']) == 20 and sum(row['cells'] for row in visual['sheets']) == 300
    assert len({row['file'] for row in visual['sheets']}) == 20
    assert {row['file']: row['sha256'] for row in visual['sheets']} == {
        row['file']: row['sha256'] for row in failure['all_planned_visual_sheets']}
    for row in visual['sheets']:
        assert sha(ROOT / row['file']) == row['sha256']
        assert row['all15_cells_directly_viewed'] and row['requested_image_detail'] == 'original'
        assert row['observations'] and not row['candidate_after_training']
    red, green = [read(ANALYSIS / n) for n in ['loss_forward_red.json', 'loss_forward_green.json']]
    check_forward_rows(red, green)
    assert review['failed_CPU_row'] == red['failures'][0]
    assert record['returned_protocol_sha256'] == sha(RETURN / 'protocol.json') == PIN
    returned_archive = OUT / 'cctv-dgp-conditioning-learning-v1-r3-results.tar.gz'
    assert record['returned_archive_sha256'] == sha(returned_archive)
    old = read(RETURN / 'protocol.json')
    for name, expected_sha in old['assets_sha256'].items():
        assert sha(RETURN / name) == expected_sha == sha(OUT / old['name'] / name)

    packet = OUT / 'cctv_dgp_conditioning_learning_v1_r4_vm'
    stem = 'cctv-dgp-conditioning-learning-v1-r4'
    p = read(packet / 'protocol.json')
    pin = sha(packet / 'protocol.json')
    worker = packet / 'scripts/cctv_dgp_conditioning_learning_v1_r4_vm.py'
    importer = ROOT / 'scripts/import_cctv_dgp_conditioning_learning_v1_r4_return.py'
    auditor = ROOT / 'scripts/audit_cctv_dgp_conditioning_learning_v1_r4_return.py'
    assert pin == record['next_manual_protocol_sha256'] == constant(importer, 'PIN')
    for path in [worker, importer]:
        assert constant(path, 'NAME') == packet.name and constant(path, 'STEM') == stem
    assert constant(worker, 'STAGE') == 'outputs/conditioning_learning_v1_r4'
    assert constant(packet / 'conditioning_model.py', 'NAME') == packet.name
    transfer = read(OUT / 'cctv_dgp_conditioning_learning_v1_r4_transfer_audit.json')
    assert transfer['complete'] and transfer['protocol_sha256'] == pin
    assert transfer['future_importer_sha256'] == sha(importer)
    assert transfer['future_auditor_sha256'] == sha(auditor)
    assert transfer['archive_files'] == 14 and transfer['parent_assets_verified'] == 5852
    assert transfer['all100_saved_raw_and_PNG_checker_baselines_replayed']
    assert transfer['NaN_and_infinity_metrics_rejected']
    assert len(transfer['actual_prior_rejected_joint_proposals_still_rejected']) == 4
    assert audit(packet, p) == transfer['loss_forward_and_unchanged_recipe_audit']
    prepared = read(OUT / 'cctv_dgp_conditioning_learning_v1_r4_preparation.json')
    archive = OUT / (stem + '-execution.tar.gz')
    archive_sha = sha(archive)
    assert archive_sha == prepared['archive_sha256'] == record['next_manual_archive_sha256']
    assert archive.stat().st_size == prepared['archive_bytes'] == transfer['archive_bytes']
    assert Path(str(archive) + '.sha256').read_bytes() == f'{archive_sha}  {archive.name}\n'.encode('ascii')
    bindings = {**p['assets_sha256'], 'protocol.json': pin}
    assert len(bindings) == 14
    for name, expected_sha in bindings.items():
        assert sha(packet / name) == expected_sha
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        assert len(members) == len(bindings)
        assert {m.name for m in members} == {packet.name + '/' + n for n in bindings}
        import hashlib
        for member in members:
            assert member.isfile() and not member.issym() and not member.islnk()
            payload = tar.extractfile(member).read()
            assert hashlib.sha256(payload).hexdigest() == bindings[member.name[len(packet.name) + 1:]]
    assert not (packet / 'outputs').exists() and not (packet / 'review.log').exists()

    text = (ROOT / 'CCTV_DGP_CONDITIONING_LEARNING_V1_R4_VM.md').read_text(encoding='utf-8')
    blocks = re.findall(r'```[^\n]*\n(.*?)```', text, re.S)
    assert len(blocks) == 5
    for index in [0, 4]:
        assert blocks[index].startswith('cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"\n')
    uploads = [v for v in blocks[0].splitlines() if v.startswith('gcloud compute scp ')]
    downloads = [v for v in blocks[4].splitlines() if v.startswith('gcloud compute scp ')]
    assert len(uploads) == 2 and len(downloads) == 3
    for line in uploads + downloads:
        assert '--project=forensic-dgp-thesis --zone=us-central1-a' in line
        assert line.count('janusdominic0@forensic-dgp-thesis:/home/janusdominic0/') == 1
    for suffix, line in zip(['-execution.tar.gz', '-execution.tar.gz.sha256'], uploads):
        assert f'"{stem + suffix}"' in line and (OUT / (stem + suffix)).is_file()
    for suffix, line in zip(['-results.tar.gz', '-results.tar.gz.sha256', '-export.json'], downloads):
        assert f'/home/janusdominic0/{stem + suffix}"' in line
        assert line.endswith('"' + stem + suffix + '"')
    assert f'sha256sum -c {stem}-execution.tar.gz.sha256' in blocks[1]
    assert 'test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv' in blocks[1]
    assert 'test -d ~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm' in blocks[1]
    assert f'test ! -e ~/forensic-dgp/{packet.name}' in blocks[1]
    assert f'tar -xzf {stem}-execution.tar.gz -C ~/forensic-dgp' in blocks[1]
    assert blocks[2].strip() == 'tmux new-session -A -s dgp_conditioning_learning_v1_r4'
    assert f'cd ~/forensic-dgp/{packet.name} &&' in blocks[3]
    assert worker.name in blocks[3]
    assert '--parent-root ../cctv_dgp_bank_photometric_guard_v1_vm' in blocks[3]
    assert f'--protocol-sha {pin} --verify-transfer' in blocks[3]
    assert f'bash scripts/run_learning.sh {pin}' in blocks[3]
    assert f"{prepared['archive_bytes']:,}bytes" in text and archive_sha in text
    assert '9GiB free after installation' in text and 'At most two accepted training updates' in text
    assert '1% structure gain globally and per-source on raw/PNG' in text
    assert 'R3' not in '\n'.join(blocks) and 'v1_r3' not in '\n'.join(blocks)
    bash = read(OUT / 'cctv_dgp_conditioning_learning_v1_r4_bash_syntax.json')
    assert bash['complete']
    for path in [importer, auditor, worker, publisher, Path(__file__)]:
        ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 10))
    report = (ROOT / 'CCTV_DGP_CONDITIONING_LEARNING_V1_R3_RESULTS.md').read_text(encoding='utf-8')
    for phrase in ['115.99s', 'zero accepted updates/epochs', 'All420', 'all600 values pass',
                   'CPU reproduction', 'original failed GPU scalar was not exported',
                   'No fresh', 'post-failure RAM-state hash', 'two processing stops']:
        assert phrase in report, phrase
    protected = read(ANALYSIS / 'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT / name) == expected_sha for name, expected_sha in protected.items())
    assert record['actual_learning_worker_processing_stops_since_activation'] == 2
    assert record['next_training_pilot_unrun'] and not record['model_qualified'] and not record['goal_complete']
    write(folder / 'publication_audit.json', dict(
        complete=True, UTC=datetime.now(timezone.utc).isoformat(), checker_sha256=sha(__file__),
        publication_sha256=sha(folder / 'publication.json'), exact_history_suffixes_and_backups_checked=6,
        command_blocks_checked=5, new_archive_files_and_names_checked=14,
        all_upload_worker_protocol_export_import_download_names_agree=True,
        single_remote_source_per_PuTTY_download=True, LF_ASCII_checksum_exact=True,
        returned_420_queries_zero_updates_before_structure_gate_verified=True,
        original_CPU_mismatch_and_all600_corrected_loss_values_verified=True,
        GPU_failure_scalar_unknown_and_partial_gradients_not_neurally_replayed=True,
        all20_visual_sheet_hashes_and300_direct_review_cell_records_checked=True,
        scientific_architecture_initialization_limits_losses_and_quality_gates_unchanged=True,
        original_R3_packet_return_and_failure_retained=True,
        protected_original_app_checkpoint_unchanged=True, R4_manual_training_verified_and_unrun=True,
        no_agent_VM_training_or_cleanup=True, model_qualified=False, goal_complete=False))
    print(dict(complete=True, command_blocks_checked=5, exact_histories_preserved=6,
               saved_loss_checks=600, visual_sheets=20, returned_updates=0,
               R4_manual_training_unrun=True))


if __name__ == '__main__':
    main()
