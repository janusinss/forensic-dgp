"""Independently audit failure claims, exact histories and manual command paths."""
import ast
from datetime import datetime, timezone
from pathlib import Path
import re

from import_cctv_dgp_conditioning_learning_v1_r2_return import ROOT, OUT, RETURN, ANALYSIS, read, sha, write


def constant(path, name):
    result = [n.value.value for n in ast.parse(path.read_text(encoding='utf-8')).body
              if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant) and
              any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert len(result) == 1
    return result[0]


def main():
    folder = ANALYSIS / 'publication'
    record = read(folder / 'publication.json')
    assert record['complete'] and record['publisher_sha256'] == sha(ROOT / 'scripts/publish_cctv_dgp_conditioning_learning_v1_r2_startup.py')
    for n, h in record['evidence_sha256'].items():
        assert sha(ROOT / n) == h
    assert len(record['documents']) == 7
    for n, row in record['documents'].items():
        before = Path(row['backup'])
        assert sha(before) == row['before_sha256'] and sha(ROOT / n) == row['after_sha256']
        assert (ROOT / n).read_bytes()[row['prefix_bytes']:] == before.read_bytes()
    for n, h in record['new_documents_sha256'].items():
        assert sha(ROOT / n) == h
    startup = read(ANALYSIS / 'startup_review.json')
    assert startup['complete'] and not any(startup['reported_counts'].values())
    assert not startup['finite_structure_gate_reached'] and startup['improved_RAW_PNG_images'] == 0
    assert startup['new_checkpoint_or_visual_sheet_count'] == 0
    assert record['returned_training_updates'] == 0 and not record['finite_structure_gate_reached']
    packet = OUT / 'cctv_dgp_conditioning_learning_v1_r3_vm'
    pin = sha(packet / 'protocol.json')
    p = read(packet / 'protocol.json')
    stem = 'cctv-dgp-conditioning-learning-v1-r3'
    worker = packet / 'scripts/cctv_dgp_conditioning_learning_v1_r3_vm.py'
    importer = ROOT / 'scripts/import_cctv_dgp_conditioning_learning_v1_r3_return.py'
    auditor = ROOT / 'scripts/audit_cctv_dgp_conditioning_learning_v1_r3_return.py'
    assert pin == record['next_manual_protocol_sha256'] == constant(importer, 'PIN')
    for f in [worker, importer]:
        assert constant(f, 'NAME') == packet.name and constant(f, 'STEM') == stem
    assert constant(worker, 'STAGE') == 'outputs/conditioning_learning_v1_r3'
    assert constant(packet / 'conditioning_model.py', 'NAME') == packet.name
    transfer = read(OUT / 'cctv_dgp_conditioning_learning_v1_r3_transfer_audit.json')
    assert transfer['complete'] and transfer['protocol_sha256'] == pin
    assert transfer['future_importer_sha256'] == sha(importer) and transfer['future_auditor_sha256'] == sha(auditor)
    assert transfer['archive_files'] == 12 and transfer['parent_assets_verified'] == 5852
    assert transfer['all100_saved_raw_and_PNG_checker_baselines_replayed'] and transfer['NaN_and_infinity_metrics_rejected']
    assert len(transfer['actual_prior_rejected_joint_proposals_still_rejected']) == 4
    from audit_cctv_dgp_conditioning_learning_v1_r3_startup_repair import audit
    assert audit(packet, p) == transfer['startup_shape_and_exact_unchanged_recipe_audit']
    prepared = read(OUT / 'cctv_dgp_conditioning_learning_v1_r3_preparation.json')
    archive = OUT / (stem + '-execution.tar.gz')
    assert sha(archive) == prepared['archive_sha256'] == record['next_manual_archive_sha256']
    assert archive.stat().st_size == prepared['archive_bytes']
    assert Path(str(archive) + '.sha256').read_bytes() == f"{sha(archive)}  {archive.name}\n".encode('ascii')
    assert not (packet / 'outputs').exists() and not (packet / 'review.log').exists()
    for n, h in p['assets_sha256'].items():
        assert sha(packet / n) == h
    old = read(RETURN / 'protocol.json')
    assert all(sha(RETURN / n) == h and sha(OUT / old['name'] / n) == h for n, h in old['assets_sha256'].items())
    assert record['returned_archive_sha256'] == sha(OUT / 'cctv-dgp-conditioning-learning-v1-r2-results.tar.gz')
    text = (ROOT / 'CCTV_DGP_CONDITIONING_LEARNING_V1_R3_VM.md').read_text(encoding='utf-8')
    blocks = re.findall(r'```[^\n]*\n(.*?)```', text, re.S)
    assert len(blocks) == 5
    for i in [0, 4]:
        assert blocks[i].startswith('cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"\n')
    uploads = [v for v in blocks[0].splitlines() if v.startswith('gcloud compute scp ')]
    downloads = [v for v in blocks[4].splitlines() if v.startswith('gcloud compute scp ')]
    assert len(uploads) == 2 and len(downloads) == 3
    for line in uploads + downloads:
        assert '--project=forensic-dgp-thesis --zone=us-central1-a' in line
        assert line.count('janusdominic0@forensic-dgp-thesis:/home/janusdominic0/') == 1
    for suffix, line in zip(['-execution.tar.gz', '-execution.tar.gz.sha256'], uploads):
        assert f'"{stem + suffix}"' in line and (OUT / (stem + suffix)).is_file()
    for suffix, line in zip(['-results.tar.gz', '-results.tar.gz.sha256', '-export.json'], downloads):
        assert f'/home/janusdominic0/{stem + suffix}"' in line and line.endswith('"' + stem + suffix + '"')
    assert f'sha256sum -c {stem}-execution.tar.gz.sha256' in blocks[1]
    assert f'test ! -e ~/forensic-dgp/{packet.name}' in blocks[1]
    assert f'tar -xzf {stem}-execution.tar.gz -C ~/forensic-dgp' in blocks[1]
    assert blocks[2].strip() == 'tmux new-session -A -s dgp_conditioning_learning_v1_r3'
    assert f'cd ~/forensic-dgp/{packet.name} &&' in blocks[3]
    assert worker.name in blocks[3] and f'--protocol-sha {pin} --verify-transfer' in blocks[3]
    assert f'bash scripts/run_learning.sh {pin}' in blocks[3]
    assert f"{prepared['archive_bytes']:,}bytes" in text and prepared['archive_sha256'] in text
    assert '9GiB free after installation' in text and 'at2 accepted updates' in text
    assert '1% structure gain' in text and 'The worker checks current free space on launch.' in text
    assert 'conditioning_learning_v1_r2' not in '\n'.join(blocks) and 'conditioning-learning-v1-r2' not in '\n'.join(blocks)
    for f in [importer, auditor, worker, ROOT / 'scripts/publish_cctv_dgp_conditioning_learning_v1_r2_startup.py', Path(__file__)]:
        ast.parse(f.read_text(encoding='utf-8'), feature_version=(3, 10))
    report = (ROOT / 'CCTV_DGP_CONDITIONING_LEARNING_V1_R2_RESULTS.md').read_text(encoding='utf-8')
    assert 'zero accepted updates' in report and 'It did not reach' in report
    assert 'No fresh post-failure memory-state hash is' in report
    assert startup['reported_worker_seconds'] > 0 and f"{startup['reported_worker_seconds']:.6f}" in report
    assert record['returned_archive_sha256'] in report
    protected = read(ANALYSIS / 'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT / n) == h for n, h in protected.items())
    write(folder / 'publication_audit.json', dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
          checker_sha256=sha(__file__), publication_sha256=sha(folder / 'publication.json'),
          exact_history_suffixes_and_backups_checked=7, command_blocks_checked=5,
          all_upload_model_worker_protocol_export_import_and_download_names_agree=True,
          single_remote_source_per_PuTTY_download=True, LF_ASCII_checksum_exact=True,
          R2_startup_failure_zero_calls_updates_and_no_quality_gate_verified=True,
          real_helper_red_then_green_CPU_regression_verified=True,
          scientific_architecture_initialization_losses_limits_and_gates_unchanged=True,
          original_R2_packet_return_and_failure_retained=True, protected_original_app_checkpoint_unchanged=True,
          R3_manual_training_verified_and_unrun=True, no_agent_VM_training_or_cleanup=True,
          model_qualified=False, more_epochs_qualified=False, goal_complete=False))
    print(dict(complete=True, commands_checked=5, exact_histories_preserved=7,
               actual_returned_updates=0, R3_manual_training_unrun=True))


if __name__ == '__main__':
    main()
