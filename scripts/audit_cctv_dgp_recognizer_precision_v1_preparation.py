"""Independent transfer/guard audit; only hash-only worker CLI is executed."""
import ast
from datetime import datetime, timezone
import subprocess
import sys
import tarfile

from import_cctv_dgp_recognizer_precision_v1_return import ANALYSIS, OUT, INPUT, RETURN, SOURCE, BANK, ROOT, NAME, STEM, PIN, member_names, read, sha, write
from cctv_dgp_recognizer_precision_v1_contract import select_references


def main():
    packet = OUT / NAME; p = read(packet / 'protocol.json')
    prepared = read(OUT / 'cctv_dgp_recognizer_precision_v1_preparation.json')
    archive = OUT / (STEM + '-execution.tar.gz')
    assert prepared['complete'] and prepared['manual_UNRUN'] and not prepared['VM_launch']
    assert PIN == sha(packet / 'protocol.json') == prepared['protocol_sha256'] and len(PIN) == 64
    assert sha(archive) == prepared['archive_sha256'] and archive.stat().st_size == prepared['archive_bytes']
    checksum = (OUT / (archive.name + '.sha256')).read_bytes()
    assert checksum == (prepared['archive_sha256'] + '  ' + archive.name + '\n').encode('ascii')
    assert not RETURN.exists() and not ANALYSIS.exists() and not (packet / 'outputs').exists() and not (packet / 'review.log').exists()
    with tarfile.open(archive, 'r:gz') as tar:
        members = member_names(tar.getmembers())
        expected = {**p['assets_sha256'], 'protocol.json': PIN}
        assert set(members) == set(expected)
        import hashlib
        for n, m in members.items():
            assert sha(packet / n) == expected[n] and (packet / n).stat().st_size == m.size
            with tar.extractfile(m) as f:
                assert hashlib.sha256(f.read()).hexdigest() == expected[n]
    assert p['case_ids'] == [v for b in p['reference_blocks'] for v in b['case_ids']] and len(p['case_ids']) == 10
    sources = ['dataset/asian_faces', 'dataset/thumbnails128x128']
    selection = select_references(read(packet / 'evidence/boundary_review.json')['case_records'], read(BANK / 'protocol.json')['cases'], sources)
    assert selection == p['selected_references']
    assert [r['failure_case'] for r in selection] == ['v9_tr_asian_06782_lowlight_lr32', 'v9_tr_ffhq_59043_compound_lr24']
    assert len(p['reference_blocks']) * len(p['precisions']) * len(p['labels']) == p['forward_cap'] == 16
    assert 2 * 2 * 2 * 5 == p['gradient_cap'] == 40
    assert p['measurement_seconds'] == 360 and p['external_measurement_seconds'] == 480
    assert p['FP64_projection_cap_seconds'] == 300 and p['export_seconds'] == 180 and p['external_export_seconds'] == 210
    assert p['minimum_free_bytes'] == 3 * 1024**3 and p['disk_reserve_bytes'] == 1024**3 and p['output_byte_cap'] == 256 * 1024**2
    assert p['scientific_thresholds'] == read(BANK / 'protocol.json')['scientific_thresholds']
    assert p['original_restorer_checkpoint_sha256'] == '646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'
    assert p['FP64_diagnostic_only_not_official_scientific_metric'] and p['scientific_gates_unchanged']
    assert not p['scientific_and_existing_failure_bounds_relaxed'] and not p['app_promotion'] and not p['automatic_follow_on']
    for n in ['scripts/cctv_dgp_recognizer_precision_v1_vm.py', 'precision_contract.py', 'cctv_dgp_pilot.py', 'models/identity_loss.py']:
        ast.parse((packet / n).read_text(encoding='utf-8'), feature_version=(3, 10))
    text = (packet / 'scripts/cctv_dgp_recognizer_precision_v1_vm.py').read_text(encoding='utf-8')
    assert 'torch.cat([rgb, baseline, target])' in text and 'vectors = F.normalize(model.encoder(crop * 2 - 1), dim=1)' in text
    assert 'copy.deepcopy(original).double()' in text and 'assert roundtrip_state(models[\'FP64\']) == STATE' in text
    shell = (packet / 'scripts/run_precision.sh').read_bytes()
    assert b'\r' not in shell and b'480s' in shell and b'210s' in shell and b'--kill-after=30s' in shell
    assert b'test ! -e review.log' in shell and b'export_code=$?' in shell
    fixture = read(OUT / 'cctv_dgp_recognizer_precision_v1_fixture.json')
    assert fixture['complete'] and fixture['actual_pixel_projection_corruption_rejected'] and fixture['wrong_frozen_baseline_rejected'] and fixture['contradictory_activation_signs_rejected']
    assert fixture['return_auditor_sha256'] == sha(ROOT / 'scripts/audit_cctv_dgp_recognizer_precision_v1_return.py')
    guard = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/check_cctv_dgp_recognizer_precision_v1_guard.py')], capture_output=True, text=True, timeout=30)
    assert guard.returncode == 0, guard.stdout + guard.stderr
    command = [sys.executable, '-B', str(packet / 'scripts/cctv_dgp_recognizer_precision_v1_vm.py'),
        '--root', str(packet), '--input-root', str(INPUT), '--response-root', str(SOURCE), '--bank-root', str(BANK), '--protocol-sha', PIN, '--verify-transfer']
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    import json
    receipt = json.loads(result.stdout)
    assert receipt['complete'] and receipt['assets'] == len(p['assets_sha256']) and receipt['cases'] == 10
    assert receipt['actual_neural_or_gradient_calls'] == receipt['optimizer_updates'] == 0
    protected = read(OUT / 'cctv_dgp_appearance_input_response_v1_analysis/local_protected_before.json')['files_sha256']
    assert all(sha(ROOT / n) == h for n, h in protected.items())
    write(OUT / 'cctv_dgp_recognizer_precision_v1_transfer_audit.json', dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        checker_sha256=sha(__file__), protocol_sha256=PIN, archive_sha256=sha(archive), packet_bindings=len(expected),
        dependency_bindings=prepared['dependency_bindings'], hash_only_worker_receipt=receipt,
        hash_only_free_space_observation_is_Windows_not_VM=True, manual_guard_passed=True,
        exact10_case_selection_and16_forward40_gradient_caps=True, ASCII_LF_checksum_and_wrapper=True,
        corrected_control_baseline_and_full_FP64_normalization_checked=True, saved_array_fixture_passed=True,
        original_gates_and_protected_files_unchanged=True, manual_UNRUN=True, local_neural_or_training_calls=0,
        automatic_VM_launch=False, model_qualified=False, more_epochs_qualified=False, goal_complete=False))
    print(dict(complete=True, archive_bindings=len(expected), cases=10, actual_neural_calls=0, manual_UNRUN=True))


if __name__ == '__main__':
    main()
