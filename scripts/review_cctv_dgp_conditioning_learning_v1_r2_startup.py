"""Bind the returned failure to the sealed source and failed real utility test."""
from datetime import datetime, timezone
from import_cctv_dgp_conditioning_learning_v1_r2_return import ROOT, OUT, RETURN, ANALYSIS, PIN, read, sha, write


def main():
    imported = read(ANALYSIS / 'import.json')
    checked = read(ANALYSIS / 'failure_audit.json')
    fixture = read(ANALYSIS / 'pixel_fixture_red.json')
    p = read(RETURN / 'protocol.json')
    failure_path = RETURN / 'outputs/conditioning_learning_v1_r2/failure.json'
    failure = read(failure_path)
    assert imported['complete'] and checked['complete']
    assert imported['archive_sha256'] == 'a90ad5b133bec9dba10d66883300e43d1a2d50ab571e08cf27c85a4de601a12a'
    assert imported['archive_bytes'] == 589270 and imported['returned_file_bindings'] == 16
    assert imported['protocol_sha256'] == sha(RETURN / 'protocol.json') == PIN
    assert failure == checked['failure'] and failure['type'] == 'AssertionError'
    assert failure['progress']['stage'] == 'initialize'
    expected = dict(B=0, recognizer=0, gradient_queries=0, backwards=0, accepted_updates=0, parameter_proposals=0)
    assert failure['counts'] == failure['progress']['counts'] == expected
    assert 'delivered_surrogates(a,torch.from_numpy(probe).cuda())' in failure['traceback']
    assert 'raw.shape[1:] == (3, 256, 256)' in failure['traceback']
    assert failure['rollback_to_last_accepted_component_error'] is None
    assert not fixture['complete'] and fixture['type'] == 'AssertionError'
    assert fixture['actual_fixture_shape'] == [1, 3, 16, 16]
    assert fixture['protocol_sha256'] == PIN and fixture['worker_sha256'] == p['assets_sha256']['scripts/cctv_dgp_conditioning_learning_v1_r2_vm.py']
    assert fixture['helper_sha256'] == p['assets_sha256']['conditioning_losses.py']
    assert 'raw.shape[1:] == (3, 256, 256)' in fixture['traceback']
    stage = RETURN / 'outputs/conditioning_learning_v1_r2'
    assert {f.name for f in stage.iterdir()} == {'failure.json'}
    assert (RETURN / 'review_exit_code.txt').read_text().strip() == '1'
    assert sha(ROOT / 'scripts/check_cctv_dgp_conditioning_preflight_pixels_v1.py') == fixture['checker_sha256']
    protected = read(ANALYSIS / 'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT / n) == h for n, h in protected.items())
    write(ANALYSIS / 'startup_review.json', dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
          checker_sha256=sha(__file__), import_sha256=sha(ANALYSIS / 'import.json'),
          failure_audit_sha256=sha(ANALYSIS / 'failure_audit.json'), fixture_red_sha256=sha(ANALYSIS / 'pixel_fixture_red.json'),
          returned_failure_sha256=sha(failure_path), protocol_sha256=PIN,
          startup_location='run/canonical-pixel fixture before model construction',
          demonstrated_cause='The sealed NumPy fixture is 1x3x16x16; its unchanged helper requires 1x3x256x256.',
          narrowly_supported_fix='Use a 256x256 all-byte-level fixture; retain the helper and scientific gates.',
          reported_counts=expected, reported_worker_seconds=failure['seconds'],
          output_files_in_run_stage=['failure.json'], improved_RAW_PNG_images=0,
          new_checkpoint_or_visual_sheet_count=0, finite_structure_gate_reached=False,
          no_initial_or_post_failure_model_state_hash_claimed=True,
          CPU_shape_failure_reproduced_not_a_CUDA_numerical_reproduction=True,
          all_returned_packet_and_failure_hashes_retained=True, protected_app_unchanged=True,
          model_qualified=False, more_epochs_qualified=False, goal_complete=False))
    print(dict(complete=True, cause='16x16 preflight fixture', actual_updates=0,
               model_calls_reported=0, quality_gate_reached=False))


if __name__ == '__main__':
    main()
