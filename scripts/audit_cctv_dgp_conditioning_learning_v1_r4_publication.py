"""Audit published return claims, exact histories and the unrun design boundary."""
import ast
from collections import Counter
from datetime import datetime, timezone
import math
from pathlib import Path
import re

import numpy as np

from import_cctv_dgp_conditioning_learning_v1_r4_return import ROOT, OUT, RETURN, ANALYSIS, PIN, read, sha, write
from cctv_dgp_conditioning_learning_v1_r4_contract import decision


def main():
    folder = ANALYSIS / 'publication'
    record = read(folder / 'publication.json')
    publisher = ROOT / 'scripts/publish_cctv_dgp_conditioning_learning_v1_r4_results.py'
    assert record['complete'] and sha(publisher) == record['publisher_sha256']
    stop = read(folder / 'publication_stop.json')
    assert stop['exit_code'] == 1 and stop['master_document_changes_before_stop'] == 0
    backup = Path(stop['backup'])
    assert backup.resolve().is_relative_to(folder.resolve()) and sha(backup) == stop['original_publisher_sha256']
    restored = publisher.read_text(encoding='utf-8').replace("    (stage / 'before').mkdir(exist_ok=True)\n", '')
    restored = restored.replace("stage / 'before' / name, expected[name]", "stage / 'before', expected[name]")
    restored = restored.replace("stage / 'before' / used, expected[used]", "stage / 'before', expected[used]")
    assert restored == backup.read_text(encoding='utf-8')
    assert len(record['documents']) == 6
    for name, row in record['documents'].items():
        path = Path(row['backup'])
        assert path.resolve().is_relative_to(folder.resolve()) and path.name == name
        assert sha(path) == row['before_sha256'] and sha(ROOT / name) == row['after_sha256']
        assert (ROOT / name).read_bytes()[row['prefix_bytes']:] == path.read_bytes()
    assert len(record['new_documents_sha256']) == 2
    for name, digest in {**record['new_documents_sha256'], **record['evidence_sha256']}.items():
        assert sha(ROOT / name) == digest, name
    imported = read(ANALYSIS / 'import.json')
    audited = read(ANALYSIS / 'failure_audit.json')
    visual = read(ANALYSIS / 'visual_review.json')
    prediction = read(ANALYSIS / 'gradient_prediction_review.json')
    human = read(ANALYSIS / 'diagnostic_decision.json')
    assert all(v['complete'] for v in [imported, audited, visual, prediction, human])
    assert record['returned_protocol_sha256'] == imported['protocol_sha256'] == PIN == sha(RETURN / 'protocol.json')
    archive = OUT / 'cctv-dgp-conditioning-learning-v1-r4-results.tar.gz'
    assert sha(archive) == imported['archive_sha256'] == record['returned_archive_sha256']
    assert archive.stat().st_size == imported['archive_bytes'] == 1384730860
    assert imported['returned_file_bindings'] == 663 and imported['parent_file_bindings'] == 5852
    assert audited['actual_update_count_reported'] == record['returned_accepted_updates'] == 0
    assert audited['completed_trials'] == record['actual_rejected_trials'] == 3
    assert audited['accepted_trials_verified'] == [] and len(audited['gradient_audits']) == 1
    assert audited['failure']['counts'] == dict(B=100, recognizer=200, gradient_queries=640,
                                               backwards=0, accepted_updates=0, parameter_proposals=3)
    assert audited['failure']['message'] == 'No actual raw/PNG preserving conditioning step; retain stop'
    assert audited['failure']['no_post_failure_fresh_state_hash_claimed']
    assert audited['underlying_scientific_metric_replay_bound_unchanged'] == 1e-13
    assert audited['original_checker_sha256'] == sha(ROOT / 'scripts/audit_cctv_dgp_conditioning_learning_v1_r4_return.py')
    assert audited['checker_sha256'] == sha(ROOT / 'scripts/audit_cctv_dgp_conditioning_learning_v1_r4_return_r1.py')
    assert audited['receipt_helper_sha256'] == sha(ROOT / 'scripts/conditioning_decision_receipt_check_v1.py')
    green = read(ANALYSIS / 'receipt_check_green.json')
    assert green['complete'] and len(green['real_replayed_trials_verified']) == 3
    assert len(green['corruptions_rejected']) == 18 and not green['new_scientific_or_numeric_replay_tolerance_introduced']
    assert visual['all20_sheets_and600_cells_directly_viewed'] and len(visual['sheets']) == 20
    assert sum(v['cells'] for v in visual['sheets']) == 600
    assert not visual['independent_final_review'] and not visual['native_CCTV_used']
    assert not visual['automatic_or_assisted_completion_qualification']
    assert {v['file']: v['sha256'] for v in visual['sheets']} == {
        v['file']: v['sha256'] for v in audited['all_planned_visual_sheets']}
    for row in visual['sheets']:
        assert sha(ROOT / row['file']) == row['sha256'] and row['all30_cells_directly_viewed']
        assert row['requested_image_detail'] == 'original' and row['observations']

    stage = RETURN / 'outputs/conditioning_learning_v1_r4'
    assert not (stage / 'run_results.json').exists() and not list(stage.glob('accepted*'))
    baseline = read(stage / 'initial/snapshot.json')['groups']
    gradient = read(stage / 'round00/gradients.json')
    gram = np.asarray(gradient['certificate']['normalized_Gram'], np.float64)
    for name, matrix in [('RAW32', gram[:32, :32]), ('PNG32', gram[32:, 32:]), ('joint64', gram)]:
        eigen = np.linalg.eigvalsh(matrix)
        assert eigen[0] > 0
        values = prediction['eigensystems'][name]
        assert np.isclose(values['condition_number'], eigen[-1] / eigen[0], rtol=1e-12)
    correlations = np.diag(gram[:32, 32:])
    for key, value in [('minimum', correlations.min()), ('maximum', correlations.max()), ('mean', correlations.mean())]:
        assert prediction['paired_RAW_PNG_gradient_cosines'][key] == float(value)
    assert len(prediction['real_group_trial_comparisons']) == record['saved_group_trial_predictions_reviewed'] == 192
    ledger = Counter()
    report = (ROOT / 'CCTV_DGP_CONDITIONING_LEARNING_V1_R4_RESULTS.md').read_text(encoding='utf-8')
    for row in prediction['trials']:
        label = row['trial']
        saved = read(stage / label / 'snapshot.json')['groups']
        receipt = read(stage / label / 'decision.json')
        assert decision(baseline, baseline, saved) == receipt['decision']
        assert not receipt['decision']['accept_step'] and not receipt['decision']['one_percent_checkpoint_pass']
        for s in ['raw', 'png']:
            summary = row['detail_and_gates'][s]
            expected = receipt['decision']['stages'][s]
            assert summary['actual_global_detail_gain'] == expected['relative_feature_gain']
            assert summary['actual_preservation_failed_checks'] == len(expected['initial_preservation_failures'])
            assert f"{summary['actual_global_detail_gain'] * 100:.6f}%" in report
        for item in [v for v in prediction['real_group_trial_comparisons'] if v['trial'] == label]:
            ledger[(label, item['stage'])] += 1
            metric = item['metric']
            key = item['source'] + '/' + item['profile']
            field = {'MSE': 'MSE', 'SSIM_loss': 'SSIM', 'ArcFace_loss': 'ArcFace_observed_fixed',
                     'feature_MSE': 'landmark_high_frequency_MSE'}[metric]
            before, after = [g[item['stage']][key][field] for g in [baseline, saved]]
            if metric in ['SSIM_loss', 'ArcFace_loss']:
                before, after = 1 - before, 1 - after
            assert item['before_loss'] == before and item['after_loss'] == after
            assert item['actual_scientific_loss_delta'] == after - before
            assert item['predicted_delta_for_actual_rounded_parameter_change'] < 0
            assert math.isfinite(item['finite_minus_linear_prediction_residual'])
    assert len(ledger) == 6 and set(ledger.values()) == {32}
    assert human['human_answer'] == 'apply best approach' and len(human['three_stops_retained']) == 3
    assert human['saved_prediction_review_sha256'] == sha(ANALYSIS / 'gradient_prediction_review.json')
    design = (ROOT / 'CCTV_DGP_POST_R4_GRADIENT_DESIGN_REVIEW_V1.md').read_text(encoding='utf-8')
    assert 'proposed optimizer comparison' in design and 'runnable packet' in design
    assert 'No PNG' in design and 'measurement or acceptance gate may be removed or relaxed' in design
    assert 'https://arxiv.org/abs/1903.05662' in design
    assert 'https://docs.pytorch.org/docs/2.9/generated/torch.autograd.gradcheck.gradcheck.html' in design
    for name in record['new_documents_sha256']:
        text = (ROOT / name).read_text(encoding='utf-8')
        for target in re.findall(r'\]\(<(C:/[^>]+)>\)', text):
            assert Path(target).is_file(), target
    protected = read(ANALYSIS / 'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT / name) == digest for name, digest in protected.items())
    p = read(RETURN / 'protocol.json')
    for name, digest in p['assets_sha256'].items():
        assert sha(RETURN / name) == digest == sha(OUT / p['name'] / name)
    for path in [publisher, Path(__file__), ROOT / 'scripts/conditioning_decision_receipt_check_v1.py']:
        ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 10))
    write(folder / 'publication_audit.json', dict(
        complete=True, UTC=datetime.now(timezone.utc).isoformat(), auditor_sha256=sha(__file__),
        publication_sha256=sha(folder / 'publication.json'), exact_history_suffixes_and_backups_verified=6,
        publisher_path_stop_retained_and_inverse_diff_checked=True, all_local_evidence_links_verified=True,
        all20_sheets600_review_cells_hash_bound=True, actual_zero_updates_and_three_rejected_trials_verified=True,
        scientific_gate_decisions_and_192_prediction_actual_score_bindings_checked=True,
        original_R4_packet_worker_checker_and_failed_trials_preserved=True,
        protected_original_app_checkpoint_unchanged=True, RAW_only_design_explicitly_not_runnable_or_qualified=True,
        no_agent_VM_training_or_new_neural_model_calls=True, model_qualified=False, goal_complete=False))
    print(dict(complete=True, history_documents=6, reviewed_sheets=20,
               group_trial_score_bindings=192, accepted_updates=0, new_runnable_packet=False))


if __name__ == '__main__':
    main()
