"""Seal the demonstrated RAW-SSIM forward repair for human-manual L4 training."""
import ast
import copy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tarfile

from import_cctv_dgp_conditioning_learning_v1_r3_return import ROOT, OUT, RETURN, ANALYSIS, PARENT, PIN, read, sha, write

NAME = 'cctv_dgp_conditioning_learning_v1_r4_vm'
STEM = 'cctv-dgp-conditioning-learning-v1-r4'


def text_file(path, text):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)


def main():
    p = read(RETURN / 'protocol.json')
    review = read(ANALYSIS / 'loss_alignment_review.json')
    green = read(ANALYSIS / 'loss_forward_green.json')
    visual = read(ANALYSIS / 'visual_review.json')
    assert all(v['complete'] for v in [review, green, visual])
    assert not review['reported_counts']['accepted_updates'] and review['more_epochs_or_actual_structure_gate_not_reached']
    assert visual['all20_sheets_and300_cells_directly_viewed'] and green['real_saved_loss_checks'] == 600
    assert sha(RETURN / 'protocol.json') == PIN
    original_members = {**p['assets_sha256'], 'protocol.json': PIN}
    assert all(sha(RETURN / n) == h and sha(OUT / p['name'] / n) == h for n, h in original_members.items())
    assert all(sha(PARENT / n) == h for n, h in p['parent_required_sha256'].items())
    protected = read(ANALYSIS / 'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT / n) == h for n, h in protected.items())
    archived = {n: sha(OUT / n) for n in [
        'cctv-dgp-conditioning-learning-v1-r3-execution.tar.gz',
        'cctv-dgp-conditioning-learning-v1-r3-results.tar.gz',
        'cctv-dgp-conditioning-learning-v1-r3-results.tar.gz.sha256',
        'cctv-dgp-conditioning-learning-v1-r3-export.json']}
    dest = OUT / NAME
    assert not dest.exists(), 'Retain every sealed/used packet'
    dest.mkdir()
    old_import = '    from cctv_dgp_group_conflicts_v1_losses import pixel_and_structure_losses as raw_losses\n'
    new_import = '    from conditioning_raw_losses import pixel_and_structure_losses as raw_losses\n'
    assertion = "                        assert abs(float(value.detach())-expected_loss)<=tolerance,'Scientific/surrogate loss alignment differs'\n"
    diagnostic = """                        if abs(float(value.detach())-expected_loss)>tolerance:
                            write(out/'loss_alignment_failure.json',dict(round=round_index,
                                begin=block['begin'],reference=block['reference'],metric_stage=stage,
                                query=query,key=key,actual_forward_value=float(value.detach()),
                                expected_scientific_value=expected_loss,alignment_bound=tolerance,
                                absolute_difference=abs(float(value.detach())-expected_loss),
                                snapshot=current_snapshot['label'],candidate_state=round_state,
                                raw_helper_sha256=sha(root/'conditioning_raw_losses.py'),
                                PNG_helper_sha256=sha(root/'conditioning_losses.py')))
"""
    for name, h in p['assets_sha256'].items():
        source = RETURN / name
        target_name = name.replace('conditioning_learning_v1_r3', 'conditioning_learning_v1_r4')
        target = dest / target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name == 'scripts/cctv_dgp_conditioning_learning_v1_r3_vm.py':
            original = source.read_text(encoding='utf-8')
            assert original.count(old_import) == original.count(assertion) == 1
            text = original.replace('conditioning_learning_v1_r3', 'conditioning_learning_v1_r4')
            text = text.replace('cctv-dgp-conditioning-learning-v1-r3', STEM)
            text = text.replace(old_import, new_import).replace(assertion, diagnostic + assertion)
            ast.parse(text, feature_version=(3, 10))
            text_file(target, text)
            text_file(ROOT / 'scripts/cctv_dgp_conditioning_learning_v1_r4_vm.py', text)
            restored = text.replace('conditioning_learning_v1_r4', 'conditioning_learning_v1_r3').replace(STEM, 'cctv-dgp-conditioning-learning-v1-r3')
            restored = restored.replace(new_import, old_import).replace(diagnostic + assertion, assertion)
            assert restored == original
        elif name in ['conditioning_model.py', 'scripts/run_learning.sh']:
            original = source.read_text(encoding='utf-8')
            text = original.replace('conditioning_learning_v1_r3', 'conditioning_learning_v1_r4').replace('cctv-dgp-conditioning-learning-v1-r3', STEM)
            assert text != original
            text_file(target, text)
        else:
            with source.open('rb') as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst)
            assert sha(target) == h
    raw_helper = ROOT / 'scripts/cctv_dgp_conditioning_learning_v1_r4_raw_losses.py'
    assert sha(raw_helper) == green['raw_helper_sha256'] == review['corrected_helper_sha256']
    for source, target in [(raw_helper, dest / 'conditioning_raw_losses.py'),
                           (ANALYSIS / 'loss_alignment_review.json', dest / 'loss_alignment_review.json')]:
        with source.open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst)
        assert sha(source) == sha(target)
    q = copy.deepcopy(p)
    q.update(format='finite-conditioning-learning-v1-r4', name=NAME, UTC=datetime.now(timezone.utc).isoformat(),
             assets_sha256={f.relative_to(dest).as_posix(): sha(f) for f in dest.rglob('*') if f.is_file()},
             loss_forward_repair_of_protocol_sha256=PIN,
             loss_forward_repair_of_return_archive_sha256=read(ANALYSIS / 'import.json')['archive_sha256'],
             loss_alignment_review_sha256=sha(ANALYSIS / 'loss_alignment_review.json'),
             RAW_SSIM_exact_scientific_forward_with_explicit_approximate_float64_backward=True,
             RAW_SSIM_backward_graph_unchanged_from_R3=True,
             alignment_failure_exports_actual_scalar_case_stage_and_bound=True,
             repair_scope='Canonical RAW SSIM forward and failure scalar logging; model, derivative, thresholds and finite limits unchanged')
    excluded = {'format', 'name', 'UTC', 'assets_sha256', 'repair_scope',
                'loss_forward_repair_of_protocol_sha256', 'loss_forward_repair_of_return_archive_sha256',
                'loss_alignment_review_sha256', 'RAW_SSIM_exact_scientific_forward_with_explicit_approximate_float64_backward',
                'RAW_SSIM_backward_graph_unchanged_from_R3', 'alignment_failure_exports_actual_scalar_case_stage_and_bound'}
    assert {k: v for k, v in q.items() if k not in excluded} == {k: v for k, v in p.items() if k not in excluded}
    write(dest / 'protocol.json', q)
    pin = sha(dest / 'protocol.json')
    archive = OUT / (STEM + '-execution.tar.gz')
    assert not archive.exists()
    with tarfile.open(archive, 'x:gz', compresslevel=1) as tar:
        for f in sorted(f for f in dest.rglob('*') if f.is_file()):
            tar.add(f, arcname=NAME + '/' + f.relative_to(dest).as_posix(), recursive=False)
    text_file(Path(str(archive) + '.sha256'), sha(archive) + '  ' + archive.name + '\n')
    assert all(sha(RETURN / n) == h and sha(OUT / p['name'] / n) == h for n, h in original_members.items())
    assert all(sha(OUT / n) == h for n, h in archived.items())
    assert all(sha(ROOT / n) == h for n, h in protected.items())
    write(OUT / 'cctv_dgp_conditioning_learning_v1_r4_preparation.json', dict(complete=True,
          UTC=datetime.now(timezone.utc).isoformat(), preparer_sha256=sha(__file__), protocol_sha256=pin,
          archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
          packet_assets=len(q['assets_sha256']), parent_file_bindings=len(q['parent_required_sha256']),
          source_protocol_sha256=PIN, returned_loss_alignment_review_sha256=sha(ANALYSIS / 'loss_alignment_review.json'),
          activation_audit_sha256=q['activation_audit_sha256'],
          old_packet_and_return_archive_files_sha256=archived,
          old_packet_and_all_failures_retained=True,
          model_derivative_quality_thresholds_training_limits_and_initial_state_unchanged=True,
          corrected_RAW_helper_sha256=sha(raw_helper), all600_saved_RAW_PNG_loss_forward_checks_passed=True,
          local_neural_gradient_optimizer_or_parameter_displacement_calls=0, manual_pilot_unrun=True,
          prospective_transfer_and_return_audits_required=True, model_qualified=False, goal_complete=False))
    print(dict(complete=True, protocol_sha256=pin, archive_bytes=archive.stat().st_size, manual_unrun=True))


if __name__ == '__main__':
    main()
