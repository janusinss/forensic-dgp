"""Prepare a finite manual test of the demonstrated zero-penalty gradient defect."""
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import tarfile

from import_cctv_dgp_bank_loss_alignment_v1_r1_return import ROOT, OUT, RETURN, ANALYSIS, PARENT, PIN, read, sha, write

NAME = 'cctv_dgp_bank_hinge_boundary_v1_vm'
STEM = 'cctv-dgp-bank-hinge-boundary-v1'


def main():
    dest = OUT / NAME
    archive = OUT / (STEM + '-execution.tar.gz')
    receipt = OUT / 'cctv_dgp_bank_hinge_boundary_v1_preparation.json'
    assert not any(f.exists() for f in [dest, archive, receipt])
    evidence = [ANALYSIS / 'import.json', ANALYSIS / 'independent_audit.json',
                ANALYSIS / 'loss_direction_review.json', OUT / 'cctv_dgp_bank_hinge_boundary_v1_loss_audit.json']
    assert all(read(path)['complete'] for path in evidence)
    audit, review, loss_audit = [read(path) for path in evidence[1:]]
    assert audit['exact_control_layout_repair_confirmed_on_L4']
    assert review['group_summaries']['all']['initial']['scalar_zero_brightness_penalty_with_nonzero_gradient'] == 20
    assert loss_audit['exact_scalar_and_metadata_parity_all40']
    old = read(RETURN / 'protocol.json')
    assert sha(RETURN / 'protocol.json') == PIN
    for name, digest in old['parent_required_sha256'].items():
        assert sha(PARENT / name) == digest
    worker_path = RETURN / 'scripts/cctv_dgp_bank_loss_alignment_v1_r1_vm.py'
    assert sha(worker_path) == old['assets_sha256']['scripts/' + worker_path.name]
    source = worker_path.read_text(encoding='utf-8')
    changes = {
        "NAME='cctv_dgp_bank_loss_alignment_v1_r1_vm'":"NAME='" + NAME + "'",
        "STEM='cctv-dgp-bank-loss-alignment-v1-r1'":"STEM='" + STEM + "'",
        'from cctv_dgp_bank_photometric_guard_v1_loss import brightness_guard':
            'from cctv_dgp_bank_hinge_boundary_v1_loss import brightness_guard\n    from cctv_dgp_bank_photometric_guard_v1_loss import brightness_guard as retained_brightness_guard',
        "photo,photo_terms=brightness_guard(pred,baseline_tensor,target,mask,NORMALIZERS[0],10.,.2)":
            """photo,photo_terms=brightness_guard(pred,baseline_tensor,target,mask,NORMALIZERS[0],10.,.2)
                with torch.no_grad():
                    retained_photo,retained_terms=retained_brightness_guard(pred.detach(),baseline_tensor,target,mask,NORMALIZERS[0],10.,.2)
                assert torch.equal(photo.detach(),retained_photo) and photo_terms==retained_terms,'Scalar penalty changed; retain stop'
                assert Path(sys.modules['cctv_dgp_bank_hinge_boundary_v1_loss'].__file__).resolve()==(root/'cctv_dgp_bank_hinge_boundary_v1_loss.py').resolve()""",
        "gradient_rows.append(tuple(g.detach() for g in grads))":
            """gradient_rows.append(tuple(g.detach() for g in grads))
                if state_label=='initial':
                    assert float(photo.detach())==0 and all(torch.count_nonzero(g).item()==0 for g in gradient_rows[6]),'Inactive initial hinge has nonzero gradient; retain stop'""",
        "gradient_queries=9,seconds=elapsed,photometric_terms=photo_terms,":
            "gradient_queries=9,seconds=elapsed,photometric_terms=photo_terms,scalar_penalty_exactly_matches_retained_guard=True,\n                    inactive_at_zero_hinge_derivative_declared=True,",
        "all40_frozen_raw_controls_retained=True,input_layout_matches_parent=True":
            "inactive_initial_guard_zero_gradient_all20_verified=True,scalar_penalty_parity_all40_verified=True,all40_frozen_raw_controls_retained=True,input_layout_matches_parent=True",
    }
    for before, after in changes.items():
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    tree = ast.parse(source)
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr in ['step', 'backward', 'Adam', 'AdamW', 'SGD'] for n in ast.walk(tree))
    dest.mkdir()
    (dest / 'scripts').mkdir()
    worker_name = 'cctv_dgp_bank_hinge_boundary_v1_vm.py'
    (dest / 'scripts' / worker_name).write_text(source, encoding='utf-8', newline='\n')
    helper = 'cctv_dgp_bank_loss_alignment_v1.py'
    (dest / helper).write_bytes((RETURN / helper).read_bytes())
    corrected_name = 'cctv_dgp_bank_hinge_boundary_v1_loss.py'
    assert sha(ROOT / 'scripts' / corrected_name) == loss_audit['corrected_helper_sha256']
    (dest / corrected_name).write_bytes((ROOT / 'scripts' / corrected_name).read_bytes())
    shell = (RETURN / 'scripts/run_alignment.sh').read_text(encoding='utf-8')
    shell = shell.replace(worker_path.name, worker_name)
    (dest / 'scripts/run_boundary.sh').write_text(shell, encoding='utf-8', newline='\n')
    p = dict(old)
    p.update(name=NAME, format='manual-inactive-at-zero-brightness-hinge-gradient-diagnostic-v1',
        diagnostic_parent_protocol_sha256=PIN,
        loss_and_optimizer_direction_configuration_unchanged=False,
        scalar_loss_values_weights_and_optimizer_direction_configuration_unchanged=True,
        brightness_guard_gradient_convention='torch.relu: inactive at and below zero for all three nonnegative hinges',
        scientific_thresholds_and_preservation_stops_unchanged=True,
        compare_initial_zero_guard_gradient_and_nominal_directions_before_any_training=True,
        initial_zero_guard_gradient_required=True, exact_scalar_guard_parity_required=True,
        nominal_direction_still_not_actual_accumulated_or_finite_update=True,
        corrected_gradient_behavior_and_training_usefulness_unrun=True)
    p['assets_sha256'] = {f.relative_to(dest).as_posix(): sha(f) for f in dest.rglob('*') if f.is_file()}
    p['evidence_sha256'] = {**old['evidence_sha256'], **{path.relative_to(ROOT).as_posix(): sha(path) for path in evidence}}
    write(dest / 'protocol.json', p)
    pin = sha(dest / 'protocol.json')
    with tarfile.open(archive, 'x:gz', compresslevel=1) as tar:
        for file in sorted(dest.rglob('*')):
            if file.is_file():
                tar.add(file, arcname=NAME + '/' + file.relative_to(dest).as_posix(), recursive=False)
    digest = sha(archive)
    with Path(str(archive) + '.sha256').open('x', encoding='ascii', newline='\n') as stream:
        stream.write(digest + '  ' + archive.name + '\n')
    write(receipt, dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), preparer_sha256=sha(__file__),
        name=NAME, stem=STEM, protocol_sha256=pin, archive_sha256=digest, archive_bytes=archive.stat().st_size,
        packet_assets=len(p['assets_sha256']), parent_required_assets=len(p['parent_required_sha256']),
        diagnostic_parent_protocol_sha256=PIN, corrected_helper_sha256=sha(dest / corrected_name),
        preceding_audit_sha256=sha(ANALYSIS / 'independent_audit.json'),
        identified_zero_penalty_nonzero_gradient_boundary_behavior=True,
        scalar_penalty_values_weights_states_cases_and_gates_unchanged=True,
        derivative_change_explicitly_declared=True, raw_exact_control_checks_retained=True,
        resource_limits_unchanged=True, no_optimizer_or_failed_state_resume=True,
        local_neural_calls=0, local_gradient_queries=0, local_optimizer_updates=0,
        manual_VM_gradient_diagnostic_pending=True, independent_transfer_and_return_checker_pending=True,
        no_agent_VM_launch=True, model_qualified=False, goal_complete=False))
    print(dict(complete=True, protocol_sha256=pin, archive_sha256=digest,
               archive_bytes=archive.stat().st_size, manual_VM_gradient_diagnostic_pending=True))


if __name__ == '__main__':
    main()
