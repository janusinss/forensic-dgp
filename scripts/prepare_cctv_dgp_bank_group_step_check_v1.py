"""Prepare a new UNRUN signed small-step check of the failed finite prediction."""
import ast
from datetime import datetime, timezone
import tarfile
from pathlib import Path
from import_cctv_dgp_bank_group_gradients_v1_r1_return import ROOT, OUT, ANALYSIS, PARENT, read, sha, write

OLD = 'cctv_dgp_bank_group_direction_forward_v1'
NEW = 'cctv_dgp_bank_group_step_check_v1'
NAME = NEW+'_vm'
OLD_STEM = 'cctv-dgp-bank-group-direction-forward-v1'
STEM = 'cctv-dgp-bank-group-step-check-v1'
RATES = ['-0.001', '-0.0001', '0.0001', '0.001']


def once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def scope(source):
    return source.replace(OLD, NEW).replace(OLD_STEM, STEM).replace(
        'group_direction_forward_v1', 'group_step_check_v1')


def source_file(path, source):
    ast.parse(source, feature_version=(3, 10))
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(source)


def main():
    dest, archive = OUT/NAME, OUT/(STEM+'-execution.tar.gz')
    receipt = OUT/(NEW+'_preparation.json')
    assert not any(path.exists() for path in [dest, archive, receipt])
    previous = OUT/(OLD+'_return')
    reviewed = OUT/(OLD+'_analysis')
    evidence = [reviewed/n for n in ['import.json', 'independent_audit.json',
        'visual_review.json', 'finite_direction_review.json', 'review_binding_audit.json']]
    assert all(read(path)['complete'] for path in evidence)
    assert read(reviewed/'visual_review.json')['all20_sheets_and800_cells_inspected']
    assert not any(r['all_preview_requirements_pass'] for r in
                   read(reviewed/'finite_metrics.json')['proposals'].values())
    old = read(previous/'protocol.json')
    assert sha(previous/'protocol.json') == '92783a76effd2409d4a77433a6a4f52ea80e48628dffc444716e3e6df2d7bd9e'
    protected = read(reviewed/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n) == h for n, h in protected.items())
    assert all(sha(PARENT/n) == h for n, h in old['parent_required_sha256'].items())
    direction_root = ANALYSIS/'common_direction'
    direction = read(direction_root/'direction_review.json')
    independent = read(direction_root/'independent_audit.json')
    assert sha(direction_root/'direction_review.json') == independent['direction_review_sha256']
    assert sha(direction_root/'directions.npz') == independent['direction_file_sha256']
    worker_source = previous/'scripts'/f'{OLD}_vm.py'
    value = scope(worker_source.read_text(encoding='utf-8'))
    value = once(value, "for rate in ['0.005', '0.01', '0.02', '0.04']]",
                 "for rate in ['-0.001', '-0.0001', '0.0001', '0.001']]")
    value = once(value, "    assert p['maximum_temporary_parameter_proposals'] == 4",
        "    assert p['maximum_temporary_parameter_proposals'] == 4\n"
        "    assert p['signed_radii'] == [.001, .0001] and p['negative_steps_diagnostic_only']\n"
        "    assert p['no_arbitrary_derivative_threshold_or_model_selection']\n"
        "    assert p['export_storage_reserve_enforced']")
    value = once(value, "            updates = {}", "            updates = {}\n            changes = {}")
    value = once(value, "                updates[name] = sha_array(value)",
        """                updates[name] = sha_array(value)
                actual = value.astype(np.float64)-initial[name].numpy().astype(np.float64)
                nominal = float(rate)*directions[proposal['direction']][i].astype(np.float64)
                changes[name] = dict(changed_values=int(np.count_nonzero(actual)),
                    values=int(actual.size), squared_L2=float(np.square(actual).sum()),
                    nominal_squared_L2=float(np.square(nominal).sum()),
                    rounding_error_squared_L2=float(np.square(actual-nominal).sum()),
                    maximum_rounding_error=float(np.max(np.abs(actual-nominal))))""")
    value = once(value, "                model_state=proposal_state, selected_parameter_value_sha256=updates,",
        "                model_state=proposal_state, selected_parameter_value_sha256=updates,\n"
        "                selected_parameter_change_statistics=changes, negative_step_diagnostic_only=bool(rate<0),")
    value = once(value, "    assert sum(f.stat().st_size for f in files) < p['output_byte_cap']+64*1024**2",
        "    payload_bytes = sum(f.stat().st_size for f in files)\n"
        "    assert payload_bytes < p['output_byte_cap']+64*1024**2\n"
        "    assert shutil.disk_usage(root).free >= p['disk_reserve_bytes']+payload_bytes+1024**2, 'Export space reserve stop'")
    value = once(value, "            tar.add(file, arcname=NAME+'/'+file.relative_to(root).as_posix(), recursive=False)",
        "            assert shutil.disk_usage(root).free >= p['disk_reserve_bytes']+file.stat().st_size, 'Export reserve stop'\n"
        "            tar.add(file, arcname=NAME+'/'+file.relative_to(root).as_posix(), recursive=False)\n"
        "            assert archive.stat().st_size <= p['output_byte_cap']+64*1024**2, 'Export byte cap stop'")
    value = once(value, "    digest = sha(archive)",
        "    assert shutil.disk_usage(root).free >= p['disk_reserve_bytes'], 'Export final reserve stop'\n"
        "    assert time.monotonic()-stamp < p['export_seconds'], 'Export hash time stop'\n"
        "    digest = sha(archive)\n"
        "    assert time.monotonic()-stamp < p['export_seconds'], 'Export final time stop'")
    path = ROOT/'scripts'/f'{NEW}_vm.py'
    source_file(path, value)
    dest.mkdir(); (dest/'scripts').mkdir()
    (dest/'scripts'/path.name).write_bytes(path.read_bytes())
    for name in ['directions.npz', 'direction_review.json']:
        (dest/name).write_bytes((direction_root/name).read_bytes())
    (dest/'scripts/run_forward.sh').write_text(scope((previous/'scripts/run_forward.sh').read_text()), encoding='utf-8', newline='\n')
    p = {key: v for key, v in old.items() if key not in ['assets_sha256', 'evidence_sha256']}
    p.update(name=NAME, format='signed-small-step-gradient-consistency-v1', UTC=datetime.now(timezone.utc).isoformat(),
        proposals=[dict(label='balanced_'+s, direction='balanced', rate=float(s)) for s in RATES],
        signed_radii=[.001, .0001], negative_steps_diagnostic_only=True,
        no_arbitrary_derivative_threshold_or_model_selection=True, export_storage_reserve_enforced=True,
        direction_derivation_original_rate_candidates_retained=[.005, .01, .02, .04],
        prospective_maximum_parameter_displacement_relative_L2=.001*direction['direction_norm']/direction['initial_parameter_norm'],
        actual_float32_parameter_rounding_and_changed_values_saved=True,
        PNG_quantization_not_treated_as_a_neural_gradient=True,
        scientific_recipe_change='Signed tenfold-separated smaller radii distinguish local slope/curvature or path mismatch; no optimizer or continued epochs',
        evidence_sha256={path.relative_to(ROOT).as_posix(): sha(path) for path in evidence})
    p['assets_sha256'] = {f.relative_to(dest).as_posix(): sha(f) for f in dest.rglob('*') if f.is_file()}
    write(dest/'protocol.json', p); pin = sha(dest/'protocol.json')
    with tarfile.open(archive, 'x:gz', compresslevel=1) as stream:
        for file in sorted(dest.rglob('*')):
            if file.is_file():
                stream.add(file, arcname=NAME+'/'+file.relative_to(dest).as_posix(), recursive=False)
    digest = sha(archive)
    with Path(str(archive)+'.sha256').open('x', encoding='ascii', newline='\n') as stream:
        stream.write(digest+'  '+archive.name+'\n')
    checks = []
    for prefix in ['import', 'audit']:
        path = ROOT/'scripts'/f'{prefix}_{NEW}_return.py'
        value = scope((ROOT/'scripts'/f'{prefix}_{OLD}_return.py').read_text(encoding='utf-8'))
        value = value.replace('92783a76effd2409d4a77433a6a4f52ea80e48628dffc444716e3e6df2d7bd9e', pin)
        if prefix == 'audit':
            value = once(value, 'import torch\n', 'import torch\nfrom cctv_dgp_signed_step_audit import signed_differences\n')
            value = once(value, '            state[name]=torch.from_numpy(value)',
                """            actual=value.astype(np.float64)-initial[name].numpy().astype(np.float64)
            nominal=float(np.float32(proposal['rate']))*directions[proposal['direction']][i].astype(np.float64)
            changes=dict(changed_values=int(np.count_nonzero(actual)), values=int(actual.size),
                squared_L2=float(np.square(actual).sum()), nominal_squared_L2=float(np.square(nominal).sum()),
                rounding_error_squared_L2=float(np.square(actual-nominal).sum()),
                maximum_rounding_error=float(np.max(np.abs(actual-nominal))))
            assert parameter['selected_parameter_change_statistics'][name]==changes
            state[name]=torch.from_numpy(value)""")
            value = once(value, "        assert not parameter['checkpoint_saved'] and parameter['optimizer_updates']==0",
                "        assert not parameter['checkpoint_saved'] and parameter['optimizer_updates']==0\n"
                "        assert parameter['negative_step_diagnostic_only']==(proposal['rate']<0)")
            value = once(value, "            source_structure_requirement_pass=source_pass,all_preview_requirements_pass=source_pass and all(g['pass_all_preview_requirements'] for g in gates.values()),",
                "            source_structure_requirement_pass=source_pass,negative_step_diagnostic_only=proposal['rate']<0,\n"
                "            all_preview_requirements_pass=proposal['rate']>0 and source_pass and all(g['pass_all_preview_requirements'] for g in gates.values()),")
            value = once(value, "    sheets=ANALYSIS/'sheets'",
                "    signed=ANALYSIS/'signed_derivative_review.json'\n"
                "    predictions=read(RETURN/'direction_review.json')['float32_group_predictions']\n"
                "    slopes=signed_differences(results,groups_before,predictions)\n"
                "    slopes.update(finite_metrics_sha256=sha(metric_receipt),direction_review_sha256=sha(RETURN/'direction_review.json'),\n"
                "        checker_sha256=sha(ROOT/'scripts/cctv_dgp_signed_step_audit.py'))\n"
                "    write(signed,slopes)\n"
                "    sheets=ANALYSIS/'sheets'")
            value = once(value, "        finite_metrics_sha256=sha(metric_receipt),comparison_sheets=sheet_rows,",
                "        finite_metrics_sha256=sha(metric_receipt),signed_derivative_review_sha256=sha(signed),comparison_sheets=sheet_rows,")
            value = once(value, "'Group step .005','Group step .01','Group step .02','Group step .04'",
                "'Signed step -.001','Signed step -.0001','Signed step +.0001','Signed step +.001'")
        source_file(path, value); checks.append(path)
    for prefix, suffix in [('prepare', 'return_checker'), ('audit', 'transfer'), ('audit', 'source_gates')]:
        path = ROOT/'scripts'/f'{prefix}_{NEW}_{suffix}.py'
        value = scope((ROOT/'scripts'/f'{prefix}_{OLD}_{suffix}.py').read_text(encoding='utf-8'))
        if suffix == 'transfer':
            value = once(value, "for s in ['0.005','0.01','0.02','0.04']]", "for s in ['-0.001','-0.0001','0.0001','0.001']]")
            value = once(value, "    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']",
                "    assert p['negative_steps_diagnostic_only'] and p['export_storage_reserve_enforced']\n"
                "    derivative=read(OUT/(BASE+'_signed_derivative_fixture_audit.json'))\n"
                "    assert derivative['complete'] and derivative['groups_checked']==32\n"
                "    assert derivative['checker_sha256']==sha(ROOT/'scripts/cctv_dgp_signed_step_audit.py')\n"
                "    assert 'selected_parameter_change_statistics' in source and 'Export space reserve stop' in source\n"
                "    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']")
            value = once(value, "        actual_manual_VM_run_unrun=True,model_qualified=False,goal_complete=False))",
                "        signed_derivative_fixture_audit_sha256=sha(OUT/(BASE+'_signed_derivative_fixture_audit.json')),\n"
                "        actual_manual_VM_run_unrun=True,model_qualified=False,goal_complete=False))")
        source_file(path, value); checks.append(path)
    assert all(sha(ROOT/n) == h for n, h in protected.items())
    write(receipt, dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), preparer_sha256=sha(__file__),
        name=NAME, stem=STEM, protocol_sha256=pin, archive_sha256=digest, archive_bytes=archive.stat().st_size,
        assets=4, parent_required_assets=5852, source_worker_sha256=sha(worker_source),
        future_checker_files_sha256={f.relative_to(ROOT).as_posix():sha(f) for f in checks},
        scientific_thresholds_and_identical_inputs_preserved=True, manual_VM_run_unrun=True,
        no_agent_VM_connection_or_launch=True, local_neural_gradient_optimizer_or_parameter_displacement_calls=0,
        model_qualified=False, goal_complete=False))
    print(dict(complete=True, archive_bytes=archive.stat().st_size, protocol_sha256=pin, manual_VM_run_unrun=True))


if __name__ == '__main__':
    main()
