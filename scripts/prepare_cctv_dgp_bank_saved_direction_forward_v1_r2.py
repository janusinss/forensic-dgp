"""Prepare a fresh, bounded alignment repair; preserve every previous packet."""
import ast
from datetime import datetime, timezone
from pathlib import Path
import tarfile

from import_cctv_dgp_bank_saved_direction_forward_v1_r1_return import (
    ROOT, OUT, RETURN, ANALYSIS, PARENT, PIN, read, sha, write,
)

NAME = 'cctv_dgp_bank_saved_direction_forward_v1_r2_vm'
STEM = 'cctv-dgp-bank-saved-direction-forward-v1-r2'

MATRICES_HELPER = '''def fixed_reference_matrices(cases, references):
    """Use frozen target->112 affines; five landmarks are not affine coefficients."""
    matrices = []
    for case in cases:
        matrix = references[case['source_person_or_reference']]['matrix112']
        if len(matrix) != 2 or any(len(row) != 3 for row in matrix):
            raise ValueError('Fixed target affine must have shape2x3')
        if not all(math.isfinite(float(value)) for row in matrix for value in row):
            raise ValueError('Nonfinite fixed target affine')
        determinant = float(matrix[0][0])*float(matrix[1][1])-float(matrix[0][1])*float(matrix[1][0])
        if abs(determinant) < 1e-8:
            raise ValueError('Singular fixed target affine')
        matrices.append(matrix)
    return matrices


'''


def corrected_source(source):
    assert source.count("grid112(c['landmarks5_canvas_xy'])") == 1
    source = source.replace('forward_v1_r1', 'forward_v1_r2').replace('forward-v1-r1', 'forward-v1-r2')
    assert source.count('import json\n') == 1
    source = source.replace('import json\n', 'import json\nimport math\n')
    assert source.count('def verify(root, parent, pin):') == 1
    source = source.replace('def verify(root, parent, pin):', MATRICES_HELPER+'def verify(root, parent, pin):')
    tail = "    assert p['native_pixels_used'] == p['final_identity_pixels'] == 0\n    return p, old"
    assert source.count(tail) == 1
    source = source.replace(tail, """    assert p['native_pixels_used'] == p['final_identity_pixels'] == 0
    refs = {reference['id']: reference for reference in old['references']}
    cases = [case for case in old['cases'] if case['id'] in p['case_ids']]
    assert len(cases) == 100
    assert len(fixed_reference_matrices(cases, refs)) == 100
    return p, old""")
    source = source.replace("[grid112(c['landmarks5_canvas_xy']) for c in cs]",
                            '[grid112(matrix) for matrix in fixed_reference_matrices(cs, refs)]')
    anchor = "                png = np.stack([deliver(a, camera, area) for a, camera, area in zip(raw, x8, observed)])"
    assert source.count(anchor) == 1
    late_save = "                stem = f'b{begin//5:03d}'\n                np.save(folder/(stem+'.npy'), raw, allow_pickle=False)\n"
    assert source.count(late_save) == 1
    source = source.replace(late_save, '')
    source = source.replace(anchor, late_save+anchor)
    raw_embedding = "                vectors_raw = identity.embedding(torch.cat([pred, base, target]), torch.cat([mask]*3), torch.cat([grids]*3))\n"
    png_embedding = "                vectors_png = identity.embedding(torch.cat([tensor(png), base_png, target]), torch.cat([mask]*3), torch.cat([grids]*3))\n"
    assert source.count(raw_embedding) == source.count(png_embedding) == source.count("                counts['recognizer'] += 2\n") == 1
    source = source.replace("                counts['recognizer'] += 2\n", '')
    source = source.replace(raw_embedding, raw_embedding+"                counts['recognizer'] += 1\n")
    source = source.replace(png_embedding, png_embedding+"                counts['recognizer'] += 1\n")
    source = source.replace('neural_calls=0, gradient_queries=0, optimizer_updates=0), flush=True)',
                            'neural_calls=0, gradient_queries=0, optimizer_updates=0, fixed_target_affines_checked=100), flush=True)')
    return source


def main():
    dest = OUT/NAME
    archive = OUT/(STEM+'-execution.tar.gz')
    receipt = OUT/'cctv_dgp_bank_saved_direction_forward_v1_r2_preparation.json'
    source_path = ROOT/'scripts/cctv_dgp_bank_saved_direction_forward_v1_r2_vm.py'
    assert not any(path.exists() for path in [dest, archive, receipt, source_path])
    failure_path = ANALYSIS/'failure_audit.json'
    failure = read(failure_path)
    assert failure['complete'] and failure['failure_is_processing_not_a_quality_gate']
    assert failure['actual_alignment_argument_rejection_reproduced_without_neural_calls']
    assert failure['all100_reference_matrices_are_valid'] and failure['partial_PNGs_checked'] == 5
    assert sha(RETURN/'protocol.json') == PIN
    visual_path = ANALYSIS/'visual_review.json'
    assert not visual_path.exists()
    write(visual_path, dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
        reviewer='Primary agent visual inspection of exact returned pixels',
        failure_audit_sha256=sha(failure_path),sheet=failure['comparison_sheet'],
        sheet_sha256=failure['comparison_sheet_sha256'],all25_sheet_cells_and5_partial_PNGs_inspected=True,
        clear_case='Glasses rims, eyes, nose, lips, chin and hair remain visible; the output is softer than the clear input/target.',
        degraded_cases='All four degraded profiles remain soft around eyes, glasses and mouth; no convincing whole-face clarity gain over the retained DGP.',
        partial_run_cannot_establish_preservation_or_quality_pass=True,
        automatic_or_assisted_completion_tested=False,independent_final_review=False,
        model_qualified=False,app_promotion=False,more_epochs_not_qualified=True))
    previous = OUT/'cctv_dgp_bank_saved_direction_forward_v1_r1_vm'
    p = read(previous/'protocol.json')
    assert sha(previous/'protocol.json') == PIN
    for name, digest in p['assets_sha256'].items():
        assert sha(previous/name) == digest
    protected = read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/name) == digest for name,digest in protected.items())
    original_source = previous/'scripts/cctv_dgp_bank_saved_direction_forward_v1_r1_vm.py'
    source = corrected_source(original_source.read_text(encoding='utf-8'))
    tree = ast.parse(source, feature_version=(3,10))
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr in ['grad','backward','step','Adam','AdamW','SGD','enable_vm_learning'] for n in ast.walk(tree))
    with source_path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(source)
    dest.mkdir()
    (dest/'scripts').mkdir()
    (dest/'scripts'/source_path.name).write_bytes(source_path.read_bytes())
    for name in ['directions.npz','direction_review.json']:
        (dest/name).write_bytes((previous/name).read_bytes())
    shell = (previous/'scripts/run_forward.sh').read_text(encoding='utf-8')
    shell = shell.replace('forward_v1_r1','forward_v1_r2').replace('forward-v1-r1','forward-v1-r2')
    (dest/'scripts/run_forward.sh').write_text(shell,encoding='utf-8',newline='\n')
    p['name'] = NAME
    p['supersedes_failed_R1_protocol_sha256'] = PIN
    p['supersedes_failed_R1_return_archive_sha256'] = read(ANALYSIS/'import.json')['archive_sha256']
    p['alignment_policy'] = 'Frozen reference matrix112 target->112; identical retained grid112 and identity_crop'
    p['all100_fixed_affines_shape_finite_determinant_checked_before_neural_import'] = True
    p['raw_output_saved_before_identity_measurement'] = True
    p['recognizer_counter_incremented_after_each_actual_forward'] = True
    p['repair_scope'] = ['Use reference matrix112 instead of five landmark coordinates',
        'Validate all100 reference matrices before any neural import',
        'Save raw arrays before identity processing and count each completed recognizer forward']
    p['evidence_sha256'].update({path.relative_to(ROOT).as_posix():sha(path)
                               for path in [failure_path,visual_path,ANALYSIS/'import.json']})
    p['assets_sha256'] = {f.relative_to(dest).as_posix():sha(f) for f in dest.rglob('*') if f.is_file()}
    write(dest/'protocol.json',p)
    pin = sha(dest/'protocol.json')
    with tarfile.open(archive,'x:gz',compresslevel=1) as stream:
        for file in sorted(dest.rglob('*')):
            if file.is_file():
                stream.add(file,arcname=NAME+'/'+file.relative_to(dest).as_posix(),recursive=False)
    digest = sha(archive)
    with Path(str(archive)+'.sha256').open('x',encoding='ascii',newline='\n') as stream:
        stream.write(digest+'  '+archive.name+'\n')
    checker_files = {}
    for prefix in ['import_','audit_']:
        old_file = ROOT/'scripts'/(prefix+'cctv_dgp_bank_saved_direction_forward_v1_r1_return.py')
        new_file = ROOT/'scripts'/(prefix+'cctv_dgp_bank_saved_direction_forward_v1_r2_return.py')
        assert not new_file.exists()
        value = old_file.read_text(encoding='utf-8').replace('forward_v1_r1','forward_v1_r2').replace('forward-v1-r1','forward-v1-r2').replace(PIN,pin)
        ast.parse(value)
        with new_file.open('x',encoding='utf-8',newline='\n') as stream:
            stream.write(value)
        checker_files[new_file.relative_to(ROOT).as_posix()] = sha(new_file)
    old_checker = ROOT/'scripts/prepare_cctv_dgp_bank_saved_direction_return_checker_v1_r1.py'
    new_checker = ROOT/'scripts/prepare_cctv_dgp_bank_saved_direction_return_checker_v1_r2.py'
    assert not new_checker.exists()
    value=old_checker.read_text(encoding='utf-8').replace('forward_v1_r1','forward_v1_r2').replace('checker_v1_r1','checker_v1_r2')
    ast.parse(value)
    with new_checker.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(value)
    checker_files[new_checker.relative_to(ROOT).as_posix()] = sha(new_checker)
    assert all(sha(ROOT/name)==digest for name,digest in protected.items())
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),preparer_sha256=sha(__file__),
        name=NAME,stem=STEM,protocol_sha256=pin,archive_sha256=digest,archive_bytes=archive.stat().st_size,
        packet_assets=len(p['assets_sha256']),parent_required_assets=len(p['parent_required_sha256']),
        failure_audit_sha256=sha(failure_path),partial_visual_review_sha256=sha(visual_path),
        original_R1_worker_sha256=sha(original_source),corrected_R2_worker_sha256=sha(source_path),
        prospective_return_checker_files_sha256=checker_files,
        directions_source_and_bytes_unchanged=True,scientific_thresholds_cases_proposals_and_resource_limits_unchanged=True,
        R1_failure_and_original_packets_retained=True,actual_VM_forward_unrun=True,
        local_neural_gradient_optimizer_or_parameter_displacement=0,no_agent_VM_launch=True,
        model_qualified=False,goal_complete=False))
    print(dict(complete=True,archive_bytes=archive.stat().st_size,protocol_sha256=pin,actual_VM_run_unrun=True))


if __name__=='__main__':
    main()
