"""Freeze a small manual tail review after the original diagnostic storage stop."""
from pathlib import Path
import ast
import hashlib
import tarfile
import time
from cctv_dgp_actual_step_tail_v1_vm import NAME, PARENT, STEM, PARENT_PIN, read, write, sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / NAME
PARENT_BUNDLE = ROOT / 'outputs' / PARENT
PARENT_RETURN = ROOT / 'outputs' / (PARENT + '_return')


def main():
    start = time.monotonic()
    assert not OUT.exists()
    parent_p = read(PARENT_BUNDLE / 'protocol.json')
    assert sha(PARENT_BUNDLE / 'protocol.json') == PARENT_PIN
    imported = read(ROOT / 'outputs/cctv_dgp_actual_step_review_v1_return_import.json')
    assert imported['complete'] and imported['archive_sha256'] == 'd380ce08cf858eebc334fcad3c6f5590e01d526084239bbf807bd4b465e8c604'
    failure = read(PARENT_RETURN / 'outputs/failure.json')
    assert failure['progress']['completed_conditions'] == 29 and failure['error'] == 'Return-storage limit; retain partial review'
    source_path = PARENT_BUNDLE / 'scripts/review_cctv_dgp_actual_steps_v1_vm.py'
    source = source_path.read_text(encoding='utf-8')
    node = next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='review')
    original = ast.get_source_segment(source,node)
    adapted = original
    changes = []
    def change(before, after):
        nonlocal adapted
        assert adapted.count(before) == 1, before
        adapted = adapted.replace(before,after,1);changes.append({'before':before,'after':after})
    change("vm_scope(root);out=root/'outputs'", "vm_scope(root);out=TAIL_ROOT/'outputs'")
    change('Need6GiB free after installation', 'Need2GiB free after installation')
    change("for probe_index,probe in enumerate(p['probes']):", "for probe_index,probe in enumerate(p['probes'][-1:]):")
    change("folder=root/output_prefix(probe['update'],proposal)", "folder=TAIL_ROOT/output_prefix(probe['update'],proposal)")
    change("'of':3150", "'of':315")
    change('projected=elapsed*10*BUDGETS', 'projected=elapsed*BUDGETS')
    change("'remaining_probe_states':9", "'remaining_probe_states':0")
    change("assert progress['forward_slots']==3150 and progress['completed_conditions']==30", "assert progress['forward_slots']==315 and progress['completed_conditions']==3")
    inverse = adapted
    for row in reversed(changes):
        assert inverse.count(row['after']) == 1;inverse=inverse.replace(row['after'],row['before'],1)
    assert inverse == original
    ast.parse(adapted,feature_version=(3,10))
    OUT.mkdir();(OUT/'scripts').mkdir()
    scripts = ['cctv_dgp_actual_step_tail_v1_vm.py','audit_cctv_dgp_actual_step_tail_v1_return.py']
    for filename in scripts:
        content=(ROOT/'scripts'/filename).read_text(encoding='utf-8');ast.parse(content,feature_version=(3,10))
        (OUT/'scripts'/filename).write_text(content,encoding='utf-8',newline='\n')
    launcher = '#!/usr/bin/env bash\nset -euo pipefail\ncd "$(dirname "$0")/.."\npython -B -u scripts/cctv_dgp_actual_step_tail_v1_vm.py --root . --parent-root ../cctv_dgp_actual_step_review_v1_vm --protocol-sha "${1:?Protocol SHA required}" --supervise\n'
    (OUT/'scripts/run_actual_step_tail.sh').write_text(launcher,encoding='utf-8',newline='\n')
    overlap = {name:digest for name,digest in imported['files_sha256'].items() if name.startswith('outputs/probes/update0045/')}
    assert len(overlap)==828
    for name,digest in overlap.items():assert sha(PARENT_RETURN/name)==digest,name
    retained_names = ['protocol.json','export_manifest.json','outputs/failure.json','supervisor_receipt.json','review_exit_code.txt']
    retained = {name:imported['files_sha256'][name] for name in retained_names}
    p = {'format':'finite-inference-final-state-storage-recovery-v1','parent_protocol_sha256':PARENT_PIN,
        'parent_stopped_archive_sha256':imported['archive_sha256'],'original_worker_sha256':sha(source_path),
        'review_source_changes':changes,'adapted_review_function_sha256':hashlib.sha256(adapted.encode()).hexdigest(),
        'updates':[45],'proposals':['zero','recorded','cone'],'forward_slots':315,'initial_exact_DGP_parity_cases':145,
        'completed_controls_repeated_only_for_exact_overlap_validation':True,'original_capacity_and_preservation_gates_unchanged':True,
        'original_storage_failure_retained':True,'retained_parent_return_sha256':retained,'overlap_files_sha256':overlap,
        'budgets':{'cache_seconds':120,'review_seconds':300,'worker_seconds':600,'export_seconds':300,
            'minimum_free_bytes':2*1024**3,'maximum_allocated_VRAM_bytes':20*1024**3,'maximum_output_bytes':512*1024**2,
            'maximum_member_bytes':16*1024**2,'maximum_return_members':1100,'maximum_forward_slots':315,'timing_safety_factor':1.25},
        'prospective_independent_audit':{'seconds':900,'all315_raw_PNG_mean_only_slots':True,'CPU_current_batch_replays':15,
            'all828_overlap_arrays_pixels_and_control_metrics_exact':True,'original_pixel_term_and_CPU_tolerances_unchanged':True,
            'derived_floored_ratio_recomputation':'predeclared R1 epsilon-bound on already failed MSE only; exact saved-row comparison and categories'},
        'prospective_auditor_sha256':sha(ROOT/'scripts/audit_cctv_dgp_actual_step_tail_v1_return.py'),
        'local_audit_source_sha256':{path.relative_to(ROOT).as_posix():sha(path) for path in [
            ROOT/'scripts/audit_cctv_dgp_actual_step_review_v1_return.py',ROOT/'scripts/cctv_dgp_actual_step_review_v1_partial_checks_r1.py',
            ROOT/'scripts/cctv_dgp_actual_step_review_v1_partial_checks.py',ROOT/'scripts/cctv_dgp_actual_step_review_v1_metrics.py',
            ROOT/'scripts/cctv_dgp_actual_step_review_v1_contract.py']},
        'assets_sha256':{path.relative_to(OUT).as_posix():sha(path) for path in OUT.rglob('*') if path.is_file()},
        'manual_VM_execution_required':True,'optimizer_updates':0,'gradient_queries':0,'new_trained_checkpoint':False,
        'native_or_DEV_or_reserved_final_used':False,'automatic_follow_on':False,'app_promotion':False,'goal_complete':False,
        'prepared_only':True,'no_new_restoration_recipe':True,'runtime_assets_reused_readonly_from_verified_parent_packet':True,
        'no_checkpoint_split_failure_or_original_output_removed':True}
    write(OUT/'protocol.json',p)
    archive=ROOT/'outputs'/(STEM+'-execution.tar.gz')
    assert not archive.exists()
    with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
        for path in sorted(path for path in OUT.rglob('*') if path.is_file()):
            tar.add(path,arcname=NAME+'/'+path.relative_to(OUT).as_posix(),recursive=False)
    digest=sha(archive)
    with Path(str(archive)+'.sha256').open('x',encoding='utf-8',newline='\n') as stream:stream.write(digest+'  '+archive.name+'\n')
    preparation=ROOT/'outputs/cctv_dgp_actual_step_tail_v1_preparation';preparation.mkdir()
    write(preparation/'preparation.json',{'complete':True,'protocol_sha256':sha(OUT/'protocol.json'),'archive_sha256':digest,
        'archive_bytes':archive.stat().st_size,'members':4,'parent_return_overlap_files':828,'source_inverse_verified':True,
        'forward_slots_planned':315,'optimizer_updates':0,'gradient_queries':0,'neural_calls':0,'VM_calls':0,
        'original_failure_retained':True,'app_promotion':False,'goal_complete':False,'seconds':time.monotonic()-start,'checker_sha256':sha(Path(__file__))})
    print(read(preparation/'preparation.json'),flush=True)


if __name__=='__main__':main()
