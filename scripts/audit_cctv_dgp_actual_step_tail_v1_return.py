"""Prospective315-slot tail return audit; immutable old failure stays separate."""
from pathlib import Path, PurePosixPath
import ast
import hashlib
import shutil
import sys
import tarfile
import time
import traceback
from cctv_dgp_actual_step_tail_v1_vm import NAME, PARENT, STEM, read, write, sha, verify_transfer, allowed_return

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs' / NAME
PARENT_BUNDLE = ROOT / 'outputs' / PARENT
RETURN = ROOT / 'outputs' / (NAME + '_return')
AUDIT = ROOT / 'outputs/cctv_dgp_actual_step_tail_v1_independent_audit.json'
CAP = 900


def main():
    start = time.monotonic()
    pin = sha(BUNDLE / 'protocol.json')
    p, parent_p, _ = verify_transfer(BUNDLE, PARENT_BUNDLE, pin)
    assert sha(Path(__file__)) == p['prospective_auditor_sha256']
    for name, digest in p['local_audit_source_sha256'].items():
        assert sha(ROOT / name) == digest, name
    archive = ROOT / 'outputs' / (STEM + '-results.tar.gz')
    export = read(ROOT / 'outputs' / (STEM + '-export.json'))
    assert export['complete'] and export['archive_sha256'] == sha(archive) and export['bytes'] == archive.stat().st_size
    assert export['optimizer_updates'] == export['gradient_queries'] == 0 and not export['new_trained_checkpoint']
    assert export['training_success_not_implied']
    assert (ROOT / 'outputs' / (STEM + '-results.tar.gz.sha256')).read_text().split() == [export['archive_sha256'], archive.name]
    assert not RETURN.exists() and not AUDIT.exists(), 'Retain every return and failed audit'
    allowed = allowed_return(parent_p)
    seen, checked_members, total = set(), [], 0
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        assert len(members) <= p['budgets']['maximum_return_members']
        for member in members:
            assert member.isfile() and not member.issym() and not member.islnk()
            parts = PurePosixPath(member.name).parts
            assert parts[0] == NAME + '_return' and len(parts) > 1
            assert not PurePosixPath(member.name).is_absolute() and all(part not in ['', '.', '..'] for part in parts)
            assert '\\' not in member.name and ':' not in member.name
            name = '/'.join(parts[1:])
            assert name in allowed and name.casefold() not in seen
            seen.add(name.casefold());total += member.size
            assert 0 <= member.size <= p['budgets']['maximum_member_bytes']
            assert total <= p['budgets']['maximum_output_bytes']
            checked_members.append((member, name))
        assert shutil.disk_usage(ROOT / 'outputs').free >= total + 256*1024**2
        RETURN.mkdir()
        for member, name in checked_members:
            assert time.monotonic() - start < CAP
            target = RETURN / name;target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream, tar.extractfile(member) as source:
                shutil.copyfileobj(source, stream)
    assert sha(RETURN / 'protocol.json') == pin
    manifest = read(RETURN / 'export_manifest.json')
    assert manifest['complete'] and manifest['protocol_sha256'] == pin
    imported = {name: sha(RETURN / name) for _, name in checked_members}
    assert set(imported) == set(manifest['files_sha256']) | {'export_manifest.json'}
    for name, digest in manifest['files_sha256'].items():
        assert imported[name] == digest, name
    write(ROOT / 'outputs/cctv_dgp_actual_step_tail_v1_return_import.json', {
        'complete':True,'archive_sha256':export['archive_sha256'],'members':len(imported),'files_sha256':imported,'returned_code_executed':False})
    supervisor = read(RETURN / 'supervisor_receipt.json')
    assert supervisor['optimizer_updates'] == supervisor['gradient_queries'] == 0
    assert not supervisor['new_trained_checkpoint'] and not supervisor['automatic_training_follow_on'] and not supervisor['app_promotion']
    assert int((RETURN / 'review_exit_code.txt').read_text()) == supervisor['review_exit_code']
    if not manifest['diagnostic_completed']:
        assert not supervisor['complete'] and supervisor['review_exit_code'] != 0 and not export['tail_completed']
        assert (RETURN / 'outputs/tail_failure.json').exists()
        write(AUDIT, {'complete':True,'scope':'safe hash-bound stopped-tail import only; scientific audit incomplete',
            'all315_outputs_checked':False,'original_storage_failure_retained':True,'members_verified':len(imported),
            'optimizer_updates':0,'gradient_queries':0,'app_promotion':False})
        print({'tail_failure_imported':True,'scientific_audit_complete':False},flush=True);return
    assert supervisor['complete'] and supervisor['review_exit_code'] == 0 and export['tail_completed']
    assert supervisor['calls'] == [{'flag':'--review','exit_code':0,'timeout':False,'external_cap_seconds':630}]
    expected = allowed - {'outputs/failure.json', 'outputs/tail_failure.json'}
    assert set(imported) == expected
    results = read(RETURN / 'outputs/results.json')
    scope = read(RETURN / 'outputs/tail_scope_receipt.json')
    overlap = read(RETURN / 'outputs/overlap_receipt.json')
    assert results['complete'] and results['protocol_sha256'] == p['parent_protocol_sha256']
    assert results['progress'] == {'optimizer_updates':0,'gradient_queries':0,'backward_calls':0,'optimizer_constructed':False,
        'new_trained_checkpoint':False,'parameter_proposals_loaded':3,'forward_slots':315,'completed_conditions':3}
    assert results['final_restored_states'] == parent_p['initial_states']
    assert results['original_initial_reference_and_recognizer_unchanged'] and not results['full_TRAIN_capacity_pass']
    assert results['source_forward_counts'] == {'original_DGP':121,'decoder':92,'reference_decoder':92,'recognizer':155}
    assert not results['native_or_DEV_or_reserved_final_used'] and not results['app_promotion']
    assert results['seconds'] <= p['budgets']['worker_seconds'] and results['allocated_peak_VRAM_bytes'] <= p['budgets']['maximum_allocated_VRAM_bytes']
    assert scope['complete'] and scope['protocol_sha256'] == pin and scope['parent_protocol_sha256'] == p['parent_protocol_sha256']
    assert scope['conditions'] == 3 and scope['forward_slots'] == 315 and scope['overlap_files_checked'] == 828
    assert scope['standalone_tail_completion_does_not_relabel_original_failure'] and not scope['full_TRAIN_capacity_pass']
    assert overlap['complete'] and overlap['files_checked'] == 828 and overlap['all_arrays_pixels_and_saved_control_metrics_exact']
    assert set(overlap['overlap_files']) == set(p['overlap_files_sha256'])
    # Independently repeat all828 overlap checks against the archived local return.
    import numpy as np
    from PIL import Image
    parent_return = ROOT / 'outputs' / (PARENT + '_return')
    for name, digest in p['overlap_files_sha256'].items():
        a, b = parent_return / name, RETURN / name
        assert sha(a) == digest
        if a.suffix == '.npz':
            with np.load(a,allow_pickle=False) as x, np.load(b,allow_pickle=False) as y:
                assert x.files == y.files
                for key in x.files:
                    assert x[key].dtype == y[key].dtype and np.array_equal(x[key],y[key]), (name,key)
        elif a.suffix == '.png':
            with Image.open(a) as x, Image.open(b) as y:
                assert x.mode == y.mode == 'RGB' and x.size == y.size == (256,256)
                assert np.array_equal(np.asarray(x),np.asarray(y)), name
        else:
            assert read(a) == read(b), name
    preflight = read(RETURN / 'outputs/preflight.json')
    cache = read(RETURN / 'outputs/cache_receipt.json')
    timing = read(RETURN / 'outputs/timing_projection.json')
    assert preflight['complete'] and preflight['states'] == parent_p['initial_states'] and preflight['initial_exact_case_parity'] == 145
    assert 'L4' in preflight['gpu'] and not preflight['grad_enabled'] and preflight['source_forward_counts'] == {'original_DGP':58,'decoder':29,'reference_decoder':29,'recognizer':29}
    assert cache['complete'] and cache['cases'] == 145 and cache['references'] == 29 and 0 < cache['seconds'] <= cache['cap_seconds'] == 120
    assert timing['completed_probe_states'] == 1 and timing['completed_forward_slots'] == 315 and timing['remaining_probe_states'] == 0
    assert timing['cap_seconds'] == 300 and timing['safety_factor'] == 1.25 and abs(timing['projected_review_seconds'] - timing['seconds']*1.25) <= 1e-9
    assert 0 < timing['projected_review_seconds'] <= 300
    from cctv_dgp_actual_step_review_v1_partial_checks_r1 import __dict__ as original_namespace
    checks_source = (ROOT / 'scripts/cctv_dgp_actual_step_review_v1_partial_checks_r1.py').read_text(encoding='utf-8')
    namespace = dict(original_namespace)
    namespace.update({'RETURN':RETURN,'BUNDLE':PARENT_BUNDLE,'CAP':CAP,'roundoff_receipts':[]})
    helper_begin = checks_source.index('roundoff_receipts=[]')
    helper_end = checks_source.index('\ndef scientific_partial_r1(', helper_begin)
    exec(compile(checks_source[helper_begin:helper_end],'<unchanged-source-bound-roundoff-check>','exec'),namespace)
    functions = {node.name:ast.get_source_segment(checks_source,node) for node in ast.parse(checks_source).body if isinstance(node,ast.FunctionDef)}
    for original_name, expected_count, new_count in [('scientific_partial_r1',3045,315),('CPU_partial_replay',145,15)]:
        source = functions[original_name]
        before = 'checked==3045' if original_name.startswith('scientific') else "result['cases']==145"
        after = 'checked==315' if original_name.startswith('scientific') else "result['cases']==15"
        assert source.count(before) == 1
        source = source.replace(before,after,1)
        if original_name.startswith('scientific'):
            source = source.replace("'of':29", "'of':3")
        if original_name == 'CPU_partial_replay':
            source = source.replace("'of':145", "'of':15")
        exec(compile(source,'<source-bound315-slot-scope>','exec'),namespace)
    complete = {(45,proposal) for proposal in ['zero','recorded','cone']}
    errors, conditions, checked = namespace['scientific_partial_r1'](parent_p,{'completed_conditions':results['conditions']},start,complete)
    replay = namespace['CPU_partial_replay'](parent_p,start,complete)
    assert checked == 315 and len(conditions) == 3 and replay['cases'] == 15 and time.monotonic()-start < CAP
    write(AUDIT, {'complete':True,'all315_outputs_checked':True,'scope':'separate final-state tail315 slots; original storage-stopped run remains failed',
        'protocol_sha256':pin,'parent_protocol_sha256':p['parent_protocol_sha256'],'archive_sha256':export['archive_sha256'],
        'members_verified':len(imported),'overlap_files_independently_checked':828,'all_control_and_partial_overlap_arrays_pixels_metrics_exact':True,
        'conditions':conditions,'raw_objective_maximum_errors':errors.tolist(),'CPU_replay':replay,
        'bounded_derived_ratio_recomputation':namespace['roundoff_receipts'],'every_original_stored_row_and_categorical_gate_exact':True,
        'optimizer_updates':0,'gradient_queries':0,'backward_calls':0,'new_trained_checkpoint':False,
        'original_storage_failure_retained':True,'full_TRAIN_capacity_pass':False,'native_or_DEV_or_reserved_final_used':False,
        'returned_code_executed':False,'app_promotion':False,'goal_complete':False,'checker_sha256':sha(Path(__file__)),
        'seconds':time.monotonic()-start,'cap_seconds':CAP})
    print({'tail_scientific_audit_complete':True,'slots':315,'CPU_replays':15,'overlap_files':828,'original_failure_retained':True},flush=True)


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        path=ROOT/'outputs/cctv_dgp_actual_step_tail_v1_audit_failure.json'
        if not path.exists():write(path,{'complete':False,'error':str(error),'type':type(error).__name__,'traceback':traceback.format_exc(),
            'original_failure_retained':True,'optimizer_updates':0,'gradient_queries':0,'new_trained_checkpoint':False,'app_promotion':False})
        raise
