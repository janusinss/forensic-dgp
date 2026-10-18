"""Explicit saved-evidence schema and strictly scoped portability adapter R2.

No learned model, new derivative, proposal or optimizer is executed. Scientific
gates and byte checks are retained; VM-scalar provenance and local scalar replay
are distinct obligations. The original and stopped R1 checkers remain intact.
"""
import argparse
import ast
import copy
import json
import math
from pathlib import Path
import types

import audit_cctv_dgp_decoder_activation_v1_return as original
import audit_cctv_dgp_decoder_activation_v1_return_r1 as prior
from cctv_dgp_decoder_activation_v1_contract import read, write, sha, verify_transfer, STAGE, STATES
from cctv_dgp_decoder_activation_v1_math import finite_objective, one_sided_objective

PRIOR_SHA = '9b33762bbabdbca9dde2363b3daf0f79faf9caed37e8958144d97497f942dc66'
DELTA_ASSERTION = "row['finite_minus_torch_objective'] == finite_saved['objective'] - row['objective']"
GATE_CALL = "close_tree(outcomes[label], inherited_outcome['gates'], atol=1e-12, rtol=1e-12)"
EVENT_CALL = "close_tree(one_sided, measured['one_sided'])"
ROW_KEYS = set('reference source case_ids raw_control_exact floor_PNG_control_exact objective raw_terms '
    'normalized_weighted_RGB_SSIM_recognizer_structure regression_barriers_RGB_SSIM_recognizer '
    'clear_blur_RGB_anchors barrier anchors photometric_guard per_image_photometric current baseline '
    'channel_mean_shift recognizer_batch embedding_sha256 finite_arithmetic finite_minus_torch_objective '
    'raw_sha256 delivered_uint8_sha256'.split())
MEASUREMENT_KEYS = set('complete reference source case_ids scalar_labels scalar_values directional_slopes '
    'gradients_shape gradients_dtype gradients_sha256 clipped_gradient_sha256 retained_direction_sha256 '
    'selected_decoder_branch_only branch_finite_validation current_slopes unqueried_clear_structure_slope_placeholder '
    'channel_mean_slopes photometric_slopes mean_control_right_derivative one_sided objective raw_terms '
    'normalized_weighted_RGB_SSIM_recognizer_structure regression_barriers_RGB_SSIM_recognizer '
    'clear_blur_RGB_anchors barrier anchors photometric_guard per_image_photometric current baseline channel_mean_shift '
    'recognizer_batch embedding_sha256 embedding_batch5_file embedding_batch5_file_sha256 embedding_batch5_array_sha256 '
    'full_gradient_norm_before_clip clip_factor clipped_gradient_replay literal_float32_step_derivative_claim '
    'optimizer_updates backwards'.split())


def typed_record_compare(actual,saved,*,atol=1e-10,rtol=1e-10):
    """Strict JSON containers and scalar types, with caller's unchanged bounds."""
    assert type(actual) is type(saved), 'Saved container/scalar type differs'
    if isinstance(actual,dict):
        assert actual.keys()==saved.keys() and all(type(key) is str for key in actual)
        return max([typed_record_compare(v,saved[k],atol=atol,rtol=rtol) for k,v in actual.items()] or [0.])
    if isinstance(actual,list):
        assert len(actual)==len(saved), 'Saved record-array length differs'
        return max([typed_record_compare(a,b,atol=atol,rtol=rtol) for a,b in zip(actual,saved)] or [0.])
    if type(actual) is float:
        assert math.isfinite(actual) and math.isfinite(saved)
        error=abs(actual-saved)
        assert error<=atol+rtol*abs(actual), 'Saved floating scalar exceeds unchanged caller bound'
        return error
    assert actual is None or type(actual) in (bool,int,str)
    assert actual==saved, 'Saved discrete value differs'
    return 0.


def verify_saved_delta(row):
    for value in [row['objective'],row['finite_arithmetic']['objective'],row['finite_minus_torch_objective']]:
        assert type(value) is float and math.isfinite(value)
    expected=row['finite_arithmetic']['objective']-row['objective']
    assert row['finite_minus_torch_objective']==expected, 'Stored VM finite-minus-Torch relation differs'
    return dict(reference=row['reference'],stored_VM_finite_objective=row['finite_arithmetic']['objective'],
        stored_Torch_objective=row['objective'],stored_delta=row['finite_minus_torch_objective'],
        exact_stored_VM_scalar_relation=True,local_reduction_not_substituted=True)


def _float_array(value,shape):
    if not shape:
        assert type(value) is float and math.isfinite(value)
        return
    assert type(value) is list and len(value)==shape[0]
    for item in value:
        _float_array(item,shape[1:])


def _count_array(value,shape):
    if not shape:
        assert type(value) is int and value>=0
        return
    assert type(value) is list and len(value)==shape[0]
    for item in value: _count_array(item,shape[1:])


def _terms(row):
    assert row['current'].keys()=={'MSE','SSIM_loss','ArcFace_loss','landmark_structure'}
    assert row['baseline'].keys()=={'MSE','SSIM_loss','ArcFace_loss'}
    for values in [row['current'],row['baseline']]:
        for array in values.values(): _float_array(array,[5])
    for key,length in [('raw_terms',4),('normalized_weighted_RGB_SSIM_recognizer_structure',4),
                       ('regression_barriers_RGB_SSIM_recognizer',3),('clear_blur_RGB_anchors',2)]:
        _float_array(row[key],[length])
    _float_array(row['channel_mean_shift'],[5,3])
    for key in ['objective','barrier','anchors','photometric_guard']:
        _float_array(row[key],[])
    assert type(row['recognizer_batch']) is int and row['recognizer_batch']==5
    photo=row['per_image_photometric']
    assert photo.keys()==set('photometric_guard degraded_mean_gain degraded_actual_gain degraded_excess '
                             'old_MSE actual_MSE mean_only_MSE per_image_guard_stronger_than_aggregate_gate clear_profile_in_guard'.split())
    for key in ['old_MSE','actual_MSE','mean_only_MSE']: _float_array(photo[key],[5])
    for key in ['photometric_guard','degraded_mean_gain','degraded_actual_gain','degraded_excess']: _float_array(photo[key],[])
    assert photo['per_image_guard_stronger_than_aggregate_gate'] is True and photo['clear_profile_in_guard'] is False


def preflight_saved_schema(root,repair,source,p,old):
    """All 80 objective rows, 8 gates and 2 derivative records before image math."""
    root,repair,source=map(Path,(root,repair,source));out=root/STAGE
    gates=prior.validate_retained_gate_schema(repair,source)
    objective_rows,initial,bindings=[],{},{}
    for state in STATES:
        path=out/state/'objective_components.json';bindings[str(path)]=sha(path)
        saved=read(path)
        assert saved.keys()=={'complete','rows','recognizer_batch','historical_batch15_separate','historical_components_sha256',
                             'state_vector_sha256','model_state','model_qualified','seconds'}
        assert saved['complete'] is saved['historical_batch15_separate'] is True and saved['model_qualified'] is False
        assert type(saved['recognizer_batch']) is int and saved['recognizer_batch']==5 and len(saved['rows'])==20
        assert type(saved['seconds']) is float and math.isfinite(saved['seconds']) and 0<saved['seconds']<=p['budgets']['snapshot_seconds']
        for row,block in zip(saved['rows'],p['reference_blocks']):
            assert row.keys()==ROW_KEYS and row['reference']==block['reference'] and row['source']==block['source']
            assert row['case_ids']==block['case_ids'] and row['raw_control_exact'] is row['floor_PNG_control_exact'] is True
            _terms(row)
            finite=finite_objective(row['current'],row['baseline'],old['normalizers'],old['active_anchor_normalizers'],
                photometric=row['per_image_photometric'],reconstruction_weights=old['reconstruction_weights'])
            proof=prior.compare_finite_objective_only(finite,row['finite_arithmetic'],p['objective_replay_max_abs'])
            # Exact typed traversal for every other finite leaf, including flags.
            typed_record_compare({k:v for k,v in finite.items() if k!='objective'},
                {k:v for k,v in row['finite_arithmetic'].items() if k!='objective'},atol=0,rtol=0)
            delta=verify_saved_delta(row)
            assert abs(finite['objective']-row['objective'])<=p['objective_replay_max_abs']
            objective_rows.append(dict(state=state,**proof,delta_provenance=delta))
            if state=='initial': initial[row['reference']]=row
    gradients=[]
    for index in p['derivative_reference_indices']:
        block=p['reference_blocks'][index];path=out/'gradients'/f'r{index:02d}'/'measurement.json'
        bindings[str(path)]=sha(path);m=read(path)
        assert m.keys()==MEASUREMENT_KEYS and m['complete'] is m['selected_decoder_branch_only'] is True
        assert m['reference']==block['reference'] and m['source']==block['source'] and m['case_ids']==block['case_ids']
        assert m['scalar_labels']==p['gradient_layout'] and m['gradients_shape']==[35,609219] and m['gradients_dtype']=='float32'
        assert all(type(v) is int for v in m['gradients_shape'])
        for key in ['scalar_values','directional_slopes']: _float_array(m[key],[35])
        _float_array(m['channel_mean_slopes'],[5,3]);_terms(m)
        assert type(m['optimizer_updates']) is type(m['backwards']) is int and m['optimizer_updates']==m['backwards']==0
        assert m['literal_float32_step_derivative_claim'] is False
        assert type(m['unqueried_clear_structure_slope_placeholder']) is float and m['unqueried_clear_structure_slope_placeholder']==0.
        assert m['branch_finite_validation']=='only the three retained nonzero amplitudes; no universal derivative claim'
        for key in ['full_gradient_norm_before_clip','clip_factor']: _float_array(m[key],[])
        assert m['full_gradient_norm_before_clip']>=0 and 0<m['clip_factor']<=1
        for key in ['gradients_sha256','clipped_gradient_sha256','retained_direction_sha256','embedding_sha256',
                    'embedding_batch5_file_sha256','embedding_batch5_array_sha256']:
            assert type(m[key]) is str and len(m[key])==64 and all(c in '0123456789abcdef' for c in m[key])
        assert m['embedding_batch5_file']=='embedding_batch5.npy'
        replay=m['clipped_gradient_replay']
        if index==p['derivative_reference_indices'][0]:
            assert type(replay) is dict and replay.keys()==set('max_abs difference_L2 cosine fresh_clipped_L2 retained_clipped_L2 '
                'bound pass_ full_gradient_norm_before_clip clip_factor actual_fitting_reference quality_gate_relaxed'.split())
            for key in ['max_abs','difference_L2','cosine','fresh_clipped_L2','retained_clipped_L2','full_gradient_norm_before_clip','clip_factor']:
                _float_array(replay[key],[])
            typed_record_compare(dict(max_abs=2e-7,cosine_min=.999999),replay['bound'],atol=0,rtol=0)
            assert replay['pass_'] is replay['actual_fitting_reference'] is True and replay['quality_gate_relaxed'] is False
            typed_record_compare(replay,read(path.parent/'clipped_gradient_replay.json'),atol=0,rtol=0)
        else: assert replay is None
        for key in ['baseline','channel_mean_shift','per_image_photometric']:
            typed_record_compare(initial[m['reference']][key],m[key],atol=0,rtol=0)
        for key in ['MSE','SSIM_loss','ArcFace_loss','landmark_structure']:
            _float_array(m['current_slopes'][key],[5])
        assert m['current_slopes'].keys()=={'MSE','SSIM_loss','ArcFace_loss','landmark_structure'}
        assert m['photometric_slopes'].keys()=={'actual_MSE','mean_only_MSE'}
        for key in ['actual_MSE','mean_only_MSE']: _float_array(m['photometric_slopes'][key],[5])
        right=m['mean_control_right_derivative']
        assert right.keys()==set('channel_mean_slopes finite_retained_state_comparisons_required frozen_float32_residual_MSE_slopes '
            'literal_float32_program_derivative_certified maximum_rounding_proxy_slope_difference mean_control_derivative_kind '
            'mean_only_MSE_slopes model_qualified observed_one_pixels_per_channel observed_pixels_per_profile '
            'observed_zero_pixels_per_channel scope selected_neural_branch_must_be_validated'.split())
        for key in ['mean_only_MSE_slopes','frozen_float32_residual_MSE_slopes']: _float_array(right[key],[5])
        _float_array(right['channel_mean_slopes'],[5,3]);_float_array(right['maximum_rounding_proxy_slope_difference'],[])
        for key in ['observed_one_pixels_per_channel','observed_zero_pixels_per_channel']: _count_array(right[key],[5,3])
        _count_array(right['observed_pixels_per_profile'],[5])
        assert right['finite_retained_state_comparisons_required'] is right['selected_neural_branch_must_be_validated'] is True
        assert right['literal_float32_program_derivative_certified'] is right['model_qualified'] is False
        assert right['mean_control_derivative_kind']=='idealized_pixel_clip' and type(right['scope']) is str
        typed_record_compare(right['channel_mean_slopes'],m['channel_mean_slopes'],atol=0,rtol=0)
        typed_record_compare(right['mean_only_MSE_slopes'],m['photometric_slopes']['mean_only_MSE'],atol=0,rtol=0)
        scalar_values=([m['current']['MSE'][i] for i in range(5)]+[m['current']['SSIM_loss'][i] for i in range(5)]
            +[m['current']['ArcFace_loss'][i] for i in range(5)]+m['current']['landmark_structure'][1:]
            +[v for values in m['channel_mean_shift'] for v in values]+[m['objective']])
        typed_record_compare(scalar_values,m['scalar_values'],atol=0,rtol=0)
        finite=finite_objective(m['current'],m['baseline'],old['normalizers'],old['active_anchor_normalizers'],
            photometric=m['per_image_photometric'],reconstruction_weights=old['reconstruction_weights'])
        assert abs(finite['objective']-m['objective'])<=p['objective_replay_max_abs']
        expected=one_sided_objective(m['current'],m['baseline'],m['current_slopes'],old['normalizers'],old['active_anchor_normalizers'],
            photometric=m['per_image_photometric'],photometric_slopes=m['photometric_slopes'],
            mean_control_derivative_kind='idealized_pixel_clip',reconstruction_weights=old['reconstruction_weights'])
        event_difference=typed_record_compare(expected,m['one_sided'])
        gradients.append(dict(reference=m['reference'],one_sided_nested_records_validated=True,
                              maximum_existing_bound_scalar_difference=event_difference))
    for row in gates:
        if row['state']!='initial':
            retained=read(repair/'outputs/decoder_hinge_repair_v1'/(row['state']+'_outcome.json'))['gates'][row['mode']]
            typed_record_compare(row['gate'],retained,atol=0,rtol=0)
        else:
            frozen=prior.load_frozen_gate(source)
            control=repair/'outputs/decoder_hinge_repair_v1'
            baseline=read(control/'baseline/metrics.json')['groups'][row['mode']]
            initial_groups=read(control/'initial/metrics.json')['groups'][row['mode']]
            typed_record_compare(row['gate'],frozen(baseline,initial_groups),atol=0,rtol=0)
    assert len(objective_rows)==80 and len(gates)==8 and len(gradients)==2
    return dict(complete=True,precedes_heavyweight_pixel_recomputation=True,objective_rows=objective_rows,
                exact_stored_VM_delta_rows=80,gate_records=8,gradient_records=gradients,bindings=bindings,
                scope='Schema/scalar provenance only; saved gradients, embeddings and 400 images require the subsequent numerical audit.',
                model_qualified=False)


def load_adapter_module(norm_proofs,objective_proofs,delta_proofs):
    assert sha(prior.__file__)==PRIOR_SHA and sha(original.__file__)==prior.ORIGINAL_SHA
    prior_module,text=prior.load_adapter_module(norm_proofs,objective_proofs)
    tree=ast.parse(text)
    matches=[node for node in ast.walk(tree) if isinstance(node,ast.Assert) and ast.unparse(node.test)==DELTA_ASSERTION]
    assert len(matches)==1
    def verify(row):
        delta_proofs.append(verify_saved_delta(row));return True
    matches[0].test=ast.Call(func=ast.Name(id='_r2_verify_saved_delta',ctx=ast.Load()),args=[ast.Name(id='row',ctx=ast.Load())],keywords=[])
    for expression in [GATE_CALL,EVENT_CALL]:
        calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and ast.unparse(node)==expression]
        assert len(calls)==1
        calls[0].func=ast.Name(id='_r2_typed_record_compare',ctx=ast.Load())
    module=types.ModuleType('retained_activation_explicit_schema_r2');module.__file__=original.__file__
    for key in ['_r1_verify_exact_direction_radius','_r1_compare_finite_objective_only']:
        module.__dict__[key]=prior_module.__dict__[key]
    module.__dict__.update(_r2_verify_saved_delta=verify,_r2_typed_record_compare=typed_record_compare)
    exec(compile(ast.fix_missing_locations(tree),original.__file__,'exec'),module.__dict__)
    module.unchanged_gate=prior.repaired_gate
    return module,ast.unparse(tree)


def audit_saved_r2(root,repair,source,parent,replay,p,rp,old,data,receipt):
    assert sha(prior.__file__)==PRIOR_SHA and sha(original.__file__)==prior.ORIGINAL_SHA
    receipt=Path(receipt).resolve();sidecar=receipt.parent/(receipt.stem+'_schema_r2.json')
    derived=sidecar.with_suffix('.derived.py');preflight_path=sidecar.with_suffix('.preflight.json')
    assert not any(path.exists() for path in [receipt,sidecar,derived,preflight_path]), 'Retain every prior attempt'
    preflight=preflight_saved_schema(root,repair,source,p,old);write(preflight_path,preflight)
    norm_proofs,objective_proofs,delta_proofs=[],[],[]
    module,text=load_adapter_module(norm_proofs,objective_proofs,delta_proofs)
    with derived.open('x',encoding='utf-8',newline='\n') as stream: stream.write(text+'\n')
    module.__file__=str(derived);unmodified_write=module.write
    changes=['315: approximate norm equality replaced by exact retained-byte dyadic radius proof; saved norm unverified',
             '381: only finite_arithmetic.objective reduction uses existing absolute1e-10; every other leaf exact',
             '383: saved VM delta checked against saved VM finite objective minus stored Torch objective exactly',
             '427: explicit strict typed record/list traversal, original1e-12 bounds retained',
             '469: explicit strict typed record/list traversal, original1e-10 bounds retained',
             'unchanged_gate: add only four exact descriptive historical gate fields']
    def labeled(path,value):
        assert Path(path).resolve()==receipt and len(norm_proofs)==1 and len(objective_proofs)==len(delta_proofs)==80
        value=dict(value,original_checker_sha256=prior.ORIGINAL_SHA,adapter_sha256=sha(__file__),
            inherited_R1_sha256=PRIOR_SHA,transformed_local_checker_sha256=sha(derived),
            original_checker_passed_unmodified=False,R1_checker_passed=False,derived_R2_numerical_checks_complete=True,
            preflight_schema_sha256=sha(preflight_path),explicit_change_list=changes,exact_direction_norm_proof=norm_proofs[0],
            saved_norm_metadata_certified=False,saved_norm_metadata_excluded_as_approximate=True,
            original_exact_objective_check_passed=all(row['original_exact_objective_check_passed'] for row in objective_proofs),
            objective_reduction_proofs=objective_proofs,exact_stored_VM_delta_proofs=delta_proofs,
            scientific_thresholds_or_bounds_changed=False,new_model_gradient_optimizer_calls=0)
        unmodified_write(path,value)
    module.write=labeled
    module.audit_saved(root,repair,source,parent,replay,p,rp,old,data,receipt)
    failure_paths=[Path(root)/'saved_audit.log',Path(root)/'saved_audit_exit.json',receipt.parent/'audit_r0.log',receipt.parent/'audit_r1.log']
    assert all(path.is_file() for path in failure_paths)
    dependencies=[Path(original.__file__),Path(prior.__file__),Path(__file__).with_name('test_cctv_dgp_decoder_activation_v1_return_r2.py')]
    write(sidecar,dict(complete=True,original_checker_passed_unmodified=False,R1_checker_passed=False,
        derived_R2_numerical_checks_complete=True,model_qualified=False,explicit_change_list=changes,
        original_checker_sha256=prior.ORIGINAL_SHA,adapter_sha256=sha(__file__),inherited_R1_sha256=PRIOR_SHA,
        transformed_local_checker_sha256=sha(derived),final_saved_audit_sha256=sha(receipt),preflight_schema_sha256=sha(preflight_path),
        exact_direction_norm_proof=norm_proofs[0],objective_reduction_proofs=objective_proofs,
        exact_stored_VM_delta_proofs=delta_proofs,scientific_thresholds_or_bounds_changed=False,
        preserved_failed_checkers={str(path):sha(path) for path in failure_paths},dependencies={str(path.resolve()):sha(path) for path in dependencies},
        local_neural_calls=0,local_gradient_queries=0,local_optimizer_updates=0))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['root','repair-root','source-root','parent-root','replay-root','receipt']: parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--protocol-sha',required=True);args=parser.parse_args()
    p,rp,old,data=verify_transfer(args.root,args.repair_root,args.source_root,args.parent_root,args.replay_root,args.protocol_sha)
    audit_saved_r2(args.root.resolve(),args.repair_root.resolve(),args.source_root.resolve(),args.parent_root.resolve(),
                   args.replay_root.resolve(),p,rp,old,data,args.receipt)


if __name__=='__main__':main()
