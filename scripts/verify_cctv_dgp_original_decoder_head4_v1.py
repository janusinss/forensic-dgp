"""Independent NumPy arithmetic and evidence readback; no neural or gradient calls."""
import ast
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_original_decoder_head4_review_v1'
RETURN = ROOT / 'outputs/cctv_dgp_original_decoder_gradient_v1_r2_return'
PARENT = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def stats(a):
    return {'shape': list(a.shape), 'dtype': str(a.dtype), 'values': a.size,
            'nonzero': int(np.count_nonzero(a)), 'positive': int(np.count_nonzero(a > 0)),
            'negative': int(np.count_nonzero(a < 0)), 'minimum': float(a.min()),
            'maximum': float(a.max()), 'maximum_absolute': float(np.abs(a).max())}


def convolution(x, weights):
    # Independent cross-correlation of every saved value, without torch/model code.
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(x, ((0,0),(0,0),(1,1),(1,1))), (3,3), axis=(2,3))
    return np.einsum('nchwij,ocij->nohw', windows, weights, optimize=True)


def main():
    started = time.monotonic()
    receipt = OUT / 'independent_readback.json'
    assert not receipt.exists()
    r, plan = read(OUT / 'results.json'), read(OUT / 'plan.json')
    assert r['complete'] and plan['complete'] and plan['frozen_before_forward_trace']
    assert r['plan_sha256'] == sha(OUT / 'plan.json')
    assert r['source_bindings_sha256'] == plan['source_bindings_sha256']
    for name, digest in r['source_bindings_sha256'].items():
        p = (ROOT / name).resolve()
        assert p.is_relative_to(ROOT) and sha(p) == digest, name
    assert len(r['source_bindings_sha256']) == 425
    diagnostic = read(ROOT / 'outputs/cctv_dgp_original_decoder_gradient_v1_r2_vm/protocol.json')
    audit = read(ROOT / 'outputs/cctv_dgp_original_decoder_gradient_v1_r2_independent_audit.json')
    failure = read(RETURN / 'outputs/failure.json')
    supervision = read(RETURN / 'supervisor_receipt.json')
    assert audit['complete'] and not audit['VM_proof_complete'] and audit['VM_failure_retained']
    assert failure['component_gradient_calls'] == 70 and failure['optimizer_updates'] == failure['epochs'] == 0
    assert failure['reference_DGP_forwards'] == failure['candidate_DGP_forwards'] == 10
    assert failure['recognizer_forwards'] == 20 and not failure['optimizer_constructed'] and not failure['new_checkpoint_created']
    assert not failure['resume_permitted']
    assert 'All14 original decoder tensors' in failure['cause'] and 'line 263' in failure['traceback']
    assert supervision['trainer_exit_code'] == 1 and supervision['within_external_bound']
    assert 0 < failure['seconds'] < 600 and 0 < supervision['seconds'] < 660
    batches = [np.load(RETURN / ('outputs/gradients/batch'+str(i)+'.npy'), allow_pickle=False) for i in range(10)]
    total = np.load(RETURN / 'outputs/gradient_components.npy', allow_pickle=False)
    assert all(a.shape == (7,609219) and a.dtype == np.float64 and np.isfinite(a).all() for a in [*batches,total])
    assert np.array_equal(total, sum(batches, np.zeros_like(total)))
    partitions = read(OUT / 'saved_gradient_partitions.json')
    zero = []
    for parameter, saved in zip(diagnostic['parameter_layout'], partitions['partitions']):
        assert saved['name'] == parameter['name']
        begin, end = parameter['start'], parameter['end']
        block = total[:,begin:end]
        assert saved['parameters'] == end-begin and saved['aggregate_nonzero_values'] == np.count_nonzero(block)
        assert saved['aggregate_improvement_gradient_norm'] == float(np.linalg.norm(block[:3].sum(0)))
        assert np.count_nonzero(block[3:]) == 0
        if not np.count_nonzero(block): zero.append(saved['name'])
        for i, (batch, row) in enumerate(zip(batches, saved['batches'])):
            v = batch[:,begin:end]
            assert row['batch'] == i and row['nonzero_values'] == np.count_nonzero(v)
            assert row['component_nonzero_values'] == np.count_nonzero(v, axis=1).tolist()
            assert row['improvement_gradient_norm'] == float(np.linalg.norm(v[:3].sum(0)))
            if saved['name'].startswith('head4.'):
                assert np.count_nonzero(v) == 0
    assert zero == r['zero_VM_gradient_tensors_all_batches'] == ['head4.block0.weight','head4.block1.weight']
    assert len(diagnostic['parameter_layout']) == len(partitions['partitions']) == 14
    assert r['active_aggregate_VM_gradient_tensors'] == partitions['active_aggregate_tensors'] == 12
    wstats = read(OUT / 'weight_statistics.json')
    weights = [np.load(OUT / (name.replace('.', '_')+'.npy'), allow_pickle=False) for name in zero]
    tiny, smallest = float(np.finfo(np.float32).tiny), float(np.finfo(np.float32).smallest_subnormal)
    for name, w in zip(zero, weights):
        assert w.dtype == np.float32 and np.isfinite(w).all()
        actual = stats(w); saved = wstats['weights'][name]
        assert all(saved[k] == value for k,value in actual.items())
        assert actual['nonzero'] == actual['values'] == saved['subnormal_nonzero_values']
        assert 0 < actual['maximum_absolute'] < tiny
        assert saved['float32_minimum_normal'] == tiny and saved['float32_minimum_subnormal'] == smallest
    assert wstats['all_head4_weight_values_nonzero_subnormal'] and r['all_head4_weights_nonzero_subnormal']
    p = read(PARENT / 'protocol.json')
    assert [c['id'] for c in p['cases']] == plan['case_ids'] == [row['id'] for row in r['cases']]
    assert len(r['cases']) == 50
    worst_relative = 0.0; arrays_checked = 0
    for case, row in zip(p['cases'], r['cases']):
        assert case['role'] == row['role'] == 'train'
        assert (case['source'],case['profile']) == (row['source'],row['profile'])
        values = {}
        for name, binding in row['saved_arrays'].items():
            path = (ROOT / binding['path']).resolve()
            assert path.is_relative_to(OUT) and sha(path) == binding['sha256']
            a = np.load(path, allow_pickle=False)
            assert np.isfinite(a).all() and stats(a) == row['statistics'][name]
            values[name] = a; arrays_checked += 1
        assert values['fpn.lateral4'].dtype == np.float32 and np.count_nonzero(values['fpn.lateral4']) > 0
        assert np.count_nonzero(values['head4']) == np.count_nonzero(values['head4.block1']) == 0
        first = convolution(values['fpn.lateral4'].astype(np.float64), weights[0].astype(np.float64))
        second = convolution(np.maximum(first,0), weights[1].astype(np.float64))
        for actual, key in [(first,'float64_head4_block0'),(second,'float64_head4_block1'),(np.maximum(second,0),'float64_head4')]:
            expected = values[key]
            assert actual.shape == expected.shape and expected.dtype == np.float64
            relative = float(np.abs(actual-expected).max()/max(np.abs(expected).max(),1e-300))
            assert relative < 1e-12, (row['id'],key,relative)
            worst_relative = max(worst_relative,relative)
        assert 0 < np.abs(values['float64_head4']).max() < smallest
    assert arrays_checked == 350
    for key in ['unchanged_CPU_head4_outputs_exact_zero_cases','CPU_fourth_map_nonzero_cases',
                'float64_head4_nonzero_cases','float64_maximum_head4_below_float32_subnormal_cases']:
        assert r[key] == 50, key
    assert r['saved_gradient_values_checked'] == audit['saved_gradient_values_independently_checked'] == 46909863
    assert r['inference_counts'] == {'original_DGP':50,'fixed_float64_convolutions':100}
    assert r['DGP_state_before_after'] == diagnostic['original_DGP_state']
    assert 0 < r['seconds'] < plan['cap_seconds'] == 300
    tree = ast.parse((ROOT / 'scripts/review_cctv_dgp_original_decoder_head4_v1.py').read_text(encoding='utf-8'))
    calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
    assert 'torch.no_grad' in calls and not any(x.endswith(('backward','autograd.grad','step','enable_grad')) for x in calls)
    assert 'torch.save' not in calls
    assert not any(isinstance(n,ast.Attribute) and n.attr in ['Adam','AdamW','SGD'] for n in ast.walk(tree))
    assert r['local_gradient_calls'] == r['local_backward_calls'] == r['local_optimizer_updates'] == 0
    assert r['failure_retained']
    for key in ['historical_all14_gate_waived','CUDA_intermediate_activations_measured', 'checkpoint_history_unique_cause_claimed',
                'quality_improvement_claimed','native_or_reserved_used','VM_actions','new_training_recipe_created',
                'app_promotion','independent_final_review','goal_complete']:
        assert r[key] is False
    record = {'complete':True, 'checker_sha256':sha(Path(__file__)), 'results_sha256':sha(OUT/'results.json'),
              'source_bindings_verified':425, 'saved_gradient_values_verified':46909863,
              'activation_arrays_verified':arrays_checked, 'independent_float64_convolutions':100,
              'maximum_float64_convolution_relative_error':worst_relative,
              'inactive_VM_tensors':zero, 'initial_active_aggregate_tensors':12,
              'unchanged_CPU_zero_head4_cases':50, 'nonzero_CPU_upstream_cases':50,
              'higher_precision_head4_below_float32_minimum_subnormal_cases':50,
              'failure_and_all14_gate_preserved':True, 'local_neural_or_gradient_calls':0,
              'VM_actions':False, 'app_promotion':False, 'independent_final_review':False,
              'goal_complete':False, 'seconds':time.monotonic()-started}
    with receipt.open('x',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps(record,indent=2))


if __name__ == '__main__':
    main()
