"""Saved-gradient arithmetic and finite forward-only trace of the inactive path."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'
RETURN = ROOT / 'outputs/cctv_dgp_original_decoder_gradient_v1_r2_return'
OUT = ROOT / 'outputs/cctv_dgp_original_decoder_head4_review_v1'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def statistics(value):
    import numpy as np
    assert np.isfinite(value).all()
    return {'shape': list(value.shape), 'dtype': str(value.dtype), 'values': value.size,
            'nonzero': int(np.count_nonzero(value)), 'positive': int(np.count_nonzero(value > 0)),
            'negative': int(np.count_nonzero(value < 0)), 'minimum': float(value.min()),
            'maximum': float(value.max()), 'maximum_absolute': float(np.abs(value).max())}


def main():
    start = time.monotonic()
    assert not OUT.exists(), 'Retain any previous or partial trace'
    p = read(PARENT / 'protocol.json')
    protocol_path = ROOT / 'outputs/cctv_dgp_original_decoder_gradient_v1_r2_vm/protocol.json'
    diagnostic = read(protocol_path)
    audit_path = ROOT / 'outputs/cctv_dgp_original_decoder_gradient_v1_r2_independent_audit.json'
    audit = read(audit_path)
    assert audit['complete'] and audit['VM_failure_retained'] and not audit['VM_proof_complete']
    assert audit['saved_gradient_values_independently_checked'] == 46909863
    summary = read(RETURN / 'outputs/gradient_summary.json')
    source_names = ['scripts/review_cctv_dgp_original_decoder_head4_v1.py', 'dgp_face_workflow_v3.py',
                    'outputs/cctv_dgp_decoder_app_normalization_v1/results.json',
                    'outputs/cctv_dgp_decoder_app_normalization_v1/independent_readback.json',
                    'outputs/cctv_dgp_original_decoder_gradient_v1_r2_vm/protocol.json',
                    'outputs/cctv_dgp_original_decoder_gradient_v1_r2_independent_audit.json',
                    'outputs/cctv_dgp_original_decoder_gradient_v1_r2_return_import.json']
    bindings = {name: sha(ROOT / name) for name in source_names}
    for name, digest in p['assets_sha256'].items():
        assert sha(PARENT / name) == digest
        bindings[(PARENT / name).relative_to(ROOT).as_posix()] = digest
    imported = read(ROOT / 'outputs/cctv_dgp_original_decoder_gradient_v1_r2_return_import.json')
    for name, digest in imported['files_sha256'].items():
        assert sha(RETURN / name) == digest, name
        bindings[(RETURN / name).relative_to(ROOT).as_posix()] = digest
    OUT.mkdir()
    write(OUT / 'plan.json', {
        'complete': True, 'date': '2026-10-06', 'frozen_before_forward_trace': True,
        'source_bindings_sha256': bindings, 'case_ids': [c['id'] for c in p['cases']],
        'hypothesis': 'The two head4 kernels are entirely float32 subnormal. Trace the unchanged fourth map, both convolution preactivations and the head output; compare an inference-only float64 two-convolution calculation to determine whether this path underflows. Do not change or accept the failed all14 gate.',
        'maximum_original_DGP_forwards': 50, 'maximum_fixed_float64_convolutions': 100,
        'cap_seconds': 300, 'local_gradient_calls': 0, 'local_optimizer_updates': 0,
        'VM_actions': False, 'native_or_reserved_used': False,
        'criteria': ['Every saved head4 gradient value checked in every batch and component',
                     'Measure nonzero upstream input, exact float32 head output and both kernel ranges',
                     'Keep higher precision forward calculations separate from the actual model output',
                     'Record original weights/state and all partial evidence without altering historical gates']})
    import numpy as np
    import torch
    from torch.nn import functional as F
    from PIL import Image
    sys.path.insert(0, str(PARENT))
    from cctv_dgp_pilot import state_hash
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    torch.set_num_threads(4)
    all_batches = [np.load(RETURN / ('outputs/gradients/batch' + str(i) + '.npy'), allow_pickle=False) for i in range(10)]
    total = np.load(RETURN / 'outputs/gradient_components.npy', allow_pickle=False)
    assert total.shape == (7, 609219) and total.dtype == np.float64 and np.isfinite(total).all()
    assert np.array_equal(sum(all_batches, np.zeros_like(total)), total)
    partitions = []
    for parameter in diagnostic['parameter_layout']:
        begin, end = parameter['start'], parameter['end']
        block = total[:, begin:end]
        per_batch = []
        for index, batch in enumerate(all_batches):
            values = batch[:, begin:end]
            per_batch.append({'batch': index, 'nonzero_values': int(np.count_nonzero(values)),
                              'component_nonzero_values': np.count_nonzero(values, axis=1).tolist(),
                              'improvement_gradient_norm': float(np.linalg.norm(values[:3].sum(0)))})
        partitions.append({'name': parameter['name'], 'parameters': end - begin,
                           'aggregate_nonzero_values': int(np.count_nonzero(block)),
                           'aggregate_improvement_gradient_norm': float(np.linalg.norm(block[:3].sum(0))),
                           'batches': per_batch})
    zero = [row['name'] for row in partitions if not row['aggregate_nonzero_values']]
    assert zero == ['head4.block0.weight', 'head4.block1.weight']
    assert all(not row['nonzero_values'] for part in partitions if part['name'] in zero for row in part['batches'])
    write(OUT / 'saved_gradient_partitions.json', {'complete': True, 'partitions': partitions,
          'zero_in_every_component_and_batch': zero, 'active_aggregate_tensors': 12,
          'saved_derivatives_only_no_local_derivatives': True})
    model, _ = load_frozen_dgp_restorer(PARENT / 'weights/dgp_v2.pth',
                                      expected_sha256=p['assets_sha256']['weights/dgp_v2.pth'], device='cpu')
    before = state_hash(model.net)
    weights = {}; weight_stats = {}
    tiny, smallest = float(np.finfo(np.float32).tiny), float(np.finfo(np.float32).smallest_subnormal)
    for name in zero:
        value = dict(model.net.named_parameters())[name].detach().cpu().numpy().copy()
        weights[name] = torch.from_numpy(value).double()
        np.save(OUT / (name.replace('.', '_') + '.npy'), value, allow_pickle=False)
        weight_stats[name] = {**statistics(value), 'float32_minimum_normal': tiny,
                              'float32_minimum_subnormal': smallest,
                              'subnormal_nonzero_values': int(((np.abs(value) < tiny) & (value != 0)).sum())}
    write(OUT / 'weight_statistics.json', {'complete': True, 'weights': weight_stats,
          'all_head4_weight_values_nonzero_subnormal': all(r['subnormal_nonzero_values'] == r['values'] for r in weight_stats.values())})
    captures = {}
    hooks = []
    names = ['fpn.enc4', 'fpn.lateral4', 'head1', 'head2', 'head3', 'head4.block0', 'head4.block1', 'head4']
    for name in names:
        module = dict(model.net.named_modules())[name]
        def hook(_module, _inputs, value, name=name):
            # Clone before FPNHead's subsequent in-place ReLU changes this value.
            captures[name] = value.detach().cpu().numpy().copy()
        hooks.append(module.register_forward_hook(hook))
    fn = next(n for n in ast.parse((ROOT / 'dgp_face_workflow_v3.py').read_text(encoding='utf-8')).body
              if isinstance(n, ast.FunctionDef) and n.name == 'canonical_tensor')
    namespace = {'np': np, 'torch': torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<actual-app-encoder-only>', 'exec'), namespace)
    arrays = OUT / 'activations'; arrays.mkdir()
    rows = []; counts = {'original_DGP': 0, 'fixed_float64_convolutions': 0}
    with torch.no_grad():
        for case in p['cases']:
            assert time.monotonic() - start < 300, 'Forward trace cap300 seconds'
            assert case['role'] == 'train'
            captures.clear()
            with Image.open(PARENT / case['input']) as im:
                camera = np.asarray(im.convert('RGB')).copy()
            model(namespace['canonical_tensor'](camera, 'cpu'))
            counts['original_DGP'] += 1
            assert set(captures) == set(names)
            upstream = torch.from_numpy(captures['fpn.lateral4']).double()
            first = F.conv2d(upstream, weights['head4.block0.weight'], padding=1)
            second = F.conv2d(first.relu(), weights['head4.block1.weight'], padding=1)
            counts['fixed_float64_convolutions'] += 2
            captures['float64_head4_block0'] = first.numpy().copy()
            captures['float64_head4_block1'] = second.numpy().copy()
            captures['float64_head4'] = second.relu().numpy().copy()
            path_stats = {name: statistics(value) for name, value in captures.items()}
            paths = {}
            for name in ['fpn.lateral4', 'head4.block0', 'head4.block1', 'head4',
                         'float64_head4_block0', 'float64_head4_block1', 'float64_head4']:
                path = arrays / (case['id'] + '__' + name.replace('.', '_') + '.npy')
                np.save(path, captures[name], allow_pickle=False)
                paths[name] = {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path)}
            rows.append({'id': case['id'], 'source': case['source'], 'profile': case['profile'], 'role': 'train',
                         'statistics': path_stats, 'saved_arrays': paths})
    for hook in hooks:
        hook.remove()
    assert counts == {'original_DGP': 50, 'fixed_float64_convolutions': 100}
    assert state_hash(model.net) == before == diagnostic['original_DGP_state']
    assert all(not value.requires_grad and value.grad is None for value in model.parameters())
    for name, digest in bindings.items():
        assert sha(ROOT / name) == digest, name
    record = {'complete': True, 'date': '2026-10-06', 'plan_sha256': sha(OUT / 'plan.json'),
              'source_bindings_sha256': bindings, 'saved_gradient_values_checked': total.size * 11,
              'zero_VM_gradient_tensors_all_batches': zero, 'active_aggregate_VM_gradient_tensors': 12,
              'all_head4_weights_nonzero_subnormal': all(r['subnormal_nonzero_values'] == r['values'] for r in weight_stats.values()),
              'cases': rows, 'inference_counts': counts, 'DGP_state_before_after': before,
              'unchanged_CPU_head4_outputs_exact_zero_cases': sum(not row['statistics']['head4']['nonzero'] for row in rows),
              'CPU_fourth_map_nonzero_cases': sum(bool(row['statistics']['fpn.lateral4']['nonzero']) for row in rows),
              'float64_head4_nonzero_cases': sum(bool(row['statistics']['float64_head4']['nonzero']) for row in rows),
              'float64_maximum_head4_below_float32_subnormal_cases': sum(row['statistics']['float64_head4']['maximum_absolute'] < smallest for row in rows),
              'failure_retained': True, 'historical_all14_gate_waived': False,
              'CUDA_intermediate_activations_measured': False,
              'checkpoint_history_unique_cause_claimed': False, 'quality_improvement_claimed': False,
              'local_gradient_calls': 0, 'local_backward_calls': 0, 'local_optimizer_updates': 0,
              'native_or_reserved_used': False, 'VM_actions': False, 'new_training_recipe_created': False,
              'app_promotion': False, 'independent_final_review': False, 'goal_complete': False,
              'seconds': time.monotonic() - start}
    write(OUT / 'results.json', record)
    print(json.dumps({key: record[key] for key in ['complete', 'zero_VM_gradient_tensors_all_batches',
          'unchanged_CPU_head4_outputs_exact_zero_cases', 'CPU_fourth_map_nonzero_cases',
          'all_head4_weights_nonzero_subnormal', 'float64_head4_nonzero_cases',
          'float64_maximum_head4_below_float32_subnormal_cases', 'seconds']}, indent=2))


if __name__ == '__main__':
    main()
