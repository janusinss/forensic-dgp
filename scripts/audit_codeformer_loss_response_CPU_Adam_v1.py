"""Old-state CPU/NumPy equation readback, never construct/call a model or optimizer.

Only hash-verified pre-existing checkpoints use torch.load(weights_only=True,
map_location='cpu',mmap=True). GPU delta arrays are absent: hash mismatches
cannot establish a per-element replay error, even when scalar norms are close.
"""
import copy
import gc
import hashlib
from pathlib import Path
import sys
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_loss_response_v1 import contract as c
from research.codeformer_loss_response_v1.worker import tree_digest

REVIEW = ROOT / 'outputs/codeformer_loss_response_v1_return_review'
DEST = REVIEW / 'CPU_Adam_v1'


def sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def equation(parameter, gradient, moment, square, counter, rate):
    """One literal FP32 Adam recurrence; no optimizer object or mutated base."""
    m = np.asarray(np.float32(.9) * moment + np.float32(.1) * gradient, dtype=np.float32)
    v = np.asarray(np.float32(.99) * square + np.float32(.01) * gradient * gradient, dtype=np.float32)
    denominator = np.sqrt(v).astype(np.float32) / np.float32((1 - .99 ** (counter + 1)) ** .5)
    denominator += np.float32(1e-8)
    delta = np.float32(rate / (1 - .9 ** (counter + 1))) * m / denominator
    return np.asarray(parameter - delta, dtype=np.float32)


def fixtures():
    p = np.asarray([1., -1., 0.], dtype=np.float32)
    zero = np.zeros_like(p)
    c.require(np.array_equal(equation(p, zero, zero, zero, 0, 1e-5), p), 'Empty-state zero-gradient identity failed')
    moved = equation(p, np.asarray([2., -2., 0.], dtype=np.float32), zero, zero, 0, 1e-5)
    c.require(moved[0] < p[0] and moved[1] > p[1] and moved[2] == p[2], 'First-step sign fixture failed')
    c.require(not np.array_equal(equation(p, zero, np.ones_like(p), np.ones_like(p), 50, 1e-5), p),
              'Retained momentum should move despite explicit current zeros')


def main():
    fixtures()
    c.require(not DEST.exists(), 'Preserve previous CPU readback')
    imported = c.read_json(REVIEW / 'return_import_v1.json')
    c.require(imported['complete'] and imported['protocol_sha256'] ==
              '23c19c394e34261861b8853cfd60aa60d1cfaefc91b12e92a2f8b8dce85112ab', 'Verified import required')
    returned = Path(imported['import_root'])
    protocol = c.read_json(returned / 'protocol.json')
    resolution = c.read_json(ROOT / 'outputs/codeformer_loss_response_design_v1/stage_v1_r1.json')['local_dependency_resolution']
    initialization = c.read_json(returned / 'outputs/loss_response_v1/initialization.json')
    names = initialization['census']['trainable_names']
    c.require(len(names) == 350 and len(set(names)) == 350, 'Exact trainable order required')
    DEST.mkdir()
    # Tensor deserialization only. No model construction, forward, autograd,
    # optimizer object, optimizer.step or CUDA tensor operation follows.
    import torch
    torch.set_num_threads(1)
    records = []
    for label, counter, asset in [('original0', 0, 'original_checkpoint'), ('stopped50', 50, 'stopped_checkpoint')]:
        declared = protocol['assets'][asset]
        checkpoint = Path(resolution[declared['path']])
        c.require(c.sha256(checkpoint) == declared['sha256'] and checkpoint.stat().st_size == declared['bytes'],
                  'Old checkpoint fingerprint differs')
        state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        c.require(state['completed_updates'] == counter and state['partial_microbatches_discarded'] == 0,
                  'Saved committed state differs')
        model, optimizer = state['model'], copy.deepcopy(state['optimizer'])
        for group in optimizer['param_groups']:
            group['lr'] = protocol['optimizer']['fixed_group_rates'][group['name']]
        directory = returned / 'outputs/loss_response_v1' / label
        source = c.read_json(directory / 'source_summary.json')
        identities = {'base_model_sha256': tree_digest(model, torch),
                      'base_Adam_sha256': tree_digest(optimizer, torch),
                      'base_RNG_sha256': tree_digest(state['rng'], torch)}
        c.require(all(source[k] == v for k,v in identities.items()), 'CPU old-state identity differs from diagnostic base')
        c.require([g['params'] for g in optimizer['param_groups']] == [list(range(278)), list(range(278,350))],
                  'Saved optimizer named-index order differs')
        components = {term: c.read_json(directory / 'gradients' / term / 'manifest.json')['parameters'] for term in c.COMPONENTS}
        branches = {condition: c.read_json(directory / 'conditions' / condition.replace(':','__') / 'branch.json')
                    for condition in c.BRANCHES if condition != 'baseline'}
        outcomes = {condition: {'parameters': {}, 'exact_after_hashes': 0, 'exact_delta_hashes': 0,
                                'trainable_exact_after_hashes': 0, 'frozen_exact_after_hashes': 0,
                                'maximum_scalar_l2_difference': 0., 'maximum_scalar_maximum_difference': 0.}
                    for condition in branches}
        for name, tensor in model.items():
            c.require(tensor.device.type == 'cpu' and tensor.dtype == torch.float32, 'CPU FP32 state required')
            base = tensor.detach().numpy()
            base_sha = sha(base)
            trainable = name in names
            if trainable:
                index = names.index(name)
                moment_state = optimizer['state'].get(index)
                if counter == 0:
                    c.require(moment_state is None, 'Original state should have empty Adam')
                    m, v = np.zeros_like(base), np.zeros_like(base)
                else:
                    c.require(moment_state['step'].item() == 50, 'Retained Adam counter differs')
                    m = moment_state['exp_avg'].numpy()
                    v = moment_state['exp_avg_sq'].numpy()
                basis = {}
                for term in c.COMPONENTS:
                    row = components[term][name]
                    if row['is_none']:
                        basis[term] = None
                    else:
                        file = row['file']['path']
                        c.require(c.sha256(returned / file) == imported['files_sha256'][file], 'Gradient file changed')
                        a = np.load(returned / file, allow_pickle=False)
                        c.require(a.dtype == np.float32 and a.shape == base.shape and np.isfinite(a).all(),
                                  'Component geometry/dtype/finite differs')
                        basis[term] = a
            for condition, branch in branches.items():
                reported = branch['step']['parameter_deltas']['parameters'][name]
                c.require(reported['base_sha256'] == base_sha and reported['shape'] == list(base.shape),
                          'Actual old parameter bytes differ')
                if trainable:
                    assigned = None
                    omitted = condition[5:] if condition.startswith('drop:') else None
                    if condition != 'momentum_only':
                        for term in c.COMPONENTS:
                            a = basis[term]
                            if term != omitted and a is not None:
                                if assigned is None:
                                    assigned = a.copy()
                                else:
                                    np.add(assigned, a, out=assigned)
                    if assigned is None:
                        assigned = np.zeros_like(base)
                    c.require(sha(assigned) == branch['step']['assigned_gradient_sha256_by_tensor'][name],
                              'Assigned gradient exact hash differs')
                    rate = protocol['optimizer']['fixed_group_rates']['fusion' if name.startswith('fuse_convs_dict.') else 'encoder_token']
                    after = equation(base, assigned, m, v, counter, rate)
                else:
                    after = base
                delta = after - base
                after_exact, delta_exact = sha(after) == reported['after_sha256'], sha(delta) == reported['delta_sha256']
                l2 = float(np.sqrt(np.square(delta.astype(np.float64)).sum(dtype=np.float64)))
                maximum = float(np.abs(delta).max())
                outcome = outcomes[condition]
                outcome['parameters'][name] = {'CPU_after_sha256': sha(after), 'CPU_delta_sha256': sha(delta),
                    'after_hash_exact': after_exact, 'delta_hash_exact': delta_exact, 'trainable': trainable,
                    'CPU_delta_l2': l2, 'CUDA_recorded_delta_l2': reported['delta_l2'],
                    'CPU_maximum_absolute_delta': maximum, 'CUDA_recorded_maximum_absolute_delta': reported['maximum_absolute_delta']}
                outcome['exact_after_hashes'] += after_exact
                outcome['exact_delta_hashes'] += delta_exact
                outcome['trainable_exact_after_hashes'] += bool(trainable and after_exact)
                outcome['frozen_exact_after_hashes'] += bool(not trainable and after_exact)
                outcome['maximum_scalar_l2_difference'] = max(outcome['maximum_scalar_l2_difference'], abs(l2-reported['delta_l2']))
                outcome['maximum_scalar_maximum_difference'] = max(outcome['maximum_scalar_maximum_difference'],
                                                                  abs(maximum-reported['maximum_absolute_delta']))
                if not trainable:
                    c.require(after_exact and delta_exact, 'Frozen CPU/returned row differs')
            if trainable:
                del basis, assigned, m, v
        exact = sum(row['exact_after_hashes'] for row in outcomes.values())
        receipt = {'snapshot': label, 'counter_before': counter, 'diagnostic_counter_after': counter+1,
            'old_checkpoint_sha256': declared['sha256'], 'base_state_identities_exact': identities,
            'parameters_per_branch': len(model), 'branches': outcomes, 'exact_after_hashes': exact,
            'per_element_CPU_CUDA_error_bound_available_when_hash_differs': False,
            'full_CUDA_delta_arrays_not_exported': True, 'actual_optimizer_steps_performed_locally': 0}
        c.write_new_json(DEST / (label + '.json'), receipt)
        records.append({'snapshot': label, 'parameters_per_branch': len(model), 'exact_after_hashes': exact,
                        'possible_after_hashes': len(model)*9, 'base_state_identities_exact': True,
                        'branches': {key:{k:v for k,v in val.items() if k!='parameters'}for key,val in outcomes.items()}})
        print({'snapshot': label, 'exact_after_hashes': exact, 'possible': len(model)*9}, flush=True)
        del state, model, optimizer, components, branches, tensor, base, after, delta, moment_state
        gc.collect()
    c.write_new_json(DEST / 'receipt.json', {'complete': True, 'scope': 'CPU/NumPy old-state and literal Adam equation readback',
        'source_sha256': c.sha256(__file__), 'import_receipt_sha256': c.sha256(REVIEW/'return_import_v1.json'),
        'snapshots': records, 'Torch_version_deserialization_only': torch.__version__,
        'actual_local_neural_gradient_optimizer_VM_calls': 0, 'old_checkpoints_loaded_weights_only_CPU': 2,
        'GPU_delta_per_element_replay_proven': all(r['exact_after_hashes']==r['possible_after_hashes']for r in records),
        'quality_gate_changed': False, 'model_qualified': False})


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        DEST.mkdir(parents=True, exist_ok=True)
        failure = DEST / 'failure.json'
        if not failure.exists():
            c.write_new_json(failure, {'complete': False, 'error': str(error), 'traceback': traceback.format_exc(),
                'source_sha256': c.sha256(__file__), 'actual_local_neural_gradient_optimizer_VM_calls': 0})
        raise
