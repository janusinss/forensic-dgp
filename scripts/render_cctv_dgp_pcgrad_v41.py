"""Exact scoped rendering from immutable V40; one learning treatment plus telemetry."""
from pathlib import Path
import ast

NAME = 'cctv_dgp_pcgrad_fit_vm_v41'
STEM = 'cctv-dgp-pcgrad-fit-v41'
FORMAT = 'own-DGP-spatial-path-PCGrad-full-TRAIN-v41'

OLD_STEP = '''            objective.backward(); progress['backwards'] += 1
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in parameters)
            torch.nn.utils.clip_grad_norm_(parameters, 1); optimizer.step(); progress['optimizer_updates'] = update'''

NEW_STEP = '''            before_values = np.concatenate([v.detach().cpu().numpy().reshape(-1) for v in parameters]).copy()
            components = []
            for component_number, term in enumerate(terms.values()):
                derivatives = torch.autograd.grad(term.mean(), parameters, retain_graph=component_number < 6, create_graph=False, allow_unused=False)
                progress['component_gradient_queries'] += 1
                assert all(torch.isfinite(v).all() for v in derivatives)
                components.append(np.concatenate([v.detach().cpu().numpy().reshape(-1) for v in derivatives]))
            components = np.stack(components).astype(np.float32)
            assert components.shape == (7, 17952)
            merged, projection = combine(components, update)
            assert projection['orders'] == p['gradient_orders'][update-1]
            offset = 0
            for parameter in parameters:
                parameter.grad = torch.from_numpy(merged[offset:offset+parameter.numel()].copy()).to(parameter.device).reshape(parameter.shape)
                offset += parameter.numel()
            assert offset == 17952 and all(torch.isfinite(v.grad).all() for v in parameters)
            torch.nn.utils.clip_grad_norm_(parameters, 1)
            applied = np.concatenate([v.grad.detach().cpu().numpy().reshape(-1) for v in parameters]).copy()
            optimizer.step(); progress['optimizer_updates'] = update
            after_values = np.concatenate([v.detach().cpu().numpy().reshape(-1) for v in parameters]).copy()
            first = np.concatenate([optimizer.state[v]['exp_avg'].detach().cpu().numpy().reshape(-1) for v in parameters]).copy()
            second = np.concatenate([optimizer.state[v]['exp_avg_sq'].detach().cpu().numpy().reshape(-1) for v in parameters]).copy()
            assert all(float(optimizer.state[v]['step']) == update for v in parameters)
            proof = out/'steps'/('step'+str(update).zfill(4)+'.npz')
            with proof.open('xb') as stream:
                np.savez(stream, components=components, combined=merged, applied=applied, before=before_values, after=after_values, exp_avg=first, exp_avg_sq=second)
            write(proof.with_suffix('.json'), {'complete': True, 'update': update, 'case_indices': ids,
                'case_ids': [items[i]['case']['id'] for i in ids], 'component_queries': 7,
                'component_means': [float(v.mean().detach()) for v in terms.values()], 'raw_objective': float(objective.detach()),
                'projection': projection, 'proof_sha256': sha(proof), 'optimizer_steps_all_equal_update': True,
                'optimizer_type': 'AdamW', 'loss_weights_or_definitions_changed': False, 'backward_calls': 0})
            progress['logged_updates'] = update'''


def worker(original):
    value = original.replace('cctv_dgp_spatial_fit_vm_v40', NAME).replace('cctv-dgp-spatial-fit-v40', STEM)
    value = value.replace('own-DGP-proven-spatial-path-full-TRAIN-v40', FORMAT)
    value = value.replace('cctv_dgp_spatial_decoder_v40', 'cctv_dgp_spatial_decoder_v41').replace('SpatialDGPCandidateV40', 'SpatialDGPCandidateV41')
    value = value.replace('V40', 'V41')
    assert value.count(OLD_STEP) == 1
    value = value.replace(OLD_STEP, NEW_STEP)
    needle = "    progress = {'optimizer_updates': 0, 'backwards': 0, 'optimizer_constructed': False, 'new_trained_checkpoint': False}"
    assert value.count(needle) == 1
    value = value.replace(needle, needle+"\n    progress.update({'component_gradient_queries': 0, 'logged_updates': 0})")
    needle = '        import numpy as np'
    assert value.count(needle) == 1
    value = value.replace(needle, needle+'\n        from cctv_dgp_pcgrad_v41 import combine')
    needle = "        progress['optimizer_constructed'] = True; fit_started = time.monotonic()"
    assert value.count(needle) == 1
    value = value.replace(needle, needle+"\n        (out/'steps').mkdir()")
    needle = "    assert p['format'] == '"+FORMAT+"'"
    assert value.count(needle) == 1
    value = value.replace(needle, needle+"\n    from cctv_dgp_pcgrad_v41 import task_orders\n    assert p['gradient_orders'] == [task_orders(i) for i in range(1,801)]")
    ast.parse(value, feature_version=(3, 10)); return value


def decoder(original):
    value = original.replace('cctv_dgp_spatial_fit_vm_v40', NAME).replace('SpatialDGPCandidateV40', 'SpatialDGPCandidateV41').replace('Distinct V40', 'Distinct V41')
    assert value.replace(NAME, 'cctv_dgp_spatial_fit_vm_v40').replace('SpatialDGPCandidateV41', 'SpatialDGPCandidateV40').replace('Distinct V41', 'Distinct V40') == original
    ast.parse(value, feature_version=(3, 10)); return value


PROOF_AUDIT = '''
def step_evidence(p, result):
    import torch
    from cctv_dgp_pcgrad_v41_evidence import read_arrays, verify_step
    n = result['optimizer_updates']; logged = result['logged_updates']
    assert n == logged and result['backwards'] == 0
    assert 7*n <= result['component_gradient_queries'] <= 7*(n+1)
    paths = sorted((OUT/'outputs/steps').glob('*.json')) if (OUT/'outputs/steps').exists() else []
    arrays_paths = sorted((OUT/'outputs/steps').glob('*.npz')) if (OUT/'outputs/steps').exists() else []
    assert len(paths) == len(arrays_paths) == n
    seed = torch.load(BUNDLE/'untrained_initial_decoder.pth', map_location='cpu', weights_only=True)
    vector = lambda state: np.concatenate([state[r['name']].detach().numpy().reshape(-1) for r in p['parameter_layout']]).copy()
    previous = vector(seed); first = np.zeros(17952, np.float32); second = first.copy(); maximum_error = 0.
    snapshot_values = {0: previous.copy()}; schedule = read(BUNDLE/'schedule.json')['batches']
    for update, (path, proof) in enumerate(zip(paths, arrays_paths), 1):
        assert path.name == 'step'+str(update).zfill(4)+'.json' and proof.with_suffix('.json') == path
        record = read(path); assert record['complete'] and record['proof_sha256'] == sha(proof)
        assert record['case_indices'] == schedule[update-1]
        assert record['case_ids'] == [p['cases'][i]['id'] for i in schedule[update-1]]
        assert len(record['component_means']) == 7 and np.isfinite(record['component_means']).all()
        assert abs(sum(record['component_means'])-record['raw_objective']) <= 2e-6*max(1.,abs(record['raw_objective']))
        arrays = read_arrays(proof); error = verify_step(arrays, record, update, previous, first, second)
        maximum_error = max(maximum_error, error)
        previous, first, second = arrays['after'], arrays['exp_avg'], arrays['exp_avg_sq']
        if update in p['snapshots']: snapshot_values[update] = previous.copy()
    for update, expected in snapshot_values.items():
        path = OUT/('outputs/update'+str(update)+'/spatial_decoder.pth')
        if path.exists(): assert np.array_equal(vector(torch.load(path,map_location='cpu',weights_only=True)), expected)
    stop = OUT/'outputs/stopped_spatial_decoder.pth'
    if stop.exists(): assert np.array_equal(vector(torch.load(stop,map_location='cpu',weights_only=True)), previous)
    return {'complete':True,'all_logged_updates_verified':n,'saved_component_queries':7*n,
        'AdamW_readback_maximum_absolute_parameter_error':maximum_error,'prospective_parameter_arithmetic_tolerance':3e-7,
        'combined_gradient_roundoff_allowance':'Two float32 ULPs plus1e-12',
        'parameter_chain_and_snapshot_tensors_exact':True,'original_loss_weights_gates_and_schedule_retained':True,
        'saved_gradients_not_independently_differentiated':True,'local_gradient_calls':0,'local_optimizer_updates':0}

'''


def auditor(original):
    value = original.replace('cctv_dgp_spatial_fit_vm_v40', NAME).replace('cctv_dgp_spatial_fit_v40_return', 'cctv_dgp_pcgrad_fit_v41_return')
    value = value.replace('cctv-dgp-spatial-fit-v40', STEM).replace('cctv_dgp_spatial_decoder_v40', 'cctv_dgp_spatial_decoder_v41')
    value = value.replace('SpatialDGPCandidateV40', 'SpatialDGPCandidateV41')
    value = value.replace('cctv_dgp_spatial_fit_v40_preparation', 'cctv_dgp_pcgrad_fit_v41_preparation')
    value = value.replace('cctv_dgp_spatial_fit_v40_return_import.json', 'cctv_dgp_pcgrad_fit_v41_return_import.json')
    value = value.replace('cctv_dgp_spatial_fit_v40_independent_audit.json', 'cctv_dgp_pcgrad_fit_v41_independent_audit.json')
    needle = '    for update in p[\'snapshots\']:'
    assert value.count(needle) == 2
    at = value.index(needle)
    value = value[:at]+"    allowed.update('outputs/steps/step'+str(i).zfill(4)+s for i in range(1,801) for s in ['.json','.npz'])\n"+value[at:]
    needle = "    assert 0 <= r['optimizer_updates'] <= 800 and r['optimizer_updates'] <= r['backwards'] <= r['optimizer_updates']+1"
    assert value.count(needle) == 1
    value = value.replace(needle, "    assert 0 <= r['optimizer_updates'] <= 800 and r['backwards'] == 0")
    needle = "        assert r['optimizer_updates'] == r['backwards'] == 800 and r['completed_epochs'] == 1"
    assert value.count(needle) == 1
    value = value.replace(needle, "        assert r['optimizer_updates'] == r['logged_updates'] == 800 and r['component_gradient_queries'] == 5600 and r['completed_epochs'] == 1")
    needle = "    reviewed = []; gates = []; replays = []"
    assert value.count(needle) == 1
    value = value.replace(needle, "    learning_evidence = step_evidence(p, r)\n"+needle)
    needle = "'members_verified': len(members), 'snapshots': reviewed"
    assert value.count(needle) == 1
    value = value.replace(needle, "'members_verified': len(members), 'per_update_learning_evidence': learning_evidence, 'snapshots': reviewed")
    needle = '\ndef audit(digest, size):'
    assert value.count(needle) == 1
    value = value.replace(needle, '\n'+PROOF_AUDIT+needle)
    ast.parse(value, feature_version=(3, 10)); return value
