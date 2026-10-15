"""Test the real pinned initializer with file_digest unavailable; no forwards/gradients."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import types

import numpy as np
import torch

import cctv_dgp_bank_pinned_initializer_v2 as loader

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / 'outputs/cctv_dgp_bank_comparison_v1_r3_B_return'
OUT = ROOT / 'outputs/cctv_dgp_bank_comparison_v1_r3_B_analysis'


def main():
    receipt = OUT / 'hash_compat_loader_regression.json'
    assert not receipt.exists()
    started = time.monotonic()
    exe = json.loads((OUT / 'failure_execution_audit.json').read_text(encoding='utf-8'))
    assert exe['complete'] and exe['missing_hash_API_reproduced']
    assert exe['actual_optimizer_updates'] == 0
    prior_path = PACKET / 'cctv_dgp_bank_pinned_initializer_v1.py'
    fixed_path = Path(loader.__file__).resolve()
    old, new = [ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 10))
                for path in [prior_path, fixed_path]]
    functions = lambda tree: {node.name: ast.dump(node, include_attributes=False)
                             for node in tree.body if isinstance(node, ast.FunctionDef)}
    old_functions, new_functions = functions(old), functions(new)
    assert old_functions.keys() == new_functions.keys()
    assert {name for name in old_functions if old_functions[name] != new_functions[name]} == {'sha'}
    # Simulate precisely the unavailable stdlib API, without altering global hashlib.
    loader.hashlib = types.SimpleNamespace(sha256=hashlib.sha256)
    assert not hasattr(loader.hashlib, 'file_digest')
    sys.path.insert(0, str(PACKET))
    from cctv_dgp_bank_comparison_v1_model import ConditionedBankDGP
    from cctv_dgp_generative_bank_v1 import load_prior
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    p = json.loads((PACKET / 'protocol.json').read_text(encoding='utf-8'))
    path = PACKET / 'initializers/initial_B.pth'
    pin = p['assets_sha256']['initializers/initial_B.pth']
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    hash_cases = []
    for name in ['initializers/initial_B.pth', 'protocol.json', 'weights/dgp_v2.pth']:
        expected = hashlib.sha256((PACKET / name).read_bytes()).hexdigest()
        assert loader.sha(PACKET / name) == expected
        hash_cases.append(dict(name=name, sha256=expected))
    with torch.no_grad():
        original, _ = load_frozen_dgp_restorer(PACKET / 'weights/dgp_v2.pth', expected_sha256=p['original_checkpoint_sha256'])
        prior = load_prior(PACKET / 'prior_source', PACKET / 'bank_implementation', p['standalone_prior_sha256'])
        model = ConditionedBankDGP(original.net, prior,
            torch.from_numpy(np.load(PACKET / 'initializers/mean_style.npy', allow_pickle=False))).eval().requires_grad_(False)
        before = loader.state_sha(model.state_dict())
        exact = loader.load_pinned_initializer(model, path, pin, 'B', {})
        assert exact['every_loaded_tensor_exact'] and exact['loaded_state_sha256'] == before
        model.style_conditioner[0].weight.flatten()[0].add_(.125)
        assert loader.state_sha(model.state_dict()) != before
        repaired = loader.load_pinned_initializer(model, path, pin, 'B', {})
        assert repaired['loaded_state_sha256'] == before
        assert [row['name'] for row in repaired['constructor_differences']] == ['style_conditioner.0.weight']
        rejected = []

        def reject(name, mutation=None):
            if mutation is not None:
                value = mutation.detach().clone()
                mutation.flatten()[0].add_(.125)
            changed = loader.state_sha(model.state_dict())
            try:
                loader.load_pinned_initializer(model, path, '0'*64 if name == 'wrong_initializer_hash' else pin, 'B', {})
            except AssertionError as exc:
                assert loader.state_sha(model.state_dict()) == changed
                rejected.append(dict(case=name, message=str(exc), no_state_overwrite=True))
            else:
                raise AssertionError('Unsafe initializer mutation accepted: ' + name)
            if mutation is not None:
                mutation.copy_(value)
            assert loader.state_sha(model.state_dict()) == before

        reject('frozen_prior_change', model.prior.constant_input.weight)
        reject('zero_fusion_change', model.fusions['16'].bias)
        reject('wrong_initializer_hash')
    assert loader.sha(path) == pin
    record = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        checker_sha256=loader.sha(__file__), loader_sha256=loader.sha(fixed_path),
        original_loader_sha256=loader.sha(prior_path), failure_execution_audit_sha256=loader.sha(OUT / 'failure_execution_audit.json'),
        only_loader_function_changed='sha', hashlib_file_digest_unavailable_in_checked_module=True,
        actual_local_Python310_interpreter_used=False, hash_API_absence_simulation=True,
        actual_B_initializer_sha256=pin, pinned_model_state_sha256=before,
        hashes_checked=hash_cases, cases_checked=5,
        exact_existing_constructor_load=exact, permitted_constructor_difference_restore=repaired,
        unsafe_changes_rejected=rejected, every_loaded_tensor_bit_exact=True,
        synthetic_constructor_perturbation_not_original_VM_tensor_diagnosis=True,
        local_neural_forwards=0, local_gradient_queries=0, local_optimizer_updates=0,
        VM_loader_and_training_still_unverified=True, model_qualified=False,
        goal_complete=False, seconds=time.monotonic()-started)
    with receipt.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(dict(complete=True, real_loader_cases=5, hashes_checked=3,
               file_digest_unavailable=True, model_forwards_or_updates=0), flush=True)


if __name__ == '__main__':
    main()
