"""Real initializer regression checks without forwards, gradients or optimizer."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from cctv_dgp_bank_pinned_initializer_v1 import load_pinned_initializer, sha, state_sha

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / 'outputs/cctv_dgp_bank_comparison_v1_r2_B_return'
OUT = ROOT / 'outputs/cctv_dgp_bank_comparison_v1_r2_B_analysis'


def main():
    receipt = OUT / 'pinned_loader_regression.json'
    assert not receipt.exists()
    started = time.monotonic()
    sys.path.insert(0, str(PACKET))
    from cctv_dgp_bank_comparison_v1_model import ConditionedBankDGP
    from cctv_dgp_generative_bank_v1 import load_prior
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    p = json.loads((PACKET / 'protocol.json').read_text())
    path = PACKET / 'initializers/initial_B.pth'
    pin = p['assets_sha256']['initializers/initial_B.pth']
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with torch.no_grad():
        original, _ = load_frozen_dgp_restorer(PACKET / 'weights/dgp_v2.pth', expected_sha256=p['original_checkpoint_sha256'])
        prior = load_prior(PACKET / 'prior_source', PACKET / 'bank_implementation', p['standalone_prior_sha256'])
        model = ConditionedBankDGP(original.net, prior,
            torch.from_numpy(np.load(PACKET / 'initializers/mean_style.npy', allow_pickle=False))).eval().requires_grad_(False)
        before = state_sha(model.state_dict())
        exact = load_pinned_initializer(model, path, pin, 'B', {})
        assert exact['every_loaded_tensor_exact'] and exact['loaded_state_sha256'] == before
        parameter = model.style_conditioner[0].weight
        parameter.flatten()[0].add_(.125)
        assert state_sha(model.state_dict()) != before, 'Original fresh-equality requirement must reject this case'
        repaired = load_pinned_initializer(model, path, pin, 'B', {})
        assert repaired['loaded_state_sha256'] == before
        assert [r['name'] for r in repaired['constructor_differences']] == ['style_conditioner.0.weight']
        rejected = []

        def reject(name, mutation=None):
            if mutation is not None:
                value = mutation.detach().clone()
                mutation.flatten()[0].add_(.125)
            changed = state_sha(model.state_dict())
            try:
                load_pinned_initializer(model, path, '0'*64 if name == 'wrong_initializer_hash' else pin, 'B', {})
            except AssertionError as exc:
                assert state_sha(model.state_dict()) == changed, 'Rejected state must not be overwritten'
                rejected.append(dict(case=name, message=str(exc), no_state_overwrite=True))
            else:
                raise AssertionError('Unsafe initializer mutation accepted: '+name)
            if mutation is not None:
                mutation.copy_(value)
            assert state_sha(model.state_dict()) == before

        reject('frozen_prior_change', model.prior.constant_input.weight)
        reject('zero_fusion_change', model.fusions['16'].bias)
        reject('wrong_initializer_hash')
    assert sha(path) == pin
    record = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), checker_sha256=sha(__file__),
        loader_sha256=sha(Path(__file__).with_name('cctv_dgp_bank_pinned_initializer_v1.py')),
        actual_B_initializer_sha256=pin, pinned_model_state_sha256=before,
        original_fresh_state_equality_failure_reproduced=True,
        synthetic_constructor_perturbation_not_VM_tensor_diagnosis=True,
        exact_existing_constructor_load=exact, permitted_constructor_difference_restore=repaired,
        unsafe_changes_rejected=rejected, cases_checked=5,
        every_loaded_tensor_bit_exact=True, no_model_forwards=True,
        local_gradient_queries=0, local_optimizer_updates=0,
        VM_loader_and_training_still_unverified=True, model_qualified=False,
        goal_complete=False, seconds=time.monotonic()-started)
    with receipt.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print(dict(complete=True, cases_checked=5, unsafe_changes_rejected=len(rejected),
               model_forwards_or_updates=0, VM_validation_pending=True))


if __name__ == '__main__':
    main()
