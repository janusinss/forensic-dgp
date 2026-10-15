"""Inspect fresh CPU construction against the sealed initializer; no forwards or gradients."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / 'outputs/cctv_dgp_bank_comparison_v1_r2_B_return'
OUT = ROOT / 'outputs/cctv_dgp_bank_comparison_v1_r2_B_analysis'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    receipt = OUT / 'CPU_initializer_inspection.json'
    assert not receipt.exists()
    started = time.monotonic()
    imported = json.loads((OUT / 'import.json').read_text())
    assert imported['complete'] and sha(PACKET / 'protocol.json') == imported['protocol_sha256']
    protocol = json.loads((PACKET / 'protocol.json').read_text())
    sys.path.insert(0, str(PACKET))
    from cctv_dgp_bank_comparison_v1_model import ConditionedBankDGP
    from cctv_dgp_generative_bank_v1 import load_prior
    from cctv_dgp_pilot import state_hash
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    bound = {n: sha(PACKET / n) for n in protocol['assets_sha256']
             if n.startswith(('initializers/', 'weights/', 'prior_source/')) or n.endswith('_model.py')}
    torch.manual_seed(20261010)
    with torch.no_grad():
        original, _ = load_frozen_dgp_restorer(PACKET / 'weights/dgp_v2.pth',
            expected_sha256=protocol['original_checkpoint_sha256'])
        prior = load_prior(PACKET / 'prior_source', PACKET / 'bank_implementation', protocol['standalone_prior_sha256'])
        mean = torch.from_numpy(np.load(PACKET / 'initializers/mean_style.npy', allow_pickle=False))
        model = ConditionedBankDGP(original.net, prior, mean).eval().requires_grad_(False)
        initial = torch.load(PACKET / 'initializers/initial_B.pth', map_location='cpu', weights_only=True)
        actual = model.state_dict()
        assert set(actual) == set(initial)
        differences = []
        for name, value in actual.items():
            expected = initial[name]
            assert value.dtype == expected.dtype and value.shape == expected.shape
            assert torch.isfinite(value).all() and torch.isfinite(expected).all()
            if not torch.equal(value, expected):
                differences.append(dict(name=name, elements=value.numel(),
                    maximum_absolute_error=float((value.double()-expected.double()).abs().max()),
                    changed_elements=int((value != expected).sum())))
        fresh_hash = state_hash(model)
        model.load_state_dict(initial, strict=True)
        assert state_hash(model) == state_hash(initial)
        frozen_names = set(initial) - set(model.learning_names())
        assert all(torch.equal(model.state_dict()[n], initial[n]) for n in frozen_names)
    assert all(sha(PACKET / n) == h for n, h in bound.items())
    record = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), checker_sha256=sha(__file__),
        protocol_sha256=imported['protocol_sha256'], source_bindings_unchanged=bound,
        torch_CPU_version=torch.__version__, VM_torch_version='2.9.1+cu129',
        fresh_CPU_state_hash=fresh_hash, pinned_initializer_state_hash=state_hash(initial),
        fresh_CPU_equals_pinned=fresh_hash == state_hash(initial), differences=differences,
        strict_authoritative_load_exact=True, frozen_tensors_exact_after_load=True,
        VM_constructor_tensor_differences_not_saved_in_original_failure=True,
        local_neural_forwards=0, local_gradient_queries=0, local_optimizer_updates=0,
        model_qualified=False, goal_complete=False, seconds=time.monotonic()-started)
    with receipt.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print(dict(complete=True, fresh_CPU_equals_pinned=record['fresh_CPU_equals_pinned'],
               changed_tensors=[r['name'] for r in differences], forwards_or_updates=0))


if __name__ == '__main__':
    main()
