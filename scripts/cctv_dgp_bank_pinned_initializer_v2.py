"""Load the authoritative initializer exactly; never regenerate its random weights.

Only the new style conditioner's constructor values may differ across runtimes.
All retained/frozen/zero-fusion state must already equal the pinned initializer.
No model forwards, gradients, optimizer or tolerance changes are performed.
"""
import hashlib
from pathlib import Path

import torch


def sha(path):
    # hashlib.file_digest is unavailable on the recorded Python 3.10 VM.
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_sha(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        a = value.detach().cpu().contiguous()
        digest.update(name.encode()+b'\0'+str(a.dtype).encode()+b'\0')
        digest.update(str(tuple(a.shape)).encode()+b'\0'+a.numpy().tobytes())
    return digest.hexdigest()


def load_pinned_initializer(model, path, expected_sha256, label, record):
    path = Path(path)
    assert label in {'A', 'B'}
    assert sha(path) == expected_sha256, 'Pinned initializer file hash differs'
    initial = torch.load(path, map_location='cpu', weights_only=True)
    current = model.state_dict()
    assert isinstance(initial, dict) and set(initial) == set(current), 'Initializer keys differ'
    learning_names = set(model.learning_names())
    differences = []
    for name, value in current.items():
        expected = initial[name]
        assert isinstance(expected, torch.Tensor)
        assert value.dtype == expected.dtype and value.shape == expected.shape, 'Initializer tensor schema differs: '+name
        assert torch.isfinite(value).all() and torch.isfinite(expected).all(), 'Initializer tensor is nonfinite: '+name
        if not torch.equal(value.detach().cpu(), expected):
            differences.append(dict(name=name, constructor_sha256=tensor_sha(value),
                authoritative_sha256=tensor_sha(expected), elements=value.numel(),
                permitted_random_constructor=label == 'B' and name.startswith('style_conditioner.') and name in learning_names))
    record.update(route=label, initializer_file_sha256=expected_sha256,
        fresh_state_sha256=state_sha(current), authoritative_state_sha256=state_sha(initial),
        constructor_differences=differences, strict_load_complete=False,
        constructor_regeneration_not_cross_runtime_requirement=True,
        model_forwards=0, gradient_queries=0, optimizer_updates=0)
    assert all(r['permitted_random_constructor'] for r in differences), 'Retained, frozen, mean-style or zero-fusion constructor tensors differ'
    with torch.no_grad():
        model.load_state_dict(initial, strict=True)
    assert state_sha(model.state_dict()) == state_sha(initial), 'Loaded authoritative initializer differs'
    assert all(torch.equal(value.detach().cpu(), initial[name]) for name, value in model.state_dict().items())
    assert sha(path) == expected_sha256, 'Initializer changed while loading'
    record.update(strict_load_complete=True, every_loaded_tensor_exact=True,
        frozen_tensors_exact=True, loaded_state_sha256=state_sha(model.state_dict()))
    return record
