"""Fixed-weight decoder/bank cross-composition; no forward or learning calls."""

VARIANTS = ['initial', 'joint', 'decoder_only', 'bank_only']
BANK_PREFIXES = ('style_conditioner.', 'spatial_conditioners.', 'fusions.')


def validate_states(initial, trained, learning_names):
    import torch
    assert set(initial) == set(trained)
    names = set(learning_names)
    decoder = {n for n in names if n.startswith('current.')}
    bank = {n for n in names if n.startswith(BANK_PREFIXES)}
    assert len(names) == 37 and len(decoder) == 15 and len(bank) == 22
    assert names == decoder | bank and len(initial) - len(names) == 755
    changed = set()
    for name, a in initial.items():
        b = trained[name]
        assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
        assert a.device.type == b.device.type == 'cpu'
        assert a.shape == b.shape and a.dtype == b.dtype
        assert torch.isfinite(a).all() and torch.isfinite(b).all(), name
        if not torch.equal(a, b):
            changed.add(name)
    assert changed == names, 'All37 learned tensors changed; every frozen tensor stays exact'
    assert all(torch.count_nonzero(a) == 0 for n, a in initial.items()
               if n.startswith(('spatial_conditioners.', 'fusions.')))
    return {'decoder': decoder, 'bank': bank, 'frozen': set(initial) - names}


def compose(initial, trained, partitions, variant):
    assert variant in VARIANTS
    chosen = set()
    if variant in ('joint', 'decoder_only'):
        chosen |= partitions['decoder']
    if variant in ('joint', 'bank_only'):
        chosen |= partitions['bank']
    return {n: (trained[n] if n in chosen else initial[n]) for n in initial}
