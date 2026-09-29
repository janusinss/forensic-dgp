"""GPU-only execution and fixed sampling for the VM feature-head pilot."""
import numpy as np
import torch

def require_cuda(min_free_gib=6):
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires the VM GPU; CPU execution is disabled')
    free,total=torch.cuda.mem_get_info()
    if free<min_free_gib*1024**3:
        raise RuntimeError(f'At least {min_free_gib} GiB free GPU memory required; free={free/1024**3:.2f} GiB')
    return torch.device('cuda')

def balanced_batches(groups,seed):
    if len(groups)!=6 or not all(groups) or max(map(len,groups))>160:
        raise ValueError('Expected six nonempty groups, each at most160 examples')
    rng=np.random.default_rng(seed);streams=[]
    for group in groups:
        stream=[]
        while len(stream)<160:stream.extend(rng.permutation(group).tolist())
        streams.append(stream[:160])
    for step in range(80):
        yield [i for stream in streams for i in stream[step*2:step*2+2]]
