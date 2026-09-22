"""Training state, kept separate from weights exported for inference."""
import os
import random
import numpy as np
import torch


def save_training_state(path, model, optimizer, scheduler, epoch, best_psnr, config, extra=None):
    numpy_state = np.random.get_state()
    state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                 scheduler=scheduler.state_dict(), epoch=epoch, best_psnr=best_psnr,
                 config=config, torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                 numpy_rng=(numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
                 cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    if extra is not None:
        state['extra'] = extra
    temporary = str(path) + '.tmp'
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_training_state(path, model, optimizer=None, scheduler=None, device='cpu'):
    state = torch.load(path, map_location=device, weights_only=True)
    is_training_state = 'model' in state and 'optimizer' in state
    model.load_state_dict(state['model'] if is_training_state else state, strict=True)
    if not is_training_state:
        return None
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])
    torch.set_rng_state(state['torch_rng'].cpu())
    random.setstate(state['python_rng'])
    rng = state['numpy_rng']
    np.random.set_state((rng[0], np.array(rng[1], dtype=np.uint32), *rng[2:]))
    if torch.cuda.is_available() and state['cuda_rng']:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
    return state
