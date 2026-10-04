"""L4-only diagnostic of the retained model's objective; zero optimizer updates."""
import argparse
import math
from pathlib import Path
import platform
import shutil
import sys
import time

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from cctv_dgp_pilot import (read, write, sha, state_hash, PilotDataset, PilotPerceptual,
    FixedObservedIdentity, composite, masked_charbonnier)
from cctv_dgp_perceptual_training_v3 import load_starting_state, START_SHA, START_STATE
from cctv_dgp_objective_diagnostic_v4 import verify_recipe, TERM_WEIGHTS, TEACHERS, gradient_summary
from run_cctv_dgp_perceptual_vm_v3 import require_vm, DGPSynthesizer


def run(root, bundle_dir, output, perceptual_bundle_dir=None, parent_return=None, v3_return=None):
    require_vm(root)  # Never create output, load models or traverse a gradient graph locally.
    started = time.monotonic()
    deadline = started + 600
    protocol, manifest = verify_recipe(root, bundle_dir, perceptual_bundle_dir, parent_return, v3_return)
    if output.exists():
        raise ValueError('Preserve the existing diagnostic; no automatic repeat')
    output.mkdir(parents=True)
    calls = 0
    try:
        runtime = output / 'runtime_sources'
        runtime.mkdir()
        for name in manifest['assets_sha256']:
            target = runtime / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(bundle_dir / name, target)
        torch.set_num_threads(4)
        torch.manual_seed(protocol['seed'])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = DGPSynthesizer().cuda().eval().requires_grad_(True)
        model.load_state_dict(load_starting_state(root), strict=True)
        identity = FixedObservedIdentity(root / protocol['weights']['arcface'], 'cuda')
        perceptual = PilotPerceptual(root / protocol['weights']['vgg_trunk'], 'cuda')
        teachers_before = {'identity': state_hash(identity), 'perceptual': state_hash(perceptual)}
        if state_hash(model) != START_STATE or teachers_before != TEACHERS:
            raise ValueError('Starting student/teacher tensors differ')
        parameters = tuple(model.parameters())
        dimension = sum(p.numel() for p in parameters)
        write(output / 'execution.json', {
            'protocol_sha256': sha(bundle_dir / 'objective_diagnostic_protocol_v4.json'),
            'host': platform.node(), 'device': 'cuda', 'gpu': torch.cuda.get_device_name(0),
            'torch': torch.__version__, 'python': sys.version, 'starting_checkpoint_sha256': START_SHA,
            'starting_state_hash': START_STATE, 'teachers_before': teachers_before,
            'student_parameter_dimension': dimension, 'optimizer_constructed': False,
            'runtime_cap_seconds': 600, 'actual_training': False,
            'normalization_stats_cloned': True, 'grid_sample_cuda_backward_may_be_nondeterministic': True,
        })
        from models.losses import SobelGradientLoss
        sobel = SobelGradientLoss(device='cuda')
        for number, group in enumerate(manifest['groups']):
            if time.monotonic() >= deadline:
                raise TimeoutError('Ten-minute diagnostic budget exceeded')
            dataset = PilotDataset(root, protocol, group['cases'])
            batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)))
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            generated = composite(model(batch['low']), batch['low'], batch['mask'])
            with torch.no_grad():
                target_embedding = identity.embedding(batch['target'], batch['mask'], batch['grid'])
            terms = {
                'pixel': masked_charbonnier(generated, batch['target'], batch['mask']),
                'color': torch.nn.functional.l1_loss(torch.nn.functional.avg_pool2d(generated, 11, 1, 5),
                                                    torch.nn.functional.avg_pool2d(batch['target'], 11, 1, 5)),
                'vgg': perceptual(generated, batch['target']),
                'sobel': sobel(generated, batch['target']),
                'identity': (1 - (identity.embedding(generated, batch['mask'], batch['grid']) * target_embedding).sum(1)).mean(),
            }
            weighted = {k: v * TERM_WEIGHTS[k] for k, v in terms.items()}
            if any(not torch.isfinite(v) for v in weighted.values()):
                raise FloatingPointError('Nonfinite objective term')
            parameter_vectors, input_vectors = {}, {}
            for index, (name, term) in enumerate(weighted.items()):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Ten-minute diagnostic budget exceeded')
                gradients = torch.autograd.grad(term, (*parameters, generated),
                                                retain_graph=index < 4, allow_unused=True)
                calls += 1
                input_gradient = gradients[-1]
                if input_gradient is None or not torch.isfinite(input_gradient).all():
                    raise FloatingPointError('Missing/nonfinite restoration-input gradient')
                parameter_vectors[name] = torch.cat([
                    (g.detach() if g is not None else torch.zeros_like(p)).reshape(-1)
                    for p, g in zip(parameters, gradients[:-1])]).cpu()
                input_vectors[name] = (input_gradient.detach() * batch['mask']).reshape(-1).cpu()
            parameter_summary = gradient_summary(parameter_vectors)
            input_summary = gradient_summary(input_vectors)
            if (any(p.grad is not None for m in (model, identity, perceptual) for p in m.parameters())
                    or any(p.requires_grad for m in (identity, perceptual) for p in m.parameters())):
                raise ValueError('Diagnostic accumulated parameter gradients')
            if (state_hash(model) != START_STATE
                    or {'identity': state_hash(identity), 'perceptual': state_hash(perceptual)} != TEACHERS):
                raise ValueError('Diagnostic changed student/teacher tensors')
            row = {'group': group, 'unweighted_losses': {k: float(v.detach()) for k, v in terms.items()},
                   'weighted_losses': {k: float(v.detach()) for k, v in weighted.items()},
                   'student_parameters': parameter_summary, 'restoration_input': input_summary,
                   'input_gradient_support': 'Observed pixels only; uncaptured padding gradient zeroed',
                   'state_unchanged': True, 'optimizer_updates': 0, 'autograd_grad_calls': 5,
                   'cumulative_autograd_grad_calls': calls, 'elapsed_seconds': time.monotonic() - started}
            write(output / f'group_{number:02d}.json', row)
            print(f"Objective diagnostic {number+1}/10: {group['id']}; zero updates; elapsed={row['elapsed_seconds']:.0f}s", flush=True)
            del generated, terms, weighted, gradients, input_gradient, parameter_vectors, input_vectors
        elapsed = time.monotonic() - started
        if calls != 50 or elapsed > 600:
            raise ValueError('Finite gradient-call/timing budget differs')
        artifacts = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
        write(output / 'results.json', {
            'complete': True, 'protocol_sha256': sha(bundle_dir / 'objective_diagnostic_protocol_v4.json'),
            'artifacts_sha256': artifacts, 'groups': 10, 'training_references': 40, 'student_forwards': 10,
            'autograd_grad_calls': calls, 'optimizer_updates': 0, 'optimizer_constructed': False,
            'student_state_before': START_STATE, 'student_state_after': state_hash(model),
            'teachers_before': TEACHERS, 'teachers_after': {'identity': state_hash(identity), 'perceptual': state_hash(perceptual)},
            'parameter_grads_accumulated': False, 'seconds': elapsed,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
            'validation_references_used': 0, 'native_cases_used': 0, 'native_reserved_used': False,
            'actual_training': False, 'improved_model_claimed': False, 'production_checkpoint_promoted': False,
            'limitation': 'Weighted gradient direction at one fixed state on40 training cases is diagnostic; it does not prove useful outputs or predict Adam training outcomes.',
        })
    except Exception as error:
        if not (output / 'failure.json').exists():
            write(output / 'failure.json', {'complete': False, 'error_type': type(error).__name__, 'error': str(error),
                  'autograd_grad_calls_recorded': calls, 'optimizer_updates': 0, 'automatic_resume_permitted': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--bundle-dir', type=Path, default=ROOT)
    parser.add_argument('--perceptual-bundle-dir', type=Path)
    parser.add_argument('--parent-return', type=Path)
    parser.add_argument('--v3-return', type=Path)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        _, protocol = verify_recipe(args.root, args.bundle_dir, args.perceptual_bundle_dir, args.parent_return, args.v3_return)
        print({'diagnostic_verified': True, 'training_references': 40, 'maximum_gradient_calls': 50,
               'optimizer_updates': 0, 'vm_execution_pending': True})
    else:
        run(args.root, args.bundle_dir, args.root / 'outputs/cctv_dgp_objective_diagnostic_v4',
            args.perceptual_bundle_dir, args.parent_return, args.v3_return)
