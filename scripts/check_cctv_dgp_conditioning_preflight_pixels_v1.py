"""Exercise the worker's actual pixel fixture without importing its run/model code.

CPU utility checks only. No model construction, CUDA, gradient query, backward,
optimizer, parameter displacement or image-quality evidence is involved.
"""
import argparse
import ast
from datetime import datetime, timezone
import importlib.util
import traceback

import numpy as np
import torch

from import_cctv_dgp_conditioning_learning_v1_r2_return import read, sha, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--packet', required=True, type=__import__('pathlib').Path)
    parser.add_argument('--record', required=True, type=__import__('pathlib').Path)
    args = parser.parse_args()
    packet = args.packet.resolve()
    p = read(packet / 'protocol.json')
    worker_name = next(n for n in p['assets_sha256'] if n.startswith('scripts/') and n.endswith('_vm.py'))
    worker = packet / worker_name
    helper = packet / 'conditioning_losses.py'
    assert sha(worker) == p['assets_sha256'][worker_name]
    assert sha(helper) == p['assets_sha256']['conditioning_losses.py']
    # Extract ONLY the reviewed NumPy fixture expression, never the worker run.
    tree = ast.parse(worker.read_text(encoding='utf-8'))
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    probes = [n for n in ast.walk(run) if isinstance(n, ast.Assign) and
              any(isinstance(t, ast.Name) and t.id == 'probe' for t in n.targets)]
    assert len(probes) == 1
    expression = ast.get_source_segment(worker.read_text(encoding='utf-8'), probes[0].value)
    assert expression.startswith('np.') and '__' not in expression
    probe = eval(compile(ast.Expression(probes[0].value), '<sealed-pixel-fixture>', 'eval'),
                 {'np': np, '__builtins__': {}})
    torch.set_grad_enabled(False)
    spec = importlib.util.spec_from_file_location('sealed_conditioning_losses', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = torch.from_numpy(probe.astype(np.float32) / np.float32(255))
    proof = dict(UTC=datetime.now(timezone.utc).isoformat(), checker_sha256=sha(__file__),
                 protocol_sha256=sha(packet / 'protocol.json'), worker_sha256=sha(worker),
                 helper_sha256=sha(helper), actual_fixture_expression=expression,
                 actual_fixture_shape=list(probe.shape), device='CPU',
                 helper_called_without_importing_worker=True,
                 local_model_CUDA_gradient_optimizer_or_parameter_displacement_calls=0,
                 CPU_utility_check_not_manual_L4_run_or_model_quality=True)
    try:
        rgb, feature = module.delivered_surrogates(raw, torch.from_numpy(probe))
        assert probe.dtype == np.uint8 and probe.shape == (1, 3, 256, 256)
        assert np.array_equal(np.unique(probe), np.arange(256, dtype=np.uint8))
        assert rgb.dtype == torch.float32 and feature.dtype == torch.float64
        assert np.array_equal(rgb.numpy(), probe.astype(np.float32) / np.float32(255))
        assert np.array_equal(feature.numpy(), probe.astype(np.float64) / 255.)
        assert not rgb.requires_grad and not feature.requires_grad
        proof.update(complete=True, all256_levels_canonical_RGB32_and_feature64=True)
    except BaseException as error:
        proof.update(complete=False, type=type(error).__name__, message=str(error),
                     traceback=traceback.format_exc())
        write(args.record, proof)
        print(dict(complete=False, actual_fixture_shape=proof['actual_fixture_shape'],
                   record=str(args.record)), flush=True)
        raise
    write(args.record, proof)
    print(dict(complete=True, actual_fixture_shape=proof['actual_fixture_shape'],
               local_model_calls=0, record=str(args.record)), flush=True)


if __name__ == '__main__':
    main()
