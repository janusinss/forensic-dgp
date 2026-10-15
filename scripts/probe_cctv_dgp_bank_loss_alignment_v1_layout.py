"""Three frozen CPU forwards isolate input layout; no gradients or training."""
import ast
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image
import torch

from import_cctv_dgp_bank_loss_alignment_v1_return import ROOT, OUT, RETURN, ANALYSIS, PARENT, read, sha, write


def main():
    start = time.monotonic()
    dest = ANALYSIS / 'layout_reproduction'
    assert not dest.exists(), 'Retain the earlier frozen reproduction'
    assert read(ANALYSIS / 'independent_failure_audit.json')['complete']
    p = read(RETURN / 'protocol.json')
    old = read(PARENT / 'protocol.json')
    dependencies = ['cctv_dgp_bank_comparison_v1_model.py', 'cctv_dgp_generative_bank_v1.py',
                    'cctv_dgp_pilot.py', 'dgp_frozen_inference_v2.py', 'cctv_dgp_head4_lossless_v1.py',
                    'initializers/initial_B.pth', 'weights/dgp_v2.pth', 'initializers/mean_style.npy']
    for name in dependencies:
        assert sha(PARENT / name) == p['parent_required_sha256'][name]
    protected = read(ANALYSIS / 'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT / name) == digest for name, digest in protected.items())
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    sys.path.insert(0, str(PARENT))
    from cctv_dgp_bank_comparison_v1_model import ConditionedBankDGP
    from cctv_dgp_generative_bank_v1 import load_prior
    from cctv_dgp_pilot import state_hash, buffer_hash
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_head4_lossless_v1 import unpack
    original, _ = load_frozen_dgp_restorer(PARENT / 'weights/dgp_v2.pth', expected_sha256=old['original_checkpoint_sha256'])
    prior = load_prior(PARENT / 'prior_source', PARENT / 'bank_implementation', old['standalone_prior_sha256'])
    model = ConditionedBankDGP(original.net, prior, torch.from_numpy(np.load(PARENT / 'initializers/mean_style.npy', allow_pickle=False)))
    initial = torch.load(PARENT / 'initializers/initial_B.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(initial, strict=True)
    model.eval().requires_grad_(False)
    before, buffers = state_hash(model), buffer_hash(model)
    assert before == p['model_states']['initial']
    block = old['cases'][180:185]
    refs = {r['id']: r for r in old['references']}
    cameras, masks = [], []
    for case in block:
        with Image.open(PARENT / case['input']) as image:
            cameras.append(np.asarray(image.convert('RGB')).copy())
        with Image.open(PARENT / refs[case['source_person_or_reference']]['observed']) as image:
            masks.append(np.asarray(image.convert('L')) > 0)
    legacy = torch.from_numpy(np.asarray(cameras).astype(np.float32) / np.float32(255)).permute(0, 3, 1, 2)
    canonical = torch.cat([torch.from_numpy(q.astype(np.float32) / np.float32(255)).permute(2, 0, 1)[None] for q in cameras])
    assert torch.equal(legacy, canonical) and canonical.is_contiguous() and not legacy.is_contiguous()
    assert legacy.is_contiguous(memory_format=torch.channels_last)
    observed = torch.from_numpy(np.stack(masks))[:, None]
    dest.mkdir()
    counts = dict(original=0, B=0, gradient_queries=0, backwards=0, optimizer_updates=0)
    original.net.register_forward_hook(lambda *_: counts.__setitem__('original', counts['original'] + 1))
    model.register_forward_hook(lambda *_: counts.__setitem__('B', counts['B'] + 1))
    arrays = {}
    with torch.inference_mode():
        for label, net, x in [('original_canonical', original.net, canonical),
                              ('initial_canonical', model, canonical), ('initial_legacy', model, legacy)]:
            assert time.monotonic() - start < 180 and shutil.disk_usage(OUT).free >= 1024**3
            assert not torch.is_grad_enabled()
            arrays[label] = torch.where(observed, net(x), x).permute(0, 2, 3, 1).numpy().copy()
            np.save(dest / (label + '.npy'), arrays[label], allow_pickle=False)
    expected, _ = unpack(PARENT / 'outputs/bank_comparison_v1/baseline/packs/b036.npz', [c['id'] for c in block])
    differences = {}
    for a, b in [('initial_canonical', 'original_canonical'), ('initial_legacy', 'initial_canonical')]:
        delta = np.abs(arrays[a].astype(np.float64) - arrays[b].astype(np.float64))
        differences[a + '_vs_' + b] = dict(exact=np.array_equal(arrays[a], arrays[b]),
            maximum_absolute_error=float(delta.max()), changed_values=int(np.count_nonzero(delta)))
    for label, a in arrays.items():
        differences[label + '_vs_saved_L4_baseline'] = dict(maximum_absolute_error=float(np.abs(a - expected).max()),
                                                          exact=np.array_equal(a, expected))
    assert state_hash(model) == before and buffer_hash(model) == buffers
    assert state_hash(original.net) == old['original_state']
    assert all(not value.requires_grad and value.grad is None for net in [original, model] for value in net.parameters())
    assert counts == dict(original=1, B=2, gradient_queries=0, backwards=0, optimizer_updates=0)
    assert all(sha(ROOT / name) == digest for name, digest in protected.items())
    write(dest / 'run_results.json', dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        worker_sha256=sha(__file__), prior_failure_audit_sha256=sha(ANALYSIS / 'independent_failure_audit.json'),
        case_ids=[c['id'] for c in block], counts=counts, input_values_exact=True,
        canonical_stride=list(canonical.stride()), failed_diagnostic_stride=list(legacy.stride()),
        differences=differences, state_unchanged=before, buffers_unchanged=buffers,
        source_parent_worker_sha256=sha(PARENT / 'scripts/cctv_dgp_bank_comparison_v1_vm.py'),
        source_failed_worker_sha256=sha(RETURN / 'scripts/cctv_dgp_bank_loss_alignment_v1_vm.py'),
        output_sha256={name: sha(dest / (name + '.npy')) for name in arrays},
        local_CPU_reproduction_not_L4_confirmation=True, local_training_or_gradients=False,
        seconds=time.monotonic() - start, cap_seconds=180, protected_bindings_unchanged=True,
        model_qualified=False, goal_complete=False))
    print(differences, flush=True)


if __name__ == '__main__':
    main()
