"""Limited frozen CPU replay of every path on two TRAIN reference batches."""
from datetime import datetime, timezone
import hashlib
import time
import sys

import numpy as np
import torch

from import_cctv_dgp_bank_path_ablation_v1_return import ROOT, RETURN, PARENT, ANALYSIS, PIN, read, sha, write
from audit_cctv_dgp_bank_path_ablation_v1_return import image


def main():
    started = time.monotonic()
    receipt = ANALYSIS/'limited_frozen_replay.json'
    assert not receipt.exists() and read(ANALYSIS/'independent_audit.json')['complete']
    p, old = read(RETURN/'protocol.json'), read(PARENT/'protocol.json')
    assert sha(RETURN/'protocol.json') == PIN
    for name, expected in p['parent_required_sha256'].items():
        assert sha(PARENT/name) == expected
    sys.path.insert(0, str(PARENT))
    from cctv_dgp_bank_comparison_v1_model import ConditionedBankDGP
    from cctv_dgp_generative_bank_v1 import load_prior
    from cctv_dgp_pilot import FixedObservedIdentity, grid112, state_hash, buffer_hash
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_head4_lossless_v1 import unpack
    from frozen_raw_metrics import deliver
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    original, _ = load_frozen_dgp_restorer(PARENT/'weights/dgp_v2.pth', expected_sha256=old['original_checkpoint_sha256'])
    recognizer = FixedObservedIdentity(PARENT/'weights/w600k_r50.onnx', 'cpu')
    prior = load_prior(PARENT/'prior_source', PARENT/'bank_implementation', old['standalone_prior_sha256'])
    model = ConditionedBankDGP(original.net, prior, torch.from_numpy(np.load(PARENT/'initializers/mean_style.npy', allow_pickle=False)))
    stored = torch.load(PARENT/'outputs/bank_comparison_v1/state_B50/full_state.pth', map_location='cpu', weights_only=True)
    assert stored['updates'] == 50 and not stored['manual_failed_resume_allowed']
    model.load_state_dict(stored['model'], strict=True); del stored
    model.eval().requires_grad_(False); recognizer.eval().requires_grad_(False)
    before, buffers = state_hash(model), buffer_hash(model)
    assert before == p['rejected_B50_model_state']
    active = {'drop': set()}; handles = []
    for key, layer in model.fusions.items():
        def hook(_layer, _args, output, key=key):
            return torch.zeros_like(output) if key in active['drop'] else output
        handles.append(layer.register_forward_hook(hook))
    refs = {r['id']: r for r in old['references']}
    rows, parity = [], []
    counts = dict(B=0, recognizer=0, original=0, gradients=0, backwards=0, optimizer_updates=0)
    model.register_forward_hook(lambda *_: counts.__setitem__('B', counts['B']+1))
    recognizer.encoder.register_forward_hook(lambda *_: counts.__setitem__('recognizer', counts['recognizer']+1))
    with torch.inference_mode():
        for begin in p['parity_begins']:
            assert time.monotonic()-started < 1200, 'Finite frozen local audit cap'
            cases = old['cases'][begin:begin+5]; ids = [c['id'] for c in cases]
            cameras = [image(PARENT/c['input']) for c in cases]
            targets = [image(PARENT/refs[c['source_person_or_reference']]['target']) for c in cases]
            masks = np.stack([image(PARENT/refs[c['source_person_or_reference']]['observed'], 'L') > 0 for c in cases])
            tensor = lambda a: torch.from_numpy(np.asarray(a).astype(np.float32)/np.float32(255)).permute(0, 3, 1, 2)
            x, mm = tensor(cameras), torch.from_numpy(masks)[:, None]
            grids = torch.from_numpy(np.stack([grid112(refs[c['source_person_or_reference']]['matrix112']) for c in cases]))
            active['drop'] = set(model.fusions)
            zeroed = model(x)
            active['drop'] = set()
            disabled = model(x, bank_enabled=False)
            assert torch.equal(zeroed, disabled)
            parity.append(dict(begin=begin, source=cases[0]['source'], maximum_absolute_error=0.))
            for variant in p['variants']:
                assert time.monotonic()-started < 1200
                active['drop'] = {variant[4:]} if variant.startswith('drop') else set()
                actual = torch.where(mm, model(x, bank_enabled=variant != 'disabled'), x).permute(0, 2, 3, 1).numpy()
                expected, embeddings = unpack(RETURN/'outputs/path_ablation_v1'/variant/'packs'/f'b{begin//5:03d}.npz', ids)
                assert np.allclose(actual, expected, rtol=1e-4, atol=1e-4), (variant, 'CPU/CUDA float parity')
                pngs = [deliver(a, camera, mask) for a, camera, mask in zip(expected, cameras, masks)]
                vectors = recognizer.embedding(torch.cat([torch.from_numpy(expected).permute(0, 3, 1, 2), tensor(pngs), tensor(targets)]),
                    torch.cat([mm.float()]*3), torch.cat([grids]*3)).numpy()
                rebuilt = np.stack([vectors[:5], vectors[5:10], vectors[10:15]], 1)
                assert np.allclose(rebuilt, embeddings, rtol=2e-4, atol=2e-4)
                rows.append(dict(variant=variant, begin=begin, source=cases[0]['source'], cases=5,
                    raw_maximum_absolute_error=float(np.abs(actual-expected).max()),
                    embedding_maximum_absolute_error=float(np.abs(rebuilt-embeddings).max())))
                print(dict(variant=variant, source=cases[0]['source'], cases=5, seconds=time.monotonic()-started), flush=True)
    for handle in handles:
        handle.remove()
    assert state_hash(model) == before and buffer_hash(model) == buffers
    assert state_hash(original.net) == old['original_state'] and state_hash(recognizer) == old['recognizer_state']
    assert all(not m.training for net in [model, original, recognizer] for m in net.modules())
    assert all(not value.requires_grad and value.grad is None for net in [model, original, recognizer] for value in net.parameters())
    assert not torch.is_grad_enabled()
    assert counts == dict(B=18, recognizer=14, original=0, gradients=0, backwards=0, optimizer_updates=0)
    for name, expected in read(ANALYSIS/'local_protected_before.json')['files_sha256'].items():
        assert sha(ROOT/name) == expected
    write(receipt, dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), checker_sha256=sha(__file__),
        protocol_sha256=PIN, independent_audit_sha256=sha(ANALYSIS/'independent_audit.json'),
        frozen_model_state_unchanged=before, frozen_buffers_unchanged=buffers, counts=counts,
        replay_rows=rows, zero_all_readouts_parity=parity, replayed_variant_compositions=70,
        tolerance_raw=dict(rtol=1e-4, atol=1e-4), tolerance_embedding=dict(rtol=2e-4, atol=2e-4),
        CPU_CUDA_numerical_tolerance_not_scientific_gate_relaxation=True,
        all_700_neural_outputs_replayed=False, all_700_saved_arrays_recomputed_separately=True,
        no_app_change_no_new_checkpoint=True, model_qualified=False, goal_complete=False,
        seconds=time.monotonic()-started))
    print(dict(complete=True, counts=counts, frozen_replay_only=True), flush=True)


if __name__ == '__main__':
    main()
