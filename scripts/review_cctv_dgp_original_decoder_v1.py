"""Source/state and exact initial forward parity review; no local differentiation."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'
OUT = ROOT / 'outputs/cctv_dgp_original_decoder_review_v1'
PIN = '22f76701e317b38d945838a6767239c5636cff534b61ddfbf248cc42ba08a82e'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    start = time.monotonic()
    assert not OUT.exists(), 'Preserve previous source/parity review'
    assert sha(BUNDLE / 'protocol.json') == PIN
    p = read(BUNDLE / 'protocol.json')
    for name, digest in p['assets_sha256'].items():
        assert sha(BUNDLE / name) == digest, name
    bindings = {(BUNDLE / name).relative_to(ROOT).as_posix(): digest for name, digest in p['assets_sha256'].items()}
    for name in ['scripts/cctv_dgp_original_decoder_candidate_v1.py', 'scripts/review_cctv_dgp_original_decoder_v1.py',
                 'outputs/dgp_feature_skips_v27_audit_milestone/milestone.json',
                 'outputs/dgp_feature_skips_v27_audit_milestone/independent_readback.json',
                 'outputs/cctv_dgp_post_v27_architecture_review_v1/user_architecture_decision.json']:
        bindings[name] = sha(ROOT / name)
    assert read(ROOT / 'outputs/dgp_feature_skips_v27_audit_milestone/independent_readback.json')['complete']
    # Import only the pinned inference implementation. Every neural call below is no-grad.
    sys.path.insert(0, str(BUNDLE))
    import numpy as np
    from PIL import Image
    import torch
    from cctv_dgp_pilot import state_hash
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from dgp_face_restoration import as_tensor
    from cctv_dgp_original_decoder_candidate_v1 import OriginalDecoderCandidate
    torch.set_num_threads(4)
    original, provenance = load_frozen_dgp_restorer(BUNDLE / 'weights/dgp_v2.pth',
        expected_sha256=p['assets_sha256']['weights/dgp_v2.pth'], device='cpu')
    before = state_hash(original.net)
    candidate = OriginalDecoderCandidate(original.net)
    assert state_hash(candidate.net) == before == 'd89f4e52777b5260e94a3e533aa21a0204cbd79b578137881fc6a4c67f1bb0c3'
    selected = [(name, value) for name, value in candidate.net.named_parameters() if value.requires_grad]
    layout = []
    offset = 0
    for name, value in selected:
        layout.append({'name': name, 'shape': list(value.shape), 'start': offset, 'end': offset + value.numel()})
        offset += value.numel()
    assert offset == 609219 and len(layout) == 14
    original_ids = {id(value) for value in original.net.parameters()} | {id(value) for value in original.net.buffers()}
    assert not original_ids & ({id(value) for value in candidate.net.parameters()} | {id(value) for value in candidate.net.buffers()})
    rows = []
    counts = {'original': 0, 'candidate': 0}
    original.net.register_forward_hook(lambda *_: counts.__setitem__('original', counts['original'] + 1))
    candidate.net.register_forward_hook(lambda *_: counts.__setitem__('candidate', counts['candidate'] + 1))
    final_outputs = []
    hook = candidate.net.final.register_forward_hook(lambda _m, _x, y: final_outputs.append(y.detach()))
    try:
        with torch.no_grad():
            for case in p['cases']:
                assert time.monotonic() - start < 300, 'Forward review cap300 seconds'
                assert case['role'] == 'train'
                with Image.open(BUNDLE / case['input']) as im:
                    camera = np.asarray(im.convert('RGB')).copy()
                with Image.open(BUNDLE / case['observed']) as im:
                    support = np.asarray(im).copy() > 0
                x = as_tensor(camera, 'cpu')
                mask = torch.from_numpy(support)[None, None]
                # Call the protected inference wrapper for the reference, the candidate net for its graph.
                expected = torch.where(mask, original(x), x)
                current = candidate(x, mask)
                assert torch.equal(current, expected), 'Initial raw parity: ' + case['id']
                raw = current[0].permute(1, 2, 0).numpy().copy()
                base = expected[0].permute(1, 2, 0).numpy().copy()
                delivered = np.where(support[..., None], np.floor(raw * np.float32(255)), camera).astype(np.uint8)
                baseline = np.where(support[..., None], np.floor(base * np.float32(255)), camera).astype(np.uint8)
                assert np.array_equal(delivered, baseline)
                cached = np.load(BUNDLE / case['raw_dgp'], allow_pickle=False)
                cached_error = float(np.abs(raw[support] - cached[support]).max())
                assert cached_error <= 1e-5, 'Historical GPU/CPU tolerance: ' + case['id']
                logits = final_outputs.pop()
                preclip = torch.tanh(logits) + x * 2 - 1
                clipped = (preclip <= -1) | (preclip >= 1)
                assert not current.requires_grad
                rows.append({'id': case['id'], 'profile': case['profile'], 'source': case['source'],
                    'role': 'train', 'raw_parity_exact': True, 'PNG_parity_exact': True,
                    'initial_raw_rgb_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
                    'initial_PNG_rgb_sha256': hashlib.sha256(delivered.tobytes()).hexdigest(),
                    'historical_GPU_cache_maximum_difference': cached_error,
                    'preclip_saturated_fraction': float(clipped.double().mean()),
                    'tanh_logit_maximum_absolute': float(logits.abs().max())})
    finally:
        hook.remove()
    # Actual failure probes stop before any additional neural call or derivative.
    rejected = {}
    probes = {'implicit_local_derivative': (x, mask), 'empty_support': (x, torch.zeros_like(mask)),
              'wrong_canvas': (x[..., :255], mask), 'nonfinite_input': (torch.full_like(x, float('nan')), mask)}
    for name, args in probes.items():
        try:
            if name == 'implicit_local_derivative':
                candidate(*args)
            else:
                with torch.no_grad():
                    candidate(*args)
        except (RuntimeError, ValueError) as exc:
            rejected[name] = str(exc)
        else:
            raise AssertionError('Missing candidate boundary: ' + name)
    assert counts == {'original': 50, 'candidate': 50}
    # Forward-only support preservation includes partially observed padding.
    partial = mask.clone()
    partial[..., :32, :] = False
    with torch.no_grad():
        partial_output = candidate(x, partial)
    assert torch.equal(partial_output[..., :32, :], x[..., :32, :])
    assert state_hash(original.net) == state_hash(candidate.net) == before
    assert all(v.grad is None for v in candidate.parameters())
    assert all(not v.requires_grad and v.grad is None for v in original.parameters())
    assert all(not v.requires_grad and v.grad is None for v in candidate.net.fpn.parameters())
    for name, digest in bindings.items():
        assert sha(ROOT / name) == digest, name
    source = ast.parse((BUNDLE / 'models/dgp_synthesizer.py').read_text(encoding='utf-8'))
    original_class = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'DGPSynthesizer')
    original_forward = next(n for n in original_class.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    prefix_counts = {prefix: sum(v.numel() for name, v in selected if name.startswith(prefix + '.'))
                     for prefix in ['head1', 'head2', 'head3', 'head4', 'smooth', 'smooth2', 'final']}
    aliases = list(candidate.net.named_parameters(remove_duplicate=False))
    result = {'complete': True, 'date': '2026-10-06', 'scope': 'Separate original own-DGP decoder source/state and initial CPU inference review only',
        'source_bindings_sha256': bindings, 'retained_protocol_sha256': PIN, 'DGP_state_before_after': before,
        'DGP_provenance': provenance, 'original_forward_AST_sha256': hashlib.sha256(ast.dump(original_forward, include_attributes=False).encode()).hexdigest(),
        'trained_parameters_proposed': offset, 'selected_parameter_tensors': len(layout), 'parameter_layout': layout,
        'parameter_counts_by_module': prefix_counts, 'unique_original_parameters': sum(v.numel() for v in original.net.parameters()),
        'original_parameter_alias_names': len(aliases), 'original_unique_parameter_tensors': len(list(candidate.net.parameters())),
        'original_model_and_candidate_share_no_tensors': True, 'encoder_FPN_and_all_stored_buffers_frozen': True,
        'normalization_layers': 5, 'initial_cases': rows, 'neural_forward_counts': counts,
        'forward_graph_uses_original_implementation': True, 'raw_and_PNG_initial_parity_cases': 50,
        'boundary_failures_verified': rejected, 'partial_observed_support_preserved': True,
        'no_target_or_identity_conditioning': True, 'no_new_weights_or_checkpoint_created': True,
        'local_gradient_calls': 0, 'local_backward_calls': 0, 'local_optimizer_updates': 0,
        'VM_gradient_proof_pending': True, 'new_training_protocol_created': False,
        'native_or_reserved_used': False, 'independent_final_review': False, 'app_promotion': False,
        'goal_complete': False, 'seconds': time.monotonic() - start}
    OUT.mkdir()
    with (OUT / 'review.json').open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ['complete','trained_parameters_proposed','selected_parameter_tensors',
        'raw_and_PNG_initial_parity_cases','neural_forward_counts','boundary_failures_verified','seconds']}, indent=2))


if __name__ == '__main__':
    main()
