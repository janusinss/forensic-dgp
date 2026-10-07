"""Local CPU forward-only parity of the new diagnostic parameter partition."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_vm'
PARENT = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'
ACTIVE = ROOT / 'outputs/cctv_dgp_active_original_decoder_vm_v28'
V31 = ROOT / 'outputs/cctv_dgp_profile_batches_vm_v31'
MIXED = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'
OUT = ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_forward_review'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    start = time.monotonic()
    assert not OUT.exists()
    p = read(PACKET / 'protocol.json')
    audit = read(ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_preparation_r1/independent_packet_audit.json')
    assert audit['complete'] and sha(PACKET / 'protocol.json') == audit['protocol_sha256']
    for base, key in [(PARENT, 'parent_dependencies_sha256'), (ACTIVE, 'active_decoder_dependencies_sha256')]:
        for name, digest in p[key].items():
            assert sha(base / name) == digest
    for path in [PARENT, ACTIVE, V31]:
        sys.path.insert(0, str(path))
    import numpy as np
    from PIL import Image
    import torch
    from cctv_dgp_pilot import state_hash
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_mean_centered_decoder_v29 import MeanCenteredOriginalDecoderV29
    from cctv_dgp_app_input_v28 import canonical_tensor
    torch.set_num_threads(4)
    original, _ = load_frozen_dgp_restorer(PARENT / 'weights/dgp_v2.pth', expected_sha256=p['original_checkpoint_sha256'], device='cpu')
    candidate = MeanCenteredOriginalDecoderV29(original.net)
    for name, value in candidate.net.named_parameters():
        if name in p['fusion_parameter_names']:
            value.requires_grad_(True)
    layout, offset = [], 0
    for name, value in candidate.net.named_parameters():
        if value.requires_grad:
            layout.append({'name': name, 'shape': list(value.shape), 'start': offset, 'end': offset + value.numel()})
            offset += value.numel()
    assert layout == p['parameter_layout'] and offset == 978243 and len(layout) == 23
    assert not any(value.requires_grad for value in candidate.net.fpn.features.parameters())
    assert not any(value.requires_grad for value in candidate.net.head4.parameters())
    states = [state_hash(model.net) for model in [original, candidate]]
    assert states == [p['original_DGP_state']] * 2
    ids = ['v9_tr_ffhq_00084_clear', 'v9_tr_ffhq_00084_blur_lr24',
           'v9_tr_asian_00048_clear', 'v9_tr_asian_00048_blur_lr24']
    cases = {c['id']: c for c in read(V31 / 'protocol.json')['case_rows']}
    rows, bindings = [], {}
    with torch.no_grad():
        for cid in ids:
            case = cases[cid]
            assert case['role'] == 'train' and time.monotonic() - start < 120
            for name in [case['input'], case['observed']]:
                bindings[(MIXED / name).relative_to(ROOT).as_posix()] = sha(MIXED / name)
            with Image.open(MIXED / case['input']) as image:
                camera = np.asarray(image.convert('RGB')).copy()
            with Image.open(MIXED / case['observed']) as image:
                mask = torch.from_numpy((np.asarray(image).copy() > 0).astype(np.float32))[None, None]
            x = canonical_tensor(camera, 'cpu')
            baseline = torch.where(mask.bool(), original(x), x)
            observed = candidate(x, mask, baseline)
            assert not baseline.requires_grad and not observed.requires_grad and observed.grad_fn is None
            assert torch.equal(observed, baseline)
            assert torch.equal(torch.floor(observed * 255), torch.floor(baseline * 255))
            rows.append({'id': cid, 'exact_raw_parity': True, 'exact_PNG_parity': True, 'grad_fn': None})
    assert [state_hash(model.net) for model in [original, candidate]] == states
    assert all(value.grad is None for model in [original, candidate] for value in model.parameters())
    assert not original.training and not candidate.training
    for path in [Path(__file__), PACKET / 'protocol.json',
                 ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_preparation_r1/independent_packet_audit.json',
                 V31 / 'cctv_dgp_mean_centered_decoder_v29.py', ACTIVE / 'cctv_dgp_active_original_decoder_v28.py',
                 ACTIVE / 'cctv_dgp_original_decoder_candidate_v1.py', PARENT / 'weights/dgp_v2.pth']:
        bindings[path.relative_to(ROOT).as_posix()] = sha(path)
    receipt = {'complete': True, 'reviewer_sha256': sha(Path(__file__)), 'protocol_sha256': sha(PACKET / 'protocol.json'),
               'source_bindings_sha256': bindings, 'rows': rows, 'selected_tensors': 23, 'selected_parameters': 978243,
               'actual_named_parameter_order_matches_packet': True,
               'original_and_candidate_state_unchanged': states, 'original_DGP_forwards': 4, 'candidate_DGP_forwards': 4,
               'gradient_queries': 0, 'backwards': 0, 'optimizer_updates': 0,
               'FPN_gradient_connectivity_still_unmeasured': True,
               'native_or_reserved_used': False, 'app_promotion': False, 'goal_complete': False,
               'seconds': time.monotonic() - start}
    OUT.mkdir()
    with (OUT / 'review.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps({k: receipt[k] for k in ['complete', 'actual_named_parameter_order_matches_packet',
                      'original_DGP_forwards', 'candidate_DGP_forwards', 'gradient_queries', 'seconds']}))


if __name__ == '__main__':
    main()
