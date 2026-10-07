"""Review the actual new candidate partition with CPU forward-only parity."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / 'outputs/cctv_dgp_feature_fusion_vm_v32'
PARENT = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'
ACTIVE = ROOT / 'outputs/cctv_dgp_active_original_decoder_vm_v28'
MIXED = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'
OUT = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_forward_review'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    started = time.monotonic()
    assert not OUT.exists()
    p = read(PACKET / 'protocol.json')
    audit = read(ROOT / 'outputs/cctv_dgp_feature_fusion_v32_preparation/independent_packet_audit.json')
    assert audit['complete'] and audit['protocol_sha256'] == sha(PACKET / 'protocol.json')
    for name, digest in p['assets_sha256'].items(): assert sha(PACKET / name) == digest
    for name, digest in read(PARENT / 'protocol.json')['assets_sha256'].items(): assert sha(PARENT / name) == digest
    dependency = read(ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_vm/protocol.json')['active_decoder_dependencies_sha256']
    for name, digest in dependency.items(): assert sha(ACTIVE / name) == digest
    for path in [PARENT, ACTIVE, PACKET]: sys.path.insert(0, str(path))
    import numpy as np
    from PIL import Image
    import torch
    from cctv_dgp_pilot import state_hash
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_app_input_v28 import canonical_tensor
    from cctv_dgp_feature_fusion_v32 import FeatureFusionCandidateV32
    from cctv_dgp_feature_fusion_v32_policy import SELECTED_NAMES, validate_layout
    torch.set_num_threads(4)
    original, _ = load_frozen_dgp_restorer(PARENT / 'weights/dgp_v2.pth', expected_sha256=p['original_checkpoint_sha256'], device='cpu')
    candidate = FeatureFusionCandidateV32(original.net)
    layout, offset = [], 0
    for name, value in candidate.net.named_parameters():
        assert value.requires_grad == (name in SELECTED_NAMES)
        if value.requires_grad:
            layout.append({'name': name, 'shape': list(value.shape), 'start': offset, 'end': offset + value.numel()})
            offset += value.numel()
    validate_layout(layout)
    assert layout == p['parameter_layout'] and offset == 978243
    candidate.train(True)
    assert not candidate.training and not candidate.net.training and all(not m.training for m in candidate.net.modules())
    assert all(not v.requires_grad for v in candidate.net.fpn.features.parameters())
    assert all(not v.requires_grad for v in candidate.net.head4.parameters())
    before = [state_hash(m.net) for m in [original, candidate]]
    assert before == [p['original_DGP_state']] * 2
    buffers = {n: v.detach().clone() for n, v in candidate.net.named_buffers()}
    ids = ['v9_tr_ffhq_00084_clear', 'v9_tr_ffhq_00084_blur_lr24',
           'v9_tr_asian_00048_clear', 'v9_tr_asian_00048_blur_lr24']
    cases = {c['id']: c for c in p['case_rows']}
    rows, bindings = [], {}
    with torch.no_grad():
        for cid in ids:
            case = cases[cid]
            assert case['role'] == 'train' and time.monotonic() - started < 120
            for name in [case['input'], case['observed']]: bindings[(MIXED / name).relative_to(ROOT).as_posix()] = sha(MIXED / name)
            with Image.open(MIXED / case['input']) as im: camera = np.asarray(im.convert('RGB')).copy()
            with Image.open(MIXED / case['observed']) as im: mask = torch.from_numpy((np.asarray(im).copy() > 0).astype(np.float32))[None, None]
            x = canonical_tensor(camera, 'cpu')
            baseline = torch.where(mask.bool(), original(x), x)
            prediction = candidate(x, mask, baseline)
            assert not prediction.requires_grad and prediction.grad_fn is None
            assert torch.equal(prediction, baseline) and torch.equal(torch.floor(prediction * 255), torch.floor(baseline * 255))
            rows.append({'id': cid, 'exact_raw_parity': True, 'exact_PNG_parity': True, 'grad_fn': None})
    assert [state_hash(m.net) for m in [original, candidate]] == before
    assert all(torch.equal(value, buffers[name]) for name, value in candidate.net.named_buffers())
    assert all(v.grad is None for m in [original, candidate] for v in m.parameters())
    for path in [Path(__file__), PACKET / 'protocol.json',
                 ROOT / 'outputs/cctv_dgp_feature_fusion_v32_preparation/independent_packet_audit.json',
                 PACKET / 'cctv_dgp_feature_fusion_v32.py', PACKET / 'cctv_dgp_feature_fusion_v32_policy.py',
                 PACKET / 'cctv_dgp_mean_centered_decoder_v29.py', ACTIVE / 'cctv_dgp_active_original_decoder_v28.py',
                 ACTIVE / 'cctv_dgp_original_decoder_candidate_v1.py', PARENT / 'weights/dgp_v2.pth']:
        bindings[path.relative_to(ROOT).as_posix()] = sha(path)
    receipt = {'complete': True, 'reviewer_sha256': sha(Path(__file__)), 'protocol_sha256': sha(PACKET / 'protocol.json'),
               'source_bindings_sha256': bindings, 'rows': rows, 'selected_tensors': 23, 'selected_parameters': 978243,
               'actual_named_parameter_order_matches_packet': True, 'only_measured11_fusion_and12_decoder_enabled': True,
               'original_and_candidate_states_unchanged': before, 'train_true_retains_evaluation_normalization': True,
               'all_buffers_unchanged': True, 'original_DGP_forwards': 4, 'candidate_DGP_forwards': 4,
               'gradient_queries': 0, 'backwards': 0, 'optimizer_updates': 0,
               'actual_VM_training_started': False, 'native_or_reserved_used': False, 'app_promotion': False,
               'goal_complete': False, 'seconds': time.monotonic() - started}
    OUT.mkdir()
    with (OUT / 'review.json').open('x', encoding='utf-8', newline='\n') as stream: json.dump(receipt, stream, indent=2)
    print(json.dumps({k: receipt[k] for k in ['complete', 'actual_named_parameter_order_matches_packet', 'all_buffers_unchanged',
                     'original_DGP_forwards', 'candidate_DGP_forwards', 'gradient_queries', 'seconds']}, indent=2))


if __name__ == '__main__':
    main()
