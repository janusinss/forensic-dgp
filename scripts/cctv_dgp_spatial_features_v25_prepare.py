"""Finite VM feature extraction for the own-DGP spatial capacity experiment."""
import hashlib
import json
from pathlib import Path
import shutil
import time


def prepare_features(root, p, require_vm):
    require_vm(root, idle=True)
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    from cctv_dgp_pilot import FixedObservedIdentity, grid112, state_hash
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_spatial_features_v25 import SpatialFeatureHead, frozen_fpn_features, tensor_receipt
    torch.set_num_threads(4)
    torch.manual_seed(p['design']['seed'])
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    assert shutil.disk_usage(root).free >= p['budgets']['minimum_free_disk_bytes'], 'Need3GiB free; preserve historical runs'
    counts = {'detail_head': 0, 'DGP': 0, 'DGP_CPU': 0, 'fixed_recognizer': 0}
    def counter(name):
        def hook(*_): counts[name] += 1
        return hook
    def rgb(name):
        with Image.open(root / name) as image: value = np.asarray(image).copy()
        assert value.shape == (256, 256, 3) and value.dtype == np.uint8
        return value
    def tensor(value):
        return torch.from_numpy(np.asarray(value).copy()).cuda()
    def write(name, record):
        with (root / name).open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(record, indent=2, allow_nan=False) + '\n')
    head = SpatialFeatureHead().cuda().eval()
    head.register_forward_hook(counter('detail_head')); head.audit_forward_counts = counts
    assert sum(value.numel() for value in head.parameters()) == p['design']['trainable_parameters'] == 53781
    original_head = state_hash(head)
    dgp, provenance = load_frozen_dgp_restorer(root / 'weights/dgp_v2.pth',
        expected_sha256=p['assets_sha256']['weights/dgp_v2.pth'], device='cuda')
    dgp.net.register_forward_hook(counter('DGP'))
    dgp_before = state_hash(dgp.net)
    items = []; feature_rows = []; cpu_rows = []
    refs = {ref['id']: ref for ref in p['references']}
    evidence_name = 'feature_preflight_' + str(time.time_ns()) + '.json'
    def evidence(complete, cause=None):
        write(evidence_name, {'complete': complete, 'cause': cause, 'counts': dict(counts),
            'DGP_state_before': dgp_before, 'DGP_state_after': state_hash(dgp.net),
            'head_state': state_hash(head), 'CUDA_feature_rows': feature_rows, 'CPU_comparison_rows': cpu_rows,
            'CUDA_cache_raw_tolerance': 2e-6, 'CPU_DGP_raw_tolerance': 1e-5, 'CPU_DGP_feature_tolerance': 5e-5,
            'optimizer_constructed': False, 'backward_calls': 0,
            'scope': 'DGP feature extraction/numerical compatibility only, not restoration quality or canonical-app qualification'})
    try:
        for case in p['cases']:
            camera, target = rgb(case['input']), rgb(case['target'])
            with Image.open(root / case['observed']) as image: mask = np.asarray(image).copy() > 0
            raw = np.load(root / case['raw_dgp'], allow_pickle=False); baseline = rgb(case['png_dgp'])
            assert raw.dtype == np.float32 and raw.shape == camera.shape and np.isfinite(raw).all()
            assert np.array_equal(baseline, np.where(mask[..., None], np.floor(raw * np.float32(255)), camera).astype(np.uint8))
            x = tensor(camera.astype(np.float32) / np.float32(255)).permute(2, 0, 1)[None]
            # Keep the declared inherited CUDA-scalar cache convention separate.
            dgp_input = tensor(camera).permute(2, 0, 1).float()[None] / 255
            fresh, fpn = frozen_fpn_features(dgp.net, dgp_input)
            error = float(np.abs(fresh[0].permute(1, 2, 0).cpu().numpy() - raw).max())
            feature_rows.append({'id': case['id'], 'fresh_cached_raw_maximum': error,
                'features': [tensor_receipt(value) for value in fpn]})
            assert error <= 2e-6, 'Fresh CUDA DGP cache parity failed: ' + case['id']
            assert len(dgp.net.fpn._forward_hooks) == 0
            feature = np.zeros((256, 256), bool)
            for point in case['landmarks5_canvas_xy']:
                xx, yy = np.floor(point).astype(int)
                feature[max(0, yy-12):min(256, yy+12), max(0, xx-12):min(256, xx+12)] = True
            interior = cv2.erode(mask.astype(np.uint8), np.ones((13, 13), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
            valid = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
            assert (feature & interior).any()
            items.append({'case': case, 'camera': camera, 'target8': target, 'mask8': mask, 'baseline8': baseline,
                'x': x, 'target': tensor(target.astype(np.float32) / np.float32(255)).permute(2, 0, 1)[None],
                'base': tensor(raw).permute(2, 0, 1)[None], 'mask': tensor(mask.astype(np.float32))[None, None],
                'feature': tensor((feature & interior).astype(np.float32))[None, None],
                'interior': tensor(interior.astype(np.float32))[None, None], 'valid7': tensor(valid.astype(np.float32))[None, None],
                'grid': tensor(grid112(refs[case['source_person_or_reference']]['matrix112']))[None], 'fpn': fpn})
        assert state_hash(dgp.net) == dgp_before and all(not value.requires_grad and value.grad is None for value in dgp.net.parameters())
        with torch.no_grad():
            for item in items:
                assert torch.equal(head(item['x'], item['base'], item['mask'], item['fpn']), item['base']), 'Initial head is not exact cached DGP'
        assert state_hash(head) == original_head
        cpu, cpu_provenance = load_frozen_dgp_restorer(root / 'weights/dgp_v2.pth',
            expected_sha256=p['assets_sha256']['weights/dgp_v2.pth'], device='cpu')
        cpu.net.register_forward_hook(counter('DGP_CPU'))
        cpu_before = state_hash(cpu.net)
        for cid in p['fresh_DGP_parity_cases']:
            item = next(item for item in items if item['case']['id'] == cid)
            cpu_x = torch.from_numpy(item['camera'].copy()).permute(2, 0, 1).float()[None] / 255
            fresh_cpu, features_cpu = frozen_fpn_features(cpu.net, cpu_x)
            raw_error = float((fresh_cpu - item['base'].cpu()).abs().max())
            errors = [float((left - right.cpu()).abs().max()) for left, right in zip(features_cpu, item['fpn'])]
            cpu_rows.append({'id': cid, 'CPU_DGP_raw_maximum': raw_error, 'CPU_DGP_feature_maxima': errors})
            assert raw_error <= 1e-5 and max(errors) <= 5e-5, 'DGP CPU/CUDA feature compatibility preflight failed: ' + cid
        assert state_hash(cpu.net) == cpu_before == dgp_before and all(not value.requires_grad and value.grad is None for value in cpu.net.parameters())
        del cpu
        evidence(True)
    except BaseException as exc:
        # Preserve a partial receipt if the failure is in writing it; never
        # replace the original error with FileExistsError from an exclusive write.
        if not (root / evidence_name).exists():
            evidence(False, str(exc))
        raise
    fresh_errors = [{'id': row['id'], 'maximum_raw_error': row['fresh_cached_raw_maximum']}
                    for row in feature_rows if row['id'] in p['fresh_DGP_parity_cases']]
    del dgp; torch.cuda.empty_cache()
    identity = FixedObservedIdentity(root / 'weights/w600k_r50.onnx', 'cuda')
    identity.encoder.register_forward_hook(counter('fixed_recognizer')); identity_before = state_hash(identity)
    with torch.no_grad():
        for item in items:
            item['truth'] = identity.embedding(item['target'], item['mask'], item['grid'])
            item['base_cosine'] = (identity.embedding(item['base'] * item['mask'] + item['x'] * (1-item['mask']), item['mask'], item['grid']) * item['truth']).sum(1)
    return head, identity, items, {'initial_head_state': original_head, 'recognizer_state': identity_before,
        'DGP_state_unchanged': dgp_before, 'DGP_provenance': provenance, 'fresh_DGP_parity': fresh_errors,
        'initial_exact_cached_DGP_cases': 50, 'trainable_parameters': 53781,
        'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'neural_forward_counts': dict(counts),
        'frozen_feature_cache_rows': feature_rows, 'CPU_DGP_comparison_rows': cpu_rows,
        'feature_preflight_file': evidence_name, 'all_feature_tensors_ordinary_detached': True}
