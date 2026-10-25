"""Export an audited stopped state for explicitly selected research tests.

No model construction, inference, gradient, optimizer, VM or default selection.
The failed training checkpoint and its optimizer/RNG evidence stay intact.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'outputs/codeformer_filter_fusion_return_review_20261019_v1'
DEST = ROOT / 'outputs/codeformer_filter_fusion_training_v1_return'
MODEL = ROOT / 'outputs/codeformer_filter_fusion_training_v1_model'
PROTOCOL = 'c1f7d8386856bd95ab3af3a8623f33fcecddd4b7e08f0696ea7694a0542c1949'
ARCHIVE = '3d6feaf30d4477fe84bd51c57d560cb10d7b4cd6509db627fc61e03accb35d16'
SOURCE = 'outputs/filter_fusion_training_v1/snapshot_0050/checkpoint.pt'


def require(value, message):
    if not value:
        raise ValueError(message)


def ordinary(path):
    path = Path(path).absolute()
    require(path.resolve(strict=False).is_relative_to(ROOT), 'Workspace path required')
    for item in (path, *path.parents):
        if item.exists() or item.is_symlink():
            info = item.lstat()
            require(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024,
                    'Linked/reparse path refused')
    if path.exists():
        info = path.stat()
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'Unique regular file required')
    return path


def pin(path):
    path = ordinary(path); before = path.stat()
    with path.open('rb') as stream:
        value = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), 'Evidence changed during hashing')
    return {'bytes': before.st_size, 'sha256': value}


def load(path):
    path = ordinary(path)
    require(path.stat().st_size <= 8 * 1024**2, 'Receipt cap exceeded')
    return json.loads(path.read_bytes())


def tensor_digest(state, torch):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode()
        result.update(len(header).to_bytes(8, 'big')); result.update(header)
        result.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def main():
    start = time.monotonic()
    paths = {'import': REVIEW / 'return_import_v1.json', 'state': REVIEW / 'state_v1/state_audit_v1.json',
             'pixels': REVIEW / 'pixels_v1/audit.json', 'visual': REVIEW / 'root_visual_review_v1.json'}
    pins = {name: pin(path) for name, path in paths.items()}
    imported, audited, pixels, visual = (load(paths[name]) for name in ('import', 'state', 'pixels', 'visual'))
    for value in (imported, audited, pixels, visual):
        require(value['complete'] is True and value['archive_sha256'] == ARCHIVE and
                value['protocol_sha256'] == PROTOCOL and value['model_qualified'] is False,
                'Completed exact return reviews required; quality failure remains explicit')
    require(Path(imported['import_root']).resolve() == DEST.resolve() and audited['audit_pass'] is True and
            audited['all443_frozen_tensors_exact'] is True and
            audited['saved_checkpoint50_and_stopped_exact_committed_state'] is True and
            audited['actual_changed_fusion_tensor_count'] > 0 and
            audited['import_receipt_sha256'] == pins['import']['sha256'], 'Exact independent state lineage required')
    require(pixels['audit_pass'] is True and pixels['comparison_RAW_PNG_pairs_checked'] == 2390 and
            pixels['TRAIN_guard_RAW_PNG_pairs_checked'] == 48 and
            pixels['import_receipt_sha256'] == pins['import']['sha256'], 'Complete saved-pixel audit required')
    require(visual['root_personal_review_complete'] is True and visual['reviewed_sheets'] == 63 and
            visual['pixel_audit_sha256'] == pins['pixels']['sha256'], 'Root review of all planned sheets required')
    for row in visual['sheets']:
        require(pin(ROOT / row['path']) == {k: row[k] for k in ('bytes', 'sha256')}, 'Reviewed sheet changed')
    require(len(visual['sheets']) == len({r['path'] for r in visual['sheets']}) == 63,
            'Exact63 distinct reviewed sheets required')
    require(audited['checkpoint_readback']['update50']['file'] ==
            {'path': SOURCE, 'bytes': imported['files_bytes'][SOURCE], 'sha256': imported['files_sha256'][SOURCE]},
            'Checkpoint binding differs from actual saved state audit')
    source = ordinary(DEST / SOURCE); source_pin = pin(source)
    require(source_pin == {'bytes': imported['files_bytes'][SOURCE], 'sha256': imported['files_sha256'][SOURCE]},
            'Physical training checkpoint changed')
    state_sha = audited['checkpoint_readback']['update50']['model_tensor_sha256']
    target = ordinary(MODEL / 'codeformer_filter_fusion_v1_u50.pth')
    temporary = ordinary(MODEL / 'codeformer_filter_fusion_v1_u50.pth.incomplete')
    receipt = ordinary(MODEL / 'model_export_v1.json')
    require(not any(path.exists() for path in (target, temporary, receipt)), 'Preserve previous/partial export')
    require(shutil.disk_usage(ROOT / 'outputs').free >= 4 * 1024**3 + 400 * 1024**2,
            'Export requires400MiB plus4GiB reserve')
    import torch
    torch.set_num_threads(2)
    saved = torch.load(source, weights_only=True, map_location='cpu', mmap=True)
    require(saved['protocol_sha256'] == PROTOCOL and saved['progress']['completed_training_updates'] == 50 and
            saved['progress']['partial_microbatches'] == 0 and not saved['progress']['optimizer_inflight'] and
            saved['model_qualified'] is False, 'Actual committed50 checkpoint metadata differs')
    state = saved['model']
    require(len(state) == 515 and all(isinstance(v, torch.Tensor) and v.device.type == 'cpu' and
            v.dtype == torch.float32 and torch.isfinite(v).all().item() for v in state.values()) and
            tensor_digest(state, torch) == state_sha, 'Exact finite515 model tensor values required')
    MODEL.mkdir(parents=True, exist_ok=True)
    payload = {'params_ema': {name: value.detach().cpu().clone() for name, value in state.items()}}
    with temporary.open('xb') as stream:
        torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
    require(temporary.stat().st_size <= 400 * 1024**2, 'Inference export physical cap exceeded')
    check = torch.load(temporary, weights_only=True, map_location='cpu', mmap=True)
    require(set(check) == {'params_ema'} and list(check['params_ema']) == list(state) and
            all(torch.equal(check['params_ema'][name], value) for name, value in state.items()) and
            tensor_digest(check['params_ema'], torch) == state_sha, 'Serialized model values differ')
    del check, payload
    require(pin(source) == source_pin and all(pin(paths[name]) == value for name, value in pins.items()),
            'Source evidence changed during export')
    os.replace(temporary, target)
    exported = pin(target)
    result = {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
        'model_path': str(target), 'export_bytes': exported['bytes'], 'export_sha256': exported['sha256'],
        'model_tensor_sha256': state_sha, 'source_checkpoint': SOURCE, 'source_checkpoint_sha256': source_pin['sha256'],
        'source_checkpoint_bytes': source_pin['bytes'], 'protocol_sha256': PROTOCOL, 'archive_sha256': ARCHIVE,
        'source_training_updates': 50, 'initializer': 'our retained R2 adaptation of pretrained CodeFormer',
        'all515_tensor_values_exact': True, 'review_pins': pins,
        'preservation_pass': pixels['preservation_pass'], 'parent_dark_retention_pass': pixels['parent_dark_retention_pass'],
        'R2_dark_retention_pass': pixels['R2_dark_retention_pass'],
        'additional_interim_stop_pass': pixels['additional_interim_stop_pass'],
        'model_qualified': False, 'app_default_changed': False, 'quality_stop_retained': pixels['recorded_failure'],
        'intended_use': 'Explicit experimental comparison only; original quality stops remain in force.',
        'params_ema_key_is_loader_compatibility_not_new_EMA': True,
        'original_training_optimizer_RNG_archive_retained': True,
        'neural_gradient_optimizer_VM_calls': 0, 'source_sha256': pin(__file__)['sha256'],
        'seconds': time.monotonic() - start}
    with receipt.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({'complete': True, 'model_path': str(target), **exported, 'model_tensor_sha256': state_sha,
                      'model_qualified': False, 'app_default_changed': False}), flush=True)


if __name__ == '__main__':
    main()
