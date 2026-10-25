"""Independent CPU weights-only readback of the exact experimental export.

Constructs no model and performs no neural, gradient, optimizer or VM work.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import time

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'outputs/codeformer_filter_fusion_return_review_20261019_v1'
MODEL = ROOT / 'outputs/codeformer_filter_fusion_training_v1_model'
RETURN = ROOT / 'outputs/codeformer_filter_fusion_training_v1_return'
OUTPUT = REVIEW / 'publication_peer_v1/export_closure_v1.json'
PROTOCOL = 'c1f7d8386856bd95ab3af3a8623f33fcecddd4b7e08f0696ea7694a0542c1949'
ARCHIVE = '3d6feaf30d4477fe84bd51c57d560cb10d7b4cd6509db627fc61e03accb35d16'
WEIGHTS = 'bc42ad4c55cef0c8e1e3df71e1ae50f6643611e178279fb67b5ffe8bbb511d61'
TENSORS = 'eff91cc6832bdf7ee27cd9f749a8f64a64a2351ab1443b14971b543ae2d86f35'


def need(value, message):
    if not value:
        raise ValueError(message)


def safe(path):
    path = Path(path).absolute()
    need(path.resolve(strict=True).is_relative_to(ROOT), 'Workspace evidence required')
    for parent in (path, *path.parents):
        row = parent.lstat()
        need(not stat.S_ISLNK(row.st_mode) and not getattr(row, 'st_file_attributes', 0) & 1024,
             'Linked/reparse evidence refused')
    row = path.stat()
    need(stat.S_ISREG(row.st_mode) and row.st_nlink == 1, 'Ordinary unique evidence required')
    return path


def pin(path):
    path = safe(path); before = path.stat()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    need((before.st_size, before.st_mtime_ns, before.st_ino) ==
         (after.st_size, after.st_mtime_ns, after.st_ino), 'Evidence changed during hash')
    return {'bytes': before.st_size, 'sha256': digest}


def canonical(state, torch):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(',', ':')).encode()
        digest.update(len(header).to_bytes(8, 'big')); digest.update(header)
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    start = time.monotonic()
    need(not OUTPUT.exists(), 'Preserve the earlier independent closure')
    receipt_path = MODEL / 'model_export_v1.json'
    receipt_pin = pin(receipt_path); receipt = json.loads(receipt_path.read_bytes())
    need(receipt['complete'] is True and receipt['model_qualified'] is False and
         receipt['protocol_sha256'] == PROTOCOL and receipt['archive_sha256'] == ARCHIVE and
         receipt['export_sha256'] == WEIGHTS and receipt['model_tensor_sha256'] == TENSORS and
         receipt['source_training_updates'] == 50 and receipt['app_default_changed'] is False and
         receipt['params_ema_key_is_loader_compatibility_not_new_EMA'] is True and
         receipt['source_sha256'] == '56757465c66ac8a2155b6a8eadf465b3711834729b37930b7926b1b65d4eb21c',
         'Exact stopped export lineage required')
    reviews = {'import': REVIEW / 'return_import_v1.json', 'state': REVIEW / 'state_v1/state_audit_v1.json',
               'pixels': REVIEW / 'pixels_v1/audit.json', 'visual': REVIEW / 'root_visual_review_v1.json'}
    review_pins = {name: pin(path) for name, path in reviews.items()}
    need(review_pins == receipt['review_pins'], 'Actual return audits changed after export')
    imported, state, pixels, visual = [json.loads(reviews[name].read_bytes()) for name in ('import', 'state', 'pixels', 'visual')]
    need(all(row['complete'] is True and row['model_qualified'] is False and row['archive_sha256'] == ARCHIVE and
             row['protocol_sha256'] == PROTOCOL for row in (imported, state, pixels, visual)), 'Completed exact return proof required')
    need(state['audit_pass'] is True and state['actual_changed_fusion_tensor_count'] == 72 and
         state['all443_frozen_tensors_exact'] is True and pixels['audit_pass'] is True and
         pixels['preservation_pass'] is False and pixels['R2_dark_retention_pass'] is False and
         pixels['parent_dark_retention_pass'] is True and pixels['additional_interim_stop_pass'] is False,
         'State integrity and failed-quality status remain distinct')
    need(visual['root_personal_review_complete'] is True and visual['reviewed_sheets'] == 63 and
         len({row['path'] for row in visual['sheets']}) == 63, 'Exact root63-sheet review required')
    for row in visual['sheets']:
        need(pin(ROOT / row['path']) == {key: row[key] for key in ('bytes', 'sha256')}, 'Reviewed sheet changed')
    source = RETURN / receipt['source_checkpoint']
    source_pin = pin(source)
    need(source_pin == {'bytes': receipt['source_checkpoint_bytes'], 'sha256': receipt['source_checkpoint_sha256']} ==
         {key: state['checkpoint_readback']['update50']['file'][key] for key in ('bytes', 'sha256')},
         'Retained training checkpoint differs from exact audited source')
    model = safe(Path(receipt['model_path'])); model_pin = pin(model)
    need(model == MODEL / 'codeformer_filter_fusion_v1_u50.pth' and
         model_pin == {'bytes': 376609465, 'sha256': WEIGHTS}, 'Actual exported weights differ')
    import torch
    torch.set_num_threads(2)
    saved = torch.load(model, weights_only=True, map_location='cpu', mmap=True)
    need(set(saved) == {'params_ema'}, 'Only inference state may be exported')
    tensors = saved['params_ema']
    need(isinstance(tensors, dict) and len(tensors) == 515 and all(
         isinstance(value, torch.Tensor) and value.device.type == 'cpu' and value.dtype == torch.float32 and
         torch.isfinite(value).all().item() for value in tensors.values()), 'Exact515 finite CPU FP32 tensors required')
    frozen = {name: value for name, value in tensors.items() if not name.startswith('fuse_convs_dict.')}
    fusion = {name: value for name, value in tensors.items() if name.startswith('fuse_convs_dict.')}
    digest = canonical(tensors, torch); frozen_digest = canonical(frozen, torch)
    need(digest == TENSORS == state['checkpoint_readback']['update50']['model_tensor_sha256'] and
         len(frozen) == 443 and len(fusion) == 72 and
         frozen_digest == state['checkpoint_readback']['update50']['frozen443_tensor_sha256'],
         'Exact independent tensor values or443 frozen values differ')
    need(pin(model) == model_pin and pin(receipt_path) == receipt_pin and
         all(pin(path) == review_pins[name] for name, path in reviews.items()), 'Evidence changed during CPU readback')
    result = {'complete': True, 'audit_pass': True, 'UTC': datetime.now(timezone.utc).isoformat(),
        'source_sha256': pin(__file__)['sha256'], 'archive_sha256': ARCHIVE, 'protocol_sha256': PROTOCOL,
        'model_path': str(model.relative_to(ROOT)), 'model_file': model_pin,
        'model_export_receipt': {'path': str(receipt_path.relative_to(ROOT)), **receipt_pin},
        'actual_readback_tensors': len(tensors), 'actual_fusion_tensors': len(fusion), 'actual_frozen_tensors': len(frozen),
        'actual_tensor_sha256': digest, 'actual_frozen443_sha256': frozen_digest,
        'all515_serialized_values_match_audited_stopped_checkpoint_digest': True,
        'all443_frozen_values_match_audited_R2_digest': True, 'review_pins': review_pins,
        'all63_root_reviewed_sheet_files_fresh_hash_match': True,
        'retained_training_checkpoint': {'path': str(source.relative_to(ROOT)), **source_pin},
        'no_optimizer_RNG_payload_in_inference_export': True,
        'compatibility_params_ema_key_does_not_claim_new_EMA': True,
        'preservation_pass': False, 'R2_dark_retention_pass': False, 'parent_dark_retention_pass': True,
        'model_qualified': False, 'app_default_changed': False, 'goal_complete': False,
        'execution_boundary': {'weights_only_CPU_deserializations': 1, 'model_constructions': 0,
            'neural_forwards': 0, 'gradient_queries': 0, 'optimizer_updates': 0, 'VM_calls': 0,
            'browser_launches': 0, 'publication_operations': 0}, 'seconds': time.monotonic() - start}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({'complete': True, 'audit_pass': True, 'receipt': str(OUTPUT), **pin(OUTPUT)}))


if __name__ == '__main__':
    main()
