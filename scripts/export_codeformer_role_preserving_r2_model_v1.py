"""Export exact audited inference tensor values; no architecture or inference.

The output is an unqualified research checkpoint even if state inspection
passes. Original training checkpoints, optimizer state and failure evidence
remain intact. Never updates the application's selected/default model.
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
REVIEW = ROOT / 'outputs/codeformer_role_preserving_training_v1_r2_return_review'
MODEL_DIR = ROOT / 'outputs/codeformer_role_preserving_training_v1_r2_model'
IMPORT = ROOT / 'outputs/codeformer_role_preserving_training_v1_r2_return'
PROTOCOL = '82fcd34cada1eed238ef65a341c22580d12146e3ee30c2ff3094312d3ba1c85a'
ARCHIVE = 'd47c88c8889b9bda66eeba03f498c7ab81173e4a75ffa2572a75049e73084689'
SOURCE_NAME = 'outputs/role_preserving_training_v1_r2/snapshot_0050/checkpoint.pt'
SOURCE_SHA = 'ec115accce587307d005b4e1e0128991f89fb4095b1f70d95b9e065663a8fa93'
SOURCE_BYTES = 855237558
STATE_DIGEST = 'a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f'


def need(value, message):
    if not value:
        raise ValueError(message)


def regular(path):
    path = Path(path).absolute()
    need(path.resolve(strict=False).is_relative_to(ROOT), 'Export path outside workspace')
    for item in (path, *path.parents):
        if item.exists() or item.is_symlink():
            info = item.lstat()
            need(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024,
                 'Linked/reparse export source or destination forbidden')
    if path.exists() and path.is_file():
        need(path.stat().st_nlink == 1, 'Unlinked source required')
    return path


def pin(path):
    path = regular(path); before = path.stat()
    need(stat.S_ISREG(before.st_mode), 'Regular export evidence required')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    after = path.stat()
    need((before.st_size, before.st_mtime_ns, before.st_ino) ==
         (after.st_size, after.st_mtime_ns, after.st_ino), 'Export evidence changed during hash')
    return {'sha256': h.hexdigest(), 'bytes': before.st_size}


def read(path, cap=8 * 1024**2):
    path = regular(path); need(path.stat().st_size <= cap, 'Bounded export metadata cap exceeded')
    return json.loads(path.read_text(encoding='utf-8'))


def tensor_digest(state, torch):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode('utf-8')
        h.update(len(header).to_bytes(8, 'big')); h.update(header)
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def main():
    started = time.monotonic()
    imported_path = REVIEW / 'return_import_v1.json'
    audit_path = REVIEW / 'state_v1/state_audit_v1.json'
    imported_pin, audit_pin = pin(imported_path), pin(audit_path)
    imported, audit = read(imported_path), read(audit_path)
    need(imported['complete'] is True and imported['archive_sha256'] == ARCHIVE and
         imported['protocol_sha256'] == PROTOCOL and Path(imported['import_root']).resolve() == IMPORT.resolve(),
         'Completed strict exact import required')
    need(audit['complete'] is True and audit['protocol_sha256'] == PROTOCOL and audit['archive_sha256'] == ARCHIVE and
         audit['all350_trainable_tensors_changed'] is True and audit['all165_frozen_tensors_exact'] is True and
         audit['saved_checkpoint50_and_stopped_alias_exact_committed_state'] is True and
         audit['model_qualified'] is False and audit['checkpoint_readback']['update50']['model_tensor_sha256'] == STATE_DIGEST,
         'Exact completed independent state audit required; quality failure remains explicit')
    need(imported['files_sha256'][SOURCE_NAME] == SOURCE_SHA and imported['files_bytes'][SOURCE_NAME] == SOURCE_BYTES,
         'Strict import final checkpoint binding differs')
    source = IMPORT / SOURCE_NAME; source_pin = pin(source)
    need(source_pin == {'sha256': SOURCE_SHA, 'bytes': SOURCE_BYTES}, 'Actual final training checkpoint differs')
    destination = regular(MODEL_DIR / 'codeformer_role_preserving_r2.pth')
    receipt_path = regular(MODEL_DIR / 'model_export_v1.json')
    temporary = regular(MODEL_DIR / 'codeformer_role_preserving_r2.pth.incomplete')
    need(not any(path.exists() or path.is_symlink() for path in (destination, receipt_path, temporary)),
         'Preserve previous or partial model export')
    need(shutil.disk_usage(ROOT / 'outputs').free >= 4 * 1024**3 + 400 * 1024**2,
         'Model export needs400MiB plus4GiB reserve')
    import torch
    torch.set_num_threads(2)
    saved = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
    need(saved['protocol_sha256'] == PROTOCOL and saved['completed_training_updates'] == 50 and
         saved['state_safe'] is True and saved['partial_microbatches'] == 0 and saved['optimizer_inflight'] is False and
         saved['candidate_tensor_digest'] == STATE_DIGEST and saved['model_qualified'] is False,
         'Actual loaded final50 metadata differs')
    state = saved['model']
    need(len(state) == 515 and all(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and
         value.dtype == torch.float32 and torch.isfinite(value).all().item() for value in state.values()) and
         tensor_digest(state, torch) == STATE_DIGEST, 'Exact finite515FP32CPU model values required')
    payload = {'params_ema': {name: value.detach().cpu().clone() for name, value in state.items()}}
    need(tensor_digest(payload['params_ema'], torch) == STATE_DIGEST, 'Inference copy changed model values')
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with temporary.open('xb') as stream:
        torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
    need(temporary.stat().st_size <= 400 * 1024**2, 'Inference-only export exceeds400MiB bound')
    check = torch.load(temporary, map_location='cpu', weights_only=True, mmap=True)
    need(set(check) == {'params_ema'} and list(check['params_ema']) == list(state) and
         all(torch.equal(check['params_ema'][name], value) for name, value in state.items()) and
         tensor_digest(check['params_ema'], torch) == STATE_DIGEST, 'Saved inference payload lost exact model values')
    del check
    need(pin(imported_path) == imported_pin and pin(audit_path) == audit_pin and pin(source) == source_pin,
         'Source evidence changed during export')
    os.replace(temporary, destination)
    exported_pin = pin(destination)
    receipt = {'format': 'codeformer-role-preserving-r2-research-inference-export-v1', 'complete': True,
        'UTC': datetime.now(timezone.utc).isoformat(), 'export_path': str(destination), **exported_pin,
        'export_sha256': exported_pin['sha256'], 'export_bytes': exported_pin['bytes'],
        'serialized_root_keys': ['params_ema'], 'parameter_tensors': 515,
        'dtype': 'torch.float32', 'device': 'cpu', 'canonical_model_tensor_sha256': STATE_DIGEST,
        'model_tensor_sha256': STATE_DIGEST,
        'source_checkpoint': str(source), 'source_checkpoint_relative': SOURCE_NAME,
        'source_checkpoint_sha256': SOURCE_SHA, 'source_checkpoint_bytes': SOURCE_BYTES,
        'protocol_sha256': PROTOCOL, 'archive_sha256': ARCHIVE,
        'retained_parent_model_digest': '92ac81750b86ef1da6f1f7c0480b9e64773fafe07b094e4a5ad7e88b3f7a0c8b',
        'source_new_committed_training_updates': 50, 'historical_parent_updates': 50,
        'import_receipt_sha256': imported_pin['sha256'], 'state_audit_sha256': audit_pin['sha256'],
        'all515_tensor_values_equal_source_after_serialization': True,
        'original_training_checkpoint_archive_optimizer_RNG_retained': True,
        'architecture_changed': False, 'quantization_applied': False,
        'EMA_semantics': 'params_ema is the existing application loader compatibility key; these are exact final trained model values, with no new EMA averaging.',
        'optimizer_RNG_sampler_removed_from_inference_file_only': True,
        'intended_use': 'Explicitly selected research testing only; failed quality gates, no default or automatic model promotion.',
        'preservation_pass': audit['metadata']['preservation_pass'],
        'dark_retention_pass': audit['metadata']['dark_retention_pass'],
        'model_qualified': False, 'app_default_changed': False,
        'actual_export_NN_autograd_optimizer_VM_calls': 0, 'source_sha256': pin(__file__)['sha256'],
        'seconds': time.monotonic() - started}
    with receipt_path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'complete': True, 'model_path': str(destination), **exported_pin,
        'model_tensor_digest': STATE_DIGEST, 'qualified': False, 'default_changed': False}), flush=True)


if __name__ == '__main__':
    main()
