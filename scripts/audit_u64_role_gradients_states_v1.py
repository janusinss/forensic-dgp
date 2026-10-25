"""Read saved NumPy states only: no Torch, model or parameter application."""
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs/codeformer_u64_role_gradients_v1_return_audit'
OUT = 'outputs/u64_role_gradients_v1'
HEAD_FILE_SHA = '77c0e1f236eb70d37f3f81059b75580e92d606bb2f8b5dd5eeb00da0305dca17'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path):
    before = path.stat(); h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            h.update(chunk)
    after = path.stat()
    need(not path.is_symlink() and before.st_nlink == 1 and
         (before.st_size, before.st_mtime_ns, before.st_ino) ==
         (after.st_size, after.st_mtime_ns, after.st_ino), 'Stable ordinary single-link state source required')
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': after.st_size, 'sha256': h.hexdigest()}


def canonical_update(digest, name, array):
    need(type(name) is str and re.fullmatch(r'[A-Za-z0-9_.]{1,200}', name) and '..' not in name,
         'Exact canonical parameter name required')
    need(array.dtype == np.float32 and np.isfinite(array).all(), 'Finite FP32 saved array required')
    header = json.dumps([name, 'torch.float32', list(array.shape)], separators=(',', ':')).encode()
    digest.update(len(header).to_bytes(8, 'big')); digest.update(header); digest.update(array.tobytes(order='C'))


def schema(path, count):
    with zipfile.ZipFile(path) as archive:
        rows = archive.infolist(); names = [r.filename for r in rows]
        need(len(rows) == len(set(names)) == count and
             all(name.endswith('.npy') and '/' not in name and '\\' not in name for name in names),
             'Exact unique flat saved-state NPZ namespace required')
        need(all(r.compress_type == zipfile.ZIP_STORED and not r.flag_bits & 1 and
                 not r.is_dir() and r.file_size < 32*1024**2 for r in rows), 'Bounded uncompressed nonencrypted arrays required')
        return {n[:-4] for n in names}


def main():
    inventory_path = AUDIT / 'inventory_v1.json'; inventory = json.loads(inventory_path.read_bytes())
    evidence = Path(inventory['evidence_root']); consumed = {str(inventory_path): pin(inventory_path)}
    def checked(relative):
        row = inventory['files'][relative]; path = evidence / relative
        actual = pin(path)
        need(actual['bytes'] == row['bytes'] and actual['sha256'] == row['sha256'], 'State binding differs: ' + relative)
        consumed[str(path)] = actual
        return path
    records = [json.loads(checked(OUT + '/full_states/' + phase + '.json').read_bytes()) for phase in ('initial', 'closing')]
    need(all(record['state'] == inventory['files'][record['state']['path']] for record in records),
         'Each full-state declared file pin must equal its actual inventory binding')
    initial, closing = [checked(record['state']['path']) for record in records]
    need(initial.stat().st_size == closing.stat().st_size == 377504068, 'Exact full575 state payload required')
    names = schema(initial, 575)
    need(schema(closing, 575) == names, 'Closing full namespace differs')
    prior_path = ROOT / 'outputs/codeformer_u64_cleananchors_continuation_v1_return_audit/extracted/outputs/u64_cleananchors_continuation_v1/states/initial_u64/state.json'
    prior = json.loads(prior_path.read_bytes()); consumed[str(prior_path)] = pin(prior_path)
    prior_npz = prior_path.parent / 'full_state.npz'; consumed[str(prior_npz)] = pin(prior_npz)
    need(prior['file']['sha256'] == consumed[str(prior_npz)]['sha256'] ==
         consumed[str(initial)]['sha256'] == consumed[str(closing)]['sha256'] ==
         '5a2d55fe9f136a1fcf035de3c3ca46e363defa5874572bf0dac261838845247b',
         'Full state must equal independently audited original trained-u64 initializer, never failed8')
    head_path = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_inference_export/terminal_u0064_heads.npz'
    consumed[str(head_path)] = pin(head_path)
    need(consumed[str(head_path)]['sha256'] == HEAD_FILE_SHA, 'Exact original trained-u64 head file required')
    schema(head_path, 36)
    need(all(len(record['description']['arrays']) == 575 and
             len({row['name'] for row in record['description']['arrays']}) == 575 for record in records),
         'Exactly575 unique description rows required; duplicate names refused')
    descriptions = [{r['name']: r for r in record['description']['arrays']} for record in records]
    need(all(set(s) == names for s in descriptions), 'Full description namespace differs')
    digests = {name: hashlib.sha256() for name in ('all', 'heads', 'base', 'adapters', 'frozen')}
    tensor_rows = []; counts = dict.fromkeys(('heads', 'base', 'adapters'), 0)
    with np.load(initial, allow_pickle=False) as first, np.load(closing, allow_pickle=False) as last, np.load(head_path, allow_pickle=False) as heads:
        for name in sorted(names):
            a, b = first[name], last[name]
            need(a.dtype == b.dtype == np.float32 and a.shape == b.shape and
                 a.tobytes(order='C') == b.tobytes(order='C') and np.isfinite(a).all(), 'Initial/closing state value changed: ' + name)
            kind = name.split('.')[0]
            need(kind in counts, 'Unexpected model namespace')
            counts[kind] += 1
            raw = a.tobytes(order='C')
            row = {'name': name, 'shape': list(a.shape), 'dtype': 'float32',
                   'bytes': len(raw), 'bytes_sha256': hashlib.sha256(raw).hexdigest()}
            need(all(s[name] == row for s in descriptions), 'Saved array description differs: ' + name)
            canonical_update(digests['all'], name, a)
            canonical_update(digests[kind], name.removeprefix('base.') if kind == 'base' else name, a)
            if kind != 'heads':
                canonical_update(digests['frozen'], name, a)
            else:
                need(name in heads and a.shape == heads[name].shape and
                     raw == heads[name].tobytes(order='C'), 'Retained head value differs from trained-u64 NPZ')
            tensor_rows.append(row)
    exact = {k: d.hexdigest() for k, d in digests.items()}
    need(counts == {'heads': 36, 'base': 515, 'adapters': 24}, 'Exact575 tensor census required')
    expected = {'all': '47e0707db446d3e39e9c101e8eedd631e49b45ba4f03a5f00aa2ad02e0281a8a',
        'heads': '6629d8414321b0d2ba3a9cc476e415d31911fd4bd1fa121a1a8db9097ce235be',
        'base': 'a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f',
        'adapters': 'fbfaa9d01c5ab04d2c4383485ba4e1eebb3d9f43cdab427773447248d14a586a',
        'frozen': 'f096108dc16dff56687a61a57d7ee367f8dd50accc385d2d72be8b53d4510cdf'}
    need(exact == expected, 'Complete original-u64 / frozen R2 / adapters / heads canonical binding differs')
    need(all(r['complete'] is True and r['canonical_sha256'] == exact['all'] and
             r['description']['canonical_sha256'] == exact['all'] for r in records), 'Full record canonical differs')
    for path, binding in consumed.items():
        need(pin(Path(path)) == binding, 'Consumed state source changed during readback')
    result = {'complete': True, 'saved_state_audit_pass': True,
        'all575_initial_closing_arrays_byte_exact': True, 'initial_equals_previous_audited_original_u64': True,
        'full_state_NPZ_byte_exact': True, 'tensor_counts': counts, 'canonical_sha256': exact,
        'tensor_rows': tensor_rows, 'consumed_files': list(consumed.values()),
        'source': pin(Path(__file__)), 'all_consumed_files_rehashed_after': True,
        'saved_arrays_not_live_grad_fields': True, 'local_Torch_or_checkpoint_pickle_decodes': 0,
        'neural_gradient_optimizer_training_VM_calls': 0, 'model_qualified': False}
    with (AUDIT / 'saved_state_audit_v1_r1.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('tensor_rows','consumed_files','source')}, indent=2))


if __name__ == '__main__':
    main()
