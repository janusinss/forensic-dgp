"""Copy the fully reviewed stopped55 state for optional experimental tests.

No neural model, returned Python, gradient, optimizer, training, VM or app default
is executed or changed. Original training states and quality failures remain.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import time

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT/'outputs/codeformer_skip_feature_adapter_return_review_v1'
RETURN = ROOT/'outputs/codeformer_skip_feature_adapter_v1_return'
MODEL = ROOT/'outputs/codeformer_skip_feature_adapter_v1_model'
PROTOCOL = '0d358ab29e0344c7243b6312cb1d04ca6cc0e0a1de270bca7e6b4040dd566c90'
ARCHIVE = 'f445f5ffe3a97215046cb8e265eece2dd2d36961ac0b607948a52a29a5809c76'
SOURCE = 'outputs/skip_feature_adapter_v1/snapshot_0055/checkpoint.pt'
SOURCE_SHA = 'f947643b80b6871dc507a1f7cb1f873f56317cc1fbd14bb2407f09ebc231dcbb'
MODEL_STATE = '45f3e1d39f4abb19d1adadff45e6c2d841df90becc0c5c97c381a8e5b7502762'
BASE_STATE = 'a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f'
ADAPTER_STATE = '3950cfddd938fcff85850ad846a936133efa3ae17ca405a2c63417870d42acf5'
STATE_AUDIT_SHA = '3e9d4f684fbe6ff2fcebe212268406edec65b699749910c9ac6d4d5796481434'
START = time.monotonic()


def require(value, message):
    if not value: raise ValueError(message)


def deadline():
    require(time.monotonic()-START < 300, 'Finite300s export cap exceeded; preserve partial export')


def ordinary(path, *, existing=True):
    path = Path(path).absolute()
    require(path.resolve(strict=False).is_relative_to(ROOT), 'Workspace evidence required')
    for item in (path, *path.parents):
        if item.exists() or item.is_symlink():
            info = item.lstat()
            require(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0)&1024,
                    'Linked or reparse evidence refused')
    if existing:
        info = path.stat()
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'Unique regular evidence required')
    return path


def relative(path):
    return Path(path).absolute().relative_to(ROOT).as_posix()


def pin(path):
    deadline(); path = ordinary(path); before = path.stat(); value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            deadline(); value.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), 'Evidence changed while hashing')
    return {'bytes': before.st_size, 'sha256': value.hexdigest()}


def load(path):
    path = ordinary(path); require(path.stat().st_size <= 8*1024**2, 'Bounded review JSON required')
    def unique(pairs):
        value = {}
        for name, item in pairs:
            require(name not in value, 'Duplicate JSON key'); value[name] = item
        return value
    return json.loads(path.read_bytes(), object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def safe_relative(name):
    require(isinstance(name, str) and '\\' not in name and ':' not in name and
            not name.startswith('/') and all(part not in ('', '.', '..') for part in name.split('/')),
            'Portable workspace-relative evidence required')
    return PurePosixPath(name)


def tensor_digest(state, torch):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        deadline(); require(isinstance(value, torch.Tensor), 'Tensor-only inference state required')
        value = value.detach().cpu().contiguous()
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode()
        result.update(len(header).to_bytes(8, 'big')); result.update(header)
        result.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def adapter_shapes():
    result = {}
    for site, channels in {'32':256, '64':256, '128':128, '256':128}.items():
        for name, shape in {'down.weight':(32,channels,1,1), 'down.bias':(32,),
                'depthwise.weight':(32,1,3,3), 'depthwise.bias':(32,),
                'up.weight':(channels,32,1,1), 'up.bias':(channels,)}.items():
            result['adapters.'+site+'.'+name] = shape
    require(len(result) == 24 and sum(math.prod(shape) for shape in result.values()) == 51328,
            'Exact24/51328 adapter schema required')
    return result


def main():
    require(not MODEL.exists(), 'Preserve existing or partial research model directory')
    source_script_pin = pin(__file__)
    paths = {'import': REVIEW/'return_import_v1.json', 'state': REVIEW/'state_v1/state_audit_v1.json',
             'pixels': REVIEW/'pixels_v1/audit.json', 'visual': REVIEW/'root_visual_review_v1.json'}
    pins = {name: {'path':relative(path), **pin(path)} for name, path in paths.items()}
    imported, state_audit, pixels, visual = (load(paths[name]) for name in ('import','state','pixels','visual'))
    require(imported['complete'] is True and imported['transport_pass'] is True and
            imported['archive']['sha256'] == ARCHIVE and imported['protocol_sha256'] == PROTOCOL and
            imported['returned_code_executed'] is False and
            imported['checkpoint_or_array_deserialized'] is False and
            Path(imported['import_root']).resolve() == RETURN.resolve(), 'Exact inert imported return required')
    for document in (state_audit, pixels, visual):
        require(document['complete'] is True and document['archive_sha256'] == ARCHIVE and
                document['protocol_sha256'] == PROTOCOL and document['model_qualified'] is False,
                'All exact completed independent reviews required; quality stop remains')
    require(pins['state']['sha256'] == STATE_AUDIT_SHA and state_audit['audit_pass'] is True and
            state_audit['actual_import']['sha256'] == pins['import']['sha256'] and
            state_audit['actual_completed_updates'] == 55 and state_audit['actual_microbatches'] == 220 and
            state_audit['pending_operations'] == {} and state_audit['training_process_success'] is False,
            'Exact independent committed55 state/operation audit required')
    selected = [row for row in state_audit['checkpoints'] if row['path'] == SOURCE]
    require(len(selected) == 1, 'Exactly one reviewed selected checkpoint required')
    selected = selected[0]
    require(selected['updates'] == 55 and selected['partial_microbatches'] == 0 and
            selected['optimizer_inflight'] is False and selected['sha256'] == SOURCE_SHA and
            selected['all515_frozen_base_tensors_exact'] is True and selected['all24_adapter_duplicates_exact'] is True and
            selected['model539_canonical_sha256'] == MODEL_STATE and selected['base_canonical_sha256'] == BASE_STATE and
            selected['adapter24_canonical_sha256'] == ADAPTER_STATE, 'Exact stopped55 selected state lineage required')
    require(pixels['audit_pass'] is True and pixels['import_receipt_sha256'] == pins['import']['sha256'] and
            pixels['actual_snapshot_updates'] == [0,55] and pixels['latest_completed_saved_snapshot'] == 55 and
            pixels['comparison_RAW_PNG_pairs_checked'] == 2390 and pixels['TRAIN_guard_RAW_PNG_pairs_checked'] == 48 and
            pixels['auxiliary_RAW_PNG_pairs_checked'] == 54 and pixels['exact_initial_R2_parity_slots'] == 478 and
            pixels['feature_arrays_checked'] == 220 and pixels['behavior_U_RAW_PNG_teacher_triples_checked'] == 110 and
            pixels['actual_local_neural_gradient_optimizer_VM_calls'] == 0 and pixels['FINAL_used'] is False,
            'All actual saved pixel/teacher/feature/initial parity evidence required')
    require(pixels['recorded_failure'] == state_audit['terminal_failure_retained']['error'] and
            isinstance(pixels['recorded_failure'], str) and bool(pixels['recorded_failure']),
            'Actual failed preservation/retention stop must remain explicit')
    manifest_path = ordinary(paths['pixels'].parent/'visual_manifest.json'); manifest_pin = pin(manifest_path)
    require(manifest_pin == pixels['outputs']['visual_manifest.json'], 'Audited visual manifest changed')
    manifest = load(manifest_path)
    require(manifest['complete'] is True and manifest['model_qualified'] is False and
            len(manifest['sheets']) == manifest['total_sheets'] == pixels['visual_manifest']['sheets'],
            'Actual exact planned visual sheet census required')
    require(visual['root_personal_review_complete'] is True and
            visual['pixel_audit_sha256'] == pins['pixels']['sha256'] and
            visual['reviewed_sheets'] == len(visual['sheets']) == manifest['total_sheets'],
            'Root personal review of every planned visual sheet required')
    expected_sheets = {relative(paths['pixels'].parent.joinpath(*safe_relative(row['path']).parts)):
                       {key:row[key] for key in ('bytes','sha256')} for row in manifest['sheets']}
    require(len(expected_sheets) == manifest['total_sheets'] and
            len({row['path'] for row in visual['sheets']}) == manifest['total_sheets'] and
            {row['path'] for row in visual['sheets']} == set(expected_sheets), 'Every exact distinct planned sheet must be reviewed')
    sheet_pins = {}
    for row in visual['sheets']:
        path = ROOT.joinpath(*safe_relative(row['path']).parts); measured = pin(path)
        require(measured == expected_sheets[row['path']] == {key:row[key] for key in ('bytes','sha256')} and
                isinstance(row['observations'], str) and bool(row['observations'].strip()), 'Reviewed sheet or observation missing/changed')
        sheet_pins[row['path']] = measured
    source = ordinary(RETURN.joinpath(*safe_relative(SOURCE).parts)); source_pin = pin(source)
    require(source_pin == imported['files'][SOURCE] == {'bytes':selected['bytes'], 'sha256':SOURCE_SHA},
            'Actual retained training checkpoint bytes changed')
    require(shutil.disk_usage(ROOT/'outputs').free >= 4*1024**3+400*1024**2,
            'Export requires400MiB plus4GiB free reserve')
    # The exact lineage and complete visual reviews are validated before Torch.
    import torch
    torch.set_num_threads(4)
    saved = torch.load(source, weights_only=True, map_location='cpu', mmap=True)
    require(saved['format'] == 'codeformer-skip-feature-adapter-training-state-v1' and
            saved['protocol_sha256'] == PROTOCOL and saved['progress']['completed_training_updates'] == 55 and
            saved['progress']['partial_microbatches'] == 0 and saved['progress']['optimizer_inflight'] is False and
            saved['frozen_base_tensor_sha256'] == BASE_STATE and saved['model_qualified'] is False,
            'Actual exact committed55 checkpoint metadata required')
    state = saved['model']; shapes = adapter_shapes()
    base = {name[5:]:value for name,value in state.items() if name.startswith('base.')}
    adapters = {name:value for name,value in state.items() if name.startswith('adapters.')}
    require(len(state) == 539 and len(base) == 515 and set(adapters) == set(shapes) and
            len(base)+len(adapters) == len(state) and
            all(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and value.dtype == torch.float32 and
                torch.isfinite(value).all().item() for value in state.values()) and
            all(tuple(adapters[name].shape) == shape for name,shape in shapes.items()), 'Exact finiteCPUFP32 base515+adapter24 schema required')
    require(tensor_digest(state,torch) == MODEL_STATE and tensor_digest(base,torch) == BASE_STATE and
            tensor_digest(adapters,torch) == ADAPTER_STATE, 'Selected inference model values differ')
    MODEL.mkdir(parents=True, exist_ok=False)
    target = ordinary(MODEL/'codeformer_skip_feature_adapter_v1_u55.pth', existing=False)
    temporary = ordinary(MODEL/'codeformer_skip_feature_adapter_v1_u55.pth.incomplete', existing=False)
    payload = {'format':'codeformer-skip-feature-adapter-inference-v1',
               'model':{name:value.detach().cpu().clone() for name,value in state.items()}}
    with temporary.open('xb') as stream:
        torch.save(payload,stream); stream.flush(); os.fsync(stream.fileno())
    require(temporary.stat().st_size <= 384*1024**2, 'Finite inference export cap exceeded')
    checked = torch.load(temporary, weights_only=True, map_location='cpu', mmap=True)
    require(set(checked) == {'format','model'} and checked['format'] == payload['format'] and
            list(checked['model']) == list(state) and all(torch.equal(checked['model'][name],value) for name,value in state.items()) and
            tensor_digest(checked['model'],torch) == MODEL_STATE, 'Serialized inference payload values differ')
    del checked,payload
    require(pin(source) == source_pin and pin(__file__) == source_script_pin and pin(manifest_path) == manifest_pin and
            all(pin(paths[name]) == {key:value[key] for key in ('bytes','sha256')} for name,value in pins.items()) and
            all(pin(ROOT/path) == value for path,value in sheet_pins.items()), 'Source review/model/visual evidence changed during export')
    os.rename(temporary,target); exported = pin(target)
    result = {'format':'codeformer-skip-feature-adapter-research-model-export-v1', 'complete':True,
        'UTC':datetime.now(timezone.utc).isoformat(), 'model_path':str(target), 'export_bytes':exported['bytes'],
        'export_sha256':exported['sha256'], 'model_tensor_sha256':MODEL_STATE,
        'base_tensor_sha256':BASE_STATE, 'adapter_tensor_sha256':ADAPTER_STATE,
        'source_checkpoint':relative(source), 'source_checkpoint_member':SOURCE,
        'source_checkpoint_sha256':source_pin['sha256'], 'source_checkpoint_bytes':source_pin['bytes'],
        'source_training_updates':55, 'protocol_sha256':PROTOCOL, 'archive_sha256':ARCHIVE,
        'all539_tensor_values_exact':True, 'original515_base_values_unchanged':True,
        'new24_adapter_tensors':24, 'new_trainable_scalar_parameters':51328,
        'serialized_keys':['format','model'], 'review_pins':pins,
        'reviewed_visual_manifest':{'path':relative(manifest_path), **manifest_pin},
        'reviewed_sheet_count':len(sheet_pins), 'preservation_pass':pixels['preservation_pass'],
        'parent_dark_retention_pass':pixels['parent_dark_retention_pass'],
        'R2_dark_retention_pass':pixels['R2_dark_retention_pass'],
        'additional_interim_stop_pass':pixels['additional_interim_stop_pass'],
        'quality_stop_retained':pixels['recorded_failure'], 'model_qualified':False, 'app_default_changed':False,
        'intended_use':'Explicit optional experimental comparison; every original quality failure remains in force.',
        'original_training_optimizer_RNG_archive_retained':True, 'optimizer_RNG_excluded_from_inference_payload':True,
        'neural_gradient_optimizer_VM_calls':0, 'returned_code_imports':0,
        'source_sha256':source_script_pin['sha256'], 'seconds':time.monotonic()-START}
    receipt = MODEL/'model_export_v1.json'
    with receipt.open('x',encoding='utf8',newline='\n') as stream:
        json.dump(result,stream,indent=2,allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({'complete':True, 'model_path':str(target), **exported,
        'model_tensor_sha256':MODEL_STATE, 'model_qualified':False, 'app_default_changed':False}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--export',action='store_true',required=True)
    parser.parse_args(); main()
