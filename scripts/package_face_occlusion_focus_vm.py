"""Verify epoch30 inputs and package new target-only loss files; no fitting."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.package_face_occlusion_vm import build_bundle, sha
from scripts.audit_face_occlusion_results import read_json, verified_inputs
from scripts.audit_face_occlusion_continuation_results import ARCHIVE, AUDIT_OUTPUT, EXTRACTION, PREFIX
from detector_training import load_manifest, ReviewedMasks
from scripts.train_face_occlusion_focus_vm import (
    target_maps, priority_masks, MODEL_SHA, OPTIMIZER_SHA, OLD_INVENTORIES, INVENTORY_PATH, MAP_PATH, PRIORITY_PATH,
)

V2_SHA = 'e0a28146f1f271b94e26351cc093446ca3ea6e6a843a4c53155aa81382e38180'
V3_SHA = 'e36ce5cf04c858d61885c2a0187d0182099ada9852eb03d21d645f5bdbb3ea18'


def require(condition, message):
    if not condition: raise ValueError(message)


def validate_new_members(files, prior_inventories):
    require(files and len(files) == len(set(files)), 'Require unique nonempty new files')
    prior = {path for inventory in prior_inventories for path in inventory}
    require(not (set(files) & prior), 'New package overlaps previously executed input')
    for name in files:
        path = PurePosixPath(name)
        require(not path.is_absolute() and '..' not in path.parts and '\\' not in name and
                not PureWindowsPath(name).drive and path.as_posix() == name, 'Invalid new member path')


def reflection_registry(training, old, reviewed):
    require(len(training) == 73 and all(r['split'] == 'train' for r in training), 'Training membership differs')
    cases = [i for i, r in enumerate(training) if r.get('glare_stratum') == 'strong_lens_reflection']
    require(cases == [15, 46], 'Only the two original training reflection additions are allowed')
    records = []
    for index in cases:
        row = training[index]; digest = row['image_sha256']
        require(digest in old and digest in reviewed, 'Missing reviewed reflection parent')
        original = old[digest]; v3 = reviewed[digest]
        require(original['split'] == v3['split'] == row['split'] == 'train' and
                v3['mask_sha256'] == row['mask_sha256'] and
                v3['v2_mask_sha256'] == original['mask_sha256'], 'Reflection mask/split lineage differs')
        target = ReviewedMasks([row], 256)[0][1][0].numpy().astype(bool)
        parent = ReviewedMasks([original], 256)[0][1][0].numpy().astype(bool)
        require(not np.any(parent & ~target), 'V3 removed original foreground pixels')
        added = target & ~parent; indices = np.flatnonzero(added).tolist()
        require(indices, 'Reflection addition must be nonempty')
        records.append({'batch_index': index, 'shape': [256, 256], 'image': row.get('image'),
                        'image_sha256': digest, 'mask_sha256': row['mask_sha256'],
                        'v2_mask_sha256': original['mask_sha256'], 'flat_indices': indices,
                        'added_pixels': len(indices), 'original_pixels': int(parent.sum()),
                        'priority_sha256': hashlib.sha256(added.astype('uint8').tobytes()).hexdigest()})
    return {'format': 'dgp-reviewed-training-reflections-v1', 'records': records,
            'scope': 'Existing train cases15 and46 only; V3-minus-V2 at256px; no label changes or heldout targets'}


def prepare(check_source):
    import torch
    from face_occlusion_focus import validate_source, validate_optimizer, focus_loss
    from detector_training import segmentation_loss
    from scripts.train_face_occlusion_vm import REPLAY_SHA, PROTOCOL_SHA
    torch.set_num_threads(4)
    destination = ROOT / MAP_PATH; priority_destination = ROOT / PRIORITY_PATH
    require(not destination.exists() and not priority_destination.exists(), 'Preserve existing registered target maps')
    audit = read_json(AUDIT_OUTPUT / 'results.json'); reproduction = read_json(AUDIT_OUTPUT / 'reproduction.json')
    require(reproduction['complete'] is True and reproduction['selection_agrees_with_vm'] is True and
            sha(ARCHIVE) == audit['archive_sha256'] == reproduction['archive_sha256'], 'Verified continuation return differs')
    protocol, _, data = verified_inputs()
    source = EXTRACTION / PREFIX / 'epoch_30.pth'; optimizer_path = EXTRACTION / PREFIX / 'final_optimizer.pth'
    require(sha(source) == MODEL_SHA and sha(optimizer_path) == OPTIMIZER_SHA, 'Verified start file changed')
    payload = torch.load(source, map_location='cpu', weights_only=True); validate_source(payload)
    optimizer = torch.load(optimizer_path, map_location='cpu', weights_only=True)
    def trainable(prefix):
        return [v for k, v in payload['model'].items() if k.startswith(prefix) and
                not k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))]
    groups = [{'params': trainable('network.encoder.'), 'lr': 1e-5},
              {'params': trainable('network.decoder.') + trainable('network.segmentation_head.'), 'lr': 1e-4}]
    verified_optimizer = validate_optimizer(optimizer, groups, MODEL_SHA)
    require(verified_optimizer['parameter_states'] == 92 and verified_optimizer['parameter_elements'] == 14328209,
            'Actual optimizer parameter membership differs')
    cache_path = ROOT / 'outputs/downloaded_coverage/outputs/coverage_training_vm/replay_pixels.pth'
    require(sha(cache_path) == REPLAY_SHA, 'Cached training replay changed')
    cache = torch.load(cache_path, map_location='cpu', weights_only=True)
    parent_path = ROOT / 'dataset/detector_expanded_review_v2/manifest.json'
    review_path = ROOT / 'dataset/detector_glare_review_v3/manifest.json'
    require(sha(parent_path) == V2_SHA and sha(review_path) == V3_SHA, 'Reviewed V2/V3 lineage changed')
    v3 = read_json(review_path); require(v3['parent_manifest_sha256'] == V2_SHA, 'V3 parent binding changed')
    old_rows = load_manifest(parent_path)
    old = {r['image_sha256']: r for r in old_rows}; reviewed = {r['image_sha256']: r for r in v3['records']}
    require(len(old) == len(old_rows) == 100 and len(reviewed) == len(v3['records']) == 100,
            'Reviewed source membership differs')
    priority_registry = reflection_registry(data['training_real'].rows, old, reviewed)
    priority_registry.update(v2_manifest_sha256=V2_SHA, v3_manifest_sha256=V3_SHA,
                             training_manifest_sha256=sha(ROOT / 'dataset/detector_training_extension_v2/manifest.json'))
    require([r['added_pixels'] for r in priority_registry['records']] == [1345, 686] and
            [r['original_pixels'] for r in priority_registry['records']] == [0, 11381], 'Original reflection areas differ')
    priority_bytes = (json.dumps(priority_registry, indent=2) + '\n').encode()
    maps, records = target_maps(data['training_real'], cache, priority_masks(priority_registry, data['training_real'].rows))
    require(len(records) == 711, 'Training-only target map count differs')
    metadata = {'format': 'dgp-component-focus-targets-v2', 'protocol_sha256': PROTOCOL_SHA,
                'replay_sha256': REPLAY_SHA, 'training_manifest_sha256':
                sha(ROOT / 'dataset/detector_training_extension_v2/manifest.json'),
                'priority_manifest_sha256': hashlib.sha256(priority_bytes).hexdigest(),
                'model_sha256': MODEL_SHA, 'optimizer_sha256': OPTIMIZER_SHA, 'optimizer': verified_optimizer,
                'records': records, 'optimizer_updates_locally': 0,
                'scope': '73 real training labels and638 frozen replay targets; no validation/test target maps'}
    if check_source:
        sys.path.insert(0, str(ROOT / 'outputs/face_extraction_dependencies'))
        from face_occlusion_adapter import load_adapter
        from scripts.train_face_occlusion_vm import dependency_versions
        model, _ = load_adapter(source, 'cpu'); model.requires_grad_(False).eval()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        batch = protocol['schedules']['extended']['batches'][0][0]
        pairs = [data['training_real'][i] if i < 73 else (cache[i - 73][0].float() / 255, cache[i - 73][1].float()) for i in batch]
        x, m = (torch.stack([p[j] for p in pairs]) for j in (0, 1)); w = torch.stack([maps[i] for i in batch])
        with torch.inference_mode():
            logits = model.detect(x); old = segmentation_loss(logits, m, .25, .1); focused = focus_loss(logits, m, w)
        require(torch.isfinite(old + .25 * focused), 'CPU source forward loss nonfinite')
        require(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()), 'Source forward changed state')
        metadata['source_forward'] = {'batch_indices': batch, 'device': 'cpu', 'supervised_loss': float(old),
                                      'auxiliary_loss': float(focused), 'model_state_unchanged': True,
                                      'optimizer_constructed': False, 'optimizer_updates_locally': 0,
                                      'dependencies': dependency_versions(ROOT / 'outputs/face_extraction_dependencies')}
    destination.parent.mkdir(parents=True, exist_ok=True)
    priority_destination.write_bytes(priority_bytes)
    destination.write_bytes((json.dumps(metadata, indent=2) + '\n').encode())
    print(json.dumps({'training_targets': len(records), 'optimizer': verified_optimizer,
                      'source_forward': metadata.get('source_forward'), 'manifest_sha256': sha(destination)}), flush=True)


def package():
    maps = read_json(ROOT / MAP_PATH)
    require(maps['format'] == 'dgp-component-focus-targets-v2' and
            maps['priority_manifest_sha256'] == sha(ROOT / PRIORITY_PATH) and
            maps['model_sha256'] == MODEL_SHA and maps['optimizer_sha256'] == OPTIMIZER_SHA and
            maps['optimizer_updates_locally'] == 0 and len(maps['records']) == 711 and
            maps.get('source_forward', {}).get('model_state_unchanged') is True,
            'Require verified source forward and registered training-only maps before packaging')
    files = ['face_occlusion_focus.py', 'scripts/train_face_occlusion_focus_vm.py',
             'scripts/package_face_occlusion_focus_vm.py', 'tests/test_face_occlusion_focus.py',
             'tests/test_face_occlusion_focus_vm.py', 'tests/test_face_occlusion_focus_package.py',
             'FACE_OCCLUSION_FOCUS.md', 'FACE_OCCLUSION_FOCUS_VM.md', MAP_PATH, PRIORITY_PATH]
    inventories = []
    for name, digest in OLD_INVENTORIES.items():
        require(sha(ROOT / name) == digest, 'Prior inventory changed'); inventories.append(read_json(ROOT / name))
    validate_new_members(files, inventories)
    for name in ('FACE_OCCLUSION_FOCUS.md', 'FACE_OCCLUSION_FOCUS_VM.md'):
        require((ROOT / name).is_file(), 'Fixed specification/runbook missing')
    report = build_bundle(ROOT, files, ROOT / INVENTORY_PATH, ROOT / 'outputs/face-occlusion-focus-code.tar.gz')
    (ROOT / INVENTORY_PATH).with_name('build.json').write_bytes((json.dumps(report, indent=2) + '\n').encode())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true', help='Verify starts and register training target maps; no fitting')
    parser.add_argument('--check-source', action='store_true', help='One unchanged-source CPU batch; no optimizer constructed')
    args = parser.parse_args()
    require(not args.check_source or args.prepare, '--check-source requires --prepare')
    prepare(args.check_source) if args.prepare else package()
