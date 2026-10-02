"""Independent archive/membership/support audit; no extraction or model execution."""
from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import random
import sys
import tarfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'outputs/native-expert-vm-code.tar.gz'
OUTPUT = ROOT / 'outputs/native_expert_package_validation_v1'
PREFIX = 'native_expert_vm_bundle/'
ARCHIVE_SHA = '9e2edb1c42591d01134514184b23041b7e8e6522d68b48beb6f86983b187e7c9'
PROTOCOL_SHA = '711a9def23d53cc32cb645395e8d9154571f5f88797dfbe6a404dadca06ebeab'


def require(condition, message):
    if not condition: raise ValueError(message)


def digest(data): return hashlib.sha256(data).hexdigest()


def validate_members(entries):
    found = {}
    for entry in entries:
        require(entry.isfile() and entry.name.startswith(PREFIX), 'Only in-bundle regular files are accepted')
        name = entry.name[len(PREFIX):]
        require(name and ':' not in name and '\\' not in name and not name.startswith('/')
                and PurePosixPath(name).as_posix() == name and '..' not in PurePosixPath(name).parts
                and name not in found and 0 <= entry.size <= 32 * 1024**2, 'Duplicate/unsafe/unbounded archive member')
        found[name] = entry
    return found


def independent_schedule(real, fixtures):
    a = [i for i, r in enumerate(real) if r['kind'] == 'covered']
    b = [i for i, r in enumerate(real) if r['kind'] == 'uncovered']
    c = [r['case_id'] for r in fixtures if r['style'] != 'clear']
    d = [r['case_id'] for r in fixtures if r['style'] == 'clear']
    require([len(p) for p in (a,b,c,d)] == [51,32,224,56], 'Fixed sampling strata differ')
    rng = random.Random(42); result = []
    for _ in range(6):
        pools = [list(p) for p in (a,b,c,d)]
        for pool in pools: rng.shuffle(pool)
        p, q, r, s = pools; epoch = []
        for i in range(56):
            epoch.append({'real': [p[2*i % 51], p[(2*i+1) % 51], q[i % 32]],
                          'fixture': r[4*i:4*i+4] + [s[i]]})
        require({j for batch in epoch for j in batch['real']} == set(range(83))
                and sorted(j for batch in epoch for j in batch['fixture']) == list(range(280)), 'Incomplete epoch exposure')
        result.append(epoch)
    return result


def main():
    require(not OUTPUT.exists(), 'Preserve existing independent audit')
    require(digest(ARCHIVE.read_bytes()) == ARCHIVE_SHA, 'Prepared archive bytes changed')
    checksum = ARCHIVE.with_name(ARCHIVE.name + '.sha256').read_bytes()
    require(checksum == (ARCHIVE_SHA + '  ' + ARCHIVE.name + '\n').encode('ascii'), 'Checksum must be exact LF format')
    with tarfile.open(ARCHIVE, 'r:gz') as tar:
        members = validate_members(tar.getmembers())
        def raw(name):
            require(name in members, 'Missing member: ' + name)
            data = tar.extractfile(members[name]).read(); require(len(data) == members[name].size, 'Truncated member')
            return data
        inventory = json.loads(raw('inventory.json'))
        require(set(members) == set(inventory) | {'inventory.json'} and len(members) == 281, 'Archive membership differs')
        for name, expected in inventory.items(): require(digest(raw(name)) == expected, 'Inventory member changed: ' + name)
        protocol_bytes = raw('inputs/native_expert_protocol.json')
        require(digest(protocol_bytes) == PROTOCOL_SHA, 'Prepared specification changed')
        protocol = json.loads(protocol_bytes)
        require(protocol['format'] == 'dgp-native-expert-pilot-v1' and protocol['epochs'] == 6
                and protocol['steps_per_epoch'] == 56 and protocol['updates_budget'] == 336
                and protocol['source_counters'] == {'epoch':42,'model_updates':882,'moment_step':672}
                and protocol['domain_weights'] == {'real':.5,'reflection':.5}
                and protocol['held_out_forward_images'] == protocol['local_optimizer_updates'] == 0
                and protocol['local_optimizer_constructed'] is False and protocol['promoted'] is False
                and protocol['stage_b_implemented'] is False, 'Fixed scope/source counters differ')
        for name, expected in protocol['code_sha256'].items():
            require(digest(raw(name)) == digest((ROOT / name).read_bytes()) == expected, 'Packaged/current source code differs')
            compile(raw(name), name, 'exec')  # Parse bytecode; never execute it.
        dataset = 'dataset/detector_supported_review_v1/'
        manifest_bytes = raw(dataset + 'manifest.json')
        require(digest(manifest_bytes) == protocol['registry_sha256'] == digest((ROOT / dataset / 'manifest.json').read_bytes()), 'Supported registry differs')
        manifest = json.loads(manifest_bytes); records = manifest['supported_records']
        require('records' not in manifest and len(records) == 115
                and manifest['support_semantics'] == 'valid_one_is_supervised', 'Explicit supported schema required')
        counts = Counter((r['split'],r['kind']) for r in records)
        require(counts == {('train','covered'):51,('train','uncovered'):32,
                           ('validation','covered'):15,('validation','uncovered'):10,
                           ('test','covered'):4,('test','uncovered'):3}, 'Original/native split membership differs')
        arrays = []
        def png(name):
            with Image.open(io.BytesIO(raw(name))) as image: return np.array(image)
        for row in records:
            values = {}
            for key in ('image','mask','valid','source_valid'):
                name = dataset + row[key]; values[key] = png(name)
                require(digest(raw(name)) == row[key+'_sha256'] == digest((ROOT / name).read_bytes()), 'Package/current dataset bytes differ')
            rgb, target, valid, support = (values[k] for k in ('image','mask','valid','source_valid'))
            require(rgb.shape == (256,256,3) and rgb.dtype == np.uint8, 'RGB shape/dtype differs')
            require(all(v.shape == (256,256) and v.dtype == np.uint8 and np.isin(v,[0,255]).all() for v in (target,valid,support)), 'Binary support/target required')
            require(valid.any() and not (target.astype(bool) & ~valid.astype(bool)).any()
                    and not (valid.astype(bool) & ~support.astype(bool)).any()
                    and (bool(target.any()) == (row['kind']=='covered')), 'Target/support semantics differ')
            if row.get('source_index') is None: require((valid == 255).all() and (support == 255).all(), 'Original supervised pixels changed')
            else: require((rgb[support==0] == 96).all(), 'Only true padding is neutral')
            if row.get('source_index') == 171:
                ignored = (support == 255) & (valid == 0)
                require(int(valid.astype(bool).sum()) == 39400 and int(support.astype(bool).sum()) == 51200
                        and ignored.any() and not (rgb[ignored] == 96).all(), 'Unknown observed source171 context was erased')
            arrays.append(values)
        real = [r for r in records if r['split']=='train']; real_arrays = [a for r,a in zip(records,arrays) if r['split']=='train']
        native_order = [r['source_index'] for r in real[73:]]
        require(native_order == protocol['native_source_order'] == [171,207,208,216,217,230,240,241,348,374], 'Native training order differs')
        validations = [r for r in records if r['split']=='validation']
        require(Path(validations[8]['image']).name == 'new_covered_40.png', 'Known mannequin reporting case moved')
        split_bytes = raw('inputs/phase4_split.json'); require(digest(split_bytes) == protocol['phase4_split_sha256'], 'Split binding differs')
        split = json.loads(split_bytes); training = set(split['train'])
        require(not training.intersection(split['validation']), 'Phase4 partition overlaps')
        fixtures = json.loads((ROOT / 'outputs/reflection_coverage_data_v1/manifest.json').read_text())['cases']
        require(all(r['source'] in training for r in real[73:]+fixtures), 'Native/fixture sources outside original training')
        require(independent_schedule(real,fixtures) == protocol['schedule'], 'Independently reconstructed deterministic schedule differs')
        old_targets = json.loads((ROOT / 'dataset/detector_expanded_review_v2/manifest.json').read_text())['records']
        old_lookup = {r['image_sha256']:r for r in old_targets}; lens_total = {'old':0,'new':0}
        for row in protocol['lens_maps']:
            mask = png(row['path']); index = row['real_index']; target = real_arrays[index]['mask'].astype(bool)
            require(digest(raw(row['path'])) == row['sha256'] and mask.shape == (256,256) and np.isin(mask,[0,255]).all(), 'Lens map changed')
            mask = mask.astype(bool)
            if index < 73:
                older = old_lookup[real[index]['image_sha256']]; older_path = ROOT / 'dataset/detector_expanded_review_v2' / older['mask']
                require(digest(older_path.read_bytes()) == older['mask_sha256'], 'Original lens subtraction target changed')
                with Image.open(older_path) as image: expected = target & ~np.array(image).astype(bool)
            else: expected = target
            require(np.array_equal(mask,expected) and not (mask & ~real_arrays[index]['valid'].astype(bool)).any()
                    and int(mask.sum()) == row['pixels'], 'Lens target/support membership differs')
            lens_total['old' if index<73 else 'new'] += int(mask.sum())
        require(lens_total == {'old':2031,'new':19822} and len(protocol['lens_maps']) == 6, 'Fixed lens cohort changed')
        require(len(set(protocol['preview_ids'])) == 10, 'Fixed visual rows differ')
        require(sum(n.startswith(dataset) for n in inventory) == 252, 'Dataset file inventory differs')
    report = {'complete':True,'date':'2026-10-02','archive_sha256':ARCHIVE_SHA,'protocol_sha256':PROTOCOL_SHA,
              'archive_bytes':ARCHIVE.stat().st_size,'files_verified':281,'supported_dataset_files':252,
              'supported_records':115,'training_cases':83,'fixture_cases':280,'fixed_updates':336,
              'independent_schedule_matches':True,'original_full_support_retained':True,
              'source171_unknown_rgb_retained':True,'lens_pixels':lens_total,'checksum_line_endings':'LF',
              'script_sha256':digest(Path(__file__).read_bytes()),'test_sha256':digest((ROOT/'tests/test_native_expert_package_audit.py').read_bytes()),
              'model_forward_images':0,'optimizer_constructed':False,'local_optimizer_updates':0,
              'actual_cuda_execution_verified':False,'stage_a_transfer_verified':True,'model_promoted':False}
    OUTPUT.mkdir()
    with (OUTPUT/'verification.json').open('x',encoding='utf-8',newline='\n') as stream: stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__': main()
