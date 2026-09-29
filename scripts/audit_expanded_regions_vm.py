"""VM inference only: localize irregular-mask errors without changing labels."""
import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def region_counts(prediction, core, target):
    prediction, core, target = (np.asarray(x, dtype=bool) for x in (prediction, core, target))
    if prediction.shape != core.shape or core.shape != target.shape or np.any(core & ~target):
        raise ValueError('Invalid region partition')
    border = target & ~core
    return {'core_pixels': int(core.sum()), 'core_missed': int((core & ~prediction).sum()),
            'border_pixels': int(border.sum()), 'border_missed': int((border & ~prediction).sum()),
            'outside_pixels': int((~target).sum()), 'outside_fp': int((prediction & ~target).sum())}


def main(root):
    if not torch.cuda.is_available():
        raise RuntimeError('Run on the VM GPU; no local training or cache extraction')
    root = Path(root).resolve()
    sys.path.insert(0, str(root))
    from expanded_feature_data import ExpandedCoveringDataset, local_path
    from feature_disk_cache import FeatureDiskCache
    from scripts.compare_pixel_heads_vm import PixelHead
    from scripts.compare_presence_heads_vm import PresenceHead
    from scripts.train_expanded_feature_vm import pixel_counts, head_digest

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    train = root / 'outputs/expanded_feature_training'
    prior_path = root / 'outputs/expanded_fit_audit/results.json'
    prior = json.loads(prior_path.read_text())
    protocol = json.loads((train / 'protocol.json').read_text())
    assert prior['complete'] and prior['protocol'] == protocol
    inventory_path = root / 'expanded_inventory.json'
    assert sha(inventory_path) == protocol['inventory_sha256']
    for name, digest in json.loads(inventory_path.read_text()).items():
        assert sha(local_path(root, name)) == digest, name
    manifest_path = root / 'outputs/expanded_feature_data_v1/manifest.json'
    assert sha(manifest_path) == protocol['manifest_sha256']
    dataset = ExpandedCoveringDataset(json.loads(manifest_path.read_text()), root, placement='fixed')
    out = root / 'outputs/expanded_regions_audit'
    out.mkdir(exist_ok=False)
    report = {'complete': False, 'optimizer_updates': 0, 'threshold': .5,
              'script_sha256': sha(__file__), 'prior_audit_sha256': sha(prior_path),
              'protocol': protocol, 'arms': {}}
    for arm in ('fixed', 'anatomical'):
        checkpoint = train / (arm + '_epoch_20.pth')
        assert sha(checkpoint) == prior['arms'][arm]['checkpoint_sha256']
        state = torch.load(checkpoint, weights_only=True, map_location='cpu')
        assert state['protocol'] == protocol
        pixel = PixelHead(3).cuda().eval().requires_grad_(False)
        gate = PresenceHead(4).cuda().eval().requires_grad_(False)
        pixel.load_state_dict(state['pixel']); gate.load_state_dict(state['presence'])
        before = head_digest(pixel, gate)
        context = {'arm': arm, **{k: protocol[k] for k in ('manifest_sha256', 'inventory_sha256', 'encoder_sha256')}}
        cache = FeatureDiskCache.open(train / (arm + '_cache'), context)
        arm_dir = out / arm
        arm_dir.mkdir()
        cases = []
        previews = 0
        try:
            assert cache.state['count'] == 3540
            with torch.inference_mode():
                for start in range(0, 3540, 12):
                    ids = list(range(start, min(start + 12, 3540)))
                    # Preserve original batch layout for exact count comparison.
                    if not any(i >= 68 and dataset.cases[i-68]['kind'] == 'irregular' for i in ids):
                        continue
                    f, targets, records = cache.batch(ids)
                    f = torch.from_numpy(f).cuda()
                    raw = pixel(f) >= 0
                    probabilities = gate(f).sigmoid()
                    gated = raw & (probabilities >= .5)[:, None, None, None]
                    for j, i in enumerate(ids):
                        if i < 68 or dataset.cases[i-68]['kind'] != 'irregular':
                            continue
                        item = dataset[i-68]
                        record = records[j]
                        previous = prior['arms'][arm]['cases'][i]
                        assert previous['cache_index'] == i and previous['record'] == record
                        assert record['path'] == item['path'] and record['variant'] == item['variant']
                        rgb = (item['input'].permute(1, 2, 0).numpy()*255).round().astype('uint8')
                        assert hashlib.sha256(rgb.tobytes()).hexdigest() == record['input_rgb_sha256']
                        target = item['mask'].numpy().astype(bool)
                        core = item['geometry'].numpy().astype(bool)
                        assert np.array_equal(target, targets[j].astype(bool))
                        entry = {'cache_index': i, 'record': record, 'regions': {}}
                        for mode, pred in (('raw', raw), ('gated', gated)):
                            assert pixel_counts(pred[j:j+1], torch.from_numpy(target[None]).cuda()) == previous[mode]
                            arr = pred[j].cpu().numpy()
                            entry['regions'][mode] = region_counts(arr, core, target)
                            Image.fromarray(arr[0].astype('uint8')*255).save(arm_dir / f'{i:04d}_{mode}.png')
                        Image.fromarray(core[0].astype('uint8')*255).save(arm_dir / f'{i:04d}_core.png')
                        Image.fromarray(target[0].astype('uint8')*255).save(arm_dir / f'{i:04d}_target.png')
                        if item['degraded'] and previews < 6:
                            Image.fromarray(rgb).save(arm_dir / f'{i:04d}_input.png')
                            previews += 1
                        cases.append(entry)
                    if start % 300 == 0:
                        print(arm, start, '/3540', flush=True)
        finally:
            cache.close()
        assert len(cases) == 704 and before == head_digest(pixel, gate)
        groups = {}
        for degraded in (False, True):
            subset = [c for c in cases if c['record']['degraded'] == degraded]
            groups[str(degraded)] = {mode: {k: sum(c['regions'][mode][k] for c in subset)
                                          for k in subset[0]['regions'][mode]} for mode in ('raw', 'gated')}
        report['arms'][arm] = {'checkpoint_sha256': sha(checkpoint), 'weights_unchanged': True,
                               'cases': cases, 'groups': groups}
        del pixel, gate
        torch.cuda.empty_cache()
    report['complete'] = True
    (out / 'results.json').write_text(json.dumps(report, indent=2))
    archive = root / 'expanded-regions-audit-results.tar.gz'
    if archive.exists():
        raise RuntimeError('Archive already exists; preserve it')
    with tarfile.open(archive, 'w:gz') as tar:
        for path in sorted(out.rglob('*')):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(out).as_posix())
        tar.add(__file__, arcname='audit_expanded_regions_vm.py')
    print('DONE:', archive, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    main(parser.parse_args().root)
