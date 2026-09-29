"""Training-only target/grid diagnostic; no model fitting or label mutation."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from expanded_feature_data import ExpandedCoveringDataset


def main():
    torch.set_num_threads(4)
    manifest_path = ROOT / 'outputs/expanded_feature_data_v1/manifest.json'
    audit = json.loads((ROOT / 'outputs/downloaded_expanded_fit_audit/results.json').read_text())
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == audit['protocol']['manifest_sha256']
    dataset = ExpandedCoveringDataset(json.loads(manifest_path.read_text()), ROOT, placement='fixed')
    out = ROOT / 'outputs/expanded_target_geometry'
    out.mkdir(exist_ok=False)
    cases = []
    panels = []
    # All training irregular cases; preview is first six degraded rows in manifest order.
    for index, meta in enumerate(dataset.cases):
        if meta['kind'] != 'irregular':
            continue
        item = dataset[index]
        mask = item['mask'].bool()
        core = item['geometry'].bool()
        assert not (core & ~mask).any()
        grid = F.interpolate(item['mask'][None], (64, 64), mode='area')
        reconstructed = F.interpolate(grid, (256, 256), mode='bilinear', align_corners=False)[0] >= .5
        tp = int((reconstructed & mask).sum())
        fp = int((reconstructed & ~mask).sum())
        fn = int((~reconstructed & mask).sum())
        entry = {'dataset_index': index, 'path': item['path'], 'degraded': item['degraded'],
                 'core_pixels': int(core.sum()), 'border_pixels': int((mask & ~core).sum()),
                 'target_pixels': int(mask.sum()), 'roundtrip': {'tp': tp, 'fp': fp, 'fn': fn}}
        cases.append(entry)
        if item['degraded'] and len(panels) < 6:
            rgb = (item['input'].permute(1, 2, 0).numpy() * 255).round().astype('uint8')
            colors = np.zeros_like(rgb)
            colors[core[0].numpy()] = [50, 190, 100]
            colors[(mask & ~core)[0].numpy()] = [245, 170, 50]
            row = Image.new('RGB', (256 * 3, 280), '#eff2f5')
            row.paste(Image.fromarray(rgb), (0, 24))
            row.paste(Image.fromarray(colors), (256, 24))
            row.paste(Image.fromarray(reconstructed[0].numpy().astype('uint8') * 255).convert('RGB'), (512, 24))
            ImageDraw.Draw(row).text((4, 4), f'{index}: input | core green / border amber | target grid roundtrip', fill='#152030')
            panels.append(row)
    groups = {}
    for degraded in (False, True):
        rows = [c for c in cases if c['degraded'] == degraded]
        counts = {k: sum(c['roundtrip'][k] for c in rows) for k in ('tp', 'fp', 'fn')}
        target = sum(c['target_pixels'] for c in rows)
        groups[str(degraded)] = {'cases': len(rows), 'counts': counts,
                                'roundtrip_iou': counts['tp'] / (counts['tp'] + counts['fp'] + counts['fn']),
                                'border_target_fraction': sum(c['border_pixels'] for c in rows) / target}
    assert len(cases) == 704 and len(panels) == 6
    report = {'manifest_sha256': audit['protocol']['manifest_sha256'], 'optimizer_updates': 0,
              'partition': 'train', 'groups': groups, 'cases': cases,
              'limitation': 'Label-derived area/bilinear roundtrip is neither a deployable predictor nor a capacity bound. It does not localize model errors.'}
    (out / 'results.json').write_text(json.dumps(report, indent=2))
    preview = Image.new('RGB', (768, 280 * len(panels)))
    for i, row in enumerate(panels):
        preview.paste(row, (0, 280 * i))
    preview.save(out / 'preview.png')
    print(json.dumps(groups, indent=2))


if __name__ == '__main__':
    main()
