"""Recount returned masks and verify checkpoint ancestry; no fitting."""
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_retention_logs import audit_archive
from scripts.evaluate_coverage_results import binary, recount
from scripts.train_coverage_vm import sha, PARENT_SHA, real_gate
from detector_training import load_manifest
from detector_replay import retention_passes


def main():
    archive = Path('outputs/retention-results.tar.gz')
    protocol = Path('outputs/coverage_protocol_v1/protocol.json')
    log = audit_archive(archive, Path('outputs/retention-code.tar.gz'), protocol)
    root = Path('outputs/downloaded_retention/outputs/retention_training_vm')
    report = json.loads((root/'results.json').read_text())
    baseline = json.loads((root/'baseline.json').read_text())
    run = json.loads((root/'run.json').read_text())
    inputs = json.loads(protocol.read_text())['input_hashes']
    manifest = Path('dataset/detector_training_extension_v2/manifest.json')
    assert sha(manifest) == inputs[manifest.as_posix()]
    rows = [r for r in load_manifest(manifest) if r['split']=='validation']
    bench = Path('outputs/downloaded_completion_benchmark/outputs/completion_pretrained_vm')
    assert sha(bench/'manifest.json') == inputs[(bench/'manifest.json').as_posix()]
    cases = json.loads((bench/'manifest.json').read_text())['cases']
    truth = {'real': [binary(r['mask_path']) for r in rows],
             'synthetic': [binary(bench/'mask'/r['file']) for r in cases]}
    predictions = {}
    scores = {}
    for domain, targets in truth.items():
        folder = root/'final_masks'/domain
        assert {p.name for p in folder.glob('*.png')} == {f'{i:04}.png' for i in range(len(targets))}
        predictions[domain] = [binary(folder/f'{i:04}.png') for i in range(len(targets))]
        scores[domain] = recount(predictions[domain], targets)
    for group, predicate in [('human_real', lambda r: Path(r['image']).name!='new_covered_40.png'),
                             ('mannequin', lambda r: Path(r['image']).name=='new_covered_40.png'),
                             ('glare', lambda r: r.get('glare_stratum')=='strong_lens_reflection')]:
        ids = [i for i,r in enumerate(rows) if predicate(r)]
        scores[group] = recount([predictions['real'][i] for i in ids], [truth['real'][i] for i in ids])
    for domain, metrics in scores.items():
        for key,value in metrics.items():
            assert math.isclose(value, report[domain][key], abs_tol=1e-12, rel_tol=0), (domain,key)
    real_ok = real_gate(scores['real'], baseline['real'], baseline['real']['iou'])
    synthetic_ok = retention_passes(scores['synthetic'], baseline['synthetic'])
    assert (real_ok, synthetic_ok, real_ok and synthetic_ok) == (report['real_gate'],report['synthetic_gate'],report['selected'])
    parent_path = Path('outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth')
    assert sha(parent_path)==PARENT_SHA==run['parent_sha256']
    parent = torch.load(parent_path, map_location='cpu', weights_only=True)['model']
    initial = torch.load(root/'initial.pth', map_location='cpu', weights_only=True)
    final = torch.load(root/'final.pth', map_location='cpu', weights_only=True)['model']
    assert parent.keys()==initial.keys()==final.keys()
    assert all(torch.equal(parent[k], initial[k]) for k in parent)
    assert all(torch.equal(initial[k], final[k]) for k in initial if k.startswith('generator.'))
    assert all(torch.isfinite(v).all() for v in final.values())
    changed = sum(not torch.equal(initial[k],final[k]) for k in initial if k.startswith('segmenter.'))
    out = Path('outputs/retention_validation');out.mkdir(exist_ok=False)
    sheet = Image.new('RGB',(384,1480),'white')
    for i,r in enumerate(rows[:10]):
        tiles = [Image.open(r['image_path']).convert('RGB'),
                 Image.fromarray(truth['real'][i].astype('uint8')*255).convert('RGB'),
                 Image.fromarray(predictions['real'][i].astype('uint8')*255).convert('RGB')]
        ImageDraw.Draw(sheet).text((2,i*148+2),f'{i}: input | target | retained mask',fill='black')
        for j,tile in enumerate(tiles):sheet.paste(tile.resize((128,128)),(128*j,i*148+20))
    sheet.save(out/'preview.png')
    evidence = dict(archive_sha256=sha(archive), log_audit=log, masks_recounted=425,
                    scores=scores, real_gate=real_ok, synthetic_gate=synthetic_ok,
                    initial_equals_parent=True, generator_unchanged=True,
                    changed_segmenter_tensors=changed, promoted=False,
                    limitations=['Checkpoint-to-mask inference not reproduced',
                                 'Replay ceilings and trial losses not independently inferred',
                                 'Intermediate optimizer states not archived'])
    (out/'results.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':main()
