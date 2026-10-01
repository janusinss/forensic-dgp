"""Partition verified reflection and retention errors; read-only, no fitting."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_face_occlusion_results import require, read_json, verified_inputs
from scripts.audit_reflection_coverage_results import ARCHIVE, EXTRACTION, AUDIT_OUTPUT, PREFIX, ARMS, DATA_PATH
from scripts.audit_face_occlusion_glare_pixels import added_region_counts
from scripts.evaluate_coverage_results import binary
from scripts.package_face_occlusion_vm import sha
from detector_training import load_manifest, ReviewedMasks


def transition_counts(parent, candidate, truth):
    require(all(isinstance(x, np.ndarray) and x.dtype == np.bool_ and x.ndim == 2
                for x in (parent, candidate, truth)) and parent.shape == candidate.shape == truth.shape,
            'Require equal binary two-dimensional masks')
    count = lambda x: int(np.count_nonzero(x))
    row = {'parent_tp': count(parent & truth), 'candidate_tp': count(candidate & truth),
           'parent_fp': count(parent & ~truth), 'candidate_fp': count(candidate & ~truth),
           'parent_fn': count(~parent & truth), 'candidate_fn': count(~candidate & truth),
           'lost_parent_tp': count(parent & truth & ~candidate),
           'recovered_parent_fn': count(~parent & truth & candidate),
           'removed_parent_fp': count(parent & ~truth & ~candidate),
           'added_visible_fp': count(~parent & ~truth & candidate),
           'visible': count(~truth), 'positive': int(truth.any()), 'negative': int(not truth.any()),
           'parent_empty': int(truth.any() and not parent.any()),
           'candidate_empty': int(truth.any() and not candidate.any()),
           'parent_clear_error': int(not truth.any() and parent.any()),
           'candidate_clear_error': int(not truth.any() and candidate.any())}
    require(row['candidate_tp'] == row['parent_tp'] - row['lost_parent_tp'] + row['recovered_parent_fn']
            and row['candidate_fp'] == row['parent_fp'] - row['removed_parent_fp'] + row['added_visible_fp'],
            'Pixel-transition conservation failed')
    return row


def aggregate_counts(rows):
    require(bool(rows) and all(set(r) == set(rows[0]) for r in rows), 'Missing/inconsistent partition counts')
    result = {k: sum(r[k] for r in rows) for k in rows[0]}
    result.update(cases=len(rows), covered_cases=result['positive'], negative_cases=result['negative'])
    for model in ('parent', 'candidate'):
        tp, fp, fn = (result[f'{model}_{k}'] for k in ('tp', 'fp', 'fn'))
        result.update({f'{model}_iou': tp/max(1, tp+fp+fn),
                       f'{model}_missed_fraction': fn/max(1, tp+fn),
                       f'{model}_visible_false_positive': fp/max(1, result['visible']),
                       f'{model}_negative_false_positive_cases': result[f'{model}_clear_error']})
    return result


def fixture_groups(records, cases):
    by_id = {r['case_id']: r for r in records}
    require(set(by_id) == {r['case_id'] for r in cases} and len(by_id) == len(records) == 280,
            'Fixture diagnostic membership differs')
    result = {}
    for axis in ('style', 'degraded'):
        for value in sorted({r[axis] for r in cases}, key=str):
            selected = [by_id[r['case_id']] for r in cases if r[axis] == value]
            totals = {k: sum(r[k] for r in selected) for k in ('tp', 'fp', 'fn', 'visible', 'ignored_positive_pixels')}
            totals.update(cases=len(selected), iou=totals['tp']/max(1, totals['tp']+totals['fp']+totals['fn']),
                          recall=totals['tp']/max(1, totals['tp']+totals['fn']),
                          visible_false_positive=totals['fp']/max(1, totals['visible']),
                          clear_errors=sum(r['tp']+r['fn'] == 0 and r['fp'] > 0 for r in selected),
                          empty_covered=sum(r['tp']+r['fn'] > 0 and r['tp']+r['fp'] == 0 for r in selected))
            result[f'{axis}/{value}'] = totals
    return result


def main():
    import torch
    from PIL import Image, ImageDraw
    out = ROOT/'outputs/reflection_coverage_analysis_v1'
    require(not out.exists(), 'Preserve existing analysis evidence')
    audit_path = AUDIT_OUTPUT/'results.json'; repro_path = AUDIT_OUTPUT/'reproduction.json'
    audit, repro = read_json(audit_path), read_json(repro_path)
    require(repro['complete'] is True and repro['saved_masks_compared'] == 4315 and
            repro['selection_agrees_with_vm'] is True and repro['audit_sha256'] == sha(audit_path) and
            repro['archive_sha256'] == audit['archive_sha256'] == sha(ARCHIVE) and
            audit['checksum_verified'] is True and audit['script_sha256'] == sha(ROOT/'scripts/audit_reflection_coverage_results.py') and
            repro['script_sha256'] == sha(ROOT/'scripts/reproduce_reflection_coverage_results.py'),
            'Require completed unchanged return and CPU prediction audits')
    _, validation_rows, datasets = verified_inputs()
    members = read_json(AUDIT_OUTPUT/'members.json'); returned = EXTRACTION/PREFIX
    folders = {'parent': returned/'baseline_masks', 'source30': returned/'initial_masks'}
    folders.update({f'{a}/{e}': returned/a/f'epoch_{e}_masks' for e in (36, 42) for a in ARMS})
    def saved(key, domain, i):
        path = folders[key]/domain/f'{i:04}.png'
        require(sha(path) == members[path.relative_to(EXTRACTION).as_posix()], 'Audited mask changed')
        return binary(path)
    bench = datasets['synthetic'].cases
    truth = [datasets['synthetic'][i][1][0].numpy().astype(bool) for i in range(400)]
    parents = [saved('parent', 'synthetic', i) for i in range(400)]
    synthetic = {}
    for key in folders:
        if key == 'parent': continue
        rows = [{'case': i, 'kind': b['kind'], 'degraded': b['degraded'], 'source': b['dataset_source'],
                 **transition_counts(parents[i], saved(key, 'synthetic', i), truth[i])} for i, b in enumerate(bench)]
        only_counts = lambda r: {k: v for k, v in r.items() if k not in ('case', 'kind', 'degraded', 'source')}
        groups = {'all': aggregate_counts([only_counts(r) for r in rows])}
        for axis in ('kind', 'degraded', 'source'):
            for value in sorted({r[axis] for r in rows}, key=str):
                groups[f'{axis}/{value}'] = aggregate_counts([only_counts(r) for r in rows if r[axis] == value])
        expected = audit['source']['synthetic'] if key == 'source30' else next(
            r['synthetic'] for r in audit['candidates'][key.split('/')[0]] if r['epoch'] == int(key.split('/')[1]))
        require(groups['all']['candidate_iou'] == expected['iou'], 'Synthetic micro-IoU differs')
        synthetic[key] = {'groups': groups, 'records': rows}
    data = read_json(ROOT/DATA_PATH/'manifest.json')
    fixtures = {'source30': fixture_groups(audit['source_training_reflection']['records'], data['cases'])}
    for a in ARMS:
        for row in audit['candidates'][a]:
            fixtures[f'{a}/{row["epoch"]}'] = fixture_groups(row['training_reflection']['records'], data['cases'])
    old_path = ROOT/'dataset/detector_expanded_review_v2/manifest.json'
    v3_path = ROOT/'dataset/detector_glare_review_v3/manifest.json'; v3 = read_json(v3_path)
    require(sha(old_path) == v3['parent_manifest_sha256'], 'Reflection label lineage differs')
    old = {r['image_sha256']: r for r in load_manifest(old_path)}
    glare = []; visuals = []
    keys = [k for k in folders if k != 'parent']
    for domain, rows in (('training_real', datasets['training_real'].rows), ('real', validation_rows)):
        for i, row in enumerate(rows):
            if row.get('glare_stratum') != 'strong_lens_reflection': continue
            x, target = datasets[domain][i]; target = target[0].numpy().astype(bool)
            parent = ReviewedMasks([old[row['image_sha256']]], 256)[0][1][0].numpy().astype(bool)
            predictions = {k: saved(k, domain, i) for k in keys}
            glare.append({'domain': domain, 'case': i, 'image_sha256': row['image_sha256'],
                          'states': {k: added_region_counts(p, target, parent) for k, p in predictions.items()}})
            visuals.append((domain, i, x, target & ~parent, {k: p & ~parent for k, p in predictions.items()}))
    require(len(glare) == 3 and sum(r['domain'] == 'real' for r in glare) == 1, 'Glare cohort differs')
    out.mkdir()
    synthetic_preview_ids = {}
    for pool in sorted({r['dataset_source'] for r in bench}):
        ids = [next(i for i, r in enumerate(bench) if r['dataset_source'] == pool and r['kind'] == kind
                    and r['degraded'] == camera) for kind in ('none', 'lower', 'eyes', 'irregular', 'object')
               for camera in (False, True)]
        synthetic_preview_ids[pool] = ids
        sheet = Image.new('RGB', (1024, 148*len(ids)), 'white')
        for line, i in enumerate(ids):
            x, _ = datasets['synthetic'][i]
            tiles = [Image.fromarray(np.rint(x.permute(1, 2, 0).numpy()*255).astype('uint8')),
                     Image.fromarray(truth[i].astype('uint8')*255).convert('RGB')]
            tiles.extend(Image.fromarray(saved(k, 'synthetic', i).astype('uint8')*255).convert('RGB') for k in folders)
            ImageDraw.Draw(sheet).text((2, line*148+2), f'Synthetic {i} {bench[i]["kind"]}/{bench[i]["degraded"]}: input | target | '+ ' | '.join(folders), fill='black')
            for j, tile in enumerate(tiles): sheet.paste(tile.resize((128, 128)), (j*128, line*148+20))
        sheet.save(out/f'synthetic_{Path(pool).name}.png')
    sheet = Image.new('RGB', (1024, 148*3), 'white')
    for line, (domain, i, x, added, predictions) in enumerate(visuals):
        face = Image.fromarray(np.rint(x.permute(1, 2, 0).numpy()*255).astype('uint8'))
        yy, xx = np.where(added); box = (max(0, int(xx.min())-12), max(0, int(yy.min())-12),
                                       min(256, int(xx.max())+13), min(256, int(yy.max())+13))
        tiles = [face, face.crop(box), Image.fromarray(added.astype('uint8')*255).convert('RGB').crop(box)]
        tiles.extend(Image.fromarray(predictions[k].astype('uint8')*255).convert('RGB').crop(box) for k in keys)
        ImageDraw.Draw(sheet).text((2, line*148+2), f'{domain} {i}: face | zoom | lens target | '+ ' | '.join(keys), fill='black')
        for j, tile in enumerate(tiles): sheet.paste(tile.resize((128, 128), Image.Resampling.NEAREST), (j*128, line*148+20))
    sheet.save(out/'glare.png')
    pixels_path = ROOT/DATA_PATH/'pixels.pth'
    require(sha(pixels_path) == data['pixel_cache_sha256'], 'Fixture bytes changed')
    pixels = torch.load(pixels_path, map_location='cpu', weights_only=True)['pixels']
    for camera in (False, True):
        sheet = Image.new('RGB', (640, 148*10), 'white')
        for line, i in enumerate(pos*10+style*2+int(camera) for pos in (0, 14) for style in range(5)):
            case = pixels[i]; face = Image.fromarray(case['input'].permute(1, 2, 0).numpy())
            tiles = [face, Image.fromarray(case['mask'][0].numpy()*255).convert('RGB')]
            tiles.extend(Image.fromarray(saved(k, 'training_reflection', i).astype('uint8')*255).convert('RGB')
                         for k in ('source30', 'control/42', 'reflective/42'))
            ImageDraw.Draw(sheet).text((2, line*148+2), f'Training fixture {i}: input | target | source30 | control42 | reflective42', fill='black')
            for j, tile in enumerate(tiles): sheet.paste(tile.resize((128, 128)), (j*128, line*148+20))
        sheet.save(out/f'fixture_{"degraded" if camera else "clean"}.png')
    report = {'complete': True, 'date': '2026-10-02', 'script_sha256': sha(__file__),
              'archive_sha256': audit['archive_sha256'], 'audit_sha256': sha(audit_path),
              'reproduction_sha256': sha(repro_path), 'v2_manifest_sha256': sha(old_path), 'v3_manifest_sha256': sha(v3_path),
              'synthetic': synthetic, 'training_fixtures': fixtures, 'lens_only': glare,
              'preview_fixture_positions': [0, 14], 'preview_synthetic_ids': synthetic_preview_ids,
              'preview_note': 'Fixed first source/kind/camera views, not selected by candidate errors; synthetic sheets contain one identity per source pool.',
              'optimizer_constructed': False, 'optimizer_updates_locally': 0,
              'promoted': False, 'scope': 'Audited saved VM masks; CPU differences separately recorded in reproduction.json. Training fixtures are not independent validation.',
              'limitation': 'Only two real training reflection images and one reused development validation image. Approximate assistant polygons; external identity/pretraining overlap unresolved.'}
    (out/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'complete': True, 'synthetic_cases_per_checkpoint': 400, 'fixture_cases_per_checkpoint': 280,
                      'real_reflection_cases': len(glare), 'optimizer_updates_locally': 0, 'promoted': False}), flush=True)


if __name__ == '__main__': main()
