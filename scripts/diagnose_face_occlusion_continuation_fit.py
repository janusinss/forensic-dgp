"""Continuation fit, small targets and glare; inference only, no optimization."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from completion import KINDS
from scripts.compare_replay_fit import counts, aggregate
from scripts.diagnose_face_occlusion_replay import summarize_records, verify_reference
from scripts.audit_face_occlusion_results import require, read_json, read_lines, verified_inputs
from scripts.audit_face_occlusion_continuation_results import (
    ARCHIVE, AUDIT_OUTPUT, EXTRACTION, PREFIX, START_SHA, REPLAY_SHA, PROTOCOL_SHA,
    PARENT_SHA, audit_history, verify_package,
)
from scripts.package_face_occlusion_vm import sha
from scripts.evaluate_coverage_results import binary


def area_summary(records):
    require(records and len({r['case'] for r in records}) == len(records), 'Require unique nonempty cases')
    groups = {}
    for row in records:
        c = row['counts']; fg = c['tp'] + c['fn']; total = fg + c['visible']
        require(set(c) == {'tp', 'fp', 'fn', 'visible', 'empty', 'positive', 'negative', 'false_positive'} and
                all(type(v) is int and v >= 0 for v in c.values()) and total > 0 and c['fp'] <= c['visible'],
                'Invalid pixel confusion counts')
        require(c['positive'] == int(fg > 0) and c['negative'] == int(fg == 0) and
                c['empty'] == int(fg > 0 and c['tp'] + c['fp'] == 0) and
                c['false_positive'] == int(fg == 0 and c['fp'] > 0), 'Counts disagree with target/prediction')
        band = 'clear' if fg == 0 else ('covered_le_1pct' if 100 * fg <= total else 'covered_gt_1pct')
        groups.setdefault(band, []).append(c)
    return {'all': aggregate([r['counts'] for r in records]),
            'by_area': {band: aggregate(rows) for band, rows in groups.items()}}


def main():
    import torch
    import numpy as np
    from PIL import Image, ImageDraw
    torch.set_num_threads(4)
    out = ROOT / 'outputs/face_occlusion_continuation_fit'
    require(not out.exists(), 'Preserve existing continuation fit evidence')
    audit = read_json(AUDIT_OUTPUT / 'results.json'); reproduction = read_json(AUDIT_OUTPUT / 'reproduction.json')
    require(reproduction['complete'] is True and reproduction['saved_masks_compared'] == 2915 and
            reproduction['selection_agrees_with_vm'] is True and
            sha(ARCHIVE) == audit['archive_sha256'] == reproduction['archive_sha256'],
            'Require completed, unchanged return audit and checkpoint verification')
    verify_package(); protocol, validation_rows, data = verified_inputs()
    returned = EXTRACTION / PREFIX; run = read_json(returned / 'run.json')
    complete = read_json(returned / 'complete.json'); baseline = read_json(returned / 'baseline.json')
    history = read_lines(returned / 'metrics.jsonl'); steps = read_lines(returned / 'steps.jsonl')
    base_schedule = protocol['schedules']['extended']['batches']
    audit_history(steps, history, base_schedule, baseline, complete)
    source_metrics = read_json(returned / 'source_metrics.json')
    members = read_json(AUDIT_OUTPUT / 'members.json')
    cache_path = ROOT / 'outputs/downloaded_coverage/outputs/coverage_training_vm/replay_pixels.pth'
    require(sha(cache_path) == REPLAY_SHA and sha(ROOT / 'outputs/coverage_protocol_v1/protocol.json') == PROTOCOL_SHA,
            'Frozen cache/protocol changed')
    cache = torch.load(cache_path, map_location='cpu', weights_only=True)
    indices = sorted(cache)
    weights = Counter(i - 73 for e in base_schedule for b in e for i in b if i >= 73)
    require(set(weights) == set(cache) and len(cache) == 638 and sum(weights.values()) == 840,
            'Training replay membership/exposure differs')
    for x, m in cache.values():
        require(x.dtype == m.dtype == torch.uint8 and x.shape == (3, 256, 256) and m.shape == (1, 256, 256)
                and ((m == 0) | (m == 1)).all(), 'Invalid cached tensor')
    previous_path = ROOT / 'outputs/face_occlusion_replay_fit/results.json'; previous = read_json(previous_path)
    require(previous['optimizer_updates_locally'] == 0 and previous['replay_sha256'] == REPLAY_SHA and
            previous['protocol_sha256'] == PROTOCOL_SHA and previous['checkpoints']['parent'] == PARENT_SHA and
            previous['checkpoints']['pretrained10'] == START_SHA and sha(returned / 'initial.pth') == START_SHA,
            'Prior replay initialization/provenance differs')
    reused = {}
    for key in ('parent', 'pretrained10'):
        records = previous['training'][key]['records']; verify_reference(records, cache, protocol)
        require(summarize_records(records) == previous['training'][key]['unique'] and
                summarize_records(records, weights) == previous['training'][key]['scheduled_exposure'],
                'Reused replay summary differs from counts')
        reused[key] = {'records': records, 'unique': summarize_records(records),
                       'scheduled_exposure': summarize_records(records, weights), 'area': area_summary(records)}
    result = {'script_sha256': sha(__file__), 'archive_sha256': audit['archive_sha256'],
              'reproduction_sha256': sha(AUDIT_OUTPUT / 'reproduction.json'),
              'prior_replay_report_sha256': sha(previous_path), 'protocol_sha256': PROTOCOL_SHA,
              'replay_sha256': REPLAY_SHA, 'optimizer_updates_locally': 0, 'evaluation_batch_size': 8,
              'reused_parent_and_source_records': True, 'training': reused, 'validation': {},
              'real_fit': {}, 'promoted': False}
    def saved(folder, domain, i):
        path = returned / folder / domain / f'{i:04}.png'
        require(sha(path) == members[path.relative_to(EXTRACTION).as_posix()], 'Audited mask changed')
        return binary(path)
    benchmark_path = ROOT / 'outputs/downloaded_completion_benchmark/outputs/completion_pretrained_vm/manifest.json'
    cases = read_json(benchmark_path)['cases']; require(len(cases) == 400, 'Validation membership differs')
    validation_truth = [data['synthetic'][i][1][0].numpy().astype(bool) for i in range(400)]
    choices = [('parent', 'baseline_masks', baseline), ('pretrained10', 'initial_masks', source_metrics)]
    choices.extend((f'pretrained{r["epoch"]}', f'epoch_{r["epoch"]}_masks', r) for r in history)
    for key, folder, logged in choices:
        records = [{'case': i, 'kind': case['kind'], 'degraded': case['degraded'], 'source': case['dataset_source'],
                    'counts': counts(saved(folder, 'synthetic', i), validation_truth[i])}
                   for i, case in enumerate(cases)]
        groups = summarize_records(records)
        require({k: v for k, v in groups['all'].items() if k not in ('cases', 'unique_cases')} == logged['synthetic'],
                'Grouped validation does not match independently audited masks')
        result['validation'][key] = {'records': records, 'groups': groups, 'area': area_summary(records),
                                    'clear_false_positive_cases': [r['case'] for r in records if r['counts']['false_positive']]}
    training_rows = data['training_real'].rows
    source_exposure = Counter(i for e in base_schedule for b in e for i in b if i < 73)
    continued_schedule = base_schedule + base_schedule
    require(len(training_rows) == 73 and len(validation_rows) == 25, 'Real membership differs')
    for key, folder, logged in choices[1:]:
        additional = 0 if key == 'pretrained10' else logged['additional_epoch']
        exposure = source_exposure + Counter(i for e in continued_schedule[:additional] for b in e for i in b if i < 73)
        domains = {}
        for domain, rows in [('training_real', training_rows), ('real', validation_rows)]:
            records = []
            for i, row in enumerate(rows):
                t = data[domain][i][1][0].numpy().astype(bool); c = counts(saved(folder, domain, i), t)
                records.append({'case': i, 'image': row['image'], 'kind': row['kind'],
                                'glare_stratum': row.get('glare_stratum'), 'counts': c,
                                'target_pixels': int(t.sum()), 'target_fraction': float(t.mean()),
                                'training_exposures': exposure[i] if domain == 'training_real' else 0})
            areas = area_summary(records)
            require({k: v for k, v in areas['all'].items() if k != 'cases'} == logged[domain],
                    'Real training/validation recount differs')
            glare = [r for r in records if r['glare_stratum'] == 'strong_lens_reflection']
            require(len(glare) == (2 if domain == 'training_real' else 1), 'Glare case membership differs')
            domains[domain] = {'records': records, 'area': areas,
                               'glare': {'records': glare, 'metrics': aggregate([r['counts'] for r in glare])}}
        result['real_fit'][key] = domains
    result['loss_curve'] = []
    for additional in range(1, 21):
        rows = steps[(additional - 1) * 21:additional * 21]
        result['loss_curve'].append({'global_epoch': 10 + additional,
                                    'mean_total_loss': sum(r['loss'] for r in rows) / 21,
                                    'mean_pre_clip_norm': sum(r['pre_clip_norm'] for r in rows) / 21,
                                    'note': 'Combined loss only; separate supervised/teacher terms were not logged'})
    out.mkdir(); (out / 'partial.json').write_text(json.dumps(result, indent=2) + '\n')
    sys.path.insert(0, str(ROOT / 'outputs/face_extraction_dependencies'))
    from face_occlusion_adapter import load_adapter
    from scripts.train_face_occlusion_vm import dependency_versions
    result['dependencies'] = dependency_versions(ROOT / 'outputs/face_extraction_dependencies')
    checkpoint = returned / 'epoch_30.pth'
    require(sha(checkpoint) == reproduction['checkpoint_checks']['30']['sha256'] == complete['final_sha256'],
            'Verified final checkpoint changed')
    model, payload = load_adapter(checkpoint, 'cpu')
    require(payload['epoch'] == 30 and payload['optimizer_updates'] == 630 and payload['continuation_metadata'] == run,
            'Final checkpoint metadata differs')
    model.requires_grad_(False).eval(); before = {k: v.clone() for k, v in model.state_dict().items()}
    records = []; mask_dir = out / 'train_masks/pretrained30'; mask_dir.mkdir(parents=True)
    with torch.inference_mode():
        for start in range(0, len(indices), 8):
            batch = indices[start:start + 8]; x = torch.stack([cache[i][0].float() / 255 for i in batch])
            predictions = (model.detect(x).sigmoid()[:, 0] >= .5).numpy()
            for i, p in zip(batch, predictions):
                t = cache[i][1][0].numpy().astype(bool)
                records.append({'case': i, 'kind': KINDS[i % 5], 'degraded': i % 10 >= 5,
                                'source': protocol['sources'][i // 10]['source'], 'counts': counts(p, t)})
                Image.fromarray(p.astype('uint8') * 255).save(mask_dir / f'{i:04}.png')
            if start % 160 == 0: print('pretrained30 replay', start, len(indices), flush=True)
    require(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()), 'Read-only inference changed model')
    verify_reference(records, cache, protocol)
    result['training']['pretrained30'] = {'records': records, 'unique': summarize_records(records),
                                        'scheduled_exposure': summarize_records(records, weights),
                                        'area': area_summary(records), 'state_unchanged': True, 'checkpoint_sha256': sha(checkpoint)}
    del model, payload, before
    # Hold the earlier source10 training failure cases fixed for the before/after grid.
    ids = previous['preview_selection']['cases']
    require(len(ids) == len(set(ids)) == 10 and all(i in cache for i in ids), 'Prior preview membership differs')
    source_records = {r['case']: r for r in result['training']['pretrained10']['records']}
    sheet = Image.new('RGB', (640, 1480), 'white')
    for line, i in enumerate(ids):
        image = cache[i][0].permute(1, 2, 0).numpy(); t = cache[i][1][0].numpy().astype(bool)
        old = binary(ROOT / f'outputs/face_occlusion_replay_fit/train_masks/pretrained10/{i:04}.png')
        require(counts(old, t) == source_records[i]['counts'], 'Reused preview mask disagrees with prior counts')
        p = binary(mask_dir / f'{i:04}.png'); overlay = image.copy()
        overlay[t & ~p] = (220, 55, 65); overlay[~t & p] = (55, 135, 220)
        tiles = [Image.fromarray(image), Image.fromarray(t.astype('uint8') * 255).convert('RGB'),
                 Image.fromarray(old.astype('uint8') * 255).convert('RGB'),
                 Image.fromarray(p.astype('uint8') * 255).convert('RGB'), Image.fromarray(overlay)]
        ImageDraw.Draw(sheet).text((2, line * 148 + 2), f'{i}: input | target | source10 | epoch30 | red=miss blue=extra', fill='black')
        for j, tile in enumerate(tiles): sheet.paste(tile.resize((128, 128)), (j * 128, line * 148 + 20))
    sheet.save(out / 'synthetic_preview.png')
    result['synthetic_preview'] = {'training_only': True, 'source10_failure_ranked_cases_held_fixed': ids}
    glare_rows = [(d, r['case']) for d in ('training_real', 'real')
                  for r in result['real_fit']['pretrained30'][d]['glare']['records']]
    sheet = Image.new('RGB', (1024, 148 * len(glare_rows)), 'white')
    for line, (domain, i) in enumerate(glare_rows):
        x, m = data[domain][i]; image = np.rint(x.permute(1, 2, 0).numpy() * 255).astype('uint8')
        t = m[0].numpy().astype(bool); yy, xx = np.where(t)
        box = (max(0, int(xx.min()) - 12), max(0, int(yy.min()) - 12),
               min(256, int(xx.max()) + 13), min(256, int(yy.max()) + 13))
        tiles = [Image.fromarray(image), Image.fromarray(image).crop(box),
                 Image.fromarray(t.astype('uint8') * 255).convert('RGB').crop(box)]
        tiles.extend(Image.fromarray(saved(folder, domain, i).astype('uint8') * 255).convert('RGB').crop(box)
                     for _, folder, _ in choices[1:])
        ImageDraw.Draw(sheet).text((2, line * 148 + 2),
                                  f'{domain} {i}: face | zoom | target | source10 | epoch11 | epoch15 | epoch20 | epoch30', fill='black')
        for j, tile in enumerate(tiles): sheet.paste(tile.resize((128, 128)), (j * 128, line * 148 + 20))
    sheet.save(out / 'glare_preview.png')
    failed_clear = result['validation']['pretrained30']['clear_false_positive_cases']
    result['validation_clear_false_positive_cases'] = failed_clear
    if failed_clear:
        sheet = Image.new('RGB', (768, 148 * min(5, len(failed_clear))), 'white')
        for line, i in enumerate(failed_clear[:5]):
            x, m = data['synthetic'][i]; image = np.rint(x.permute(1, 2, 0).numpy() * 255).astype('uint8')
            t = m[0].numpy().astype(bool); p = saved('epoch_30_masks', 'synthetic', i); overlay = image.copy()
            overlay[t & ~p] = (220, 55, 65); overlay[~t & p] = (55, 135, 220)
            masks = [t, saved('baseline_masks', 'synthetic', i), saved('initial_masks', 'synthetic', i), p]
            tiles = [Image.fromarray(image)] + [Image.fromarray(a.astype('uint8') * 255).convert('RGB') for a in masks] + [Image.fromarray(overlay)]
            ImageDraw.Draw(sheet).text((2, line * 148 + 2), f'Validation {i}: input | target | parent | source10 | epoch30 | blue=extra', fill='black')
            for j, tile in enumerate(tiles): sheet.paste(tile.resize((128, 128)), (j * 128, line * 148 + 20))
        sheet.save(out / 'clear_error_preview.png')
    result['complete'] = True
    (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'training_replay': {k: v['unique']['all'] for k, v in result['training'].items()},
                      'glare': {k: {d: v['glare']['metrics'] for d, v in domains.items()} for k, domains in result['real_fit'].items()},
                      'optimizer_updates_locally': 0}, indent=2), flush=True)


if __name__ == '__main__': main()
