"""Existing L4 inference only; fresh parity and fixed paired development cohort."""
import argparse
from pathlib import Path
import shutil
import sys
import time


def run(root, parent, r2, mixed, baseline, pin):
    sys.path.insert(0, str(parent))
    from cctv_dgp_targets_v6 import require_vm
    require_vm(root)
    sys.path.insert(0, str(root))
    import cctv_dgp_input_selection_v19 as q
    p, v, old = q.verify(root, parent, r2, mixed, baseline, pin)
    import numpy as np
    from PIL import Image
    import torch
    from dgp_input_selector_v19 import select_restoration
    from dgp_broader_code_conditioner_v16 import DGPBroaderCodePrior
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from pretrained_face_restoration_portable_v12 import load_face_restorer
    from dgp_structure_conditioner_v18 import DGPStructureResidualHead, prior_features64
    from cctv_dgp_pilot import FixedObservedIdentity, grid112, state_hash
    torch.set_num_threads(4); torch.manual_seed(q.DESIGN['seed'])
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    start = time.monotonic()
    out = root / 'outputs/input_selection_v19'
    q.require(not out.exists(), 'Preserve previous/partial V19; no resume')
    q.require(shutil.disk_usage(root).free >= q.DESIGN['free_disk_required_bytes'], 'Need4GiB free disk')
    out.mkdir(parents=True)
    pp = v.read(parent / 'face_code_fit_protocol_v12.json')
    dgp, dp = load_frozen_dgp_restorer(parent / pp['weights']['dgp'],
        expected_sha256=pp['assets_sha256'][pp['weights']['dgp']], device='cuda')
    prior, cp = load_face_restorer(parent / pp['weights']['prior'], 'cuda')
    core = DGPBroaderCodePrior(dgp.net, prior.net).eval().cuda()
    core.conditioner.load_state_dict(torch.load(root / 'parent_v18/weights/r2_conditioner_epoch8.pth',
                                               map_location='cuda', weights_only=True), strict=True)
    head = DGPStructureResidualHead().cuda().eval()
    head.load_state_dict(torch.load(root / 'weights/structure_update600.pth', map_location='cuda', weights_only=True), strict=True)
    identity = FixedObservedIdentity(parent / pp['weights']['arcface'], 'cuda')
    frozen = {'dgp': core.core.dgp, 'prior': core.core.prior, 'r2_classifier': core.conditioner,
              'unused_v11': core.core.conditioner, 'recognizer': identity, 'spatial_decoder': head}
    for net in frozen.values(): net.requires_grad_(False).eval()
    before = {name: state_hash(net) for name, net in frozen.items()}
    q.require(before['spatial_decoder'] == p['terminal_state_hash'], 'Loaded terminal state differs')
    prior_receipt = v.read(root / 'lineage/v18_neural_receipt.json')
    q.require({k: before[k] for k in prior_receipt['frozen_before']} == prior_receipt['frozen_before'], 'Used frozen lineage/state differs')
    counts, handles = {}, []
    for name, net in [('dgp', core.core.dgp), ('prior_encoder', core.core.prior.encoder.blocks[0]),
        ('prior_classifier', core.core.prior.idx_pred_layer), ('r2_head', core.conditioner),
        ('prior_generator', core.core.prior.generator.blocks[0]), ('prior_RGB_tail', core.core.prior.generator.blocks[-1]),
        ('residual_head', head), ('recognizer', identity.encoder), ('unused_v11', core.core.conditioner)]:
        counts[name] = 0
        def count(_a, _b, _c, key=name): counts[key] += 1
        handles.append(net.register_forward_hook(count))
    artifacts, parity, rows, cached = {}, [], [], []
    def clock():
        torch.cuda.synchronize()
        q.require(time.monotonic() - start <= 1200, 'V19 inference exceeds1200s')
        q.require(torch.cuda.max_memory_allocated() <= q.DESIGN['peak_vram_cap_bytes'], 'V19 VRAM exceeds20GiB')
    def tensor(rgb):
        return torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()[None].cuda() / 255
    def save(name, value):
        path = q.safe(out, name); path.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith('.npy'):
            with path.open('xb') as stream: np.save(stream, value, allow_pickle=False)
        elif name.endswith('.npz'):
            with path.open('xb') as stream: np.savez(stream, **value)
        else:
            q.require(not path.exists(), 'No PNG overwrite'); Image.fromarray(value).save(path)
        artifacts[name] = q.sha(path); return name
    def embedding(rgb, ref, support):
        return identity.embedding(tensor(rgb), torch.from_numpy(support.astype(np.float32))[None, None].cuda(),
            torch.from_numpy(grid112(ref['matrix112']))[None].cuda())[0].cpu().numpy().copy()
    def predict(camera):
        x = tensor(camera)
        base, features, logits = core.frozen_inputs(x)
        codes = core.conditioner(x, base, features, logits)
        prior64 = prior_features64(core.core.prior, codes)
        result = head(x, base, prior64)
        return (base[0].permute(1, 2, 0).cpu().numpy().copy(),
                result[0].permute(1, 2, 0).cpu().numpy().copy(),
                {'dgp_base': base[0].cpu().numpy().copy(), 'prior64': prior64[0].cpu().numpy().copy()})
    q.write(out / 'execution.json', {'complete': True, 'protocol_sha256': pin,
        'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0), 'states_before': before,
        'provenance': {'dgp': dp, 'prior': cp}, 'design': q.DESIGN,
        'training': False, 'optimizer_constructed': False, 'backward_calls': 0, 'optimizer_updates': 0})
    artifacts['execution.json'] = q.sha(out / 'execution.json')
    train_refs = {r['id']: r for r in p['training_parity_references']}
    train_first = [p['training_parity_references'][0]['id'], p['training_parity_references'][5]['id']]
    control = {r['id']: r for r in v.read(root / 'lineage/processing_results_v19.json')['decisions']}
    baselines = v.read(baseline / 'results.json')
    base_rows = {r['id']: r for r in baselines['rows']}
    refs = {r['id']: r for r in p['references']}
    with torch.inference_mode():
        for c in p['training_parity_cases']:
            clock(); cid = c['id']; ref = train_refs[c['reference_id']]
            camera = v.rgb(mixed / c['input'])
            with Image.open(mixed / ref['observed']) as im: support = np.asarray(im).copy() > 0
            base, raw, arrays = predict(camera)
            decision = select_restoration(camera, support)
            q.require(decision['branch'] == control[cid]['branch'], 'Fresh training input decision differs')
            deltas = {}
            for key, value, folder in [('retained_dgp_v2', base, 'dgp_base'), ('structure_v18_update600', raw, 'structure_update600')]:
                previous = np.load(root / ('parity/' + folder + '/' + cid + '.npy'), allow_pickle=False)
                deltas[key] = float(np.max(np.abs(value - previous)))
                q.require(deltas[key] <= 2e-6 and np.array_equal(v.png(value, camera, support), v.png(previous, camera, support)),
                          'Fresh VM training raw/PNG parity differs; preserve failure')
            item = {'id': cid, 'decision': decision, 'maximum_float_differences': deltas,
                    'base_raw': save('parity/base/' + cid + '.npy', base),
                    'terminal_raw': save('parity/terminal/' + cid + '.npy', raw)}
            if c['reference_id'] in train_first and c['profile'] in ['clear', 'blur_lr24']:
                item['cache'] = save('cache_replay/' + cid + '.npz', arrays)
                cached.append({'id': cid, 'role': 'train', 'file': item['cache'], 'raw': item['terminal_raw']})
            parity.append(item)
        q.write(out / 'fresh_training_parity.json', {'complete': True, 'cases': parity, 'training': False})
        artifacts['fresh_training_parity.json'] = q.sha(out / 'fresh_training_parity.json')
        print('V19 fresh-image parity50 cases passed; zero training updates', flush=True)
        truth = {}
        for rid, ref in refs.items():
            clock()
            with Image.open(mixed / ref['observed']) as im: support = np.asarray(im).copy() > 0
            truth[rid] = embedding(v.rgb(mixed / ref['target']), ref, support)
            save('target_embeddings/' + rid + '.npy', truth[rid])
        durations = []
        for index, c in enumerate(p['cases'], 1):
            clock(); step = time.monotonic(); cid = c['id']; ref = refs[c['reference_id']]
            camera = v.rgb(mixed / c['input']); target = v.rgb(mixed / ref['target'])
            with Image.open(mixed / ref['observed']) as im: support = np.asarray(im).copy() > 0
            decision = select_restoration(camera, support)  # Before using a target/output for metrics.
            base, raw, arrays = predict(camera)
            base_png, terminal_png = v.png(base, camera, support), v.png(raw, camera, support)
            previous = base_rows[cid]
            q.require(np.array_equal(base_png, v.rgb(baseline / previous['arms']['retained_dgp_v2']['prediction'])),
                      'Fresh DGP PNG differs from declared V15 baseline')
            terminal_vec = embedding(terminal_png, ref, support)
            row = {k: c[k] for k in ['id', 'reference_id', 'source', 'profile']}
            row.update({'role': 'validation', 'decision': decision, 'arms': {}})
            for key, image, raw_value in [('retained_dgp_v2', base_png, base), ('structure_v18_update600', terminal_png, raw)]:
                values = v.metrics(image, target, support)
                if key == 'retained_dgp_v2':
                    vec = np.load(baseline / previous['arms'][key]['embedding'], allow_pickle=False)
                else: vec = terminal_vec
                row['arms'][key] = {**values, 'ArcFace_observed_fixed': float(np.clip(vec @ truth[c['reference_id']], -1, 1)),
                    'raw': save('raw/' + key + '/' + cid + '.npy', raw_value),
                    'prediction': save('predictions/' + key + '/' + cid + '.png', image),
                    'embedding': save('embeddings/' + key + '/' + cid + '.npy', vec),
                    'embedding_reused': key == 'retained_dgp_v2'}
            for key, source in [('basic_resizing', previous['input_metrics']),
                                ('pretrained_codeformer_none', previous['arms']['starting_prior_none'])]:
                image = camera if key == 'basic_resizing' else v.rgb(baseline / source['prediction'])
                vec = np.load(baseline / source['embedding'], allow_pickle=False)
                row['arms'][key] = {**v.metrics(image, target, support),
                    'ArcFace_observed_fixed': float(np.clip(vec @ truth[c['reference_id']], -1, 1)),
                    'prediction': save('predictions/' + key + '/' + cid + '.png', image),
                    'embedding': save('embeddings/' + key + '/' + cid + '.npy', vec), 'embedding_reused': True}
            selected = row['arms'][decision['branch']]
            row['arms']['automatic_v19'] = selected.copy()  # Exact raw/PNG/embedding alias; no display change.
            if c['reference_id'] in p['preview_reference_ids'] and c['profile'] in ['clear', 'blur_lr24']:
                cache_name = save('cache_replay/' + cid + '.npz', arrays)
                cached.append({'id': cid, 'role': 'validation', 'file': cache_name,
                               'raw': row['arms']['structure_v18_update600']['raw']})
            rows.append(row)
            clock(); durations.append(time.monotonic() - step)
            if index == 20:
                elapsed = time.monotonic() - start
                projection = elapsed + 500 * float(np.mean(durations[1:])) * 1.25 + 60
                q.write(out / 'inference_timing.json', {'cases': 20, 'seconds': elapsed,
                    'steady_sample_seconds': durations[1:], 'projected_seconds': projection, 'cap_seconds': 1200})
                artifacts['inference_timing.json'] = q.sha(out / 'inference_timing.json')
                q.require(projection <= 1200, 'V19 inference timing projection exceeds1200s')
            if index == 1 or index % 50 == 0:
                print({'validation_cases': index, 'total': 520, 'seconds': time.monotonic() - start}, flush=True)
    summaries = {arm: q.aggregate([{**{k: r[k] for k in ['source', 'profile']}, **r['arms'][arm]} for r in rows]) for arm in q.ARMS}
    guards = {arm: old.strict_preservation(summaries[arm], summaries['retained_dgp_v2'])
              for arm in ['structure_v18_update600', 'automatic_v19']}
    lookup = {(r['reference_id'], r['profile']): r for r in rows}
    grids = []
    font = __import__('PIL.ImageFont', fromlist=['ImageFont']).load_default()
    from PIL import ImageDraw
    for profile in v.PROFILES:
        grid = Image.new('RGB', (1560, 2904), (238, 238, 238)); draw = ImageDraw.Draw(grid)
        for col, label in enumerate(['camera', 'retained DGP', 'pretrained CodeFormer none', 'fixed V18 update600', 'automatic V19', 'clean proxy']):
            draw.text((col * 260 + 2, 2), label, fill=(0, 0, 0), font=font)
        for index, rid in enumerate(p['preview_reference_ids']):
            ref, row = refs[rid], lookup[(rid, profile)]
            images = [v.rgb(mixed / next(c['input'] for c in p['cases'] if c['id'] == row['id']))]
            images += [v.rgb(out / row['arms'][arm]['prediction']) for arm in q.ARMS[1:]]
            images += [v.rgb(mixed / ref['target'])]
            for col, value in enumerate(images):
                x, y = col * 260 + 2, 24 + index * 288
                draw.text((x, y + 2), row['id'], fill=(0, 0, 0), font=font)
                grid.paste(Image.fromarray(value), (x, y + 28))
        name = 'grids/' + profile + '.png'
        grids.append(save(name, np.asarray(grid)))
        clock()
    after = {name: state_hash(net) for name, net in frozen.items()}
    q.require(before == after and counts == q.expected_counts() and len(cached) == 24, 'Frozen state/neural scope differs')
    q.require(all(not net.training and not any(t.requires_grad or t.grad is not None for t in net.parameters())
                  for net in frozen.values()), 'Inference-only freeze/gradient invariant differs')
    q.write(out / 'neural_receipt.json', {'complete': True, 'counts': counts, 'before': before, 'after': after,
        'optimizer_constructed': False, 'backward_calls': 0, 'optimizer_updates': 0,
        'peak_allocated_vram_bytes': torch.cuda.max_memory_allocated(), 'cached_head_cases': cached})
    artifacts['neural_receipt.json'] = q.sha(out / 'neural_receipt.json')
    clock()
    q.write(out / 'results.json', {'complete': True, 'protocol_sha256': pin, 'rows': rows,
        'summaries': summaries, 'preservation': guards, 'grids': grids, 'artifacts_sha256': artifacts,
        'seconds': time.monotonic() - start, 'training': False, 'optimizer_updates': 0, 'backward_calls': 0,
        'validation_used': True, 'teacher_used': False, 'native_used': False, 'native_reserved_used': False,
        'best_checkpoint_selected': False, 'thresholds_refitted': False, 'production_promoted': False,
        'limitation': 'Previously used photographic development validation, not independent final evaluation or native CCTV evidence. Pretrained corpus overlap unknown; input selector does not qualify insufficient/covered crops.'})
    print({'complete': True, 'validation_cases': 520, 'optimizer_updates': 0, 'preservation': guards,
           'seconds': time.monotonic() - start}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'parent', 'r2', 'mixed', 'baseline']: parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-sha', required=True)
    a = parser.parse_args()
    run(a.root.resolve(), a.parent.resolve(), a.r2.resolve(), a.mixed.resolve(), a.baseline.resolve(), a.expected_sha)
