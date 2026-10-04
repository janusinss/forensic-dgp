"""Frozen training-only code/render controls; CUDA inference, no fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
PARENT_SHA = '06f056ea4b1b04e6d8c631e5e78df129345c78279230c7946f6b5deb2e226846'
RESULTS_SHA = '95fb1993e93aba1972ae89d024df1ad8d50574d1fefd7eacf5dc977b6c9c4796'
PLAN_NAME = 'render_controls_protocol_v13.json'
ARMS = ['initial_no_stats', 'learned300_no_stats', 'teacher_observed_stats', 'teacher_no_stats']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def prepare(destination):
    require(not destination.exists(), 'Preserve frozen/partial preparation')
    parent = ROOT / 'outputs/cctv_dgp_face_code_fit_vm_v12_r2'
    returned = ROOT / 'outputs/cctv_dgp_face_code_fit_return_v12_r2_verified'
    require(sha(parent / 'face_code_fit_protocol_v12.json') == PARENT_SHA, 'Parent protocol differs')
    audit = read(returned / 'local_independent_audit.json')
    require(audit['complete'] and audit['results_sha256'] == RESULTS_SHA, 'Require completed parent audit')
    p = read(parent / 'face_code_fit_protocol_v12.json')
    names = ['scripts/render_cctv_dgp_face_code_controls_v13.py',
             'scripts/audit_cctv_dgp_face_code_controls_v13.py']
    source_pins = {name: sha(ROOT / name) for name in names}
    plan = {
        'format': 'cctv-dgp-face-code-render-controls-v13', 'date': '2026-10-04',
        'parent_protocol_sha256': PARENT_SHA, 'parent_results_sha256': RESULTS_SHA,
        'parent_assets_sha256': p['assets_sha256'], 'sources_sha256': source_pins,
        'references': p['references'], 'cases': p['cases'], 'profiles': p['profiles'], 'arms': ARMS,
        'oracle_policy': 'Clean teacher code labels are target-informed capacity controls, never restored outputs or deployable inference.',
        'rendering': 'Frozen original CodeFormer codebook and generator; fidelity w0; compare saved initial/learned top1 codes and clean labels, with or without original observed encoder AdaIN statistics.',
        'decision': 'Compare learned no-stat outputs with teacher no-stat and teacher observed-stat controls. Repair code prediction and/or degraded feature transfer only after review. No selection or promotion.',
        'budget': {'runtime_seconds': 240, 'generator_forwards': 162, 'backward_calls': 0, 'optimizer_updates': 0},
        'training': False, 'validation_used': False, 'native_used': False,
        'native_reserved_used': False, 'production_promoted': False,
        'parent_local_audit_sha256': sha(returned / 'local_independent_audit.json'),
    }
    destination.mkdir(parents=True)
    for name in names:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    write(destination / PLAN_NAME, plan)
    (destination / 'render_controls_protocol_v13.sha256').write_text(sha(destination / PLAN_NAME) + '\n', encoding='ascii')
    archive = destination.parent / 'cctv-dgp-face-code-controls-v13-execution.tar.gz'
    require(not archive.exists(), 'Preserve prior execution archive')
    with tarfile.open(archive, 'w:gz') as stream:
        for path in sorted(destination.rglob('*')):
            if path.is_file():
                stream.add(path, arcname=path.relative_to(destination).as_posix(), recursive=False)
    Path(str(archive) + '.sha256').write_text(sha(archive) + '  ' + archive.name + '\n', encoding='ascii')
    print(json.dumps({'prepared': True, 'protocol_sha256': sha(destination / PLAN_NAME),
                      'archive_sha256': sha(archive), 'bytes': archive.stat().st_size,
                      'cases': 50, 'training': False, 'oracle_target_access': True}), flush=True)


def verify(root, parent, expected_sha):
    require(sha(root / PLAN_NAME) == expected_sha ==
            (root / 'render_controls_protocol_v13.sha256').read_text().strip(), 'Control protocol differs')
    p = read(root / PLAN_NAME)
    require(p['format'] == 'cctv-dgp-face-code-render-controls-v13' and p['arms'] == ARMS
            and p['parent_protocol_sha256'] == PARENT_SHA and p['parent_results_sha256'] == RESULTS_SHA,
            'Control design differs')
    require(all(p[key] is False for key in ['training','validation_used','native_used','native_reserved_used','production_promoted']), 'Control scope differs')
    for name, pin in p['sources_sha256'].items():
        require(sha(root / name) == pin, 'Control source differs: ' + name)
    require(sha(parent / 'face_code_fit_protocol_v12.json') == PARENT_SHA, 'Parent protocol differs')
    parent_plan = read(parent / 'face_code_fit_protocol_v12.json')
    require(parent_plan['assets_sha256'] == p['parent_assets_sha256'] and
            parent_plan['cases'] == p['cases'] and parent_plan['references'] == p['references'], 'Parent cohort/assets differ')
    for name, pin in p['parent_assets_sha256'].items():
        require(sha(parent / name) == pin, 'Parent asset differs: ' + name)
    return p


def run(root, parent, expected_sha):
    sys.path.insert(0, str(parent))
    from cctv_dgp_face_code_fit_v12 import require_vm
    require_vm(parent)
    p = verify(root, parent, expected_sha)
    source = parent / 'outputs/cctv_dgp_face_code_fit_v12'
    require(sha(source / 'results.json') == RESULTS_SHA, 'Parent result differs')
    recorded = read(source / 'results.json')
    out = root / 'outputs/cctv_dgp_face_code_render_controls_v13'
    require(not out.exists(), 'Preserve partial/completed controls; no resume')
    out.mkdir(parents=True)
    import numpy as np
    from PIL import Image
    import torch
    from pretrained_face_restoration_portable_v12 import load_face_restorer
    from dgp_face_code_conditioner_v11 import DGPFaceCodePrior
    from cctv_dgp_pilot import state_hash, exported_pixel_metrics
    from face_prior_grid_v10 import render_grid

    require(torch.cuda.is_available(), 'CUDA unavailable')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    deadline = started + p['budget']['runtime_seconds']
    prior, provenance = load_face_restorer(parent / read(parent / 'face_code_fit_protocol_v12.json')['weights']['prior'], device='cuda')
    model = DGPFaceCodePrior(torch.nn.Identity(), prior.net).cuda().eval().requires_grad_(False)
    before = state_hash(prior.net)
    counts = {'generator': 0, 'encoder': 0, 'transformer_head': 0, 'conditioner': 0}
    handles = []
    for key, module in [('generator', prior.net.generator.blocks[0]), ('encoder', prior.net.encoder.blocks[0]),
                        ('transformer_head', prior.net.idx_pred_layer), ('conditioner', model.conditioner)]:
        def hook(_, __, ___, key=key): counts[key] += 1
        handles.append(module.register_forward_hook(hook))
    artifacts = {}

    def clock():
        torch.cuda.synchronize()
        require(time.monotonic() <= deadline, 'Control runtime budget exceeded')

    def source_array(name):
        path = source / name
        require(sha(path) == recorded['artifacts_sha256'][name], 'Saved parent probe differs: ' + name)
        return np.load(path, allow_pickle=False)

    def rgb(path):
        with Image.open(path) as im:
            require(im.mode == 'RGB' and im.size == (256,256), 'Invalid RGB256')
            return np.asarray(im).copy()

    def saved_array(name, value):
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        np.save(out / name, value, allow_pickle=False)
        artifacts[name] = sha(out / name)

    def codes(row):
        logits = torch.from_numpy(source_array(row['code_logits'])).cuda()
        return torch.topk(torch.softmax(logits, 2), 1, dim=2)[1].squeeze(-1)

    stages = {u: {(r['id'], r['fidelity']): r for r in read(source / f'update{u}/metrics.json')['rows']} for u in [0,300]}
    refs = {r['id']: r for r in p['references']}
    oracle = {}
    rows = []
    parity = []
    with torch.inference_mode():
        for ref in p['references']:
            clock()
            rid = ref['id']
            teacher = torch.from_numpy(source_array(f'teacher/{rid}_codes.npy')).cuda()
            oracle[rid] = model.decode(teacher, fidelity=0.0)[0].permute(1,2,0).cpu().numpy()
            saved_array(f'raw/teacher_no_stats/{rid}.npy', oracle[rid])
        for i, case in enumerate(p['cases']):
            clock()
            rid = case['reference_id']
            initial = stages[0][(case['id'], 0.0)]
            learned = stages[300][(case['id'], 0.0)]
            features = torch.from_numpy(source_array(initial['code_features'])).cuda()
            clean = torch.from_numpy(source_array(f'teacher/{rid}_codes.npy')).cuda()
            first, last = codes(initial), codes(learned)
            if i < 2:
                actual = model.decode(first, features, fidelity=0.0)[0].permute(1,2,0).cpu().numpy()
                maximum = float(np.abs(actual - source_array(initial['raw'])).max())
                require(maximum <= 2e-6, 'Cached original-stat w0 renderer parity differs')
                parity.append({'id': case['id'], 'maximum_float_difference': maximum})
            values = {
                'initial_no_stats': model.decode(first, fidelity=0.0)[0].permute(1,2,0).cpu().numpy(),
                'learned300_no_stats': model.decode(last, fidelity=0.0)[0].permute(1,2,0).cpu().numpy(),
                'teacher_observed_stats': model.decode(clean, features, fidelity=0.0)[0].permute(1,2,0).cpu().numpy(),
                'teacher_no_stats': oracle[rid],
            }
            target = rgb(parent / refs[rid]['target'])
            observed = np.asarray(Image.open(parent / refs[rid]['observed'])) > 0
            camera = rgb(parent / case['input'])
            for arm, raw in values.items():
                require(raw.dtype == np.float32 and np.isfinite(raw).all() and 0 <= raw.min() <= raw.max() <= 1, 'Invalid control render')
                name = f'raw/{arm}/{rid if arm == "teacher_no_stats" else case["id"]}.npy'
                if arm != 'teacher_no_stats': saved_array(name, raw)
                image = (raw * 255).astype(np.uint8)
                image[~observed] = camera[~observed]
                prediction = f'images/{arm}/{case["id"]}.png'
                (out / prediction).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(image).save(out / prediction)
                artifacts[prediction] = sha(out / prediction)
                rows.append({'id': case['id'], 'reference_id': rid, 'source': case['source'], 'profile': case['profile'],
                             'arm': arm, 'oracle': arm.startswith('teacher_'), 'raw': name, 'prediction': prediction,
                             **exported_pixel_metrics(image, target, observed)})
            if (i + 1) % 10 == 0: print(f'Render controls {i+1}/50 elapsed={time.monotonic()-started:.1f}s', flush=True)
    clock()
    after = state_hash(prior.net)
    require(before == after and counts == {'generator':162,'encoder':0,'transformer_head':0,'conditioner':0}, 'Frozen state/counts differ')
    # Save neural proof before postprocessing so a gallery failure cannot erase it.
    write(out / 'neural_receipt.json', {'complete':True, 'protocol_sha256':expected_sha, 'frozen_before':before,
          'frozen_after':after, 'counts':counts, 'parity':parity, 'backward_calls':0, 'optimizer_updates':0,
          'seconds':time.monotonic()-started, 'torch':torch.__version__, 'provenance':provenance,
          'peak_vram_bytes':torch.cuda.max_memory_allocated(), 'native_used':False, 'validation_used':False})
    lookup = {(r['id'],r['arm']):r for r in rows}
    grids = []
    for profile in p['profiles']:
        gallery = []
        for ref in p['references']:
            case = next(c for c in p['cases'] if c['reference_id']==ref['id'] and c['profile']==profile)
            images = [rgb(parent/case['input']), rgb(source/stages[300][(case['id'],0.0)]['prediction'])]
            images.extend(rgb(out/lookup[(case['id'],arm)]['prediction']) for arm in ARMS)
            images.append(rgb(parent/ref['target']))
            gallery.append({'id':case['id'],'images':images})
        name = f'grids/{profile}_10_rows.png'
        (out / name).parent.mkdir(parents=True,exist_ok=True)
        render_grid(['camera','learned300_original_stats','initial_no_stats','learned300_no_stats',
                     'ORACLE_teacher_observed_stats','ORACLE_teacher_no_stats','clean_training_target'], gallery).save(out/name)
        artifacts[name] = sha(out/name)
        grids.append(name)
    write(out/'results.json', {'complete':True, 'protocol_sha256':expected_sha,'rows':rows,'grids':grids,
          'artifacts_sha256':artifacts,'counts':counts,'backward_calls':0,'optimizer_updates':0,
          'training':False,'oracle_clean_target_access':True,'native_used':False,'validation_used':False,
          'native_reserved_used':False,'production_promoted':False,'seconds':time.monotonic()-started,
          'limitation':'Training-only saved-code renderer controls. Clean-label oracle arms use target information, not learned restoration. No identity metric or native usefulness is established.'})
    for handle in handles: handle.remove()
    archive = root/'cctv-dgp-face-code-controls-v13-results.tar.gz'
    require(not archive.exists(), 'Preserve prior control export')
    with tarfile.open(archive,'w:gz') as stream:
        files = ([root/PLAN_NAME,root/'render_controls_protocol_v13.sha256'] +
                 [root/name for name in p['sources_sha256']] + [f for f in out.rglob('*') if f.is_file()])
        for path in sorted(files): stream.add(path,arcname=path.relative_to(root).as_posix(),recursive=False)
    Path(str(archive)+'.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='ascii')
    completion = {'complete':True,'protocol_sha256':expected_sha,'results_sha256':sha(out/'results.json'),
                  'archive_sha256':sha(archive),'bytes':archive.stat().st_size,'counts':counts,
                  'backward_calls':0,'optimizer_updates':0,'seconds':time.monotonic()-started}
    write(root/'completion.json',completion)
    print(json.dumps(completion),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--root',type=Path,default=ROOT/'outputs/cctv_dgp_face_code_render_controls_vm_v13')
    parser.add_argument('--parent-bundle',type=Path)
    parser.add_argument('--expected-protocol-sha')
    args=parser.parse_args()
    if args.prepare: prepare(args.root.resolve())
    else:
        require(args.parent_bundle is not None and args.expected_protocol_sha is not None,'Require pinned parent and protocol')
        run(args.root.resolve(),args.parent_bundle.resolve(),args.expected_protocol_sha)
