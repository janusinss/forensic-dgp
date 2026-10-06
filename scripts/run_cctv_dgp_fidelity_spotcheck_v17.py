"""Finite local inference control of camera-feature connections; no training.

Uses only audited R2 training-preview initial/final logits. This is a rendering
spotcheck, not fresh-image model verification, validation or a promoted DGP path.
"""
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs/cctv_dgp_broader_codes_vm_v16_r2'
PARENT = ROOT / 'outputs/cctv_dgp_face_code_fit_vm_v12_r2'
MIXED = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'
RETURN = ROOT / 'outputs/cctv_dgp_broader_codes_failure_return_v16_r2'
OUT = ROOT / 'outputs/cctv_dgp_fidelity_spotcheck_v17'
PIN = '4f64cf2204458f8fadcab1824fc4bc293c65803e123fdebb304f7b5bd3a502a0'
COUNTS = {'prior_encoder': 0, 'prior_classifier': 0, 'prior_generator': 0}


def run():
    sys.path.insert(0, str(PARENT)); sys.path.insert(0, str(BUNDLE))
    import cctv_dgp_broader_codes_v16 as v
    p = v.verify(BUNDLE, PARENT, MIXED,
                 ROOT / 'outputs/cctv_dgp_generalization_return_v15/outputs/generalization_v15', PIN)
    rdir = RETURN / 'outputs/broader_codes_v16_r2'
    r = v.read(rdir / 'results.json')
    audit = v.read(ROOT / 'outputs/cctv_dgp_broader_codes_v16_r2_audit_recovery_1/local_full_audit.json')
    v.require(audit['complete'] and audit['protocol_sha256'] == PIN
              and audit['results_sha256'] == v.sha(rdir / 'results.json'), 'Require audited R2 results')
    v.require(not OUT.exists(), 'Preserve prior/partial spotcheck; no repeat/resume')
    ids = p['train_preview_reference_ids'][:2] + p['train_preview_reference_ids'][5:7]
    refs = {item['id']: item for item in p['references']}
    profiles = ['clear', 'blur_lr24', 'compound_lr24']
    cases = [next(c for c in p['training_cases'] if c['reference_id'] == rid and c['profile'] == profile)
             for profile in profiles for rid in ids]
    v.require(len(cases) == 12 and len(set(ids)) == 4
              and all(refs[rid]['role'] == 'train' for rid in ids)
              and sorted(sum(refs[rid]['source'] == source for rid in ids)
                         for source in v.SOURCES) == [2, 2], 'Training-only balanced control differs')
    snapshots = {s['epoch']: {row['id']: row for row in v.read(rdir / s['metrics'])['rows']}
                 for s in r['snapshots'] if s['epoch'] in [0, 8]}
    OUT.mkdir()
    source = Path(__file__).read_bytes(); (OUT / 'runner.py').write_bytes(source)
    plan = {'format': 'v17-camera-feature-fidelity-training-spotcheck', 'date': '2026-10-05',
        'parent_protocol_sha256': PIN, 'parent_results_sha256': audit['results_sha256'],
        'runner_sha256': v.sha(OUT / 'runner.py'), 'case_ids': [c['id'] for c in cases],
        'reference_ids': ids, 'profiles': profiles, 'role': 'train', 'device': 'cpu',
        'arms': ['starting_codes_w1_none', 'own_epoch8_codes_w1_none'],
        'statistics': 'none', 'fidelity': 1.0, 'neural_cap_seconds': 600,
        'external_cap_seconds': 720, 'cross_device_w0_float_cap': 2e-5,
        'timing_sample_cases': 4, 'timing_steady_case_indices': [1, 3],
        'timing_safety_factor': 1.25, 'timing_reserve_seconds': 30,
        'teacher_used': False, 'validation_used': False, 'native_used': False,
        'native_reserved_used': False, 'checkpoint_selected': False, 'production_promoted': False,
        'limitation': 'Cached training-only code rendering; source folders are provenance, not ethnicity. New CPU float parity tolerance does not replace historical CUDA/fresh-image gates. No full model or app readiness claim.'}
    v.write(OUT / 'protocol.json', plan)
    import numpy as np
    from PIL import Image
    import torch
    from dgp_face_code_conditioner_v11 import DGPFaceCodePrior
    from pretrained_face_restoration_portable_v12 import load_face_restorer
    from cctv_dgp_pilot import state_hash
    from face_prior_grid_v10 import render_grid
    torch.set_num_threads(4); torch.manual_seed(20261005)
    started = time.monotonic(); deadline = started + 600
    pp = v.read(PARENT / 'face_code_fit_protocol_v12.json')
    prior, provenance = load_face_restorer(PARENT / pp['weights']['prior'], 'cpu')
    v.require(not prior.net.training and not any(x.requires_grad for x in prior.net.parameters()),
              'Inference-only prior must be frozen')
    before = state_hash(prior.net)
    # These existing helper methods depend only on the prior. No dummy DGP,
    # unused conditioner, teacher, new classifier or trainable model is created.
    context = SimpleNamespace(prior=prior.net)
    handles = []
    for name, module in [('prior_encoder', prior.net.encoder.blocks[0]),
                         ('prior_classifier', prior.net.idx_pred_layer),
                         ('prior_generator', prior.net.generator.blocks[0])]:
        def count(_a, _b, _c, key=name): COUNTS[key] += 1
        handles.append(module.register_forward_hook(count))
    artifacts = {}; rows = []; parity = []; durations = []

    def clock(): v.require(time.monotonic() <= deadline, 'Finite600s CPU inference cap exceeded')

    def save(name, value):
        path = v.safe(OUT, name); path.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith('.npy'):
            with path.open('xb') as stream: np.save(stream, value, allow_pickle=False)
        else:
            v.require(not path.exists(), 'No artifact overwrite'); Image.fromarray(value).save(path)
        artifacts[name] = v.sha(path); return name

    def bound(name):
        v.require(v.sha(v.safe(rdir, name)) == r['artifacts_sha256'][name], 'R2 probe changed:' + name)
        return v.safe(rdir, name)

    with torch.inference_mode():
        for i, case in enumerate(cases):
            clock(); step_start = time.monotonic(); cid = case['id']; ref = refs[case['reference_id']]
            camera = v.rgb(MIXED / case['input']); target = v.rgb(MIXED / ref['target'])
            with Image.open(MIXED / ref['observed']) as image: support = np.asarray(image) > 0
            image = torch.from_numpy(camera.copy()).permute(2, 0, 1).float()[None] / 255
            _, skips = DGPFaceCodePrior.encode(context, image)
            measured = {}
            for epoch, arm in [(0, 'starting_codes_w1_none'), (8, 'own_epoch8_codes_w1_none')]:
                logits = np.load(bound(snapshots[epoch][cid]['logits']), allow_pickle=False)
                v.require(logits.shape == (256, 1024) and logits.dtype == np.float32
                          and np.isfinite(logits).all(), 'Invalid cached logits')
                codes = torch.from_numpy(logits.argmax(1).astype(np.int64))[None]
                if epoch == 8 and i in [0, 2]:
                    raw0 = DGPFaceCodePrior.decode(context, codes, fidelity=0.).squeeze(0).permute(1, 2, 0).numpy().copy()
                    old = np.load(bound(snapshots[8][cid]['raw']), allow_pickle=False)
                    delta = float(np.max(np.abs(raw0 - old)))
                    item = {'id': cid, 'maximum_float_difference': delta,
                            'passed': delta <= plan['cross_device_w0_float_cap']}
                    parity.append(item)
                    v.write(OUT / ('parity_' + str(i) + '.json'), item)
                    v.require(item['passed'], 'CPU/CUDA w0 drift exceeds separate spotcheck tolerance')
                raw = DGPFaceCodePrior.decode(context, codes, skips=skips, fidelity=1.).squeeze(0).permute(1, 2, 0).numpy().copy()
                v.require(raw.shape == (256, 256, 3) and raw.dtype == np.float32
                          and np.isfinite(raw).all(), 'Invalid rendering')
                png = v.png(raw, camera, support)
                measured[arm] = {'raw': save(arm + '/' + cid + '.npy', raw),
                    'prediction': save(arm + '/' + cid + '.png', png), **v.metrics(png, target, support)}
                clock()
            rows.append({**{key: case[key] for key in ['id','reference_id','source','profile']},
                         'role': 'train', 'arms': measured})
            durations.append(time.monotonic() - step_start)
            if i == 3:
                projected = time.monotonic() - started + (len(cases) - 4) * float(np.mean([durations[1],durations[3]])) * 1.25 + 30
                v.write(OUT / 'timing.json', {'cases':4, 'seconds':time.monotonic()-started,
                    'projected_seconds':projected, 'cap_seconds':600})
                v.require(projected <= 600, 'CPU projection exceeds600s; preserve spotcheck failure')
            print({'case':i+1,'total':len(cases),'id':cid,'seconds':time.monotonic()-started},flush=True)

    after = state_hash(prior.net); v.require(before == after, 'Frozen renderer changed')
    v.require(COUNTS == {'prior_encoder':12,'prior_classifier':0,'prior_generator':26}, 'Neural counts differ')
    for handle in handles: handle.remove()
    lookup = {row['id']:row for row in rows}; grids = []
    for profile in profiles:
        cells = []
        for rid in ids:
            case = next(c for c in cases if c['reference_id'] == rid and c['profile'] == profile); cid = case['id']
            files = [MIXED/case['input'], bound(snapshots[0][cid]['dgp_preview']),
                     bound(snapshots[8][cid]['prediction'])] + [OUT/lookup[cid]['arms'][arm]['prediction']
                        for arm in plan['arms']] + [MIXED/refs[rid]['target']]
            cells.append({'id':cid,'images':[v.rgb(path) for path in files]})
        name = 'grids/' + profile + '_4_rows.png'; path = OUT/name; path.parent.mkdir(exist_ok=True)
        render_grid(['camera','retained DGP','our w0 none','starting w1 none','our w1 none','clean reference'],cells).save(path)
        artifacts[name] = v.sha(path); grids.append(name)
    clock()
    v.write(OUT/'results.json', {'complete':True,'protocol_sha256':v.sha(OUT/'protocol.json'),
        'prior_provenance':provenance,'torch':torch.__version__,'counts':COUNTS,
        'frozen_before':before,'frozen_after':after,'rows':rows,'parity':parity,'grids':grids,
        'artifacts_sha256':artifacts,'seconds':time.monotonic()-started,
        'teacher_used':False,'validation_used':False,'native_used':False,'native_reserved_used':False,
        'neural_dgp_forwards':0,'conditioner_forwards':0,'recognizer_forwards':0,
        'backward_calls':0,'optimizer_updates':0,'checkpoint_selected':False,'production_promoted':False,
        'limitation':plan['limitation']+' No recognizer metrics or held-out usefulness evaluation.'})
    print({'complete':True,'cases':12,'counts':COUNTS,'seconds':time.monotonic()-started},flush=True)


if __name__ == '__main__':
    try:
        run()
    except BaseException as error:
        if OUT.is_dir() and not (OUT/'failure.json').exists():
            with (OUT/'failure.json').open('x',encoding='utf-8') as stream:
                json.dump({'complete':False,'error_type':type(error).__name__,'error':str(error),
                           'counts':COUNTS,'backward_calls':0,'optimizer_updates':0,
                           'resume_permitted':False,'production_promoted':False},stream,indent=2)
        raise
