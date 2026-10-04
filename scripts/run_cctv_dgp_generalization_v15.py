"""L4 inference only: fresh-image capacity parity and fixed development validation."""
import argparse
from pathlib import Path
import sys
import time


def run(root, parent, capacity, expected_sha):
    sys.path.insert(0, str(parent)); sys.path.insert(0, str(root))
    from cctv_dgp_targets_v6 import require_vm
    require_vm(root)  # This finite GPU job must never fall back to local CPU.
    import cctv_dgp_generalization_v15 as v
    p = v.verify(root, parent, expected_sha)
    import numpy as np
    from PIL import Image
    import torch
    from dgp_direct_face_code_v14 import DGPDirectFaceCodePrior, render_codes
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from pretrained_face_restoration_portable_v12 import load_face_restorer
    from cctv_dgp_pilot import FixedObservedIdentity, grid112, state_hash
    from face_prior_grid_v10 import render_grid
    torch.set_num_threads(4); torch.manual_seed(v.DESIGN['seed'])
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    start = time.monotonic(); deadline = start + v.DESIGN['runner_cap_seconds']
    out = root / 'outputs/generalization_v15'
    v.require(not out.exists(), 'Preserve prior/partial V15; no resume')
    out.mkdir(parents=True)
    try:
        pp = v.read(parent / 'face_code_fit_protocol_v12.json')
        dgp, dp = load_frozen_dgp_restorer(parent / pp['weights']['dgp'],
            expected_sha256=pp['assets_sha256'][pp['weights']['dgp']], device='cuda')
        prior, cp = load_face_restorer(parent / pp['weights']['prior'], device='cuda')
        model = DGPDirectFaceCodePrior(dgp.net, prior.net).cuda().eval()
        model.conditioner.load_state_dict(torch.load(root / 'trained_conditioner.pth',
            map_location='cpu', weights_only=True), strict=True)
        identity = FixedObservedIdentity(parent / pp['weights']['arcface'], 'cuda')
        frozen = {'dgp': model.core.dgp, 'prior': model.core.prior,
                  'conditioner': model.conditioner, 'unused_v11': model.core.conditioner,
                  'recognizer': identity}
        v.require(all(not m.training and not any(x.requires_grad for x in m.parameters())
                      for m in frozen.values()), 'Only frozen inference permitted')
        before = {name: state_hash(m) for name, m in frozen.items()}
        v.require(before['conditioner'] == p['trained_head_state_hash'], 'Loaded head differs')
        counts = {}; handles = []
        for name, mod in [('dgp',model.core.dgp), ('prior_encoder',model.core.prior.encoder.blocks[0]),
                ('prior_transformer_head',model.core.prior.idx_pred_layer),
                ('direct_conditioner',model.conditioner), ('prior_generator',model.core.prior.generator.blocks[0]),
                ('unused_v11',model.core.conditioner), ('recognizer',identity.encoder)]:
            counts[name] = 0
            def count(_a, _b, _c, name=name): counts[name] += 1
            handles.append(mod.register_forward_hook(count))
        artifacts = {}; parity = []; rows = []; target_vec = {}
        def clock():
            torch.cuda.synchronize()
            v.require(time.monotonic() <= deadline, 'V15 runner exceeds1200 seconds')
            v.require(torch.cuda.max_memory_allocated() <= v.DESIGN['peak_vram_cap_bytes'], 'VRAM exceeds20GiB')
        def tensor(rgb):
            return torch.from_numpy(rgb.astype(np.float32) / 255).permute(2,0,1)[None].cuda()
        def save(name, value):
            path = v.safe(out, name); path.parent.mkdir(parents=True, exist_ok=True)
            if name.endswith('.npy'):
                with path.open('xb') as f: np.save(f, value, allow_pickle=False)
            else:
                v.require(not path.exists(), 'No image overwrite'); Image.fromarray(value).save(path)
            artifacts[name] = v.sha(path); return name
        def embedding(rgb, support, matrix):
            return identity.embedding(tensor(rgb), torch.from_numpy(support.astype(np.float32))[None,None].cuda(),
                torch.from_numpy(grid112(matrix))[None].cuda())[0].cpu().numpy().copy()
        def checked_capacity(name):
            path = v.safe(capacity, name)
            v.require(v.sha(path) == p['capacity_artifacts_sha256'][name], 'Capacity evidence differs:' + name)
            return path
        v.require(v.sha(capacity / 'results.json') == v.CAPACITY_PIN, 'V14 results differ')
        v.write(out / 'execution.json', {'complete':True,'protocol_sha256':expected_sha,
            'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'states_before':before,
            'provenance':{'dgp':dp,'prior':cp},'optimizer_constructed':False,
            'backward_calls':0,'optimizer_updates':0})
        with torch.inference_mode():
            # Prove this fresh-image path reproduces the cached capacity fit before validation.
            refs = {r['id']:r for r in p['parity_references']}
            for c in p['parity_cases']:
                clock(); camera = v.rgb(parent / c['input']); ref = refs[c['reference_id']]
                support = np.asarray(Image.open(parent / ref['observed'])) > 0
                dgprgb, features, logits = model.frozen_inputs(tensor(camera))
                pred = model.conditioner(tensor(camera), dgprgb, features, logits)
                codes = pred['logits'].argmax(2)
                raw = render_codes(model.core.prior, codes)[0].permute(1,2,0).cpu().numpy().copy()
                image = v.png(raw, camera, support)
                oldraw = np.load(checked_capacity('update1000/none/raw/' + c['id'] + '.npy'), allow_pickle=False)
                oldpng = v.rgb(checked_capacity('update1000/none/images/' + c['id'] + '.png'))
                delta = float(np.max(np.abs(raw-oldraw)))
                v.require(delta <= 2e-6 and np.array_equal(image, oldpng), 'Fresh/cached capacity parity failed')
                parity.append({'id':c['id'],'maximum_float_difference':delta,'png_equal':True,
                    'raw':save('parity/raw/' + c['id'] + '.npy',raw),
                    'prediction':save('parity/images/' + c['id'] + '.png',image)})
            v.write(out / 'fresh_image_parity.json', {'complete':True,'cases':parity}); artifacts['fresh_image_parity.json'] = v.sha(out/'fresh_image_parity.json')
            print('V15 fresh-image parity passed:50 training-cohort cases, zero updates',flush=True)
            refs = {r['id']:r for r in p['references']}
            for r in p['references']:
                clock(); support = np.asarray(Image.open(root / r['observed'])) > 0
                vec = embedding(v.rgb(root / r['target']),support,r['matrix112'])
                target_vec[r['id']] = vec; save('target_embeddings/' + r['id'] + '.npy',vec)
            eval_start = time.monotonic()
            for i,c in enumerate(p['cases'],1):
                clock(); cid = c['id']; ref = refs[c['reference_id']]
                camera = v.rgb(root / c['input']); target = v.rgb(root / ref['target'])
                support = np.asarray(Image.open(root / ref['observed'])) > 0
                x = tensor(camera); dgprgb, features, logits = model.frozen_inputs(x)
                pred = model.conditioner(x,dgprgb,features,logits)
                v.require(torch.isfinite(pred['logits']).all(), 'Nonfinite direct codes')
                raw = {'retained_dgp_v2': dgprgb,
                       'starting_prior_none': render_codes(model.core.prior,logits.argmax(2)),
                       'trained_conditioner_none': render_codes(model.core.prior,pred['logits'].argmax(2))}
                input_vec = embedding(camera,support,ref['matrix112'])
                row = {k:c[k] for k in ['id','reference_id','source','profile']}
                row['input_metrics'] = {**v.metrics(camera,target,support),
                    'ArcFace_observed_fixed':float(np.clip(input_vec @ target_vec[ref['id']],-1,1)),
                    'embedding':save('input_embeddings/' + cid + '.npy',input_vec)}
                row['arms'] = {}
                for arm in v.ARMS:
                    value = raw[arm][0].permute(1,2,0).cpu().numpy().copy(); image = v.png(value,camera,support)
                    vec = embedding(image,support,ref['matrix112'])
                    item = {**v.metrics(image,target,support),
                        'ArcFace_observed_fixed':float(np.clip(vec @ target_vec[ref['id']],-1,1)),
                        'prediction':save('predictions/' + arm + '/' + cid + '.png',image),
                        'embedding':save('embeddings/' + arm + '/' + cid + '.npy',vec)}
                    if ref['id'] in p['preview_reference_ids']:
                        item['raw'] = save('raw_previews/' + arm + '/' + cid + '.npy',value)
                    row['arms'][arm] = item
                rows.append(row)
                if i == v.DESIGN['timing_at_case']:
                    elapsed = time.monotonic()-eval_start
                    projected = time.monotonic()-start + elapsed/i*(520-i) + 60
                    v.write(out/'timing.json',{'cases':i,'seconds':elapsed,'projected_seconds':projected,'cap_seconds':1200})
                    v.require(projected <= 1200,'V15 projection exceeds finite cap')
                if i == 1 or i % 40 == 0: print(f'V15 validation{i}/520 elapsed={time.monotonic()-start:.1f}s',flush=True)
        lookup = {r['id']:r for r in rows}; grids = []
        for profile in v.PROFILES:
            cells = []
            for rid in p['preview_reference_ids']:
                c = next(c for c in p['cases'] if c['reference_id']==rid and c['profile']==profile)
                cells.append({'id':c['id'],'images':[v.rgb(root/c['input'])] +
                    [v.rgb(out/lookup[c['id']]['arms'][a]['prediction']) for a in v.ARMS] + [v.rgb(root/refs[rid]['target'])]})
            name = 'grids/' + profile + '_10_rows.png'; path = out/name; path.parent.mkdir(exist_ok=True)
            render_grid(['camera']+v.ARMS+['clean reference'],cells).save(path)
            artifacts[name]=v.sha(path); grids.append(name)
        clock(); after = {name:state_hash(m) for name,m in frozen.items()}
        expected = {'dgp':570,'prior_encoder':570,'prior_transformer_head':570,
                    'direct_conditioner':570,'prior_generator':1090,'unused_v11':0,'recognizer':2184}
        v.require(after==before and counts==expected,'Frozen state/count proof differs')
        summaries = {a:v.aggregate([{**{k:r[k] for k in ['id','source','profile']},**r['arms'][a]} for r in rows]) for a in v.ARMS}
        summaries['input'] = v.aggregate([{**{k:r[k] for k in ['id','source','profile']},**r['input_metrics']} for r in rows])
        v.write(out/'neural_execution_receipt.json',{'complete':True,'protocol_sha256':expected_sha,
            'states_before':before,'states_after':after,'counts':counts,'expected_counts':expected,
            'peak_vram_bytes':torch.cuda.max_memory_allocated(),'backward_calls':0,'optimizer_updates':0})
        artifacts['neural_execution_receipt.json']=v.sha(out/'neural_execution_receipt.json')
        v.verify(root,parent,expected_sha)
        v.write(out/'results.json',{'complete':True,'protocol_sha256':expected_sha,
            'rows':rows,'summaries':summaries,'guard_report':v.guard_report(summaries),
            'grids':grids,'artifacts_sha256':artifacts,'seconds':time.monotonic()-start,
            'optimizer_updates':0,'backward_calls':0,'validation_used':True,
            'native_used':False,'native_reserved_used':False,'production_promoted':False,
            'checkpoint_selected':False,'training':False,
            'limitation':'Own-training-disjoint development photography proxies; not blind final evaluation, unseen-by-prior proof or Zamboanga CCTV truth. Raw float verification covers the predeclared150 previews and50 parity renders; all1560 delivered PNGs/embeddings are audited.'})
        print({'complete':True,'seconds':time.monotonic()-start,'optimizer_updates':0},flush=True)
    except BaseException as error:
        v.write(out/'failure.json',{'complete':False,'error_type':type(error).__name__,'error':str(error),
            'optimizer_updates':0,'backward_calls':0,'resume_permitted':False})
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--capacity',type=Path,required=True);parser.add_argument('--expected-sha',required=True)
    args=parser.parse_args();run(args.root.resolve(),args.parent.resolve(),args.capacity.resolve(),args.expected_sha)
