"""Fixed CPU loss decomposition on saved V23 outputs; no head fitting/backward."""
import ast
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import audit_cctv_dgp_detail_skip_v23 as a


def main():
    import numpy as np
    import torch
    from torch.nn import functional as F
    started = time.monotonic(); torch.set_num_threads(4)
    bundle = a.BUNDLE; returned = ROOT / 'outputs/cctv_dgp_detail_skip_v23_return'
    out = ROOT / 'outputs/cctv_dgp_detail_skip_v23_loss_audit_v1'
    out.mkdir()
    p = a.verify_bundle(bundle)
    source = bundle / 'scripts/cctv_dgp_detail_skip_v23.py'
    plan = {
        'date': '2026-10-06', 'scope': 'Fixed CPU decomposition of original loss on saved exposed TRAINING outputs, not optimizer trajectory or gradient evidence',
        'hypothesis': 'Clear-case normalized detail improvements explain loss reduction while degraded landmark error fails; evaluate original terms without changing them',
        'protocol_sha256': a.PIN, 'source_sha256': a.sha(source), 'runner_sha256': a.sha(Path(__file__)),
        'independent_return_audit_sha256': a.sha(ROOT / 'outputs/cctv_dgp_detail_skip_v23_independent_audit.json'),
        'snapshots': [0, 50], 'cases': 50, 'original_head_forwards': 0, 'maximum_recognizer_forwards': 210,
        'seconds_cap': 300, 'compiled_loss_absolute_tolerance': 1e-6,
        'backward_calls': 0, 'optimizer_updates': 0, 'parameter_search': False,
        'VM_actions': False, 'app_promotion': False, 'goal_complete': False,
    }
    a.write(out / 'plan.json', plan)
    assert a.read(ROOT / 'outputs/cctv_dgp_detail_skip_v23_diagnostic/visual_review.json')['complete']
    sys.path.insert(0, str(bundle))
    from cctv_dgp_pilot import FixedObservedIdentity
    identity = FixedObservedIdentity(bundle / 'weights/w600k_r50.onnx', 'cpu')
    identity_before = a.state_hash(identity.state_dict())
    head = a.make_head(bundle); head_before = a.state_hash(head.state_dict())
    counts = {'recognizer_forwards': 0, 'saved_prediction_replays': 0}
    current = {}
    class SavedOutput:
        def __call__(self, x, b, mask):
            assert torch.equal(x, current['batch']['x']) and torch.equal(b, current['batch']['base']) and torch.equal(mask, current['batch']['mask'])
            counts['saved_prediction_replays'] += 1
            return current['pred']
        def high(self, value):
            return head.high(value)
    def batch(ids):
        assert ids == [0]
        return current['batch']
    namespace = {'torch': torch, 'F': F, 'head': SavedOutput(), 'identity': identity, 'batch': batch}
    tree = ast.parse(source.read_text(encoding='utf-8'))
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    nodes = [next(n for n in run.body if isinstance(n, ast.FunctionDef) and n.name == name)
             for name in ['mean', 'feature_errors', 'ssim', 'loss']]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pinned-original-V23-loss-only>', 'exec'), namespace)
    mean = namespace['mean']; feature_errors = namespace['feature_errors']; ssim = namespace['ssim']
    refs = {r['id']: r for r in p['references']}; truth_cache = {}; rows = []; maximum_difference = 0.
    source_bindings = {}
    def bind(path):
        source_bindings[path.relative_to(ROOT).as_posix()] = a.sha(path)
    for path in [source, bundle/'protocol.json', ROOT/'outputs/cctv_dgp_detail_skip_v23_independent_audit.json', bundle/'weights/w600k_r50.onnx']:
        bind(path)
    def tensor(value):
        return torch.from_numpy(np.asarray(value).copy()).permute(2, 0, 1)[None]
    with torch.inference_mode():
        for c in p['cases']:
            assert time.monotonic() - started <= 300
            camera, target8 = a.rgb(bundle/c['input']), a.rgb(bundle/c['target'])
            base = a.raw_rgb(bundle/c['raw_dgp']); mask8 = a.observed(bundle/c['observed'])
            prediction = a.raw_rgb(returned/'outputs/update50'/(c['id']+'.npy'))
            for path in [bundle/c[k] for k in ('input','target','observed','raw_dgp')]+[returned/'outputs/update50'/(c['id']+'.npy')]:
                bind(path)
            x, target, b = tensor(camera.astype(np.float32)/np.float32(255)), tensor(target8.astype(np.float32)/np.float32(255)), tensor(base)
            m = torch.from_numpy(mask8.astype(np.float32))[None,None]
            feature = torch.from_numpy(a.feature_mask(c,mask8).astype(np.float32))[None,None]
            interior = torch.from_numpy(a.interior(mask8,6).astype(np.float32))[None,None]
            valid7 = torch.from_numpy(a.interior(mask8,3).astype(np.float32))[None,None]
            grid = torch.from_numpy(a.fixed_grid(refs[c['source_person_or_reference']]['matrix112']))[None]
            reference = c['source_person_or_reference']
            if reference not in truth_cache:
                truth_cache[reference] = identity.embedding(target,m,grid); counts['recognizer_forwards'] += 1
            truth = truth_cache[reference]
            base_cosine = (identity.embedding(b*m+x*(1-m),m,grid)*truth).sum(1); counts['recognizer_forwards'] += 1
            base_feature, base_interior = feature_errors(b,target,feature,interior)
            base_pixel = mean((b-target).square(),m); base_ssim = ssim(b,target,valid7)
            batch_value = {'x':x,'base':b,'mask':m,'target':target,'feature':feature,'interior':interior,'valid7':valid7,
                           'grid':grid,'truth':truth,'base_cosine':base_cosine}
            saved = {'id':c['id'],'source':c['source'],'profile':c['profile'],'states':{}}
            for update, pred in [(0,b),(50,tensor(prediction))]:
                f,i = feature_errors(pred,target,feature,interior); pixel = mean((pred-target).square(),m)
                score = ssim(pred,target,valid7)
                # Baseline vector is reused exactly; the compiled reference performs its own fixed CPU call.
                if update==0: cosine=base_cosine
                else:
                    cosine=(identity.embedding(pred*m+x*(1-m),m,grid)*truth).sum(1); counts['recognizer_forwards'] += 1
                terms = {'landmark_detail':f/base_feature.clamp_min(1e-6), 'observed_detail':.25*i/base_interior.clamp_min(1e-6),
                         'pixel':.05*pixel/base_pixel.clamp_min(1e-5), 'pixel_regression':2*F.relu((pixel-base_pixel)/base_pixel.clamp_min(1e-5)),
                         'SSIM_regression':5*F.relu(base_ssim-score), 'ArcFace_regression':5*F.relu(base_cosine-cosine)}
                current['batch']=batch_value; current['pred']=pred
                compiled=float(namespace['loss']([0])); counts['recognizer_forwards'] += 1
                total=float(sum(terms.values()).mean());difference=abs(total-compiled)
                maximum_difference=max(maximum_difference,difference)
                assert difference<=1e-6
                if update==0:assert abs(total-1.3)<=1e-6
                saved['states'][str(update)]={'terms':{k:float(v.mean()) for k,v in terms.items()},'objective':total,
                    'compiled_objective':compiled,'raw_feature_MSE':float(f.mean()),'raw_interior_MSE':float(i.mean()),
                    'raw_pixel_MSE':float(pixel.mean()),'raw_SSIM':float(score.mean()),'raw_ArcFace':float(cosine.mean())}
            rows.append(saved)
    assert counts['recognizer_forwards']==210 and counts['saved_prediction_replays']==100
    assert a.state_hash(identity.state_dict())==identity_before and a.state_hash(head.state_dict())==head_before
    assert not any(v.grad is not None or v.requires_grad for v in list(identity.parameters())+list(head.parameters()))
    term_names=list(rows[0]['states']['0']['terms']); groups={}
    for group in ['all','clear','degraded']:
        selected=[r for r in rows if group=='all' or (r['profile']=='clear')==(group=='clear')]
        states={str(u):{'objective':float(np.mean([r['states'][str(u)]['objective'] for r in selected])),
                       'terms':{k:float(np.mean([r['states'][str(u)]['terms'][k] for r in selected])) for k in term_names}} for u in [0,50]}
        groups[group]={'cases':len(selected),'states':states,
            'objective_reduction':states['0']['objective']-states['50']['objective'],
            'contribution_to_equal50_case_objective_reduction':len(selected)/50*(states['0']['objective']-states['50']['objective']),
            'term_reductions':{k:states['0']['terms'][k]-states['50']['terms'][k] for k in term_names}}
    result={'complete':True,'plan_sha256':a.sha(out/'plan.json'),'runner_sha256':a.sha(Path(__file__)),
        'seconds':time.monotonic()-started,'rows':rows,'groups':groups,'counts':counts,'source_bindings_sha256':source_bindings,
        'maximum_compiled_loss_difference':maximum_difference,'frozen_recognizer_state':identity_before,
        'local_CPU_objective_only':True,'GPU_training_trajectory_or_gradient_causality_proven':False,
        'head_forwards':0,'backward_calls':0,'optimizer_updates':0,'VM_actions':False,'app_promotion':False,'goal_complete':False}
    a.write(out/'results.json',result)
    print(json.dumps({'complete':True,'groups':groups,'counts':counts,'seconds':result['seconds']},indent=2))


if __name__=='__main__':
    main()
