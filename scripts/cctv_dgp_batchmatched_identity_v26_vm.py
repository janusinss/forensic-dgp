"""Finite manual-L4-only detail capacity pilot. No application promotion."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.request


def sha(path):
    with Path(path).open('rb') as f:
        digest = hashlib.sha256()
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
        return digest.hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify(root, pin):
    assert sha(root / 'protocol.json') == pin, 'Protocol changed'
    p = json.loads((root / 'protocol.json').read_text())
    assert p['format'] == 'dgp-spatial-batchmatched-identity-capacity-v26'
    assert len(p['cases']) == 50 and all(c['role'] == 'train' for c in p['cases'])
    for name, expected in p['assets_sha256'].items():
        path = (root / name).resolve()
        assert path.is_relative_to(root) and sha(path) == expected, name
    return p


def require_vm(root, idle=False):
    assert sys.platform == 'linux' and platform.node().split('.')[0] == 'forensic-dgp-thesis', 'Existing Linux VM only'
    assert root.is_relative_to((Path.home() / 'forensic-dgp').resolve()), '~/forensic-dgp only'
    request = urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/machine-type',
                                    headers={'Metadata-Flavor': 'Google'})
    with urllib.request.urlopen(request, timeout=3) as response:
        assert response.read().decode().rstrip().endswith('/g2-standard-4'), 'Existing g2-standard-4 only'
    import torch
    assert torch.cuda.is_available() and 'L4' in torch.cuda.get_device_name(0), 'NVIDIA L4 CUDA required'
    if idle:
        text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True, timeout=10)
        assert not [line for line in text.splitlines() if line.strip().isdigit() and int(line) != os.getpid()], 'Another GPU process exists; no task is stopped'


def prepare(root, p):
    require_vm(root, idle=True)
    from cctv_dgp_spatial_features_v25_prepare import prepare_features
    from cctv_dgp_batchmatched_identity_v26_preflight import prove
    head, identity, items, preflight = prepare_features(root, p, require_vm)
    preflight['batchmatched_identity_proof'] = prove(root, p, head, identity, items, require_vm)
    preflight['neural_forward_counts'] = dict(head.audit_forward_counts)
    return head, identity, items, preflight


def run(root, p, pin, preflight_only):
    require_vm(root, idle=True)
    if not preflight_only:assert not (root/'outputs').exists(), 'Preserve prior/partial run before neural work'
    start = time.monotonic()
    head, identity, items, preflight = prepare(root, p)
    signal.alarm(0)
    write(root / ('preflight_' + str(time.time_ns()) + '.json'), {'complete': True, 'protocol_sha256': pin, 'seconds': time.monotonic()-start, **preflight})
    if preflight_only:
        print(json.dumps(preflight)); return
    import cv2
    import numpy as np
    from PIL import Image
    from scipy.ndimage import convolve1d
    import torch
    from torch.nn import functional as F
    from cctv_dgp_pilot import exported_pixel_metrics, state_hash
    from cctv_dgp_batchmatched_identity_v26 import cohort_normalizers, objective_terms
    normalizers, loss_setup = cohort_normalizers(items, head)
    out = root / 'outputs'
    assert not out.exists(), 'Preserve prior/partial run; no resume or overwrite'
    out.mkdir()
    write(out/'cohort_loss_setup.json', loss_setup)
    cache = out / 'frozen_DGP_features'; cache.mkdir()
    feature_bindings = {}
    for item in items:
        for index, value in enumerate(item['fpn']):
            name = item['case']['id'] + '_fpn' + str(index) + '.npy'
            np.save(cache / name, value[0].detach().cpu().numpy().copy(), allow_pickle=False)
            feature_bindings[name] = sha(cache / name)
    write(out/'frozen_DGP_features.json', {'complete': True, 'cases': 50, 'feature_arrays': 250,
        'files_sha256': feature_bindings, 'CUDA_feature_rows': preflight['frozen_feature_cache_rows'],
        'DGP_state_unchanged': preflight['DGP_state_unchanged'], 'optimizer_constructed': False,
        'scope': 'Frozen inference-only own-DGP features from camera inputs, no target feature conditioning'})

    progress = {'updates': 0, 'backwards': 0}
    snapshots = []
    fit_start = None
    step_times = []
    optimizer_constructed = False
    def execution(terminal):
        now = time.monotonic()
        write(out/'execution_receipt.json', {'complete': True, 'protocol_sha256': pin, 'terminal': terminal,
            **progress, 'worker_seconds': now-start, 'fit_seconds': None if fit_start is None else now-fit_start,
            'fit_cap_seconds': 1500, 'worker_cap_seconds': 1800,
            'step_times_seconds': step_times, 'neural_forward_counts': dict(head.audit_forward_counts),
            'optimizer_constructed': optimizer_constructed, 'optimizer': 'AdamW new head only',
            'peak_allocated_VRAM_bytes': torch.cuda.max_memory_allocated(), 'VRAM_cap_bytes': 20*1024**3,
            'recognizer_state': state_hash(identity), 'initial_recognizer_state': preflight['recognizer_state'],
            'head_state': state_hash(head), 'app_promotion': False, 'goal_complete': False})
    deadline = start + 1800
    def clock():
        torch.cuda.synchronize()
        assert time.monotonic() < deadline, 'Total worker cap 1800 seconds'
        assert torch.cuda.max_memory_allocated() <= 20 * 1024**3, 'VRAM cap 20 GiB'
    def batch(ids):
        values = {key: torch.cat([items[i][key] for i in ids]) for key in ['x','base','mask','target','feature','interior','valid7','grid','truth','base_cosine','degraded_weight','clear_weight']}
        values['fpn'] = tuple(torch.cat([items[i]['fpn'][index] for i in ids]) for index in range(5))
        return values
    def mean(error, mask):
        return (error*mask).sum((1,2,3))/(mask.sum((1,2,3))*error.shape[1]).clamp_min(1)
    def feature_errors(value, target, feature, interior):
        weights = value.new_tensor([.299,.587,.114])[None,:,None,None]
        delta = head.high((value*weights).sum(1,keepdim=True)) - head.high((target*weights).sum(1,keepdim=True))
        return mean(delta.square(),feature),mean(delta.square(),interior)
    def ssim(value, target, valid):
        pool=lambda x:F.avg_pool2d(x,7,1,3)
        u,v=pool(value),pool(target)
        va,vb=(pool(value.square())-u.square())*(49/48),(pool(target.square())-v.square())*(49/48)
        cov=(pool(value*target)-u*v)*(49/48)
        score=((2*u*v+.01**2)*(2*cov+.03**2))/((u.square()+v.square()+.01**2)*(va+vb+.03**2))
        return mean(score,valid)
    def loss(ids):
        b=batch(ids); pred=head(b['x'],b['base'],b['mask'],b['fpn'])
        terms=objective_terms(b,pred,identity,mean,feature_errors,ssim,normalizers)
        result=sum(terms.values()).mean()
        assert torch.isfinite(result)
        return result
    positions=np.arange(-6,7,dtype=np.float64);kernel=np.exp(-.5*(positions/2)**2);kernel/=kernel.sum()
    def detail_metric(png, target8, support):
        y=(png.astype(np.float64)/255*np.array([.299,.587,.114])).sum(2)
        t=(target8.astype(np.float64)/255*np.array([.299,.587,.114])).sum(2)
        hp=lambda a:a-convolve1d(convolve1d(a,kernel,axis=0,mode='reflect'),kernel,axis=1,mode='reflect')
        return float(np.square(hp(y)-hp(t))[support].mean())
    def snapshot(update):
        clock();folder=out/('update'+str(update));folder.mkdir();rows=[]
        torch.save(head.state_dict(),folder/'head.pth')
        with torch.no_grad():
            for item in items:
                clock(); c=item['case'];raw=head(item['x'],item['base'],item['mask'],item['fpn'])[0].permute(1,2,0).cpu().numpy().copy()
                delivered=np.where(item['mask8'][...,None],np.floor(raw*np.float32(255)),item['camera']).astype(np.uint8)
                np.save(folder/(c['id']+'.npy'),raw,allow_pickle=False);Image.fromarray(delivered).save(folder/(c['id']+'.png'))
                prediction=torch.from_numpy(delivered.astype(np.float32)/np.float32(255)).cuda().permute(2,0,1)[None]
                vector=identity.embedding(prediction,item['mask'],item['grid'])[0].cpu().numpy().copy()
                truth=item['truth'][0].cpu().numpy().copy();np.save(folder/(c['id']+'_embedding.npy'),vector,allow_pickle=False)
                if update==0:np.save(folder/(c['id']+'_target_embedding.npy'),truth,allow_pickle=False)
                metric=exported_pixel_metrics(delivered,item['target8'],item['mask8'])
                metric['ArcFace_observed_fixed']=float(vector@truth)
                feature=item['feature'][0,0].cpu().numpy()>0
                metric['landmark_high_frequency_MSE']=detail_metric(delivered,item['target8'],feature)
                shift=(raw-item['base'][0].permute(1,2,0).cpu().numpy())[item['mask8']].astype(np.float64).mean(0)
                mean_only=np.clip(item['base'][0].permute(1,2,0).cpu().numpy().astype(np.float64)+shift,0,1).astype(np.float32)
                mean_png=np.where(item['mask8'][...,None],np.floor(mean_only*np.float32(255)),item['camera']).astype(np.uint8)
                metric['constant_mean_shift_only_MSE']=exported_pixel_metrics(mean_png,item['target8'],item['mask8'])['MSE']
                if update==0:assert np.array_equal(delivered,item['baseline8']), 'Initial delivered baseline changed'
                rows.append({'id':c['id'],'source':c['source'],'profile':c['profile'],'metrics':metric,'postclip_mean_RGB_shift':shift.tolist()})
        groups={}
        for key in sorted({'all','clear','degraded'}|{c['source']+'/'+k for c in p['cases'] for k in ['all','clear','degraded',c['profile']]}):
            selected=[r for r in rows if key in {'all','clear' if r['profile']=='clear' else 'degraded',r['source']+'/all',r['source']+('/clear' if r['profile']=='clear' else '/degraded'),r['source']+'/'+r['profile']}]
            assert selected
            groups[key]={'cases':len(selected),**{k:float(np.mean([r['metrics'][k] for r in selected])) for k in ['MSE','SSIM','ArcFace_observed_fixed','landmark_high_frequency_MSE','constant_mean_shift_only_MSE']}}
        receipt={'update':update,'seconds':time.monotonic()-start,'rows':rows,'groups':groups,'head_state':state_hash(head)}
        write(folder/'metrics.json',receipt);snapshots.append(receipt)
        print(json.dumps({'snapshot':update,'training_outputs':50,'feature_error':groups['degraded']['landmark_high_frequency_MSE']}),flush=True)
        return receipt
    try:
        baseline=snapshot(0)
        require_vm(root)
        head.train();first=loss(list(range(5)));first.backward();progress['backwards']+=1
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in head.parameters()), 'Finite one-batch head gradients required'
        assert sum(float(v.grad.double().square().sum()) for v in head.parameters())>0, 'Nonzero head gradient required'
        assert all(v.grad is None and not v.requires_grad for v in identity.parameters()) and state_hash(identity)==preflight['recognizer_state']
        direct_gradient = float(head.direct.weight.grad.double().square().sum())
        assert direct_gradient > 0, 'Nonzero full-resolution bypass gradient required'
        write(out/'one_batch_gradient_preflight.json', {'complete': True, 'backwards': 1,
            'optimizer_constructed': False, 'direct_gradient_sum_squares': direct_gradient,
            'per_tensor_gradient_sum_squares': {name: float(value.grad.double().square().sum()) for name,value in head.named_parameters()},
            'recognizer_has_no_gradients': all(value.grad is None for value in identity.parameters()),
            'neural_forward_counts': dict(head.audit_forward_counts)})
        head.zero_grad(set_to_none=True)
        require_vm(root)
        optimizer=torch.optim.AdamW(head.parameters(),lr=.0003,weight_decay=.01)
        schedule=json.loads((root/'schedule.json').read_text())['batches'];assert len(schedule)==800
        optimizer_constructed = True
        fit_start=time.monotonic()
        for update,ids in enumerate(schedule,1):
            clock();assert time.monotonic()-fit_start<1500,'Training cap 1500 seconds'
            require_vm(root) if update==1 else None
            step=time.monotonic();optimizer.zero_grad(set_to_none=True);objective=loss(ids)
            objective.backward();progress['backwards']+=1
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in head.parameters())
            if update == 2:
                projection_gradients = {str(index): float(layer.weight.grad.double().square().sum()) for index,layer in enumerate(head.projections)}
                assert all(value > 0 for value in projection_gradients.values()), 'Nonzero gradients through all five own-DGP feature projections required'
                write(out/'feature_path_gradient_update2.json', {'complete': True, 'update_before_optimizer': 2,
                    'per_projection_gradient_sum_squares': projection_gradients,
                    'frozen_feature_tensors_have_no_gradients': all(not value.requires_grad and value.grad is None for item in items for value in item['fpn']),
                    'DGP_state_unchanged': preflight['DGP_state_unchanged']})
            torch.nn.utils.clip_grad_norm_(head.parameters(),1);optimizer.step();progress['updates']=update
            torch.cuda.synchronize();step_times.append(time.monotonic()-step)
            if update==20:
                elapsed=time.monotonic()-fit_start;projected=elapsed+(800-20)*float(np.mean(step_times[1:]))*1.25+120
                write(out/'timing_update20.json',{'updates':20,'seconds':elapsed,'steady_sample_seconds':step_times[1:].copy(),'remaining_updates':780,'safety_factor':1.25,'overhead_seconds':120,'projected_seconds':projected,'cap_seconds':1500})
                assert projected<=1500,'Training timing projection exceeds cap'
            if update in [50,400,800]:
                current=snapshot(update)
                if update==50:
                    gain=1-current['groups']['degraded']['landmark_high_frequency_MSE']/baseline['groups']['degraded']['landmark_high_frequency_MSE']
                    write(out/'early_structure_stop.json',{'update':50,'relative_feature_error_gain':gain,'minimum':.01,'pass':gain>=.01})
                    assert gain>=.01,'No one-percent early structural gain; retain stop'
            if update==1 or update%50==0:print(f'V26 update {update}/800 objective={float(objective.detach()):.6f}',flush=True)
        assert time.monotonic()-fit_start <= 1500, 'Final fitting elapsed exceeds1500 seconds'
        head.eval();candidate=snapshots[-1];failures=[]
        for key,b in baseline['groups'].items():
            a=candidate['groups'][key];assert a['cases']==b['cases']
            for metric in ['MSE','SSIM','ArcFace_observed_fixed']:
                bad=a[metric]>b[metric]+1e-12 if metric=='MSE' else a[metric]<b[metric]-1e-6
                if bad:failures.append({'group':key,'metric':metric,'baseline':b[metric],'candidate':a[metric]})
        gain=1-candidate['groups']['degraded']['landmark_high_frequency_MSE']/baseline['groups']['degraded']['landmark_high_frequency_MSE']
        source_gains={src:1-candidate['groups'][src+'/degraded']['landmark_high_frequency_MSE']/baseline['groups'][src+'/degraded']['landmark_high_frequency_MSE'] for src in sorted({c['source'] for c in p['cases']})}
        b=baseline['groups']['degraded'];a=candidate['groups']['degraded']
        photometric_gain=max(0,b['MSE']-a['constant_mean_shift_only_MSE'])
        photometric_fraction=photometric_gain/max(b['MSE']-a['MSE'],1e-12)
        if photometric_fraction>.2:failures.append({'group':'degraded','metric':'brightness_gain_fraction','candidate':photometric_fraction,'maximum':.2})
        assert state_hash(identity)==preflight['recognizer_state']
        result={'complete':True,'protocol_sha256':pin,'seconds':time.monotonic()-start,**progress,'preservation_failures':failures,'degraded_feature_MSE_relative_gain':gain,'source_feature_gains':source_gains,'brightness_gain_fraction':photometric_fraction,'necessary_capacity_pass':not failures and gain>=.1 and all(v>=0 for v in source_gains.values()),'selection':'Final800 only; no checkpoint selected for application','recognizer_unchanged':True,'DGP_checkpoint_unchanged':sha(root/'weights/dgp_v2.pth')==p['assets_sha256']['weights/dgp_v2.pth'],'native_or_reserved_used':False,'independent_audit_pending':True,'visual_review_pending':True,'app_promotion':False,'goal_complete':False}
        execution('completed800')
        write(out/'results.json',result);print(json.dumps(result),flush=True)
    except BaseException as exc:
        write(out/'failure.json',{'complete':False,'protocol_sha256':pin,'seconds':time.monotonic()-start,**progress,'cause':str(exc),'traceback':traceback.format_exc(),'resume_permitted':False,'app_promotion':False})
        torch.save(head.state_dict(),out/'stopped_head.pth')
        execution('failed_partial')
        raise


def export(root, p, pin):
    assert sys.platform=='linux' and root.is_relative_to((Path.home()/'forensic-dgp').resolve())
    started=time.monotonic();home=Path.home();name='cctv-dgp-batchmatched-identity-v26-results.tar.gz';dest=home/name
    assert not dest.exists() and not (home/(name+'.sha256')).exists(), 'Preserve previous export'
    files=[root/'protocol.json',root/'schedule.json',root/'scripts/cctv_dgp_batchmatched_identity_v26_vm.py',root/'scripts/run_v26.sh']
    files+=sorted((root/'outputs').rglob('*')) if (root/'outputs').exists() else []
    files+=sorted(root.glob('preflight_*.json'))
    files+=sorted(root.glob('feature_preflight_*.json'))
    files+=sorted(root.glob('batchmatched_identity_preflight_*.json'))
    files+=[root/name for name in ['cctv_dgp_batchmatched_identity_v26.py', 'cctv_dgp_batchmatched_identity_v26_preflight.py', 'scripts/install_v26.py', 'installation_receipt.json']]

    files+=[root/name for name in ['cctv_dgp_spatial_features_v25.py','cctv_dgp_spatial_features_v25_prepare.py','cctv_dgp_degraded_objective_v24.py']]
    files+=[root/name for name in ['trainer.log','trainer_exit_code.txt','supervisor_receipt.json'] if (root/name).exists()]
    bindings={f.relative_to(root).as_posix():sha(f) for f in files if f.is_file()}
    write(root/'export_manifest.json',{'complete':True,'protocol_sha256':pin,'files_sha256':bindings,'scope':'Training capacity results only; failed gates retained; independent local audit required'})
    files.append(root/'export_manifest.json')
    with tarfile.open(dest,'x:gz',compresslevel=3) as archive:
        for f in files:
            assert time.monotonic()-started<120, 'Export cap120 seconds'
            if f.is_file():archive.add(f,arcname='cctv_dgp_batchmatched_identity_v26_return/'+f.relative_to(root).as_posix(),recursive=False)
    fingerprint=sha(dest)
    with (home/(name+'.sha256')).open('x',encoding='ascii',newline='\n') as f:f.write(fingerprint+'  '+name+'\n')
    receipt={'complete':True,'archive_sha256':fingerprint,'bytes':dest.stat().st_size,'seconds':time.monotonic()-started,'training_success_not_implied':True,'run_results_present':(root/'outputs/results.json').exists(),'failure_present':(root/'outputs/failure.json').exists()}
    write(home/'cctv-dgp-batchmatched-identity-v26-export.json',receipt);print(json.dumps(receipt))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path);parser.add_argument('--protocol-sha',required=True)
    modes=parser.add_mutually_exclusive_group(required=True)
    for mode in ['verify-transfer','preflight','run','export','record-supervision']:modes.add_argument('--'+mode,action='store_true')
    parser.add_argument('--supervisor-start',type=float);parser.add_argument('--supervisor-end',type=float);parser.add_argument('--trainer-exit',type=int)
    args=parser.parse_args();root=args.root.resolve();sys.path.insert(0,str(root));p=verify(root,args.protocol_sha)
    if args.verify_transfer:print(json.dumps({'complete':True,'assets':len(p['assets_sha256']),'cases':50,'neural_or_training_calls':0}));return
    if args.record_supervision:
        assert sys.platform=='linux' and root.is_relative_to((Path.home()/'forensic-dgp').resolve())
        import math
        assert all(value is not None and math.isfinite(value) for value in [args.supervisor_start,args.supervisor_end]) and args.supervisor_end>=args.supervisor_start and args.trainer_exit is not None
        seconds=args.supervisor_end-args.supervisor_start
        write(root/'supervisor_receipt.json', {'complete': True, 'protocol_sha256': args.protocol_sha,
            'seconds': seconds, 'cap_seconds': 2100, 'kill_grace_seconds': 30,
            'within_external_bound': seconds<=2130, 'trainer_exit_code': args.trainer_exit,
            'measurement': 'Parent shell samples Linux system monotonic clock before/after child pipeline; parent creates no CUDA context',
            'training_acceptance_not_implied': True})
        return
    if args.export:export(root,p,args.protocol_sha);return
    if sys.platform=='linux':
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Preflight cap300 seconds')))
        signal.alarm(300)
    if args.run:signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('External supervisor deadline')))
    run(root,p,args.protocol_sha,args.preflight)


if __name__=='__main__':main()
