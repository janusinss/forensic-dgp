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
    assert p['format'] == 'dgp-degraded-detail-cohort-capacity-v24'
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
    require_vm(root, idle=True)  # Before new neural/gradient/optimizer work.
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    from torch import nn
    from torch.nn import functional as F
    from cctv_dgp_pilot import FixedObservedIdentity, grid112, state_hash
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    torch.set_num_threads(4)
    torch.manual_seed(p['design']['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert shutil.disk_usage(root).free >= 3 * 1024**3, 'Need 3 GiB free; preserve old runs'

    class DetailHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.direct = nn.Conv2d(13, 3, 3, padding=1)
            self.stem = nn.Conv2d(13, 16, 3, padding=1)
            self.refine = nn.Conv2d(16, 16, 3, padding=1)
            self.tail = nn.Conv2d(16, 3, 1)
            nn.init.zeros_(self.direct.weight)
            nn.init.zeros_(self.direct.bias)
            nn.init.zeros_(self.tail.weight)
            nn.init.zeros_(self.tail.bias)
            z = torch.arange(-6, 7, dtype=torch.float32)
            k = torch.exp(-0.5 * (z / 2).square()); k = k / k.sum()
            self.register_buffer('kernel', k)
            self.register_buffer('reflect_indices', torch.cat((torch.arange(5, -1, -1), torch.arange(256), torch.arange(255, 249, -1))))

        def blur(self, value):
            c = value.shape[1]
            vertical = self.kernel.view(1, 1, 13, 1).expand(c, 1, 13, 1)
            horizontal = self.kernel.view(1, 1, 1, 13).expand(c, 1, 1, 13)
            value = F.conv2d(value.index_select(2, self.reflect_indices), vertical, groups=c)
            return F.conv2d(value.index_select(3, self.reflect_indices), horizontal, groups=c)

        def high(self, value):
            return value - self.blur(value)

        def forward(self, x, base, mask):
            features = torch.cat((x, base, self.high(x), self.high(base), mask), dim=1)
            shallow = F.silu(self.refine(F.silu(self.stem(features))))
            q = 0.05 * torch.tanh(self.direct(features) + self.tail(shallow))
            low = self.blur(q * mask) / self.blur(mask).clamp_min(1e-8)
            band = (q - low) * mask
            mean = band.sum((2, 3), keepdim=True) / mask.sum((2, 3), keepdim=True)
            band = band - mean * mask
            return (base + band).clamp(0, 1)

    def rgb(name):
        with Image.open(root / name) as im:
            a = np.asarray(im).copy()
        assert a.shape == (256, 256, 3) and a.dtype == np.uint8
        return a

    def tensor(a):
        return torch.from_numpy(np.asarray(a).copy()).cuda()

    refs = {r['id']: r for r in p['references']}
    items = []
    counts = {'detail_head': 0, 'DGP': 0, 'fixed_recognizer': 0}
    def counter(name):
        def hook(*_): counts[name] += 1
        return hook
    head = DetailHead().cuda().eval()
    head.register_forward_hook(counter('detail_head'))
    head.audit_forward_counts = counts
    assert sum(v.numel() for v in head.parameters()) == p['design']['trainable_parameters'] == 4613
    original_head_state = state_hash(head)
    for c in p['cases']:
        camera, target = rgb(c['input']), rgb(c['target'])
        with Image.open(root / c['observed']) as im:
            mask = np.asarray(im).copy() > 0
        raw = np.load(root / c['raw_dgp'], allow_pickle=False)
        assert raw.shape == camera.shape and raw.dtype == np.float32 and np.isfinite(raw).all()
        baseline = rgb(c['png_dgp'])
        assert np.array_equal(baseline, np.where(mask[..., None], np.floor(raw * np.float32(255)), camera).astype(np.uint8))
        feature = np.zeros((256, 256), bool)
        for point in c['landmarks5_canvas_xy']:
            x, y = np.floor(point).astype(int)
            feature[max(0, y-12):min(256, y+12), max(0, x-12):min(256, x+12)] = True
        interior = cv2.erode(mask.astype(np.uint8), np.ones((13, 13), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        valid7 = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        assert (feature & interior).any()
        items.append({'case': c, 'camera': camera, 'target8': target, 'mask8': mask,
            'baseline8': baseline, 'x': tensor(camera.astype(np.float32) / np.float32(255)).permute(2, 0, 1)[None],
            'target': tensor(target.astype(np.float32) / np.float32(255)).permute(2, 0, 1)[None],
            'base': tensor(raw).permute(2, 0, 1)[None], 'mask': tensor(mask.astype(np.float32))[None, None],
            'feature': tensor((feature & interior).astype(np.float32))[None, None],
            'interior': tensor(interior.astype(np.float32))[None, None], 'valid7': tensor(valid7.astype(np.float32))[None, None],
            'grid': tensor(grid112(refs[c['source_person_or_reference']]['matrix112']))[None]})
    with torch.no_grad():
        for item in items:
            assert torch.equal(head(item['x'], item['base'], item['mask']), item['base']), 'Initial head must be exact cached DGP'
    assert state_hash(head) == original_head_state
    dgp, provenance = load_frozen_dgp_restorer(root / 'weights/dgp_v2.pth', expected_sha256=p['assets_sha256']['weights/dgp_v2.pth'], device='cuda')
    dgp.net.register_forward_hook(counter('DGP'))
    before = state_hash(dgp.net)
    fresh_errors = []
    with torch.no_grad():
        for cid in p['fresh_DGP_parity_cases']:
            item = next(i for i in items if i['case']['id'] == cid)
            # Reproduce the declared historical CUDA-scalar convention for this
            # capacity cache. Canonical app normalization is a later gate.
            x = tensor(item['camera']).permute(2, 0, 1).float()[None] / 255
            actual = dgp.net(x)
            err = float((actual-item['base']).abs().max())
            fresh_errors.append({'id': cid, 'maximum_raw_error': err})
            assert err <= 2e-6, 'Retained DGP CUDA cache parity failed'
    assert state_hash(dgp.net) == before and not any(v.requires_grad for v in dgp.net.parameters())
    del dgp
    torch.cuda.empty_cache()
    identity = FixedObservedIdentity(root / 'weights/w600k_r50.onnx', 'cuda')
    identity.encoder.register_forward_hook(counter('fixed_recognizer'))
    identity_state = state_hash(identity)
    with torch.no_grad():
        for item in items:
            item['truth'] = identity.embedding(item['target'], item['mask'], item['grid'])
            item['base_cosine'] = (identity.embedding(item['base']*item['mask']+item['x']*(1-item['mask']), item['mask'], item['grid'])*item['truth']).sum(1)
    return head, identity, items, {'initial_head_state': original_head_state, 'recognizer_state': identity_state,
        'DGP_state_unchanged': before, 'DGP_provenance': provenance, 'fresh_DGP_parity': fresh_errors,
        'initial_exact_cached_DGP_cases': 50, 'trainable_parameters': sum(v.numel() for v in head.parameters()),
        'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'neural_forward_counts': dict(counts)}


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
    from cctv_dgp_degraded_objective_v24 import cohort_normalizers, objective_terms
    normalizers, loss_setup = cohort_normalizers(items, head)
    out = root / 'outputs'
    assert not out.exists(), 'Preserve prior/partial run; no resume or overwrite'
    out.mkdir()
    write(out/'cohort_loss_setup.json', loss_setup)
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
        return {key: torch.cat([items[i][key] for i in ids]) for key in ['x','base','mask','target','feature','interior','valid7','grid','truth','base_cosine','degraded_weight','clear_weight']}
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
        b=batch(ids); pred=head(b['x'],b['base'],b['mask'])
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
                clock(); c=item['case'];raw=head(item['x'],item['base'],item['mask'])[0].permute(1,2,0).cpu().numpy().copy()
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
            if update==1 or update%50==0:print(f'V24 update {update}/800 objective={float(objective.detach()):.6f}',flush=True)
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
    started=time.monotonic();home=Path.home();name='cctv-dgp-degraded-detail-v24-results.tar.gz';dest=home/name
    assert not dest.exists() and not (home/(name+'.sha256')).exists(), 'Preserve previous export'
    files=[root/'protocol.json',root/'schedule.json',root/'scripts/cctv_dgp_degraded_detail_v24.py',root/'scripts/run_v24.sh']
    files+=sorted((root/'outputs').rglob('*')) if (root/'outputs').exists() else []
    files+=sorted(root.glob('preflight_*.json'))
    files+=[root/name for name in ['trainer.log','trainer_exit_code.txt','supervisor_receipt.json'] if (root/name).exists()]
    bindings={f.relative_to(root).as_posix():sha(f) for f in files if f.is_file()}
    write(root/'export_manifest.json',{'complete':True,'protocol_sha256':pin,'files_sha256':bindings,'scope':'Training capacity results only; failed gates retained; independent local audit required'})
    files.append(root/'export_manifest.json')
    with tarfile.open(dest,'x:gz',compresslevel=3) as archive:
        for f in files:
            assert time.monotonic()-started<120, 'Export cap120 seconds'
            if f.is_file():archive.add(f,arcname='cctv_dgp_degraded_detail_v24_return/'+f.relative_to(root).as_posix(),recursive=False)
    fingerprint=sha(dest)
    with (home/(name+'.sha256')).open('x',encoding='ascii',newline='\n') as f:f.write(fingerprint+'  '+name+'\n')
    receipt={'complete':True,'archive_sha256':fingerprint,'bytes':dest.stat().st_size,'seconds':time.monotonic()-started,'training_success_not_implied':True,'run_results_present':(root/'outputs/results.json').exists(),'failure_present':(root/'outputs/failure.json').exists()}
    write(home/'cctv-dgp-degraded-detail-v24-export.json',receipt);print(json.dumps(receipt))


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
