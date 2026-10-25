"""Frozen R2 clean-SFT-input intervention; local inference only, never training."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

# Reuse only inert hash/read helpers; importing this module does not import Torch.
import run_codeformer_hq_code_oracle_local_v1_r1 as prior

ROOT = prior.ROOT
INPUT = prior.INPUT
OUT = ROOT / 'outputs/codeformer_clean_skip_oracle_local_20261019_v1'
DESIGN = ROOT / 'outputs/codeformer_clean_skip_design_20261019_v1/design_frozen_v1.json'
SITES = {'32':256,'64':256,'128':128,'256':128}
need, safe, pin, new_json, read = prior.need, prior.safe, prior.pin, prior.new_json, prior.read
tensor_digest = prior.tensor_digest


def verify():
    import psutil
    need(psutil.__version__ == '7.0.0', 'Pinned installed resource guard required before spawn')
    need(pin(INPUT/'manifest_v1.json')['sha256']==prior.MANIFEST_SHA,'Exact18 frozen inputs required')
    manifest=read(INPUT/'manifest_v1.json'); review=read(INPUT/'root_input_review_v1_r1.json')
    need(review['complete'] and review['inference_only_diagnostic_admitted'] and not review['training_admitted'],
         'Existing input-only diagnostic review required')
    need(len(manifest['cases'])==18 and len(manifest['targets'])==6 and
         {r['id'] for r in manifest['cases']}=={r['id'] for r in review['cases']},'Exact reviewed case identities required')
    design=read(DESIGN)
    need(design['census']['total_top_level_model_calls']==42 and design['census']['optimizer_updates']==0,
         'Prospective42-call zero-update design required')
    evidence={}
    for path,digest in [(prior.R2,prior.R2_SHA),*[(ROOT/'third_party/codeformer'/n,s) for n,s in prior.ARCH.items()]]:
        value=pin(path);need(value['sha256']==digest,'Frozen model/architecture changed')
        evidence[str(path.relative_to(ROOT)).replace('\\','/')]=value
    for row in manifest['cases']:
        need(row['split']=='TRAIN','TRAIN observations only')
        for key in ('input','metrics_target256'):
            p=row[key];value=pin(ROOT/p['path'])
            need(value=={k:p[k] for k in ('bytes','sha256')},'Frozen pixels changed')
            evidence[p['path']]=value
    for path in (INPUT/'manifest_v1.json',INPUT/'root_input_review_v1_r1.json',DESIGN,Path(prior.__file__),Path(__file__)):
        evidence[str(path.relative_to(ROOT)).replace('\\','/')]=pin(path)
    return manifest,review,evidence


def stop_owned_tree(pid,created):
    import psutil
    try:parent=psutil.Process(pid)
    except psutil.NoSuchProcess:return []
    need(parent.create_time()==created,'Owned PID reused; refuse unrelated process')
    candidates=list(reversed(parent.children(recursive=True)))+[parent]; verified=[]
    for p in candidates:
        try:
            cmd=p.cmdline();paths=[Path(w).absolute() for w in cmd if w.endswith('.py')]
            need(Path(__file__).absolute() in paths and '--worker' in cmd,'Unrelated descendant; refuse kill')
            verified.append((p,p.create_time()))
        except psutil.NoSuchProcess:pass
    stopped=[]
    for p,stamp in verified:
        try:
            need(p.create_time()==stamp,'Owned descendant PID reused');p.kill();stopped.append(p.pid)
        except psutil.NoSuchProcess:pass
    _,alive=psutil.wait_procs([p for p,_ in verified],timeout=15)
    need(not alive,'Owned timed-out worker remains alive')
    return stopped


def worker():
    manifest,review,evidence=verify(); launch=read(OUT/'launch.json')
    import psutil
    need(launch['source_sha256']==pin(__file__)['sha256'] and
         launch['parent_pid'] in {p.pid for p in psutil.Process().parents()},'Exact live supervisor required')
    need(not (OUT/'worker_started.json').exists(),'Preserve earlier/partial worker evidence')
    new_json(OUT/'worker_started.json',{'pid':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat()})
    sys.path.insert(0,str(ROOT))
    import numpy as np
    from PIL import Image
    import torch
    from torch.nn import functional as F
    from third_party.codeformer.codeformer_arch import CodeFormer
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(20261019)
    torch.use_deterministic_algorithms(True)
    start=time.monotonic(); calls=0; files={};written=0; process=psutil.Process()
    resources={'maximum_RSS_bytes':0,'minimum_free_disk_bytes':shutil.disk_usage(ROOT).free}
    def check():
        rss=process.memory_info().rss;free=shutil.disk_usage(ROOT).free
        resources['maximum_RSS_bytes']=max(resources['maximum_RSS_bytes'],rss)
        resources['minimum_free_disk_bytes']=min(resources['minimum_free_disk_bytes'],free)
        need(time.monotonic()-start<900 and rss<=10*1024**3 and free>=4*1024**3,'900s/10GiB RSS/4GiB reserve stop')
    def emit(name,array):
        nonlocal written
        check();path=OUT/name;safe(path);path.parent.mkdir(parents=True,exist_ok=True)
        need(written+array.nbytes+4096<=1024**3,'OneGiB array/PNG payload cap')
        with path.open('xb') as f:
            if path.suffix=='.npy':np.save(f,array,allow_pickle=False)
            else:Image.fromarray(array).save(f,format='PNG')
        value=pin(path);files[name]=value;written+=value['bytes']
    def rgb(path):
        with Image.open(safe(ROOT/path)) as im:
            need(im.mode=='RGB' and im.size==(256,256),'Exact RGB256 required')
            return np.asarray(im).copy()
    def xvalue(row):
        a=rgb(row['input']['path']);x=torch.from_numpy(a.astype(np.float32)/np.float32(255)).permute(2,0,1)[None]
        return F.interpolate(x,(512,512),mode='bilinear',align_corners=False)*2-1
    def td(t):return tensor_digest({'value':t},torch)
    def result_pixels(prefix,native):
        need(native.shape==(1,3,512,512) and torch.isfinite(native).all().item(),'Finite native512 required')
        raw=F.interpolate(((native.float()+1)/2).clamp(0,1),(256,256),mode='bilinear',align_corners=False)[0].permute(1,2,0).contiguous().numpy().copy()
        png=np.floor(raw*np.float32(255)).astype(np.uint8)
        emit(prefix+'-raw.npy',raw);emit(prefix+'.png',png)
        return raw,png
    state=torch.load(prior.R2,weights_only=True,map_location='cpu',mmap=True)['params_ema']
    net=CodeFormer(dim_embd=512,codebook_size=1024,n_head=8,n_layers=9,connect_list=['32','64','128','256'])
    net.load_state_dict(state,strict=True);net.float().eval().requires_grad_(False)
    need(all(not module.training for module in net.modules()),'All frozen modules must remain in evaluation mode')
    before=tensor_digest(net.state_dict(),torch)
    need(before=='a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f','Exact R2 tensor state required')
    original_lookup=net.quantize.get_codebook_feat
    clean={};captures={}; rows=[]; all_site_events=[]
    def run(row,arm,prefix):
        nonlocal calls
        check();need(calls<42,'Exact42-call ceiling')
        handles=[];sites={}; latent=[]; codes=[]; features={}
        need('get_codebook_feat' not in net.quantize.__dict__,'Unexpected lookup override')
        def lookup(indices,shape):
            codes.append(indices.detach().clone());return original_lookup(indices,shape)
        def site_hook(site):
            def hook(module,args):
                need(len(args)==3 and args[2]==1.0 and site not in sites,'One exact SFT invocation/site')
                observed,decoder,w=args
                need(observed.shape==(1,SITES[site],int(site),int(site)),'SFT feature geometry changed')
                used=clean[row['reference_id']][site] if arm=='clean_skip_oracle' else observed
                need(used.shape==observed.shape and decoder.shape==observed.shape and
                     all(t.dtype==torch.float32 and t.device.type=='cpu' and torch.isfinite(t).all().item()
                         for t in (observed,used,decoder)), 'Finite FP32 CPU SFT fields required')
                sites[site]={'observed_encoder_sha256':td(observed),'used_encoder_sha256':td(used),
                    'decoder_input_sha256':td(decoder),'decoder_argument_passed_through':True}
                if arm=='capture':features[site]=observed.detach().clone()
                return (used,decoder,w) if arm=='clean_skip_oracle' else None
            return hook
        def journal(event):
            with (OUT/'operations.jsonl').open('a',encoding='utf8') as f:
                f.write(json.dumps({'event':event,'number':calls,'id':row['id'],'arm':arm})+'\n');f.flush();os.fsync(f.fileno())
        try:
            for site,block in net.fuse_convs_dict.items():handles.append(block.register_forward_pre_hook(site_hook(site)))
            handles.append(net.generator.blocks[0].register_forward_pre_hook(lambda module,args:latent.append(args[0].detach().clone())))
            net.quantize.get_codebook_feat=lookup
            calls+=1;journal('attempt')
            native,logits,lq=net(xvalue(row),w=1.,detach_16=True,adain=True)
            journal('return');check()
        finally:
            for handle in handles:handle.remove()
            if 'get_codebook_feat' in net.quantize.__dict__:del net.quantize.__dict__['get_codebook_feat']
        need(set(sites)==set(SITES) and len(latent)==len(codes)==1,'Four SFT calls, one code lookup and postAdaIN field required')
        need(logits.shape==(1,256,1024) and lq.shape==latent[0].shape==(1,256,16,16) and
             all(t.dtype==torch.float32 and torch.isfinite(t).all().item() for t in (logits,lq,latent[0])),
             'Finite FP32 upstream fields with exact geometry required')
        raw,png=result_pixels(prefix,native)
        for suffix,value in [('logits',logits),('lq',lq),('postAdaIN',latent[0]),('codes',codes[0].reshape(1,256))]:
            emit(prefix+'-'+suffix+'.npy',value.numpy().copy())
        upstream={k:td(t) for k,t in [('logits',logits),('lq',lq),('postAdaIN',latent[0]),('codes',codes[0].reshape(1,256))]}
        all_site_events.append({'id':row['id'],'arm':arm,'sites':sites})
        return raw,png,upstream,sites,features
    with torch.inference_mode():
        for rid in manifest['source_order']:
            row=next(r for r in manifest['cases'] if r['reference_id']==rid and r['profile']=='clear')
            raw,png,upstream,sites,features=run(row,'capture','capture/'+rid)
            clean[rid]=features;captures[rid]=(raw,png,upstream)
            for site,tensor in features.items():emit('capture/'+rid+'-encoder'+site+'.npy',tensor.numpy().copy())
        for row in manifest['cases']:
            target=rgb(row['metrics_target256']['path']).astype(np.float64)/255.;arms={};paired=None
            for arm in ('natural','clean_skip_oracle'):
                raw,png,upstream,sites,_=run(row,arm,'cases/'+row['id']+'/'+arm)
                if paired is None:paired=(upstream,sites)
                else:
                    need(upstream==paired[0],'Oracle changed logits/LQ/code selections/postAdaIN')
                    need(all(sites[s]['observed_encoder_sha256']==paired[1][s]['observed_encoder_sha256'] for s in SITES),
                         'Observed branch features changed upstream')
                    need(all(sites[s]['used_encoder_sha256']==td(clean[row['reference_id']][s]) for s in SITES),'Wrong clean source/site used')
                if row['profile']=='clear':
                    c=captures[row['reference_id']]
                    need(np.array_equal(raw,c[0]) and np.array_equal(png,c[1]) and upstream==c[2],
                         'Clear self-substitution/natural instrumentation parity failed')
                arms[arm]={'RAW_MAE':float(np.abs(raw.astype(np.float64)-target).mean()),
                    'PNG_MAE':float(np.abs(png.astype(np.float64)/255.-target).mean()),'upstream':upstream,'SFT_sites':sites}
            rows.append({'id':row['id'],'reference_id':row['reference_id'],'profile':row['profile'],'arms':arms})
            print(json.dumps({'case':len(rows),'of':18,'calls':calls,'seconds':time.monotonic()-start}),flush=True)
    after=tensor_digest(net.state_dict(),torch)
    need(before==after and calls==42 and all(p.grad is None and not p.requires_grad for p in net.parameters()) and
         all(not module.training for module in net.modules()),'State/modes/gradients/call census changed')
    need(all(not b._forward_pre_hooks and not b._forward_hooks for b in net.modules()) and 'get_codebook_feat' not in net.quantize.__dict__,
         'Instrumentation remained attached')
    need(all(pin(ROOT/name)==value for name,value in evidence.items()),'Input/source changed')
    new_json(OUT/'site_events.json',all_site_events);files['site_events.json']=pin(OUT/'site_events.json')
    new_json(OUT/'result_v1.json',{'complete':True,'UTC':datetime.now(timezone.utc).isoformat(),'source_pins':evidence,
        'model_state_before':before,'model_state_after':after,'neural_forward_attempts':calls,'neural_forward_returns':calls,
        'clear_capture_forwards':6,'paired_forwards':36,'SFT_invocations':168,'optimizer_updates':0,'gradient_queries':0,'backward_calls':0,
        'VM_calls':0,'device':'cpu','dtype':'FP32','files':files,'cases':rows,'resources':resources,'seconds':time.monotonic()-start,
        'payload_bytes':written,'operations_journal':pin(OUT/'operations.jsonl'),'clear_self_substitution_all6_exact':True,
        'upstream_all18_pairs_exact':True,'hooks_restored':True,'training_admitted':False,'model_qualified':False,
        'app_default_changed':False,'oracle_not_deployable':True,'FINAL_used':False,'native_CCTV_used':False})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',action='store_true');parser.add_argument('--run',action='store_true');parser.add_argument('--worker',action='store_true')
    args=parser.parse_args();need(sum((args.verify,args.run,args.worker))==1,'Choose one explicit operation')
    if args.worker:return worker()
    _,_,evidence=verify()
    if args.verify:
        print(json.dumps({'complete':True,'cases':18,'planned_model_calls':42,'actual_model_calls':0,'optimizer_updates':0}));return
    need(not OUT.exists(),'Preserve previous/partial run; no automatic retry')
    need(shutil.disk_usage(ROOT).free>=5*1024**3,'FiveGiB initial free required')
    OUT.mkdir(parents=True);new_json(OUT/'launch.json',{'parent_pid':os.getpid(),'source_sha256':pin(__file__)['sha256'],
        'source_pins':evidence,'worker_seconds':900,'external_seconds':960,'planned_model_calls':42})
    try:
        import psutil
        with (OUT/'worker.log').open('xb') as f:
            child=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'--worker'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            created=psutil.Process(child.pid).create_time();new_json(OUT/'child.json',{'pid':child.pid,'created':created})
            try:code=child.wait(timeout=960)
            except subprocess.TimeoutExpired:
                stopped=stop_owned_tree(child.pid,created);new_json(OUT/'timeout_stop.json',{'owned_pids_stopped':stopped})
                child.wait(timeout=30);raise TimeoutError('External960s stop; retain partial outputs')
        need(code==0,'Worker failed; retain worker.log and do not retry')
        need((OUT/'result_v1.json').is_file(),'Actual result required')
        print(json.dumps({'complete':True,'result':str(OUT/'result_v1.json'),'neural_calls':42,'optimizer_updates':0}),flush=True)
    except BaseException:
        new_json(OUT/'failure.json',{'complete':False,'traceback':traceback.format_exc(),'retry_allowed':False});raise


if __name__=='__main__':main()
