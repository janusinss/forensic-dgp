"""Finite local TRAIN feature diagnosis: encoder prefix and adapters, zero updates.

Prepared only until independently reviewed. --preflight performs no neural call.
--run supervises one fresh --worker for at most900s; it never resumes/repeats.
This does not execute CodeFormer.forward, transformer, quantizer or decoder.
It cannot qualify face output quality, preservation, native CCTV or identity.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RETURN = ROOT/'outputs/codeformer_skip_feature_adapter_v1_return'
REVIEW = ROOT/'outputs/codeformer_skip_feature_adapter_return_review_v1'
OUT = ROOT/'outputs/codeformer_skip_encoder_adapter_role_diagnostic_v1'
REFS = ('tr_ffhq_03761','tr_ffhq_55791','tr_ffhq_60978','tr_ffhq_34217','tr_ffhq_31449','tr_ffhq_52422')
ROLES = ('clear','moderate','strong','dark','local_mosaic8','global_mosaic8')
SITES = ('32','64','128','256')
CHANNELS = {'32':256,'64':256,'128':128,'256':128}
TAPS = {5:'256',8:'128',11:'64',14:'32'}
POOL = 16
LIMIT = 512*1024**2
RESERVE = 4*1024**3
CAP_SECONDS = 900
RSS_CAP = 6*1024**3
MOMENTS = ('wanted_abs','final_residual_abs','wanted_sq','correction_sq',
           'final_residual_sq','correction_dot_wanted','wanted_spatial_sq',
           'correction_spatial_sq','spatial_dot','observed_sq','target_sq')
STATIC = {
 'research_skip_feature_restoration_v1.py':'e3783c5ac323829dcd2b1f935825cfc8725fc394163bacbd220a55f1c2c9415d',
 'research/codeformer_filter_pairs_v1/filters.py':'a806eaa0fe61f965e7a071fa6af084033af0ff55c84bb9a45e708baac1fa189c',
 'outputs/codeformer_skip_feature_adapter_v1_return/auxiliary_bundle.json':'2ac6205ec5c6e85c2c9ec80e3c18567a12bf2539860604094ea3c6a1ae8a975b',
 'outputs/codeformer_skip_feature_adapter_v1_return/data_bundle.json':'3b813c550ac3b2c16aa2ab88e858a97abe259d7567140c75a527ac464729c061',
 'outputs/codeformer_skip_feature_adapter_v1_return/outputs/skip_feature_adapter_v1/teacher/manifest.json':'9f96e22a84c142fdd4d4a81191f2c02e522c3fded196793b8763d52db034bfba',
 'outputs/codeformer_skip_feature_adapter_return_review_v1/state_v1/state_audit_v1.json':'3e9d4f684fbe6ff2fcebe212268406edec65b699749910c9ac6d4d5796481434',
 'outputs/codeformer_skip_feature_adapter_v1_model/model_export_v1.json':'4473b9841c0d72208ef81cb4f85df418d08fd2c7a88a345f610e2f3bd545517c'}


def need(value, message):
    if not value: raise ValueError(message)


def utc(): return datetime.now(timezone.utc).isoformat()


def safe(path, exists=True):
    path=Path(path).absolute()
    need(path.is_relative_to(ROOT), 'Workspace path required')
    for part in (path,*path.parents):
        if part.exists() or part.is_symlink():
            item=part.lstat()
            need(not stat.S_ISLNK(item.st_mode) and not getattr(item,'st_file_attributes',0)&1024,
                 'Linked/reparse paths refused')
    if exists: need(path.is_file(), 'Ordinary existing file required: '+str(path))
    return path


def pin(path):
    path=safe(path)
    with path.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'path':path.relative_to(ROOT).as_posix(),'bytes':path.stat().st_size,'sha256':digest}


def read(path):
    path=safe(path);need(path.stat().st_size<=8*1024**2,'Finite JSON cap')
    def unique(pairs):
        answer={}
        for key,value in pairs:
            need(key not in answer,'Duplicate JSON key');answer[key]=value
        return answer
    return json.loads(path.read_bytes(),object_pairs_hook=unique,
        parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def used_bytes():
    return sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())


def room(incoming=0):
    need(shutil.disk_usage(ROOT/'outputs').free>=RESERVE+incoming,'Preserve4GiB free reserve')
    need(used_bytes()+incoming<=LIMIT,'512MiB diagnostic output cap')


def new_bytes(path, data):
    path=safe(path,False);need(path.is_relative_to(OUT),'Diagnostic output path required')
    room(len(data));path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
    return pin(path)


def new_json(path, value):
    return new_bytes(path,(json.dumps(value,indent=2,allow_nan=False)+'\n').encode('utf8'))


def array(path, value, np):
    need(value.dtype==np.float64 and np.isfinite(value).all(),'Finite float64 audit arrays required')
    buffer=io.BytesIO();np.save(buffer,np.ascontiguousarray(value),allow_pickle=False)
    return new_bytes(path,buffer.getvalue())


def header(path, np):
    with safe(path).open('rb') as stream:
        version=np.lib.format.read_magic(stream);need(version in ((1,0),(2,0)),'NPY1/2 required')
        parser=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
        shape,fortran,dtype=parser(stream);offset=stream.tell()
    need(not fortran and dtype==np.float32 and not dtype.hasobject and
         path.stat().st_size==offset+math.prod(shape)*4,'Exact contiguous FP32 cache required')
    return list(shape)


def bound_static():
    need('torch' not in sys.modules,'Preflight must not import Torch')
    pins={}
    for name,digest in STATIC.items():
        item=pin(ROOT/name);need(item['sha256']==digest,'Frozen source/proof changed: '+name);pins[name]=item
    import research_skip_feature_restoration_v1 as app
    model,export=app.skip_file()
    for proof in app.REVIEW_BINDINGS.values():pins[proof['path']]=pin(ROOT/proof['path'])
    pins[model.relative_to(ROOT).as_posix()]=pin(model)
    for name,digest in app.base.ARCHITECTURE_PINS.items():
        path=ROOT/'third_party/codeformer'/name;item=pin(path)
        need(item['sha256']==digest,'Official architecture changed');pins[item['path']]=item
    state=read(REVIEW/'state_v1/state_audit_v1.json')
    need(state['audit_pass'] is True and [r['updates'] for r in state['checkpoints']]==[0,55] and
         all(r['all515_frozen_base_tensors_exact'] and r['all24_adapter_duplicates_exact'] for r in state['checkpoints']),
         'Independent identity-initializer/frozen-base audit required')
    need(state['checkpoints'][0]['adapter24_canonical_sha256']=='20398c0179f55436cc03333ed439c3f61e9a33e026274af548423e2753fd6b4b',
         'Exact independently checked zero-final-projection initializer required')
    need(export['preservation_pass'] is False and export['R2_dark_retention_pass'] is False,'Quality failure must remain')
    need('torch' not in sys.modules,'Inert proof check imported Torch')
    return app,pins


def preflight():
    need(not OUT.exists(),'Preserve earlier/partial diagnosis; do not restart')
    need(shutil.disk_usage(ROOT/'outputs').free>=RESERVE+LIMIT,'512MiB plus4GiB reserve required')
    app,pins=bound_static()
    import numpy as np
    from PIL import Image
    from research.codeformer_filter_pairs_v1 import filters
    import psutil
    need(psutil.__version__=='7.0.0','Existing pinned psutil7.0.0 required')
    need(psutil.virtual_memory().available>=8*1024**3,'At least8GiB available host RAM required')
    auxiliary=read(RETURN/'auxiliary_bundle.json');data=read(RETURN/'data_bundle.json')['case_by_id']
    teachers=read(RETURN/'outputs/skip_feature_adapter_v1/teacher/manifest.json')
    need(tuple(dict.fromkeys(r['reference_id'] for r in auxiliary['cases']))==REFS,'Same six predeclared TRAIN references required')
    need(auxiliary['no_FINAL_or_native_inputs'] is True and auxiliary['not_independent_holdout'] is True,'TRAIN-only diagnostic boundary')
    OUT.mkdir();new_json(OUT/'preflight_started.json',{'UTC':utc(),'source':pin(__file__),'neural_calls':0})
    cases=[];features={}
    for rid in REFS:
        aux={r['profile']:r for r in auxiliary['cases'] if r['reference_id']==rid}
        selected={'clear':aux['clear'],'moderate':aux['moderate_existing_gaussian_2p4'],
                  'strong':aux['strong_fixed_blur3p2_lr64_jpeg50'],
                  'dark':next(r for r in data.values() if r['reference_id']==rid and r['active_role']=='dark')}
        clear=selected['clear']
        need(clear['input']==teachers['features'][rid]['clear_input'],'Cached clear teacher input changed')
        clear_pin=pin(RETURN/clear['input']['path'])
        need(all(clear_pin[k]==clear['input'][k] for k in ('bytes','sha256')),'Source clear pixels changed')
        with Image.open(RETURN/clear['input']['path']) as image:clear_rgb=np.asarray(image.convert('RGB')).copy()
        need(clear_rgb.shape==(256,256,3),'Exact prepared RGB256 required')
        for role in ROLES:
            if role in selected:
                row=selected[role];need(row['split']=='TRAIN','TRAIN input required');path=RETURN/row['input']['path']
                item=pin(path);need(all(item[k]==row['input'][k] for k in ('bytes','sha256')),'Exact input asset differs')
                origin={'kind':'retained prepared paired TRAIN input','id':row['id'],'profile':row['profile'],
                        'input_only_decision':row.get('input_only_decision'),'source_manifest':'auxiliary_bundle.json' if role!='dark' else 'data_bundle.json'}
            else:
                profile_id='local_mosaic_8' if role=='local_mosaic8' else 'global_mosaic_8'
                profile=next(p for p in filters.PROFILES if p['id']==profile_id)
                pixels,footprint=filters.apply_profile(clear_rgb,profile)
                buffer=io.BytesIO();Image.fromarray(pixels).save(buffer,format='PNG')
                path=OUT/'generated_inputs'/rid/(role+'.png');item=new_bytes(path,buffer.getvalue())
                origin={'kind':'new paired TRAIN diagnostic corruption; no training admission','profile':profile,
                        'source_clear_input':clear['input'],'pure_filter_source':STATIC['research/codeformer_filter_pairs_v1/filters.py'],
                        'local_rectangle_xyxy_half_open':[64,64,192,192] if role=='local_mosaic8' else None,
                        'corruption_footprint_pixels':int((footprint>0).sum()),'input_only_decision':'not newly adjudicated; diagnostic only'}
            with Image.open(path) as image:rgb=np.asarray(image.convert('RGB')).copy()
            need(rgb.shape==(256,256,3) and rgb.dtype==np.uint8,'Fixed RGB256 input required')
            pins[item['path']]=item
            cases.append({'reference_id':rid,'role':role,'split':'TRAIN','input':item,
                          'RGB8_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'origin':origin})
        target=teachers['features'][rid];features[rid]={'features':{},'scales':target['scales']}
        scale_path=RETURN/'outputs/skip_feature_adapter_v1/teacher/features'/rid/'scales.json'
        scale=read(scale_path);item=pin(scale_path);pins[item['path']]=item
        need(scale['scales']==target['scales'],'Fixed clean RMS scales changed')
        for site,fp in target['features'].items():
            path=RETURN/fp['path'];item=pin(path);pins[item['path']]=item
            need(all(item[k]==fp[k] for k in ('bytes','sha256')) and header(path,np)==[1,CHANNELS[site],int(site),int(site)],
                 'Exact cached teacher feature/hash/header required')
            features[rid]['features'][site]=item
    need(len(cases)==36 and len({(c['reference_id'],c['role']) for c in cases})==36,'Exactly36 fixed role inputs')
    predicted=36*sum(3*CHANNELS[s]*POOL*POOL*8+len(MOMENTS)*int(s)**2*8+3*CHANNELS[s]*8 for s in SITES)
    need(predicted<470*1024**2,'Prospective feature-output budget exceeded')
    pins[Path(__file__).relative_to(ROOT).as_posix()]=pin(__file__)
    need('torch' not in sys.modules,'Preflight imported Torch')
    result={'complete':True,'UTC':utc(),'source':pin(__file__),'cases':cases,'clean_features':features,'pins':pins,
        'planned_encoder_prefix_passes':36,'planned_site_adapter_evaluations':144,'planned_optimizer_updates':0,
        'planned_full_CodeFormer_transformer_quantizer_decoder_calls':0,'initial_is_identity_backed_by_state_audit':True,
        'encoder_taps':{str(k):v for k,v in TAPS.items()},'stop_encoder_after_block':14,'pool':POOL,'moment_channels':list(MOMENTS),
        'predicted_audit_array_payload_bytes':predicted,'output_cap_bytes':LIMIT,'free_reserve_bytes':RESERVE,
        'outer_cap_seconds':CAP_SECONDS,'worker_RSS_cap_bytes':RSS_CAP,'actual_neural_calls':0,
        'changelog':'Prepared24case/four-role design expanded before execution to36cases/sixroles by adding exact prior local/global8-cell mosaics. No24case diagnosis ran.',
        'limits':'Paired TRAIN feature diagnosis only. Full face RAW/PNG quality/preservation remains failed; no FINAL/native/app promotion.'}
    new_json(OUT/'preflight.json',result)
    print(json.dumps({'complete':True,'preflight':pin(OUT/'preflight.json'),'cases':36,'neural_calls':0,'predicted_payload_bytes':predicted}),flush=True)


def verify_preflight():
    report=read(OUT/'preflight.json');need(report['complete'] is True,'Completed preflight required')
    for name,expected in report['pins'].items():need(pin(ROOT/name)==expected,'Preflight proof/input changed: '+name)
    need(report['source']==pin(__file__) and report['planned_encoder_prefix_passes']==36 and
         report['planned_site_adapter_evaluations']==144,'Reviewed source/finite plan changed')
    return report


def cosine(dot, a2, b2):
    return float(dot/math.sqrt(a2*b2)) if a2>0 and b2>0 else None


def summarize(moments, means, scale, np):
    # Each moment map is the float64 mean over channels at each original pixel.
    m=dict(zip(MOMENTS,(float(v) for v in moments.mean(axis=(1,2)))))
    wanted_dc=float(np.square(means[1]).mean());correction_dc=float(np.square(means[2]).mean())
    dc_dot=float((means[1]*means[2]).mean())
    initial=m['wanted_abs']/scale;final=m['final_residual_abs']/scale
    return {'initial_normalized_L1':initial,'update55_normalized_L1':final,
        'relative_L1_gain':1-final/initial if initial>0 else None,
        'correction_RMS_over_fixed_clean_RMS':math.sqrt(m['correction_sq'])/scale,
        'wanted_RMS_over_fixed_clean_RMS':math.sqrt(m['wanted_sq'])/scale,
        'final_residual_RMS_over_fixed_clean_RMS':math.sqrt(m['final_residual_sq'])/scale,
        'correction_DC_energy':correction_dc,'correction_spatial_energy':m['correction_spatial_sq'],
        'wanted_DC_energy':wanted_dc,'wanted_spatial_energy':m['wanted_spatial_sq'],
        'correction_DC_energy_fraction':correction_dc/m['correction_sq'] if m['correction_sq']>0 else None,
        'direction_alignment':cosine(m['correction_dot_wanted'],m['correction_sq'],m['wanted_sq']),
        'DC_direction_alignment':cosine(dc_dot,correction_dc,wanted_dc),
        'spatial_direction_alignment':cosine(m['spatial_dot'],m['correction_spatial_sq'],m['wanted_spatial_sq']),
        'observed_RMS_over_fixed_clean_RMS':math.sqrt(m['observed_sq'])/scale,
        'target_RMS_over_fixed_clean_RMS':math.sqrt(m['target_sq'])/scale}


def measure(observed, used, target, scale, np):
    x=observed.astype(np.float64);t=target.astype(np.float64)
    wanted=t-x;correction=used.astype(np.float64)-x;residual=correction-wanted
    means=np.stack((x.mean(axis=(1,2)),wanted.mean(axis=(1,2)),correction.mean(axis=(1,2))))
    wa=wanted-means[1,:,None,None];ca=correction-means[2,:,None,None]
    moments=np.stack((np.abs(wanted).mean(0),np.abs(residual).mean(0),np.square(wanted).mean(0),
        np.square(correction).mean(0),np.square(residual).mean(0),(correction*wanted).mean(0),
        np.square(wa).mean(0),np.square(ca).mean(0),(ca*wa).mean(0),np.square(x).mean(0),np.square(t).mean(0)))
    def pool(v):
        c,h,w=v.shape;need(h%POOL==w%POOL==0,'Exact uniform16x16 pooling required')
        return v.reshape(c,POOL,h//POOL,POOL,w//POOL).mean(axis=(2,4))
    pooled=np.stack((pool(x),pool(wanted),pool(correction)))
    m=summarize(moments,means,scale,np)
    need(abs(float(moments[3].mean())-m['correction_DC_energy']-m['correction_spatial_energy'])<1e-9*max(1,float(moments[3].mean())),
         'Correction DC/spatial energy decomposition differs')
    m['maximum_absolute_wanted_correction']=float(np.abs(wanted).max())
    return pooled,moments,means,m


def worker():
    pre=verify_preflight();launch=read(OUT/'launch.json')
    import psutil
    need(launch['parent_pid'] in {p.pid for p in psutil.Process().parents()},'Exact live supervisor required')
    need(launch['source']==pin(__file__),'Supervisor source changed')
    need(not (OUT/'worker_started.json').exists(),'Preserve previous partial worker')
    new_json(OUT/'worker_started.json',{'UTC':utc(),'pid':os.getpid(),'source':pin(__file__)})
    import numpy as np
    from PIL import Image
    import torch
    import research_skip_feature_restoration_v1 as app
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    torch.use_deterministic_algorithms(True)
    net,provenance=app.engine.load('skip55',torch)
    need(all(not p.requires_grad and p.grad is None for p in net.parameters()) and not net.training,'Frozen eval only')
    need(net.base.fuse_encoder_block=={'512':2,'256':5,'128':8,'64':11,'32':14,'16':18},'Exact official encoder tap map')
    before=app.base.state_sha(net.state_dict(),torch)
    counters={'encoder_prefix_attempts':0,'encoder_prefix_returns':0,'adapter_attempts':0,'adapter_returns':0,
              'full_CodeFormer_transformer_quantizer_decoder_calls':0,'optimizer_gradient_backward_calls':0}
    results=[];outputs=[];started=time.monotonic()
    def progress(stage,**details):
        need(time.monotonic()-started<CAP_SECONDS,'Worker elapsed cap')
        row={'UTC':utc(),'stage':stage,'counters':dict(counters),**details}
        path=OUT/'progress.jsonl';data=(json.dumps(row,allow_nan=False)+'\n').encode()
        room(len(data))
        with path.open('ab') as f:f.write(data);f.flush();os.fsync(f.fileno())
    for case in pre['cases']:
        with Image.open(ROOT/case['input']['path']) as image:rgb=np.asarray(image.convert('RGB')).copy()
        need(hashlib.sha256(rgb.tobytes()).hexdigest()==case['RGB8_sha256'],'Actual RGB input changed')
        x=torch.from_numpy(rgb.astype(np.float32)/np.float32(255)).permute(2,0,1)[None]
        value=torch.nn.functional.interpolate(x,(512,512),mode='bilinear',align_corners=False)*2-1
        rid,role=case['reference_id'],case['role'];seen=[]
        counters['encoder_prefix_attempts']+=1;progress('encoder_prefix_attempt',reference_id=rid,role=role)
        with torch.no_grad(),torch.autocast(device_type='cpu',enabled=False):
            for index,block in enumerate(net.base.encoder.blocks):
                need(index<=14,'No16x16 encoder/transformer/decoder execution')
                value=block(value)
                if index in TAPS:
                    site=TAPS[index];seen.append(site);observed=value.detach().clone()
                    need(observed.shape==(1,CHANNELS[site],int(site),int(site)) and observed.dtype==torch.float32 and
                         torch.isfinite(observed).all().item(),'Exact finite detached prefix tap required')
                    counters['adapter_attempts']+=1;progress('adapter_attempt',reference_id=rid,role=role,site=site)
                    used=net.adapters[site](observed);counters['adapter_returns']+=1
                    need(used.shape==observed.shape and torch.isfinite(used).all().item(),'Finite exact adapter geometry')
                    fp=pre['clean_features'][rid]['features'][site];teacher=np.load(ROOT/fp['path'],mmap_mode='r',allow_pickle=False)
                    scale=pre['clean_features'][rid]['scales'][site]
                    actual_RMS=float(np.sqrt(np.square(teacher.astype(np.float64)).mean()))
                    need(abs(actual_RMS-scale)<=1e-9*max(1,scale),'Fixed cached RMS changed')
                    pooled,moments,means,metrics=measure(observed[0].numpy(),used[0].numpy(),teacher[0],scale,np)
                    stem=OUT/'arrays'/rid/role/('site'+site)
                    files={'pooled':array(stem.with_name(stem.name+'_pooled.npy'),pooled,np),
                           'moments':array(stem.with_name(stem.name+'_moments.npy'),moments,np),
                           'channel_means':array(stem.with_name(stem.name+'_channel_means.npy'),means,np)}
                    outputs.extend(files.values())
                    results.append({'reference_id':rid,'role':role,'site':site,'input':case['input'],'target':fp,
                        'fixed_clean_RMS':scale,'full_resolution_metrics':metrics,'arrays':files,
                        'initial_identity_is_observed_feature_no_initial_adapter_call':True,
                        'full_metrics_use_unpooled_maps':True,'pooled_order':['observed','wanted_correction','actual_update55_correction'],
                        'channel_means_order':['observed','wanted_correction','actual_update55_correction']})
                    del observed,used,teacher,pooled,moments,means
                if index==14:break
        counters['encoder_prefix_returns']+=1
        need(seen==['256','128','64','32'],'All exact official prefix taps required')
        progress('encoder_prefix_return',reference_id=rid,role=role)
        print(json.dumps({'cases':counters['encoder_prefix_returns'],'of':36,'site_adapters':counters['adapter_returns']}),flush=True)
    need(counters['encoder_prefix_attempts']==counters['encoder_prefix_returns']==36 and
         counters['adapter_attempts']==counters['adapter_returns']==144 and len(results)==144,'Exact finite graph census')
    need(app.base.state_sha(net.state_dict(),torch)==before and all(p.grad is None and not p.requires_grad for p in net.parameters()),'Frozen539 state/gradients changed')
    need(all(not m._forward_hooks and not m._forward_pre_hooks for m in net.modules()),'No intervention hooks may remain')
    for name,item in pre['pins'].items():need(pin(ROOT/name)==item,'Input/source/proof changed during diagnosis')
    result={'complete':True,'UTC':utc(),'source':pin(__file__),'preflight':pin(OUT/'preflight.json'),
        'counters':counters,'metrics':results,'output_arrays':outputs,'state539_before_after':before,
        'model_provenance':provenance,'CPU_cached_CUDA_clear_calibration':[r for r in results if r['role']=='clear'],
        'moment_channels':list(MOMENTS),'pool':POOL,'arithmetic':'unpooled full FP64 differences/reductions; fixed retained clean RMS; uniform16 pooling only for audit compression',
        'audit_limit':'Full original-resolution per-pixel moment maps and per-channel means permit independent scalar/decomposition recomputation. Pooled features permit inspection only, not full encoder/adapter activation replay.',
        'backend_limit':'Frozen clean teacher caches were captured on CUDA. CPU clear-role discrepancy is reported as a calibration floor and is not silently subtracted or called a learned gain.',
        'evidence':'paired synthetic TRAIN feature diagnostic, no native/FINAL images or independent holdout',
        'quality_failure_retained':True,'model_qualified':False,'main_app_default_changed':False,
        'new_training_or_completion':False,'seconds':time.monotonic()-started,'output_bytes_before_result':used_bytes()}
    new_json(OUT/'result.json',result)


def run():
    pre=verify_preflight();need(not (OUT/'launch.json').exists(),'Preserve earlier/partial run; do not restart')
    need(shutil.disk_usage(ROOT/'outputs').free>=RESERVE+LIMIT,'512MiB plus4GiB reserve required')
    import psutil
    need(psutil.__version__=='7.0.0','Pinned resource guard required')
    started=time.monotonic();new_json(OUT/'launch.json',{'UTC':utc(),'parent_pid':os.getpid(),'source':pin(__file__),
        'preflight':pin(OUT/'preflight.json'),'outer_cap_seconds':CAP_SECONDS,'worker_RSS_cap_bytes':RSS_CAP,
        'planned_encoder_prefixes':36,'planned_site_adapters':144,'training_updates':0})
    code=None;stop=None;peak=0
    with (OUT/'worker.log').open('xb') as log:
        process=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--worker'],cwd=ROOT,
            stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        owned=psutil.Process(process.pid);stamp=owned.create_time()
        new_json(OUT/'worker_process.json',{'pid':process.pid,'created':stamp,'source':pin(__file__)})
        while process.poll() is None:
            try:peak=max(peak,owned.memory_info().rss)
            except psutil.NoSuchProcess:pass
            if time.monotonic()-started>CAP_SECONDS:stop='900s outer time limit'
            elif peak>RSS_CAP:stop='6GiB worker RSS cap'
            elif used_bytes()>LIMIT:stop='512MiB output cap'
            elif shutil.disk_usage(ROOT/'outputs').free<RESERVE:stop='4GiB free reserve'
            elif (OUT/'worker.log').stat().st_size>4*1024**2:stop='4MiB worker log cap'
            if stop:
                need(owned.create_time()==stamp and str(Path(__file__).resolve()) in owned.cmdline() and '--worker' in owned.cmdline(),'Exact owned worker required before stop')
                owned.kill();process.wait(timeout=20);break
            try:process.wait(timeout=1)
            except subprocess.TimeoutExpired:pass
        code=process.returncode
    result=read(OUT/'result.json') if (OUT/'result.json').exists() else None
    final={'complete':True,'UTC':utc(),'worker_exit_code':code,'watchdog_stop':stop,'peak_worker_RSS_bytes':peak,
        'seconds':time.monotonic()-started,'actual_counters':result['counters'] if result else None,
        'result_present':result is not None,'model_qualified':False,'quality_failure_retained':True,
        'output_bytes':used_bytes(),'free_bytes':shutil.disk_usage(ROOT/'outputs').free}
    new_json(OUT/'supervisor_final.json',final);print(json.dumps(final),flush=True)
    need(code==0 and stop is None and result is not None,'Finite diagnosis did not complete; preserve partial evidence')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    for flag in ('preflight','run','worker'):group.add_argument('--'+flag,action='store_true')
    args=parser.parse_args()
    try:
        preflight() if args.preflight else run() if args.run else worker()
    except Exception as error:
        if OUT.exists():
            name='worker_failure.json' if args.worker else 'preflight_failure.json' if args.preflight else 'supervisor_failure.json'
            if not (OUT/name).exists():
                try:new_json(OUT/name,{'UTC':utc(),'error':str(error),'traceback':traceback.format_exc(),
                    'model_qualified':False,'training_updates':0,'partial_evidence_preserved':True})
                except Exception:pass
        raise
