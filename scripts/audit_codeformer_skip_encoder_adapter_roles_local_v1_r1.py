"""Independent saved-array arithmetic for the terminal R1 feature diagnosis.

No worker/app imports, Torch, models, gradients, browser, VM or training calls.
Requires the existing terminal supervisor evidence; never reruns the diagnosis.
"""
import hashlib
import json
import math
from pathlib import Path
import stat
import time
from datetime import datetime, timezone
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'outputs/codeformer_skip_encoder_adapter_role_diagnostic_v1_r1'
DEST=ROOT/'outputs/codeformer_skip_feature_adapter_return_review_v1/post_user_diagnosis_v1/encoder_adapter_roles_r1_result_audit_v1'
SOURCE_SHA='8030cc4c7ed851f3dd991883d5b83d28c5e4f54ed59b1e63365b29e7aae4e3f6'
PREFLIGHT_SHA='0e93a8fda81922aa1d1bf70f408620adf90a612e585b3a87e98890db914120dc'
REFS=('tr_ffhq_03761','tr_ffhq_55791','tr_ffhq_60978','tr_ffhq_34217','tr_ffhq_31449','tr_ffhq_52422')
ROLES=('clear','moderate','strong','dark','local_mosaic8','global_mosaic8')
SITES=('32','64','128','256')
CHANNELS={'32':256,'64':256,'128':128,'256':128}
MOMENTS=('wanted_abs','final_residual_abs','wanted_sq','correction_sq','final_residual_sq','correction_dot_wanted','wanted_spatial_sq','correction_spatial_sq','spatial_dot','observed_sq','target_sq')
BEGIN=time.monotonic();PINS={};CAP=600

def need(value,message):
    if not value:raise ValueError(message)
def deadline():need(time.monotonic()-BEGIN<CAP,'Finite600s audit cap; retain evidence')
def regular(path):
    path=Path(path).absolute();need(path.resolve().is_relative_to(ROOT),'Workspace evidence required')
    for p in (path,*path.parents):
        row=p.lstat();need(not stat.S_ISLNK(row.st_mode) and not getattr(row,'st_file_attributes',0)&1024,'Linked evidence refused')
    row=path.stat();need(stat.S_ISREG(row.st_mode) and row.st_nlink==1,'Unique ordinary evidence required')
    return path

def pin(path):
    deadline();path=regular(path);before=path.stat();h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):deadline();h.update(block)
    after=path.stat();need((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),'Changed while hashing')
    row={'path':path.relative_to(ROOT).as_posix(),'bytes':after.st_size,'sha256':h.hexdigest()};PINS[row['path']]=row;return row

def read(path,limit=8*1024**2):
    p=pin(path);need(p['bytes']<=limit,'Bounded metadata required')
    def unique(pairs):
        result={}
        for k,v in pairs:need(k not in result,'Duplicate JSON key');result[k]=v
        return result
    return json.loads(Path(path).read_bytes(),object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('NonfiniteJSON')))

def path_pin(row):
    need(type(row) is dict and set(row)=={'path','bytes','sha256'},'Exact physical binding schema')
    need(pin(ROOT/row['path'])==row,'Physical binding changed: '+row['path']);return ROOT/row['path']

def array(row,shape,dtype):
    path=path_pin(row)
    with path.open('rb') as f:
        version=np.lib.format.read_magic(f);need(version in ((1,0),(2,0)),'NPY1/2 only')
        parse=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
        actual_shape,fortran,actual_dtype=parse(f);offset=f.tell()
    need(tuple(shape)==actual_shape and not fortran and actual_dtype==dtype and not dtype.hasobject,'Array header/schema mismatch')
    need(path.stat().st_size==offset+math.prod(shape)*dtype.itemsize,'NPY exact size/EOF mismatch')
    value=np.load(path,mmap_mode='r',allow_pickle=False)
    need(np.isfinite(value).all(),'Nonfinite saved array');return value

def close(a,b,label,atol=2e-10,rtol=2e-10):
    need(math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,abs_tol=atol,rel_tol=rtol),'Scalar differs: '+label)
def cosine(dot,a2,b2):return dot/math.sqrt(a2*b2) if a2>0 and b2>0 else None

def summarize(maps,means,scale):
    v={name:float(maps[i].mean()) for i,name in enumerate(MOMENTS)}
    wdc=float((means[1]*means[1]).mean());cdc=float((means[2]*means[2]).mean());ddot=float((means[1]*means[2]).mean())
    ini=v['wanted_abs']/scale;fin=v['final_residual_abs']/scale
    return {'initial_normalized_L1':ini,'update55_normalized_L1':fin,'relative_L1_gain':1-fin/ini if ini>0 else None,
    'correction_RMS_over_fixed_clean_RMS':math.sqrt(v['correction_sq'])/scale,'wanted_RMS_over_fixed_clean_RMS':math.sqrt(v['wanted_sq'])/scale,
    'final_residual_RMS_over_fixed_clean_RMS':math.sqrt(v['final_residual_sq'])/scale,'correction_DC_energy':cdc,'correction_spatial_energy':v['correction_spatial_sq'],
    'wanted_DC_energy':wdc,'wanted_spatial_energy':v['wanted_spatial_sq'],'correction_DC_energy_fraction':cdc/v['correction_sq'] if v['correction_sq']>0 else None,
    'direction_alignment':cosine(v['correction_dot_wanted'],v['correction_sq'],v['wanted_sq']),'DC_direction_alignment':cosine(ddot,cdc,wdc),
    'spatial_direction_alignment':cosine(v['spatial_dot'],v['correction_spatial_sq'],v['wanted_spatial_sq']),
    'observed_RMS_over_fixed_clean_RMS':math.sqrt(v['observed_sq'])/scale,'target_RMS_over_fixed_clean_RMS':math.sqrt(v['target_sq'])/scale}

def target_statistics(row,site):
    c=CHANNELS[site];h=int(site);a=array(row,(1,c,h,h),np.dtype('float32'));means=np.empty(c);pooled=np.empty((c,16,16));square_sum=0.
    for start in range(0,c,8):
        x=a[0,start:start+8].astype(np.float64);square_sum+=float((x*x).sum());means[start:start+8]=x.mean((1,2))
        pooled[start:start+8]=x.reshape(len(x),16,h//16,16,h//16).mean((2,4))
    result={'RMS':math.sqrt(square_sum/a.size),'means':means,'pooled':pooled};del a;return result

def main():
    need(not DEST.exists(),'Preserve earlier independent audit')
    final=read(DATA/'supervisor_final.json');need(final['complete'] is True and final['worker_exit_code']==0 and final['watchdog_stop'] is None and final['result_present'] is True,'Terminal successful bounded diagnosis required')
    pre=read(DATA/'preflight.json');need(PINS[(DATA/'preflight.json').relative_to(ROOT).as_posix()]['sha256']==PREFLIGHT_SHA,'Exact root observed preflight required')
    result=read(DATA/'result.json');launch=read(DATA/'launch.json');started=read(DATA/'worker_started.json');process=read(DATA/'worker_process.json');prestart=read(DATA/'preflight_started.json')
    source=path_pin(pre['source']);need(pre['source']['sha256']==SOURCE_SHA and result['source']==launch['source']==started['source']==process['source']==prestart['source']==pre['source'],'Same immutable source throughout')
    need(result['complete'] is True and result['preflight']==PINS[(DATA/'preflight.json').relative_to(ROOT).as_posix()],'Exact completed result/preflight')
    for name,row in pre['pins'].items():need(row['path']==name and pin(ROOT/name)==row,'Input/source/cache proof changed')
    expected={'encoder_prefix_attempts':36,'encoder_prefix_returns':36,'adapter_attempts':144,'adapter_returns':144,'full_CodeFormer_transformer_quantizer_decoder_calls':0,'optimizer_gradient_backward_calls':0}
    need(result['counters']==final['actual_counters']==expected,'36/144 terminal counter closure differs')
    need(pre['planned_encoder_prefix_passes']==36 and pre['planned_site_adapter_evaluations']==144 and pre['planned_optimizer_updates']==0,'Full prospective scope')
    need(pre['moment_channels']==result['moment_channels']==list(MOMENTS) and pre['pool']==result['pool']==16,'Moment contract unchanged')
    need(result['model_qualified'] is False and final['model_qualified'] is False and result['main_app_default_changed'] is False and result['quality_failure_retained'] is True and final['quality_failure_retained'] is True and result['new_training_or_completion'] is False,'Quality boundary differs')
    need(pre['outer_cap_seconds']==launch['outer_cap_seconds']==900 and final['seconds']<905 and result['seconds']<900,'Finite time guard')
    need(final['peak_worker_RSS_bytes']<=2*1024**3 and result['peak_worker_RSS_sampled_bytes']<=2*1024**3,'SampledRSS guard')
    need(result['minimum_host_available_sampled_bytes']>=1024**3 and final['free_bytes']>=4*1024**3,'Recorded hostRAM/disk reserve')
    prov=result['model_provenance'];need(prov['full_CodeFormer_allocated'] is False and prov['transformer_decoder_quantizer_allocated'] is False and prov['base515_frozen_verified'] is True and prov['source539_tensors']==539 and prov['base515_tensors']==515 and prov['adapter24_tensors']==24,'Encoderonly source/provenance')
    need(prov['source_canonical_digests']=={'model539':'45f3e1d39f4abb19d1adadff45e6c2d841df90becc0c5c97c381a8e5b7502762','base515':'a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f','adapter24':'3950cfddd938fcff85850ad846a936133efa3ae17ca405a2c63417870d42acf5'},'Exact frozen source lineage')
    need(prov['official_Encoder_parameter_bytes_before_prune']==117664000 and prov['retained_prefix_parameter_bytes']==26607872 and prov['retained_encoder_blocks']==list(range(15)),'Exact encoderprefix census')
    path_pin(prov['model_export']);need(result['retained_prefix_adapter_state_before_after']['adapter24']==prov['source_canonical_digests']['adapter24'],'Retainedadapter hash differs')
    cases={(c['reference_id'],c['role']):c for c in pre['cases']};need(len(cases)==36 and set(cases)=={(r,role) for r in REFS for role in ROLES} and all(c['split']=='TRAIN' for c in cases.values()),'TRAIN six-by-six coverage')
    targets={};rows=[];expected_arrays=[]
    for row in result['metrics']:
        rid,role,site=row['reference_id'],row['role'],row['site'];need((rid,role) in cases and site in SITES,'Unexpected case/site')
        need(row['input']==cases[rid,role]['input'] and row['target']==pre['clean_features'][rid]['features'][site],'Input/target association differs')
        need(row['initial_identity_is_observed_feature_no_initial_adapter_call'] is True and row['full_metrics_use_unpooled_maps'] is True,'Identity/metric scope differs')
        need(row['pooled_order']==row['channel_means_order']==['observed','wanted_correction','actual_update55_correction'],'Array component order differs')
        c=CHANNELS[site];h=int(site);maps=array(row['arrays']['moments'],(11,h,h),np.dtype('float64'));means=array(row['arrays']['channel_means'],(3,c),np.dtype('float64'));pooled=array(row['arrays']['pooled'],(3,c,16,16),np.dtype('float64'))
        expected_arrays.extend(row['arrays'].values());scale=pre['clean_features'][rid]['scales'][site];need(scale==row['fixed_clean_RMS'] and scale>0,'Fixed normalization differs')
        if (rid,site) not in targets:targets[rid,site]=target_statistics(row['target'],site)
        target=targets[rid,site];close(target['RMS'],scale,'targetRMS',atol=1e-9,rtol=1e-9)
        need(np.allclose(pooled.mean((2,3)),means,atol=2e-10,rtol=2e-10),'Pooledchannel means differ')
        need(np.allclose(means[0]+means[1],target['means'],atol=2e-10,rtol=2e-10),'Wantedchannel means not exact target-observed')
        need(np.allclose(pooled[0]+pooled[1],target['pooled'],atol=2e-10,rtol=2e-10),'Wantedpooled features not exact target-observed')
        for i in [0,1,2,3,4,6,7,9,10]:need(float(maps[i].min())>=0,'Negative squared/absolute moment')
        need(np.allclose(maps[4],maps[3]+maps[2]-2*maps[5],atol=2e-10,rtol=2e-10),'Residualenergy identity differs')
        need(np.all(maps[5]*maps[5]<=maps[3]*maps[2]+2e-10),'Correction/wanted per-pixel Cauchy inequality')
        metrics=summarize(maps,means,scale)
        for name,value in metrics.items():
            saved=row['full_resolution_metrics'][name]
            if value is None:need(saved is None,'Zeroenergy null changed')
            else:close(value,saved,name)
        close(float(maps[3].mean()),metrics['correction_DC_energy']+metrics['correction_spatial_energy'],'CorrectionDCspatial')
        close(float(maps[2].mean()),metrics['wanted_DC_energy']+metrics['wanted_spatial_energy'],'WantedDCspatial')
        close(float(maps[5].mean()),float((means[1]*means[2]).mean())+float(maps[8].mean()),'DotDCspatial')
        need(math.isfinite(row['full_resolution_metrics']['maximum_absolute_wanted_correction']) and row['full_resolution_metrics']['maximum_absolute_wanted_correction']>=0,'Capturedmax scalar mustbefinite')
        rows.append({'reference_id':rid,'role':role,'site':site,'split':'TRAIN','recomputed':metrics,'arrays':row['arrays'],'maximum_absolute_wanted_correction_independently_verified':False})
        del maps,means,pooled
    need(len(rows)==144 and len({(r['reference_id'],r['role'],r['site']) for r in rows})==144,'Exact144unique site rows')
    need(len(expected_arrays)==432 and len({r['path'] for r in expected_arrays})==432 and result['output_arrays']==expected_arrays,'Exact432arraycensus and ordering')
    need(result['CPU_cached_CUDA_clear_calibration']==[r for r in result['metrics'] if r['role']=='clear'],'All24clearcalibration rows retained')
    # The progress log is a durable claim; no live neural replay is inferred.
    progress_pin=pin(DATA/'progress.jsonl');events=[json.loads(line) for line in (DATA/'progress.jsonl').read_text(encoding='utf8').splitlines()];need(len(events)==216,'36prefixattempt/returns and144adapterattemptevents')
    need(events[-1]['counters']==expected,'Finalprogresscounters differ')
    need(sum(e['stage']=='encoder_prefix_attempt' for e in events)==36 and sum(e['stage']=='encoder_prefix_return' for e in events)==36 and sum(e['stage']=='adapter_attempt' for e in events)==144,'Durable eventcount differs')
    pin(DATA/'worker.log');need((DATA/'worker.log').stat().st_size<=4*1024**2,'Boundedlog')
    all_files={p.relative_to(DATA).as_posix():p for p in DATA.rglob('*') if p.is_file()};allowed={Path(r['path']).relative_to(DATA.relative_to(ROOT)).as_posix() for r in expected_arrays}
    allowed|={Path(c['input']['path']).relative_to(DATA.relative_to(ROOT)).as_posix() for c in pre['cases'] if c['role'] in ['local_mosaic8','global_mosaic8']}
    allowed|={'preflight_started.json','preflight.json','launch.json','worker_process.json','worker_started.json','worker.log','progress.jsonl','result.json','supervisor_final.json'}
    need(set(all_files)==allowed,'Exact closed diagnostic file namespace')
    total=sum(p.stat().st_size for p in all_files.values());need(total<=512*1024**2,'Actual outputcap')
    need(final['output_bytes']==total-(DATA/'supervisor_final.json').stat().st_size,'Finalsize beforeitsownwrite')
    need(result['output_bytes_before_result']==total-(DATA/'supervisor_final.json').stat().st_size-(DATA/'result.json').stat().st_size,'Resultsize beforeitsownwrite')
    aggregates=[]
    for role in ROLES:
        for site in SITES:
            group=[r for r in rows if r['role']==role and r['site']==site];need(len(group)==6,'Sixsources perrole/site')
            agg={'role':role,'site':site,'observations':6,'mean_initial_normalized_L1':sum(r['recomputed']['initial_normalized_L1'] for r in group)/6,'mean_update55_normalized_L1':sum(r['recomputed']['update55_normalized_L1'] for r in group)/6,'positive_gain_observations':sum(r['recomputed']['relative_L1_gain'] is not None and r['recomputed']['relative_L1_gain']>0 for r in group)}
            ini=agg['mean_initial_normalized_L1'];agg['relative_gain_of_mean_L1']=1-agg['mean_update55_normalized_L1']/ini if ini>0 else None
            for metric in ['correction_RMS_over_fixed_clean_RMS','wanted_RMS_over_fixed_clean_RMS','final_residual_RMS_over_fixed_clean_RMS','correction_DC_energy_fraction','direction_alignment','DC_direction_alignment','spatial_direction_alignment']:
                vals=[r['recomputed'][metric] for r in group if r['recomputed'][metric] is not None];agg['mean_'+metric]=sum(vals)/len(vals) if vals else None
            aggregates.append(agg)
    for path,row in list(PINS.items()):need(pin(ROOT/path)==row,'Evidence changed duringaudit')
    source_audit_pin=pin(__file__)
    report={'format':'codeformer-skip-encoder-adapter-role-r1-independent-array-audit-v1','complete':True,'audit_pass':True,'UTC':datetime.now(timezone.utc).isoformat(),'source_auditor':source_audit_pin,'pins':PINS,'cases':36,'site_rows':144,'arrays_checked':432,'cached_target_arrays_checked':24,'moment_channels':list(MOMENTS),'all_scalar_moment_mean_decompositions_recomputed':True,'RAW_face_outputs_or_PNGs_produced_by_this_diagnosis':False,'aggregates_role_by_site':aggregates,'per_case_site':rows,'CPU_clear_calibration_not_subtracted':[r for r in aggregates if r['role']=='clear'],'actual_counts':expected,'actual_output_files':len(all_files),'actual_output_bytes':total,'actual_runtime':{'supervisor_seconds':final['seconds'],'worker_seconds':result['seconds'],'supervisor_peak_worker_RSS_sampled_bytes':final['peak_worker_RSS_bytes'],'worker_post_load_peak_RSS_sampled_bytes':result['peak_worker_RSS_sampled_bytes'],'worker_post_load_min_host_available_sampled_bytes':result['minimum_host_available_sampled_bytes'],'final_free_bytes':final['free_bytes']},'limitations':['Source-bound before/after parameter digests and executed call/reservation counts are checked for consistency, not independent neural or optimizer replay.','Original-resolution momentmaps/channelmeans permit scalar/decomposition audits; pooledfeatures do not reconstruct full observed/used activations.','maximum_absolute_wanted_correction is source-captured and finite only: it cannot be independently recomputed from saved moments/pooling, so it is not validated as exact.','CPUclear/CUDAcache difference is calibrationfloor; never subtracted. FeatureDCoffset is not a claim of imagebrightness.','New mosaic inputs have no new readability adjudication; six repeatedTRAINphotographs are not independentholdout/nativeperformance/FINAL.','A featuremetricgain is not visiblefacequality, preservationqualification or identityrecovery.'],'quality_failure_retained':True,'model_qualified':False,'app_default_changed':False,'FINAL_used':False,'audit_neural_gradient_optimizer_VM_browser_calls':0,'seconds':time.monotonic()-BEGIN}
    DEST.mkdir(parents=True,exist_ok=False);target=DEST/'audit_v1.json'
    with target.open('x',encoding='utf8',newline=chr(10)) as f:json.dump(report,f,indent=2,allow_nan=False);f.write(chr(10))
    # Emit only concise physical result, not full audit data.
    print(json.dumps({'complete':True,'audit_pass':True,'path':target.relative_to(ROOT).as_posix(),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'arrays':432,'neural_calls':0}))
if __name__=='__main__':main()
