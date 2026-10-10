"""Prospective independent returned-array audit and frozen CPU inference replay."""
from pathlib import Path
import hashlib
import shutil
import sys
import tarfile
import time
import zipfile
import numpy as np
from PIL import Image
from scipy.ndimage import convolve1d, uniform_filter
from cctv_dgp_actual_step_review_v1_contract import NAME,STEM,PROPOSALS,ROLES,read,write,sha,verify,safe_members,role_ids,output_prefix,allowed_return_names
from cctv_dgp_actual_step_review_v1_metrics import parameter_state_hash,pixel_metrics,detail_float,mean_only,review_groups,finite_comparison

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'outputs'/NAME
RETURN=ROOT/'outputs'/(NAME+'_return')
CAP=2400


def derived_equal(actual,reported,tolerance=1e-10):
    """Recomputed float aggregates may round; every categorical decision stays exact."""
    assert type(actual) is type(reported),(type(actual).__name__,type(reported).__name__)
    if isinstance(actual,dict):
        assert actual.keys()==reported.keys()
        for key in actual:derived_equal(actual[key],reported[key],tolerance)
    elif isinstance(actual,list):
        assert len(actual)==len(reported)
        for a,b in zip(actual,reported):derived_equal(a,b,tolerance)
    elif isinstance(actual,float):assert np.isfinite(actual) and np.isfinite(reported) and abs(actual-reported)<=tolerance
    else:assert actual==reported


def pixels(path):
    with Image.open(path) as image:
        assert image.mode=='RGB' and image.size==(256,256)
        return np.asarray(image).copy()


def arrays(path,zero=False):
    fields={'rgb','raw_vector','PNG_vector','truth','raw_reference_vector','terms'}|({'original_rgb'} if zero else set())
    with zipfile.ZipFile(path) as z:
        infos=z.infolist();assert len(infos)==len(fields) and {i.filename for i in infos}=={f+'.npy' for f in fields}
        assert sum(i.file_size for i in infos) <= 2*1024**2
    with np.load(path,allow_pickle=False) as z:value={k:z[k].copy() for k in fields}
    for key,a in value.items():
        assert a.dtype==np.float32 and np.isfinite(a).all()
        expected=(256,256,3) if key in {'rgb','original_rgb'} else (7,) if key=='terms' else (512,)
        assert a.shape==expected
        if key in {'rgb','original_rgb'}:assert a.min()>=0 and a.max()<=1
        if key.endswith('vector') or key=='truth':assert abs(float(a@a)-1)<1e-5
    return value


def raw_terms(a,base,target,mask,feature,p):
    actual=a['rgb'].astype(np.float64);reference=(target.astype(np.float32)/np.float32(255)).astype(np.float64)
    baseline=base.astype(np.float64);pixel=float(np.square(actual-reference)[mask].mean())
    bp=float(np.square(baseline-reference)[mask].mean());anchor=float(np.square(actual-baseline)[mask].mean())
    # Independently evaluate the fixed high-pass and seven-window SSIM formula.
    kernel=np.exp(-.5*(np.arange(-6,7,dtype=np.float32)/2)**2).astype(np.float32);kernel/=kernel.sum()
    high=lambda x:x-convolve1d(convolve1d(x,kernel.astype(np.float64),axis=0,mode='reflect'),kernel.astype(np.float64),axis=1,mode='reflect')
    delta=high((actual*np.array([.299,.587,.114])).sum(2))-high((reference*np.array([.299,.587,.114])).sum(2))
    from cctv_dgp_spatial_fit_v40_contract import erode
    interior=erode(mask,6);valid=erode(mask,3)
    def ssim(x):
        pool=lambda q:uniform_filter(q,size=(7,7,1),mode='constant',cval=0)
        u,v=pool(x),pool(reference);va=(pool(x*x)-u*u)*(49/48);vb=(pool(reference*reference)-v*v)*(49/48)
        cov=(pool(x*reference)-u*v)*(49/48)
        value=(2*u*v+.01**2)*(2*cov+.03**2)/((u*u+v*v+.01**2)*(va+vb+.03**2))
        return float(value[valid].mean())
    score=ssim(actual);bscore=ssim(baseline)
    cosine=float(a['raw_vector']@a['truth']);bcosine=float(a['raw_reference_vector']@a['truth'])
    return np.array([1.25*np.square(delta)[feature].mean()/p['normalizers']['feature'],
        1.25*.25*np.square(delta)[interior].mean()/p['normalizers']['interior'],1.25*.05*pixel/max(bp,1e-5),0,
        2*max(0,(pixel-bp)/max(bp,1e-5)),5*max(0,bscore-score),5*max(0,bcosine-cosine)],dtype=np.float64),anchor/max(bp,1e-5)*.05


def CPU_replay(p,start):
    import torch
    sys.path.insert(0,str(BUNDLE))
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_spatial_decoder_v41 import SpatialDGPCandidateV41
    from cctv_dgp_pilot import FixedObservedIdentity,state_hash,grid112
    torch.set_num_threads(4)
    original,_=load_frozen_dgp_restorer(BUNDLE/'weights/dgp_v2.pth',expected_sha256=p['original_checkpoint_sha256'],device='cpu')
    seed=torch.load(BUNDLE/'untrained_initial_decoder.pth',map_location='cpu',weights_only=True)
    candidate=SpatialDGPCandidateV41(original,seed).eval().requires_grad_(False)
    identity=FixedObservedIdentity(BUNDLE/'weights/w600k_r50.onnx','cpu')
    states=lambda:{'original':state_hash(original.net),'decoder':state_hash(candidate.decoder),'reference_decoder':state_hash(candidate.reference_decoder),'recognizer':state_hash(identity)}
    initial=states();assert initial==p['initial_states'];cases={c['id']:c for c in p['cases']};refs={r['id']:r for r in p['references']}
    canonical=lambda a:torch.from_numpy(a.astype(np.float32)/np.float32(255)).permute(2,0,1)[None]
    result={'cases':0,'raw_maximum_error':0.,'original_raw_maximum_error':0.,'PNG_maximum_byte_difference':0,'embedding_maximum_error':0.}
    with torch.inference_mode():
        for probe in p['probes']:
            with np.load(BUNDLE/probe['proposal_arrays'],allow_pickle=False) as data:values=data['values'].copy()
            for index,proposal in enumerate(PROPOSALS):
                assert time.monotonic()-start<CAP
                state={r['name']:torch.from_numpy(values[index,r['start']:r['end']].copy()).reshape(r['shape']) for r in p['parameter_layout']}
                candidate.decoder.load_state_dict(state,strict=True)
                assert state_hash(candidate.decoder)==parameter_state_hash(values[index],p['parameter_layout'])
                ids=probe['case_ids'];images=[pixels(BUNDLE/cases[cid]['input']) for cid in ids]
                masks=[]
                for cid in ids:
                    with Image.open(BUNDLE/cases[cid]['observed']) as image:masks.append(np.asarray(image.convert('L'))>0)
                x=torch.cat([canonical(a) for a in images]);mask=torch.from_numpy(np.stack(masks))[:,None]
                actual=candidate.forward_components(x,mask);raw=actual['result'].permute(0,2,3,1).numpy().copy()
                initial_raw=actual['original_raw'].permute(0,2,3,1).numpy().copy()
                folder=RETURN/output_prefix(probe['update'],proposal)/'current_batch';saved=[arrays(folder/(cid+'.npz'),proposal=='zero') for cid in ids]
                zero=[arrays(RETURN/output_prefix(probe['update'],'zero')/'current_batch'/(cid+'.npz'),True) for cid in ids]
                pngs=[pixels(folder/(cid+'.png')) for cid in ids]
                grids=torch.from_numpy(np.stack([grid112(refs[cases[cid]['reference_id']]['matrix112']) for cid in ids]))
                raw_x=torch.cat([torch.from_numpy(a['rgb']).permute(2,0,1)[None] for a in saved])
                original_x=torch.cat([torch.from_numpy(a['original_rgb']).permute(2,0,1)[None] for a in zero])
                target_x=torch.cat([canonical(pixels(BUNDLE/cases[cid]['target'])) for cid in ids])
                vectors={'raw_vector':identity.embedding(raw_x,mask.float(),grids).numpy(),
                         'PNG_vector':identity.embedding(torch.cat([canonical(a) for a in pngs]),mask.float(),grids).numpy(),
                         'truth':identity.embedding(target_x,mask.float(),grids).numpy(),
                         'raw_reference_vector':identity.embedding(original_x,mask.float(),grids).numpy()}
                for slot,(cid,camera,support) in enumerate(zip(ids,images,masks)):
                    error=float(np.abs(raw[slot]-saved[slot]['rgb']).max());assert error<=p['prospective_audit']['raw_replay_maximum_error']
                    result['raw_maximum_error']=max(result['raw_maximum_error'],error)
                    error=float(np.abs(initial_raw[slot]-zero[slot]['original_rgb']).max());assert error<=p['prospective_audit']['raw_replay_maximum_error']
                    result['original_raw_maximum_error']=max(result['original_raw_maximum_error'],error)
                    q=np.floor(raw[slot]*np.float32(255)).astype(np.uint8);q[~support]=camera[~support]
                    error=int(np.abs(q.astype(np.int16)-pngs[slot].astype(np.int16)).max());assert error<=1
                    result['PNG_maximum_byte_difference']=max(result['PNG_maximum_byte_difference'],error)
                    for key in vectors:
                        error=float(np.abs(vectors[key][slot]-saved[slot][key]).max());assert error<=p['prospective_audit']['embedding_replay_maximum_error']
                        result['embedding_maximum_error']=max(result['embedding_maximum_error'],error)
                    result['cases']+=1
    candidate.decoder.load_state_dict(seed,strict=True);assert states()==initial and result['cases']==150
    assert all(not v.requires_grad and v.grad is None for model in [candidate,identity] for v in model.parameters())
    return {**result,'all_persistent_model_states_unchanged':True,'optimizer_updates':0,'gradient_queries':0}


def main():
    start=time.monotonic();p=read(BUNDLE/'protocol.json');pin=sha(BUNDLE/'protocol.json');verify(BUNDLE,pin)
    assert sha(Path(__file__))==p['source_evidence_sha256'][Path(__file__).relative_to(ROOT).as_posix()]
    assert p['prospective_audit']['independent_audit_seconds']==CAP
    assert p['prospective_audit']['categorical_preservation_decisions_exact']
    archive=ROOT/'outputs'/(STEM+'-results.tar.gz');export=read(ROOT/'outputs'/(STEM+'-export.json'))
    assert export['complete'] and export['archive_sha256']==sha(archive) and export['bytes']==archive.stat().st_size
    assert export['optimizer_updates']==export['gradient_queries']==0 and not export['new_trained_checkpoint']
    assert export['training_success_not_implied']
    assert (ROOT/'outputs'/(STEM+'-results.tar.gz.sha256')).read_text().split()==[sha(archive),archive.name]
    assert not RETURN.exists(), 'Keep every prior import or audit failure'
    with tarfile.open(archive,'r:gz') as tar:
        members,total=safe_members(tar.getmembers(),p)
        assert shutil.disk_usage(ROOT/'outputs').free>=total+512*1024**2
        RETURN.mkdir()
        for member,name in members:
            assert time.monotonic()-start<CAP
            destination=RETURN/name;destination.parent.mkdir(parents=True,exist_ok=True)
            with destination.open('xb') as target,tar.extractfile(member) as source:shutil.copyfileobj(source,target)
    assert sha(RETURN/'protocol.json')==pin
    manifest=read(RETURN/'export_manifest.json');assert manifest['complete'] and manifest['protocol_sha256']==pin
    assert manifest['optimizer_updates']==manifest['gradient_queries']==0 and not manifest['new_trained_checkpoint']
    imported={name:sha(RETURN/name) for _,name in members}
    assert set(imported)==set(manifest['files_sha256'])|{'export_manifest.json'}
    for name,digest in manifest['files_sha256'].items():assert imported[name]==digest,name
    import_path=ROOT/'outputs/cctv_dgp_actual_step_review_v1_return_import.json'
    write(import_path,{'complete':True,'archive_sha256':sha(archive),'members':len(members),'files_sha256':imported,'returned_code_executed':False})
    supervisor=read(RETURN/'supervisor_receipt.json')
    assert supervisor['optimizer_updates']==supervisor['gradient_queries']==0 and not supervisor['new_trained_checkpoint']
    assert not supervisor['automatic_training_follow_on'] and not supervisor['app_promotion']
    assert int((RETURN/'review_exit_code.txt').read_text().strip())==supervisor['review_exit_code']
    results=read(RETURN/'outputs/results.json') if (RETURN/'outputs/results.json').exists() else None
    if results is None:
        failure=read(RETURN/'outputs/failure.json')
        assert not failure['complete'] and failure['optimizer_updates']==failure['gradient_queries']==0
        assert not failure['new_trained_checkpoint'] and not failure['app_promotion']
        assert not manifest['diagnostic_completed'] and not export['run_results_present'] and export['failure_present']
        assert not supervisor['complete'] and supervisor['review_exit_code']!=0
        write(ROOT/'outputs/cctv_dgp_actual_step_review_v1_independent_audit.json',{'complete':True,'failure_retained':True,
              'full_3150_review_complete':False,'members_verified':len(members),'protocol_sha256':pin,'app_promotion':False,
              'audit_scope':'hash-bound partial archive import only; scientific output audit incomplete',
              'optimizer_updates':0,'gradient_queries':0,'seconds':time.monotonic()-start})
        print({'complete':True,'partial_failure_retained':True,'full_3150_review_complete':False});return
    assert results['complete'] and results['progress']['forward_slots']==3150 and results['progress']['completed_conditions']==30
    assert results['optimizer_updates']==results['gradient_queries']==0 and not results['new_trained_checkpoint'] and not results['app_promotion']
    assert results['final_restored_states']==p['initial_states'] and not results['full_TRAIN_capacity_pass']
    assert results['protocol_sha256']==pin and not results['native_or_DEV_or_reserved_final_used'] and results['diagnostic_not_model_qualification']
    assert results['original_initial_reference_and_recognizer_unchanged']
    assert results['seconds']<=p['budgets']['worker_seconds'] and results['allocated_peak_VRAM_bytes']<=p['budgets']['maximum_allocated_VRAM_bytes']
    assert results['progress']['parameter_proposals_loaded']==30
    assert results['progress']['backward_calls']==0 and not results['progress']['optimizer_constructed'] and not results['progress']['new_trained_checkpoint']
    assert set(imported)==allowed_return_names(p)-{'outputs/failure.json'}
    assert manifest['diagnostic_completed'] and export['run_results_present'] and not export['failure_present']
    assert supervisor['complete'] and supervisor['review_exit_code']==0
    assert len(supervisor['calls'])==1 and supervisor['calls'][0]['flag']=='--review' and supervisor['calls'][0]['exit_code']==0
    assert not supervisor['calls'][0]['timeout'] and supervisor['calls'][0]['external_cap_seconds']==p['budgets']['worker_seconds']+30
    preflight=read(RETURN/'outputs/preflight.json');cache=read(RETURN/'outputs/cache_receipt.json');timing=read(RETURN/'outputs/timing_projection.json')
    assert preflight['complete'] and preflight['states']==p['initial_states'] and preflight['initial_exact_case_parity']==145
    assert 'L4' in preflight['gpu'] and preflight['grad_enabled']==False and preflight['optimizer_updates']==preflight['gradient_queries']==0
    assert not preflight['new_trained_checkpoint']
    assert preflight['source_forward_counts']=={'original_DGP':58,'decoder':29,'reference_decoder':29,'recognizer':29}
    assert results['source_forward_counts']=={'original_DGP':688,'decoder':659,'reference_decoder':659,'recognizer':1289}
    assert cache['complete'] and cache['cases']==145 and cache['references']==29 and cache['original_reference_and_recognizer_frozen']
    assert 0<cache['seconds']<=cache['cap_seconds']==p['budgets']['cache_seconds'] and cache['optimizer_updates']==0
    assert timing['completed_probe_states']==1 and timing['remaining_probe_states']==9 and timing['completed_forward_slots']==315
    assert timing['cap_seconds']==p['budgets']['review_seconds'] and timing['safety_factor']==p['budgets']['timing_safety_factor']
    assert abs(timing['projected_review_seconds']-timing['seconds']*10*timing['safety_factor'])<=1e-9
    assert 0<timing['projected_review_seconds']<=timing['cap_seconds']
    sys.path.insert(0,str(BUNDLE))
    from cctv_dgp_spatial_fit_v40_contract import feature_support,exported_pixel_metrics,detail_metric
    cases={c['id']:c for c in p['cases']};checked=0;term_errors=np.zeros(7);condition_list=[]
    for probe in p['probes']:
        zero_condition=None
        with np.load(BUNDLE/probe['proposal_arrays'],allow_pickle=False) as data:values=data['values'].copy()
        for index,proposal in enumerate(PROPOSALS):
            folder=RETURN/output_prefix(probe['update'],proposal);receipt=read(folder/'metrics.json')
            assert receipt['complete'] and receipt['decoder_state']==parameter_state_hash(values[index],p['parameter_layout'])
            assert receipt['parameter_vector_sha256']==hashlib.sha256(values[index].tobytes()).hexdigest()
            assert receipt['update']==probe['update'] and receipt['proposal']==proposal
            assert receipt['optimizer_updates']==receipt['gradient_queries']==0 and not receipt['new_trained_checkpoint'] and not receipt['app_promotion']
            assert receipt['first_order_proposal_arithmetic']==probe['arithmetic']
            rows_by_role={}
            for role in ROLES:
                ids=role_ids(p,probe,role);assert [r['id'] for r in receipt['rows'][role]]==ids;rows=[]
                for cid,reported in zip(ids,receipt['rows'][role]):
                    assert time.monotonic()-start<CAP
                    c=cases[cid];camera=pixels(BUNDLE/c['input']);target=pixels(BUNDLE/c['target'])
                    with Image.open(BUNDLE/c['observed']) as image:mask=np.asarray(image.convert('L'))>0
                    a=arrays(folder/role/(cid+'.npz'),proposal=='zero')
                    zero=arrays(RETURN/output_prefix(probe['update'],'zero')/role/(cid+'.npz'),True)
                    assert np.array_equal(a['rgb'][~mask],camera[~mask].astype(np.float32)/np.float32(255))
                    png=pixels(folder/role/(cid+'.png'));expected=np.floor(a['rgb']*np.float32(255)).astype(np.uint8);expected[~mask]=camera[~mask]
                    assert np.array_equal(png,expected) and np.array_equal(png[~mask],camera[~mask])
                    assert reported['raw_float32_sha256']==hashlib.sha256(a['rgb'].tobytes()).hexdigest()
                    mean_raw,mean_png,shift=mean_only(a['rgb'],zero['rgb'],camera,mask)
                    assert np.array_equal(mean_png,pixels(folder/role/(cid+'_mean_only.png')))
                    assert np.allclose(shift,reported['postclip_mean_RGB_shift'],rtol=0,atol=1e-12)
                    support=feature_support(mask,c['landmarks5_canvas_xy'])
                    raw=pixel_metrics(a['rgb'],target,mask);raw.update({'ArcFace_observed_fixed':float(a['raw_vector']@a['truth']),
                        'landmark_high_frequency_MSE':detail_float(a['rgb'],target,support),'constant_mean_shift_only_MSE':pixel_metrics(mean_raw,target,mask)['MSE']})
                    displayed=exported_pixel_metrics(png,target,mask);displayed.update({'ArcFace_observed_fixed':float(a['PNG_vector']@a['truth']),
                        'landmark_high_frequency_MSE':detail_metric(png,target,support),'constant_mean_shift_only_MSE':exported_pixel_metrics(mean_png,target,mask)['MSE']})
                    for stage,actual in [('raw',raw),('PNG',displayed)]:
                        for key,value in actual.items():
                            saved=reported[stage][key]
                            if value is None or isinstance(value,bool):assert value==saved
                            else:assert abs(value-saved)<=1e-10,(cid,stage,key)
                    assert np.array_equal(a['terms'].astype(np.float64),reported['objective_terms'])
                    independent,anchor=raw_terms(a,zero['original_rgb'],target,mask,support,p)
                    if c['profile']=='clear':independent[:3]=0;independent[3]=anchor
                    error=np.abs(independent-a['terms']);term_errors=np.maximum(term_errors,error)
                    assert np.all(error<=p['prospective_audit']['raw_objective_term_absolute_error']),(cid,error.tolist())
                    rows.append({'id':cid,'source':c['source'],'profile':c['profile'],'raw':raw,'PNG':displayed,'objective_terms':reported['objective_terms']});checked+=1
                rows_by_role[role]=rows
            grouped={role:{stage:review_groups(rows_by_role[role],stage) for stage in ['raw','PNG']} for role in ROLES}
            derived_equal(grouped,receipt['groups'],p['prospective_audit']['derived_group_and_comparison_absolute_error'])
            for role in ROLES:assert np.array_equal(np.mean([r['objective_terms'] for r in rows_by_role[role]],axis=0),receipt['objective_term_means'][role])
            comparison=None if zero_condition is None else {role:{stage:finite_comparison(zero_condition['groups'][role][stage],grouped[role][stage]) for stage in ['raw','PNG']} for role in ROLES}
            derived_equal(comparison,receipt['comparison_to_same_before_state'],p['prospective_audit']['derived_group_and_comparison_absolute_error'])
            condition_list.append({'update':probe['update'],'proposal':proposal,'metrics_sha256':sha(folder/'metrics.json'),'comparison_to_same_before_state':comparison})
            if proposal=='zero':zero_condition=receipt
    assert checked==3150
    derived_equal(condition_list,results['conditions'],p['prospective_audit']['derived_group_and_comparison_absolute_error'])
    replay=CPU_replay(p,start)
    receipt={'complete':True,'full_3150_review_complete':True,'protocol_sha256':pin,'archive_sha256':sha(archive),
        'members_verified':len(members),'all3150_raw_and_PNG_and_mean_only_outputs_checked':True,
        'raw_objective_maximum_errors':term_errors.tolist(),'CPU_replay':replay,'conditions':condition_list,
        'optimizer_updates':0,'gradient_queries':0,'new_trained_checkpoint':False,'app_promotion':False,
        'full_TRAIN_capacity_pass':False,'independent_final_review':False,'seconds':time.monotonic()-start,'cap_seconds':CAP}
    write(ROOT/'outputs/cctv_dgp_actual_step_review_v1_independent_audit.json',receipt)
    print({'complete':True,'output_slots':checked,'CPU_replay_cases':replay['cases'],'optimizer_updates':0,'seconds':time.monotonic()-start},flush=True)


if __name__=='__main__':main()
