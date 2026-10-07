"""Prospective safe scientific return audit; never execute returned code or train."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import time
from types import MethodType

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'outputs/cctv_dgp_mean_centered_decoder_vm_v29'
PARENT=ROOT/'outputs/cctv_dgp_feature_skips_vm_v27'
OUT=ROOT/'outputs/cctv_dgp_mean_centered_decoder_v29_return'
PIN='77565ba437959305f22cff4dd967fc6c3caadbf9dbd4ac91abf8a366577fa72f'
PREFIX='cctv_dgp_mean_centered_decoder_v29_return/'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def validate_members(members,allowed,total_cap=900_000_000):
    rows=[];seen=set();total=0
    for m in members:
        assert m.isfile() and not m.issym() and not m.islnk(),'Regular files only'
        assert m.name.startswith(PREFIX) and '\\' not in m.name and ':' not in m.name
        name=m.name[len(PREFIX):];parts=PurePosixPath(name).parts
        assert parts and not PurePosixPath(name).is_absolute() and all(x not in ['.','..'] for x in parts)
        assert name in allowed and name.casefold() not in seen,'Unexpected/duplicate/colliding file: '+name
        assert 0<=m.size<=64*1024**2,'Member bounds'
        seen.add(name.casefold());total+=m.size
        assert total<=total_cap and len(rows)<1000,'Archive bounds'
        rows.append((m,name))
    return rows,total


def matrix(a,shape=(7,498627)):
    assert a.shape==shape and a.dtype==np.float64 and np.isfinite(a).all(),'Gradient schema/nonfinite'
    return a


def statistics(a,layout):
    a=matrix(a,(7,layout[-1]['end']));norms=np.linalg.norm(a,axis=1);gram=a@a.T
    divisor=norms[:,None]*norms[None,:]
    cosine=np.divide(gram,divisor,out=np.zeros_like(gram),where=divisor>0);blocks={}
    for row in layout:
        v=a[:,row['start']:row['end']]
        blocks[row['name']]={'component_norms':np.linalg.norm(v,axis=1).tolist(),
                            'improvement_gradient_norm':float(np.linalg.norm(v[:3].sum(0))),
                            'total_gradient_norm':float(np.linalg.norm(v.sum(0)))}
    return norms,gram,cosine,blocks


def groups(rows):
    sources={r['source'] for r in rows};profiles={r['profile'] for r in rows};result={}
    for key in sorted({'all','clear','degraded'}|{src+'/'+k for src in sources for k in ['all','clear','degraded',*profiles]}):
        chosen=[r for r in rows if key in {'all','clear' if r['profile']=='clear' else 'degraded',r['source']+'/all',r['source']+('/clear' if r['profile']=='clear' else '/degraded'),r['source']+'/'+r['profile']}]
        assert chosen
        result[key]={'cases':len(chosen),**{metric:float(np.mean([r['metrics'][metric] for r in chosen])) for metric in ['MSE','SSIM','ArcFace_observed_fixed','landmark_high_frequency_MSE','constant_mean_shift_only_MSE']}}
    return result


def capacity(baseline,current):
    failures=[]
    for group,b in baseline.items():
        a=current[group];assert a['cases']==b['cases']
        for metric in ['MSE','SSIM','ArcFace_observed_fixed']:
            bad=a[metric]>b[metric]+1e-12 if metric=='MSE' else a[metric]<b[metric]-1e-6
            if bad:failures.append({'group':group,'metric':metric,'baseline':b[metric],'candidate':a[metric]})
    gain=1-current['degraded']['landmark_high_frequency_MSE']/baseline['degraded']['landmark_high_frequency_MSE']
    source_gains={key[:-len('/degraded')]:1-current[key]['landmark_high_frequency_MSE']/baseline[key]['landmark_high_frequency_MSE'] for key in baseline if key.endswith('/degraded')}
    b,a=baseline['degraded'],current['degraded']
    brightness=max(0,b['MSE']-a['constant_mean_shift_only_MSE'])/max(b['MSE']-a['MSE'],1e-12)
    if brightness>.2:failures.append({'group':'degraded','metric':'brightness_gain_fraction','candidate':brightness,'maximum':.2})
    return failures,gain,source_gains,brightness,not failures and gain>=.1 and all(x>=0 for x in source_gains.values())


def allowed_names(p,base):
    allowed={'protocol.json','export_manifest.json','trainer.log','trainer_exit_code.txt','supervisor_receipt.json',*p['assets_sha256']}
    allowed.update('outputs/'+name for name in ['gradient_components.npy','gradient_summary.json','cohort_loss_setup.json',
        'gradient_preflight.json','results.json','failure.json','timing_update20.json','early_structure_stop.json','execution_receipt.json','stopped_dgp_candidate_v29.pth'])
    allowed.update('outputs/gradients/batch'+str(i)+'.npy' for i in range(10))
    for c in base['cases']:
        for suffix in ['.npy','.png','_target_embedding.npy']:allowed.add('outputs/initial_baseline/'+c['id']+suffix)
        for update in [0,50,400,800]:
            for suffix in ['.npy','.png','_embedding.npy']:allowed.add('outputs/update'+str(update)+'/'+c['id']+suffix)
        allowed.add('outputs/update0/'+c['id']+'_target_embedding.npy')
    for update in [0,50,400,800]:
        allowed.update('outputs/update'+str(update)+'/'+name for name in ['metrics.json','dgp_candidate_v29.pth'])
    return allowed


def import_return(expected_sha,expected_bytes,p,base):
    archive=ROOT/'outputs/cctv-dgp-mean-centered-decoder-v29-results.tar.gz'
    assert re.fullmatch('[a-f0-9]{64}',expected_sha) and 0<expected_bytes<=900_000_000
    assert sha(archive)==expected_sha and archive.stat().st_size==expected_bytes
    assert Path(str(archive)+'.sha256').read_text(encoding='ascii').strip().split()==[expected_sha,archive.name]
    exported=read(ROOT/'outputs/cctv-dgp-mean-centered-decoder-v29-export.json')
    assert exported['complete'] and exported['archive_sha256']==expected_sha and exported['bytes']==expected_bytes
    assert exported['training_success_not_implied'] and not OUT.exists(),'Preserve existing import/partial evidence'
    with tarfile.open(archive,'r:gz') as tar:
        rows,total=validate_members(tar.getmembers(),allowed_names(p,base),p['budgets']['maximum_export_uncompressed_bytes'])
        OUT.mkdir()
        for member,name in rows:
            path=(OUT/name).resolve();assert path.is_relative_to(OUT)
            path.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as f,path.open('xb') as g:shutil.copyfileobj(f,g)
            assert path.stat().st_size==member.size
    files={name:sha(OUT/name) for _,name in rows}
    assert sha(OUT/'protocol.json')==PIN
    for name,digest in p['assets_sha256'].items():assert files[name]==digest
    manifest=read(OUT/'export_manifest.json')
    assert manifest['complete'] and manifest['protocol_sha256']==PIN and manifest['quality_acceptance_not_implied']
    assert manifest['files_sha256']=={k:v for k,v in files.items() if k!='export_manifest.json'}
    record={'complete':True,'archive_sha256':expected_sha,'archive_bytes':expected_bytes,'members':len(rows),'uncompressed_bytes':total,
            'files_sha256':files,'local_neural_or_gradient_calls':0,'app_promotion':False,'goal_complete':False}
    write(ROOT/'outputs/cctv_dgp_mean_centered_decoder_v29_return_import.json',record)
    return exported,record


def initial_CPU_replay(start,p,base,complete_proof):
    # Reuse only the previously verified local forward-only CPU audit block.
    # Never compile or import a file from the returned archive.
    source_path=ROOT/'scripts/audit_cctv_dgp_original_decoder_gradient_v1_r2_return.py'
    assert sha(source_path)=='1f240e95d22e8726041599a09d147dcef047b12cf2f7c4d70735366d3f65bb7e'
    source=source_path.read_text(encoding='utf-8')
    text=source[source.index('    replay_rows=[];cohort_rows=[];'):source.index('    audit={')]
    text=text.replace('if replay and baseline_folder.exists():','if baseline_folder.exists():')
    text=text.replace('        sys.path.insert(0,str(PARENT))','        sys.path.insert(0,str(PARENT))\n        sys.path.insert(0,str(BUNDLE))')
    text=text.replace('from dgp_face_restoration import as_tensor','from cctv_dgp_app_input_v28 import canonical_tensor as as_tensor')
    text=text.replace('as_tensor(camera)',"as_tensor(camera,'cpu')").replace('as_tensor(target)',"as_tensor(target,'cpu')")
    text='def replay(start,p,base,complete_proof):\n'+text+'    return replay_rows,cohort_rows,CPU_recognizer_forwards,state_before\n'
    tree=ast.parse(text)
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['grad','backward','step','AdamW'] for n in ast.walk(tree))
    namespace={'ROOT':ROOT,'PARENT':PARENT,'BUNDLE':BUNDLE,'OUT':OUT,'sys':sys,'np':np,'time':time,'read':read,'MethodType':MethodType}
    exec(compile(tree,'<pinned-local-CPU-replay-only>','exec'),namespace)
    return namespace['replay'](start,p,base,complete_proof)


def audit(expected_sha,expected_bytes):
    start=time.monotonic();assert sha(BUNDLE/'protocol.json')==PIN
    p=read(BUNDLE/'protocol.json');base=read(PARENT/'protocol.json')
    for name,digest in p['assets_sha256'].items():assert sha(BUNDLE/name)==digest
    for name,digest in base['assets_sha256'].items():assert sha(PARENT/name)==digest
    assert p['retained_capacity_gates']==base['prospective_gates'] and p['case_rows']==base['cases']
    for folder,key in [(ROOT/'outputs/cctv_dgp_active_original_decoder_v28_return','closed_V28_evidence_sha256'),(ROOT/'outputs/cctv_dgp_v28_preservation_diagnostic_v1_return','closed_diagnostic_evidence_sha256')]:
        for name,digest in p[key].items():assert sha(folder/name)==digest,name
    for name,digest in p['local_basis_sha256'].items():assert sha(ROOT/name)==digest,name
    exported,imported=import_return(expected_sha,expected_bytes,p,base)
    result_path,failure_path=OUT/'outputs/results.json',OUT/'outputs/failure.json'
    assert result_path.exists()!=failure_path.exists()
    terminal=read(result_path if result_path.exists() else failure_path)
    assert terminal['protocol_sha256']==PIN and 0<=terminal['optimizer_updates']<=800 and 0<=terminal['epochs']<=80
    assert exported['optimizer_updates']==terminal['optimizer_updates'] and exported['run_results_present']==result_path.exists() and exported['failure_present']==failure_path.exists()
    assert not terminal['app_promotion'] and not terminal['goal_complete']
    if result_path.exists():
        assert terminal['mean_centered_path_used'] and terminal['original_seven_loss_weights_unchanged']
        assert terminal['mean_centering_after_training_only'] is False and terminal['clipping_can_reintroduce_mean_shift']
    supervision=read(OUT/'supervisor_receipt.json');exit_code=int((OUT/'trainer_exit_code.txt').read_text())
    assert supervision['protocol_sha256']==PIN and supervision['cap_seconds']==2100 and supervision['kill_grace_seconds']==30
    assert supervision['trainer_exit_code']==exit_code and supervision['within_external_bound']==(supervision['seconds']<=2130)
    assert (exit_code==0)==result_path.exists()
    arrays=[];checked=0
    for path in sorted((OUT/'outputs/gradients').glob('batch*.npy')) if (OUT/'outputs/gradients').exists() else []:
        assert int(path.stem[5:])==len(arrays)
        a=matrix(np.load(path,allow_pickle=False));assert np.count_nonzero(a[3:])==0
        arrays.append(a);checked+=a.size
    summary_path=OUT/'outputs/gradient_summary.json';preflight_path=OUT/'outputs/gradient_preflight.json'
    if summary_path.exists():
        summary=read(summary_path);assert summary['terms']==p['terms'] and summary['parameter_layout']==p['parameter_layout'] and len(arrays)==len(summary['batches'])==10
        total=matrix(np.load(OUT/'outputs/gradient_components.npy',allow_pickle=False));checked+=total.size
        assert np.array_equal(total,sum(arrays,np.zeros_like(total))) and summary['gradient_array_sha256']==sha(OUT/'outputs/gradient_components.npy')
        values=np.zeros(7,np.float64)
        for i,(row,a) in enumerate(zip(summary['batches'],arrays)):
            assert row['batch']==i and row['ids']==[c['id'] for c in base['cases'][i*5:i*5+5]]
            assert row['gradient_array_sha256']==sha(OUT/'outputs/gradients'/('batch'+str(i)+'.npy'))
            assert row['initial_raw_and_PNG_parity_exact'] and row['initial_all12_preservation_gradients_exact_zero']
            assert np.allclose(np.linalg.norm(a,axis=1),row['component_norms'],rtol=2e-10,atol=1e-11) and row['component_values'][3:]==[0,0,0,0]
            values+=np.asarray(row['component_values'],np.float64)
        assert np.array_equal(values,np.asarray(summary['component_values'])) and summary['objective']==float(values.sum())
        norms,gram,cosine,blocks=statistics(total,p['parameter_layout'])
        for actual,saved in [(norms,summary['component_norms']),(gram,summary['component_gram']),(cosine,summary['component_cosines'])]:assert np.allclose(actual,saved,rtol=2e-10,atol=1e-11)
        for name,block in blocks.items():
            for key,value in block.items():assert np.allclose(value,summary['per_parameter_gradients'][name][key],rtol=2e-10,atol=1e-11)
        if preflight_path.exists():assert all(b['improvement_gradient_norm']>0 for b in blocks.values())
    proof=preflight_path.exists()
    if proof:
        q=read(preflight_path)
        assert q['complete'] and q['protocol_sha256']==PIN and q['parameter_layout']==p['parameter_layout']
        assert q['decoder_parameters']==498627 and q['decoder_parameter_tensors']==12 and q['raw_and_PNG_parity_exact_cases']==50
        assert q['all_selected12_improvement_gradients_nonzero'] and q['initial_preservation_gradients_exact_zero_all_batches']
        assert q['component_gradient_calls']==70 and q['reference_DGP_forwards']==q['candidate_DGP_forwards']==10 and q['recognizer_forwards']==20
        assert q['optimizer_updates']==q['epochs']==0 and not q['optimizer_constructed'] and not q['new_checkpoint_created']
        assert q['DGP_state_before_after']==p['original_DGP_state'] and q['recognizer_state_before_after']==p['frozen_recognizer_state']
        assert 0<q['seconds']<300 and q['peak_allocated_VRAM_bytes']<=p['budgets']['peak_vram_bytes']
        assert q['historical_R2_all14_failure_retained'] and not q['native_or_reserved_used']
    sys.path.insert(0,str(ROOT/'outputs/cctv_dgp_active_original_decoder_vm_v28'))
    sys.path.insert(0,str(PARENT));sys.path.insert(0,str(BUNDLE))
    replay,cohort,recognizer_count,original_state=initial_CPU_replay(start,p,base,proof)
    from cctv_dgp_pilot import FixedObservedIdentity,grid112,state_hash,exported_pixel_metrics
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_app_input_v28 import canonical_tensor
    from cctv_dgp_mean_centered_v29_training import frozen_partition
    from cctv_dgp_mean_centered_decoder_v29 import center_observed_delta
    import torch
    from PIL import Image
    from scipy.ndimage import convolve1d
    torch.set_num_threads(4)
    original=torch.load(PARENT/'weights/dgp_v2.pth',map_location='cpu',weights_only=True)
    selected={r['name'] for r in p['parameter_layout']}
    def checkpoint(path,initial=False):
        state=torch.load(path,map_location='cpu',weights_only=True)
        assert isinstance(state,dict) and set(state)==set(original)
        for name,value in state.items():
            assert isinstance(value,torch.Tensor) and value.dtype==original[name].dtype and value.shape==original[name].shape and bool(torch.isfinite(value).all())
            if name not in selected or initial:assert torch.equal(value,original[name]),'Frozen checkpoint partition changed: '+name
        return state
    recognizer=FixedObservedIdentity(PARENT/'weights/w600k_r50.onnx','cpu');recognizer_before=state_hash(recognizer)
    assert recognizer_before==p['frozen_recognizer_state'];refs={r['id']:r for r in base['references']}
    z=np.arange(-6,7,dtype=np.float64);kernel=np.exp(-.5*(z/2)**2);kernel/=kernel.sum()
    high=lambda a:a-convolve1d(convolve1d(a,kernel,axis=0,mode='reflect'),kernel,axis=1,mode='reflect')
    snapshot_rows=[];CPU_candidate=0;CPU_snapshot_recognizer=0
    for update in [0,50,400,800]:
        folder=OUT/('outputs/update'+str(update))
        if not folder.exists():continue
        assert proof,'Snapshots require selected12 proof first'
        saved=read(folder/'metrics.json');checkpoint(folder/'dgp_candidate_v29.pth',initial=update==0)
        net,_=load_frozen_dgp_restorer(folder/'dgp_candidate_v29.pth',expected_sha256=sha(folder/'dgp_candidate_v29.pth'),device='cpu')
        state_before=state_hash(net.net);frozen=frozen_partition(net.net,selected)
        assert state_before==saved['candidate_DGP_state'] and sha(folder/'dgp_candidate_v29.pth')==saved['candidate_checkpoint_sha256']
        assert frozen==saved['frozen_partition_sha256'] and saved['snapshot_batch_size']==5
        actual_rows=[];worst_raw=0.0;worst_vector=0.0
        assert len(saved['rows'])==50
        with torch.no_grad():
            for c,reported in zip(base['cases'],saved['rows']):
                assert time.monotonic()-start<900,'Local return audit cap900 seconds'
                assert reported['id']==c['id'] and reported['source']==c['source'] and reported['profile']==c['profile']
                with Image.open(PARENT/c['input']) as im:camera=np.asarray(im.convert('RGB')).copy()
                with Image.open(PARENT/c['target']) as im:target=np.asarray(im.convert('RGB')).copy()
                with Image.open(PARENT/c['observed']) as im:mask=np.asarray(im).copy()>0
                raw=np.load(folder/(c['id']+'.npy'),allow_pickle=False)
                assert raw.dtype==np.float32 and raw.shape==(256,256,3) and np.isfinite(raw).all() and (raw>=0).all() and (raw<=1).all()
                x=canonical_tensor(camera,'cpu');support=torch.from_numpy(mask)[None,None]
                baseline_raw=np.load(OUT/'outputs/initial_baseline'/(c['id']+'.npy'),allow_pickle=False)
                baseline_tensor=torch.from_numpy(baseline_raw.copy()).permute(2,0,1)[None]
                expected=center_observed_delta(torch.where(support,net(x),x),baseline_tensor,x,support)[0].permute(1,2,0).numpy().copy();CPU_candidate+=1
                error=float(np.abs(expected-raw).max());assert error<=1e-5;worst_raw=max(worst_raw,error)
                with Image.open(folder/(c['id']+'.png')) as im:png=np.asarray(im.convert('RGB')).copy()
                assert np.array_equal(png,np.where(mask[...,None],np.floor(raw*np.float32(255)),camera).astype(np.uint8))
                baseline=np.load(OUT/'outputs/initial_baseline'/(c['id']+'.npy'),allow_pickle=False)
                if update==0:assert np.array_equal(raw,baseline)
                vector=np.load(folder/(c['id']+'_embedding.npy'),allow_pickle=False)
                truth=np.load(OUT/'outputs/initial_baseline'/(c['id']+'_target_embedding.npy'),allow_pickle=False)
                assert vector.dtype==np.float32 and vector.shape==(512,) and np.isfinite(vector).all() and abs(float(vector@vector)-1)<1e-5
                grid=torch.from_numpy(grid112(refs[c['source_person_or_reference']]['matrix112']))[None]
                replay_vector=recognizer.embedding(canonical_tensor(png,'cpu'),support.float(),grid)[0].numpy().copy();CPU_snapshot_recognizer+=1
                verror=float(np.abs(vector-replay_vector).max());assert verror<=5e-5;worst_vector=max(worst_vector,verror)
                if update==0:assert np.array_equal(np.load(folder/(c['id']+'_target_embedding.npy'),allow_pickle=False),truth)
                metrics=exported_pixel_metrics(png,target,mask);metrics['ArcFace_observed_fixed']=float(vector@truth)
                padded=np.pad(mask.astype(np.int64),6);summed=np.pad(padded.cumsum(0).cumsum(1),((1,0),(1,0)))
                interior=summed[13:,13:]-summed[:-13,13:]-summed[13:,:-13]+summed[:-13,:-13]==169
                feature=np.zeros((256,256),bool)
                for point in c['landmarks5_canvas_xy']:
                    xx,yy=np.floor(point).astype(int);feature[max(0,yy-12):min(256,yy+12),max(0,xx-12):min(256,xx+12)]=True
                feature&=interior;luma=np.array([.299,.587,.114])
                metrics['landmark_high_frequency_MSE']=float(np.square(high((png.astype(np.float64)/255*luma).sum(2))-high((target.astype(np.float64)/255*luma).sum(2)))[feature].mean())
                shift=(raw-baseline)[mask].astype(np.float64).mean(0);mean=np.clip(baseline.astype(np.float64)+shift,0,1).astype(np.float32)
                mean_png=np.where(mask[...,None],np.floor(mean*np.float32(255)),camera).astype(np.uint8)
                metrics['constant_mean_shift_only_MSE']=exported_pixel_metrics(mean_png,target,mask)['MSE']
                assert np.array_equal(shift,np.asarray(reported['postclip_mean_RGB_shift']))
                for key,value in metrics.items():assert abs(value-reported['metrics'][key])<=1e-10,(c['id'],key)
                actual_rows.append({'id':c['id'],'source':c['source'],'profile':c['profile'],'metrics':metrics})
        derived=groups(actual_rows);assert set(derived)==set(saved['groups']) and len(derived)==17
        for name,g in derived.items():
            assert g['cases']==saved['groups'][name]['cases']
            for key,value in g.items():assert abs(value-saved['groups'][name][key])<=1e-10
        assert state_hash(net.net)==state_before and all(not v.requires_grad and v.grad is None for v in net.parameters())
        snapshot_rows.append({'update':update,'groups':saved['groups'],'CPU_raw_maximum_error':worst_raw,'CPU_vector_maximum_error':worst_vector,'frozen_partition_sha256':frozen})
    updates=[r['update'] for r in snapshot_rows];assert updates==[0,50,400,800][:len(updates)]
    assert state_hash(recognizer)==recognizer_before and all(not v.requires_grad and v.grad is None for v in recognizer.parameters())
    if (OUT/'outputs/stopped_dgp_candidate_v29.pth').exists():checkpoint(OUT/'outputs/stopped_dgp_candidate_v29.pth')
    if (OUT/'outputs/early_structure_stop.json').exists():
        early=read(OUT/'outputs/early_structure_stop.json');assert 0 in updates and 50 in updates
        gain=1-snapshot_rows[1]['groups']['degraded']['landmark_high_frequency_MSE']/snapshot_rows[0]['groups']['degraded']['landmark_high_frequency_MSE']
        assert early['update']==50 and early['minimum']==.01 and early['relative_feature_error_gain']==gain and early['pass']==(gain>=.01)
    if (OUT/'outputs/timing_update20.json').exists():
        timing=read(OUT/'outputs/timing_update20.json');samples=timing['steady_sample_seconds']
        assert timing['updates']==20 and len(samples)==19 and timing['remaining_updates']==780 and timing['safety_factor']==1.25 and timing['overhead_seconds']==120 and timing['cap_seconds']==1500
        assert timing['projected_seconds']==timing['seconds']+780*float(np.mean(samples))*1.25+120
    execution_path=OUT/'outputs/execution_receipt.json'
    if execution_path.exists():
        ex=read(execution_path);assert ex['complete'] and ex['protocol_sha256']==PIN
        for key in ['optimizer_updates','epochs','backwards','optimizer_constructed','new_checkpoint_created']:assert ex[key]==terminal[key]
        assert ex['fit_cap_seconds']==1500 and ex['worker_cap_seconds']==1800 and len(ex['step_times_seconds'])==terminal['optimizer_updates']
        assert all(x>=0 and np.isfinite(x) for x in ex['step_times_seconds']) and ex['epochs']==ex['optimizer_updates']//10
        assert ex['original_DGP_state']==p['original_DGP_state'] and ex['recognizer_state']==p['frozen_recognizer_state']
        assert not ex['app_promotion'] and not ex['goal_complete']
    necessary=False
    if result_path.exists():
        assert proof and updates==[0,50,400,800] and terminal['complete'] and terminal['updates']==terminal['optimizer_updates']==terminal['backwards']==800 and terminal['epochs']==80
        assert terminal['trained_parameters']==498627 and terminal['trained_tensors']==12
        failures,gain,source_gains,brightness,necessary=capacity(snapshot_rows[0]['groups'],snapshot_rows[-1]['groups'])
        assert failures==terminal['preservation_failures'] and gain==terminal['degraded_feature_MSE_relative_gain'] and source_gains==terminal['source_feature_gains']
        assert brightness==terminal['brightness_gain_fraction'] and necessary==terminal['necessary_capacity_pass']
        assert terminal['frozen_encoder_head4_and_all_buffers_unchanged'] and terminal['historical_R2_all14_failure_retained']
        assert ex['fit_seconds']<=1500 and ex['worker_seconds']<=1800 and ex['peak_allocated_VRAM_bytes']<=20*1024**3 and supervision['within_external_bound']
        assert ex['reference_DGP_forwards']==10 and ex['candidate_DGP_forwards']==850 and ex['recognizer_forwards']==860 and ex['component_gradient_calls']==70
    record={'complete':True,'checker_sha256':sha(Path(__file__)),'protocol_sha256':PIN,'archive_sha256':expected_sha,'archive_bytes':expected_bytes,
            'members_verified':imported['members'],'source_assets_verified':246,'selected12_preflight_complete':proof,
            'VM_failure_retained':failure_path.exists(),'training_completed800':result_path.exists(),'necessary_capacity_pass':necessary,
            'complete_gradient_matrices':len(arrays),'saved_gradient_values_checked':checked,'CPU_original_DGP_forwards':len(replay),
            'CPU_initial_fixed_recognizer_forwards':recognizer_count,'cohort_rows_checked':len(cohort),
            'CPU_candidate_DGP_forwards':CPU_candidate,'CPU_snapshot_recognizer_forwards':CPU_snapshot_recognizer,
            'snapshots_audited':snapshot_rows,'local_gradient_calls':0,'local_backward_calls':0,'local_optimizer_updates':0,
            'native_or_reserved_used':False,'visual_review_pending':True,'independent_final_review':False,
            'app_promotion':False,'goal_complete':False,'seconds':time.monotonic()-start}
    write(ROOT/'outputs/cctv_dgp_mean_centered_decoder_v29_independent_audit.json',record)
    print(json.dumps({k:record[k] for k in ['complete','selected12_preflight_complete','VM_failure_retained','training_completed800','necessary_capacity_pass','seconds']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--expected-sha',required=True);parser.add_argument('--expected-bytes',required=True,type=int)
    a=parser.parse_args();audit(a.expected_sha,a.expected_bytes)
