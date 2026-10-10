"""Frozen incoming-file, saved-gradient and initial-output audit. No autograd."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import sys
import tarfile
import time
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
PACKET=ROOT/'outputs/cctv_dgp_head4_reactivation_vm_v1'
sys.path.insert(0,str(PACKET))
from cctv_dgp_head4_reactivation_contract_v1 import NAME,STEM,BUDGETS,PARTS,COMPONENTS,STATE,read,write,sha,validate
RETURN=ROOT/'outputs'/(NAME+'_return')
RECEIPT=ROOT/'outputs/cctv_dgp_head4_reactivation_v1_independent_audit_r2.json'
REPAIR=ROOT/'outputs/cctv_dgp_head4_reactivation_v1_return_audit_failure/r2_plan.json'


def safe_member(member):
    name=member.name; path=PurePosixPath(name)
    assert '\\' not in name and ':' not in name and not path.is_absolute() and name==path.as_posix()
    assert '..' not in path.parts and path.parts and path.parts[0]==NAME+'_return'
    assert member.isfile() and not member.issym() and not member.islnk()
    assert member.size>=0
    return path


def close(a,b,atol=1e-12,rtol=1e-12):
    assert np.isfinite(a) and np.isfinite(b) and abs(a-b)<=atol+rtol*max(abs(a),abs(b)),(a,b)


def audit():
    start=time.monotonic(); p=read(PACKET/'protocol.json');validate(p);pin=sha(PACKET/'protocol.json')
    assert not RECEIPT.exists() and RETURN.is_dir(),'Preserve prior extraction; new R2 receipt only'
    repair=read(REPAIR)
    assert repair['new_checker_sha256']==sha(Path(__file__))
    for name,digest in repair['source_bindings'].items():assert sha(ROOT/name)==digest,name
    for name,digest in p['local_source_bindings'].items():assert sha(ROOT/name)==digest,name
    for name,digest in p['assets_sha256'].items():assert sha(PACKET/name)==digest,name
    archive=ROOT/'outputs'/(STEM+'-results.tar.gz')
    export=read(ROOT/'outputs'/(STEM+'-export.json'))
    sidecar=Path(str(archive)+'.sha256').read_text(encoding='ascii').strip().split()
    assert sidecar==[export['archive_sha256'],archive.name]
    assert export['complete'] and export['optimizer_updates']==0 and export['training_success_not_implied']
    assert archive.stat().st_size==export['bytes'] and sha(archive)==export['archive_sha256']
    assert archive.stat().st_size<=BUDGETS['return_uncompressed_bytes']+1024**2
    with tarfile.open(archive,'r:gz') as tar:
        members=[];names=set();total=0
        for member in tar:
            safe_member(member);assert member.name not in names;names.add(member.name);total+=member.size
            members.append(member)
            assert len(members)<=400 and total<=BUDGETS['return_uncompressed_bytes']+1024**2
        assert members
        prefix=NAME+'_return/'
        protocols=read_stream(tar,prefix+'protocol.json');manifest=read_stream(tar,prefix+'export_manifest.json')
        assert protocols==p and manifest['complete'] and manifest['protocol_sha256']==pin
        assert names=={prefix+n for n in manifest['files_sha256']}|{prefix+'export_manifest.json'}
        for member in members:
            if member.name==prefix+'export_manifest.json':continue
            with tar.extractfile(member) as stream:
                h=hashlib.sha256()
                for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
            assert h.hexdigest()==manifest['files_sha256'][member.name[len(prefix):]],member.name
        assert manifest['uncompressed_bytes']==total-tar.getmember(prefix+'export_manifest.json').size
        # R1 already extracted only after validating every archive member.
        # Revalidate the exact retained bytes; do not extract or overwrite again.
        actual={f.relative_to(RETURN).as_posix() for f in RETURN.rglob('*') if f.is_file()}
        assert actual==set(manifest['files_sha256'])|{'export_manifest.json'}
        for name,digest in manifest['files_sha256'].items():assert sha(RETURN/name)==digest,name
        assert read(RETURN/'export_manifest.json')==manifest
    assert sha(RETURN/'protocol.json')==pin
    for name in p['assets_sha256']:
        if name.endswith(('.py','.sh')):assert sha(RETURN/name)==p['assets_sha256'][name]
    supervision=read(RETURN/'supervisor_receipt.json')
    assert supervision['protocol_sha256']==pin and supervision['optimizer_updates']==0
    assert supervision['external_cap_seconds']==630 and supervision['kill_grace_seconds']==30
    assert supervision['within_external_bound']==(supervision['seconds']<=665)
    assert int((RETURN/'diagnostic_exit_code.txt').read_text())==supervision['diagnostic_exit_code']
    result_path=RETURN/'outputs/results.json';failed=(RETURN/'outputs/failure.json').exists()
    receipt={'complete':True,'protocol_sha256':pin,'archive_sha256':export['archive_sha256'],
        'archive_members_verified':len(members),'uncompressed_bytes':total,
        'checker_sha256':sha(Path(__file__)),'gradient_queries_locally_replayed':0,'optimizer_updates':0,
        'model_qualification':False,'app_promotion':False,'goal_complete':False,'failure_preserved':failed}
    assert export['failure_present']==failed and export['run_results_present']==result_path.exists()
    if not result_path.exists():
        assert failed and supervision['diagnostic_exit_code']!=0
        failure=read(RETURN/'outputs/failure.json');assert failure['optimizer_updates']==0
        assert failure['progress']['optimizer_updates']==failure['progress']['model_parameter_updates']==0
        assert failure['progress']['gradient_queries']<=160
        receipt.update({'diagnostic_completed':False,'partial_queries':failure['progress']['gradient_queries'],
            'seconds':time.monotonic()-start})
        write(RECEIPT,receipt);print({'artifact_audit_complete':True,'diagnostic_completed':False,'failure_preserved':True});return
    r=read(result_path);assert r['complete'] and r['protocol_sha256']==pin
    assert r['states_before']==r['states_after']
    assert r['states_before']['source']==r['states_before']['original']==STATE
    assert r['states_before']['repaired']==p['initial_repaired_state']
    assert r['states_before']['recognizer']==p['recognizer_state']
    assert r['forward_counts']=={'original':20,'repaired':20,'recognizer':60}
    assert r['progress']=={'gradient_queries':160,'optimizer_updates':0,'epochs':0,'optimizer_constructed':False,
        'model_parameter_updates':0,'completed_batches':20}
    assert r['optimizer'] is r['scheduler'] is None and r['optimizer_updates']==r['epochs']==0
    assert r['all_individual_gradient_vectors_saved'] and r['seconds']<=600
    for key in ['new_trained_checkpoint','capacity_or_quality_requirements_tested','native_DEV_or_reserved_final_used','app_promotion','goal_complete']:
        assert r[key] is False
    vectors=np.load(RETURN/'outputs/individual_gradients.npy',mmap_mode='r',allow_pickle=False)
    assert vectors.dtype==np.float64 and list(vectors.shape)==p['gradient_shape'] and np.isfinite(vectors).all()
    parts=np.cumsum([0]+[n for _,n in PARTS]);worst=0.;query_index=0
    for batch in range(20):
        for model_index,label in enumerate(['original','repaired']):
            for component_index,component in enumerate(COMPONENTS):
                row=r['gradient_queries'][query_index];query_index+=1
                assert (row['model'],row['batch'],row['component'])==(label,batch,component)
                assert row['case_ids']==[c['id'] for c in p['cases'][batch*5:batch*5+5]]
                for part in range(3):
                    values=np.asarray(vectors[model_index,batch,component_index,parts[part]:parts[part+1]])
                    norm=float(np.sqrt(np.dot(values,values)));stored=row['part_L2'][part]
                    close(norm,stored);worst=max(worst,abs(norm-stored))
                    assert np.count_nonzero(values)==row['part_nonzero_elements'][part]
    expected_summaries=[]
    for model_index,label in enumerate(['original','repaired']):
        for cohort_index,co in enumerate(p['cohorts']):
            for part_index,(name,_) in enumerate(PARTS):
                a=np.asarray(vectors[model_index,cohort_index*10:(cohort_index+1)*10,[0,2,3],parts[part_index]:parts[part_index+1]])
                expected_summaries.append({'model':label,'cohort':co['name'],'part':name,
                    'improvement_gradient_L2':float(np.linalg.norm(a.reshape(-1))),
                    'nonzero_elements':int(np.count_nonzero(a))})
    assert len(expected_summaries)==len(r['connected_gradient_summaries'])
    summary_arithmetic_error=0.
    for fresh,saved in zip(expected_summaries,r['connected_gradient_summaries']):
        assert set(fresh)==set(saved)
        for key in ['model','cohort','part','nonzero_elements']:assert fresh[key]==saved[key]
        a,b=fresh['improvement_gradient_L2'],saved['improvement_gradient_L2']
        assert (a>0)==(b>0), 'Exact route sign decision differs'
        if a==0 or b==0:assert a==b==0
        close(a,b,atol=1e-14,rtol=0)
        summary_arithmetic_error=max(summary_arithmetic_error,abs(a-b))
    route=all(s['improvement_gradient_L2']>0 for s in expected_summaries if s['model']=='repaired')
    assert route==r['connected_improvement_route_pass']
    if route:assert supervision['diagnostic_exit_code']==0 and not failed
    else:assert supervision['diagnostic_exit_code']!=0 and failed
    assert [row['id'] for row in r['initial_cases']]==[c['id'] for c in p['cases']]
    from frozen_raw_metrics import deliver,pixel_metrics,detail_float
    from frozen_capacity_contract import feature_support
    refs={ref['id']:ref for ref in p['references']};case_values={};recomputed_losses={}
    for c,row in zip(p['cases'],r['initial_cases']):
        cid=c['id'];ref=refs[c['source_person_or_reference']]
        paths={suffix:RETURN/'outputs/initial_outputs'/(cid+suffix) for suffix in ['.npy','.png','_embeddings.npy']}
        assert sha(paths['.npy'])==row['raw_sha256'] and sha(paths['.png'])==row['png_sha256']
        raw=np.load(paths['.npy'],allow_pickle=False);em=np.load(paths['_embeddings.npy'],allow_pickle=False)
        assert raw.dtype==np.float32 and raw.shape==(256,256,3) and np.isfinite(raw).all() and raw.min()>=0 and raw.max()<=1
        assert em.dtype==np.float32 and em.shape==(3,512) and np.isfinite(em).all() and np.array_equal(em[0],em[1])
        camera=pixels(PACKET/c['input']);target=pixels(PACKET/ref['target']);mask=pixels(PACKET/ref['observed'],'L')>0
        assert np.array_equal(pixels(paths['.png']),deliver(raw,camera,mask))
        assert np.array_equal(raw[~mask],camera[~mask].astype(np.float32)/np.float32(255))
        assert row['original_repaired_raw_max_abs']==row['original_repaired_embedding_max_abs']==0
        assert row['source']==c['source'] and row['profile']==c['profile']
        values=pixel_metrics(raw,target,mask)
        case_values[cid]={'MSE':values['MSE'],'SSIM_loss':1-values['SSIM'],
            'ArcFace_loss':1-float(np.dot(em[1].astype(np.float64),em[2].astype(np.float64))),
            'structure':detail_float(raw,target,feature_support(mask,c['landmarks5_canvas_xy']))}
    for batch in range(20):
        values=[case_values[c['id']] for c in p['cases'][batch*5:batch*5+5]]
        expected=[float(np.mean([v[k] for v in values])) for k in ['MSE','SSIM_loss','ArcFace_loss']]
        expected.append(float(np.mean([v['structure'] for v in values[1:]])))
        for model_index in range(2):
            for component in range(4):
                saved=r['gradient_queries'][batch*8+model_index*4+component]['loss']
                close(saved,expected[component],atol=[1e-12,3e-5,1e-12,1e-10][component],rtol=0)
    assert r['original_zero_head4_cases']==sum(row['original_head4_zero'] for row in r['initial_cases'])
    assert r['repaired_positive_head4_cases']==sum(row['repaired_head4_positive_activations']>0 for row in r['initial_cases'])
    # Two frozen CPU replays; no optimizer or autograd constructed locally.
    import torch
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_head4_reactivation_v1 import ReactivatedDGP
    from cctv_dgp_pilot import FixedObservedIdentity,grid112,state_hash
    torch.set_num_threads(4)
    original,_=load_frozen_dgp_restorer(PACKET/'weights/dgp_v2.pth',expected_sha256=p['original_checkpoint_sha256'],device='cpu')
    repaired=ReactivatedDGP(original.net).eval().requires_grad_(False)
    identity=FixedObservedIdentity(PACKET/'weights/w600k_r50.onnx','cpu');replays=[]
    assert state_hash(repaired)==p['initial_repaired_state']
    by_id={c['id']:c for c in p['cases']}
    with torch.inference_mode():
        for cid in p['independent_replay_case_ids']:
            c=by_id[cid];ref=refs[c['source_person_or_reference']]
            camera=pixels(PACKET/c['input']);mask8=pixels(PACKET/ref['observed'],'L')>0
            x=rgb_tensor(camera,torch);target=rgb_tensor(pixels(PACKET/ref['target']),torch)
            mask=torch.from_numpy(mask8.astype(np.float32))[None,None];grid=torch.from_numpy(grid112(ref['matrix112']))[None]
            a=original.net(x);b=repaired(x);assert torch.equal(a,b)
            b=torch.where(mask.bool(),b,x);raw=b[0].permute(1,2,0).numpy()
            old=np.load(RETURN/'outputs/initial_outputs'/(cid+'.npy'),allow_pickle=False)
            error=float(np.abs(raw-old).max());assert error<=3e-6
            embeddings=identity.embedding(torch.cat([b,target]),torch.cat([mask,mask]),torch.cat([grid,grid])).numpy()
            previous=np.load(RETURN/'outputs/initial_outputs'/(cid+'_embeddings.npy'),allow_pickle=False)
            ee=float(np.abs(embeddings-previous[[1,2]]).max());assert ee<=5e-5
            replays.append({'id':cid,'raw_CPU_VM_max_abs':error,'embedding_CPU_VM_max_abs':ee})
    assert state_hash(original.net)==STATE and state_hash(repaired)==p['initial_repaired_state']
    for name,digest in p['assets_sha256'].items():assert sha(PACKET/name)==digest,name
    for name,digest in p['local_source_bindings'].items():assert sha(ROOT/name)==digest,name
    receipt.update({'diagnostic_completed':True,'connected_route_pass':route,'saved_gradient_vectors_verified':160,
        'gradient_part_norms_verified':480,'maximum_independent_norm_arithmetic_error':worst,
        'all100_initial_raw_PNG_and_embedding_records_verified':True,'initial_loss_batches_verified':40,
        'fresh_CPU_replays':replays,'fresh_local_DGP_calls':4,'fresh_local_recognizer_calls':2,
        'all100_independently_replayed':False,'states_unchanged':True,'capacity_or_quality_qualified':False,
        'aggregate_norm_recomputation_atol':1e-14,'maximum_summary_norm_arithmetic_error':summary_arithmetic_error,
        'all_summary_identity_counts_and_route_sign_decisions_exact':True,
        'original_checker_failure_preserved':True,'no_reextraction_or_training_rerun':True,
        'repair_plan_sha256':sha(REPAIR),'seconds':time.monotonic()-start})
    write(RECEIPT,receipt);print({'complete':True,'route_pass':route,'vectors_verified':160,'fresh_replays':2,'optimizer_updates':0},flush=True)


def read_stream(tar,name):
    with tar.extractfile(tar.getmember(name)) as stream:return json.loads(stream.read().decode('utf-8'))
def pixels(path,mode='RGB'):
    with Image.open(path) as image:return np.asarray(image.convert(mode)).copy()
def rgb_tensor(array,torch):return torch.from_numpy(array.astype(np.float32)/np.float32(255)).permute(2,0,1)[None]


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.parse_args();audit()
