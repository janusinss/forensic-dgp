"""Safe return import, full gradient-array arithmetic and optional CPU inference.

Never execute returned source or run local derivatives. A completed export is not
training or restoration acceptance; actual VM differentiation assertions remain
source-bound evidence independently checked against every saved matrix element.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import time
from types import MethodType

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs/cctv_dgp_original_decoder_gradient_v1_r2_vm'
PARENT = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'
OUT = ROOT / 'outputs/cctv_dgp_original_decoder_gradient_v1_r2_return'
PIN = '81127e45a205c44ef685646a4f82911d02b2355dfe24c63772c23e62e66a41f2'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def validate_members(members,allowed,total_cap=900_000_000):
    seen=set();total=0;rows=[]
    prefix='cctv_dgp_original_decoder_gradient_v1_r2_return/'
    for member in members:
        assert member.isfile() and not member.issym() and not member.islnk(), 'Regular return members only'
        assert member.name.startswith(prefix) and '\\' not in member.name and ':' not in member.name
        name=member.name[len(prefix):]
        parts=PurePosixPath(name).parts
        assert parts and not PurePosixPath(name).is_absolute() and all(p not in ['.','..'] for p in parts)
        assert name in allowed and name.casefold() not in seen, 'Unexpected, duplicate or case-colliding member: '+name
        assert 0<=member.size<=64*1024**2,'Member bounds'
        seen.add(name.casefold());total+=member.size
        assert total<=total_cap and len(rows)<173,'Archive bounds'
        rows.append((member,name))
    return rows,total


def matrix(value,shape=(7,609219)):
    assert value.dtype==np.float64 and value.shape==shape and np.isfinite(value).all(),'Gradient matrix schema/nonfinite value'
    return value


def gradient_statistics(value,layout):
    value=matrix(value,(7,layout[-1]['end']))
    norms=np.linalg.norm(value,axis=1);gram=value@value.T
    denominator=norms[:,None]*norms[None,:]
    cosine=np.divide(gram,denominator,out=np.zeros_like(gram),where=denominator>0)
    blocks={}
    for row in layout:
        block=value[:,row['start']:row['end']]
        blocks[row['name']]={'component_norms':np.linalg.norm(block,axis=1).tolist(),
            'improvement_gradient_norm':float(np.linalg.norm(block[:3].sum(0))),
            'total_gradient_norm':float(np.linalg.norm(block.sum(0)))}
    return norms,gram,cosine,blocks


def allowed_names(base,p):
    allowed={'protocol.json','export_manifest.json','trainer.log','trainer_exit_code.txt','supervisor_receipt.json',*p['assets_sha256']}
    for case in base['cases']:
        for suffix in ['.npy','.png','_target_embedding.npy']:
            allowed.add('outputs/initial_baseline/'+case['id']+suffix)
    allowed.update('outputs/gradients/batch'+str(i)+'.npy' for i in range(10))
    allowed.update('outputs/'+name for name in ['gradient_components.npy','gradient_summary.json','cohort_loss_setup.json','results.json','failure.json'])
    return allowed


def import_return(expected_sha,expected_bytes,p,base):
    archive=ROOT/'outputs/cctv-dgp-original-decoder-gradient-v1-r2-results.tar.gz'
    assert re.fullmatch('[a-f0-9]{64}',expected_sha) and 0<expected_bytes<=900_000_000
    assert sha(archive)==expected_sha and archive.stat().st_size==expected_bytes,'Reported archive mismatch'
    sidecar=Path(str(archive)+'.sha256').read_text(encoding='ascii').strip().split()
    assert sidecar==[expected_sha,archive.name]
    exported=read(ROOT/'outputs/cctv-dgp-original-decoder-gradient-v1-r2-export.json')
    assert exported['complete'] and exported['archive_sha256']==expected_sha and exported['bytes']==expected_bytes
    assert exported['optimizer_updates']==0 and exported['training_success_not_implied']
    assert not OUT.exists(),'Preserve earlier import/partial evidence'
    with tarfile.open(archive,'r:gz') as tar:
        rows,total=validate_members(tar.getmembers(),allowed_names(base,p),p['budgets']['maximum_export_uncompressed_bytes'])
        OUT.mkdir()
        for member,name in rows:
            target=(OUT/name).resolve()
            assert target.is_relative_to(OUT.resolve())
            target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as source,target.open('xb') as destination:
                shutil.copyfileobj(source,destination)
            assert target.stat().st_size==member.size
    files={name:sha(OUT/name) for _,name in rows}
    assert sha(OUT/'protocol.json')==PIN
    for name,digest in p['assets_sha256'].items():assert files[name]==digest,name
    manifest=read(OUT/'export_manifest.json')
    assert manifest['complete'] and manifest['protocol_sha256']==PIN and manifest['quality_acceptance_not_implied']
    assert manifest['files_sha256']=={name:digest for name,digest in files.items() if name!='export_manifest.json'}
    record={'complete':True,'archive_sha256':expected_sha,'archive_bytes':expected_bytes,'members':len(rows),
            'uncompressed_bytes':total,'files_sha256':files,'neural_or_gradient_calls':0,'optimizer_updates':0,
            'app_promotion':False,'goal_complete':False}
    write(ROOT/'outputs/cctv_dgp_original_decoder_gradient_v1_r2_return_import.json',record)
    return exported,record


def audit(expected_sha,expected_bytes,replay=True):
    start=time.monotonic()
    assert sha(BUNDLE/'protocol.json')==PIN
    p=read(BUNDLE/'protocol.json');base=read(PARENT/'protocol.json')
    assert sha(PARENT/'protocol.json')==p['closed_V27_protocol_sha256']
    for name,digest in p['assets_sha256'].items():assert sha(BUNDLE/name)==digest,name
    for name,digest in base['assets_sha256'].items():assert sha(PARENT/name)==digest,name
    for name,digest in p['closed_V27_evidence_sha256'].items():
        assert sha(ROOT/'outputs/cctv_dgp_feature_skips_v27_return'/name)==digest
    assert p['retained_capacity_gates']==base['prospective_gates']
    exported,imported=import_return(expected_sha,expected_bytes,p,base)
    result_path=OUT/'outputs/results.json';failure_path=OUT/'outputs/failure.json'
    assert result_path.exists()!=failure_path.exists(),'Exactly one terminal result or failure is required'
    result=read(result_path if result_path.exists() else failure_path)
    assert result['protocol_sha256']==PIN and result['optimizer_updates']==result['epochs']==0
    assert not result['optimizer_constructed'] and not result['new_checkpoint_created'] and not result['app_promotion'] and not result['goal_complete']
    assert exported['run_results_present']==result_path.exists() and exported['failure_present']==failure_path.exists()
    supervision=read(OUT/'supervisor_receipt.json')
    assert supervision['protocol_sha256']==PIN and supervision['cap_seconds']==660 and supervision['kill_grace_seconds']==30
    assert supervision['within_external_bound']==(supervision['seconds']<=690)
    exit_code=int((OUT/'trainer_exit_code.txt').read_text())
    assert exit_code==supervision['trainer_exit_code'] and (exit_code==0)==result_path.exists()
    arrays=[];checked_values=0
    for path in sorted((OUT/'outputs/gradients').glob('batch*.npy')) if (OUT/'outputs/gradients').exists() else []:
        index=int(path.stem[len('batch'):]);assert index==len(arrays)
        a=matrix(np.load(path,allow_pickle=False))
        assert np.count_nonzero(a[3:])==0,'Every initial preservation gradient must be exactly zero'
        arrays.append(a);checked_values+=a.size
    summary_path=OUT/'outputs/gradient_summary.json'
    complete_proof=result_path.exists()
    total=None
    if summary_path.exists():
        summary=read(summary_path)
        assert summary['terms']==p['terms'] and summary['parameter_layout']==p['parameter_layout'] and len(summary['batches'])==len(arrays)==10
        total=matrix(np.load(OUT/'outputs/gradient_components.npy',allow_pickle=False));checked_values+=total.size
        assert np.array_equal(total,sum(arrays,np.zeros_like(total))), 'Gradient aggregation differs'
        assert summary['gradient_array_sha256']==sha(OUT/'outputs/gradient_components.npy')
        values=np.zeros(7,dtype=np.float64)
        for index,(row,a) in enumerate(zip(summary['batches'],arrays)):
            assert row['batch']==index and row['ids']==[c['id'] for c in base['cases'][index*5:index*5+5]]
            assert row['gradient_array_sha256']==sha(OUT/'outputs/gradients'/('batch'+str(index)+'.npy'))
            assert row['initial_raw_and_PNG_parity_exact'] and row['initial_all14_preservation_gradients_exact_zero']
            assert np.allclose(np.linalg.norm(a,axis=1),row['component_norms'],rtol=2e-10,atol=1e-11)
            assert row['component_values'][3:]==[0,0,0,0]
            values+=np.array(row['component_values'],dtype=np.float64)
        assert np.array_equal(values,np.array(summary['component_values'])) and summary['objective']==float(values.sum())
        norms,gram,cosine,blocks=gradient_statistics(total,p['parameter_layout'])
        for actual,saved in [(norms,summary['component_norms']),(gram,summary['component_gram']),(cosine,summary['component_cosines'])]:
            assert np.allclose(actual,saved,rtol=2e-10,atol=1e-11)
        for name,block in blocks.items():
            for key,value in block.items():assert np.allclose(value,summary['per_parameter_gradients'][name][key],rtol=2e-10,atol=1e-11)
        if complete_proof:assert all(b['improvement_gradient_norm']>0 for b in blocks.values())
    if complete_proof:
        assert total is not None and result['complete'] and result['decoder_parameters']==609219 and result['decoder_parameter_tensors']==14
        assert result['parameter_layout']==p['parameter_layout']
        assert result['raw_and_PNG_parity_exact_cases']==50 and result['all14_improvement_gradients_nonzero']
        assert result['initial_preservation_gradients_exact_zero_all_batches'] and result['original_V27_failure_retained']
        assert result['reference_DGP_forwards']==result['candidate_DGP_forwards']==10
        assert result['recognizer_forwards']==20 and result['component_gradient_calls']==70
        assert result['DGP_state_before_after']==p['original_DGP_state'] and result['recognizer_state_before_after']==p['frozen_recognizer_state']
        assert 0<=result['seconds']<600 and result['peak_allocated_VRAM_bytes']<=p['budgets']['peak_vram_bytes']
        assert supervision['within_external_bound'] and exported['seconds']<90 and not result['native_or_reserved_used']
        assert not result['automatic_follow_on']
    replay_rows=[];cohort_rows=[];CPU_recognizer_forwards=0
    baseline_folder=OUT/'outputs/initial_baseline'
    state_before=None
    if replay and baseline_folder.exists():
        # Replay the verified local model; never import any returned .py file.
        sys.path.insert(0,str(PARENT))
        import torch
        from PIL import Image
        from cctv_dgp_pilot import FixedObservedIdentity,grid112,state_hash
        from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
        from dgp_face_restoration import as_tensor
        torch.set_num_threads(4)
        original,_=load_frozen_dgp_restorer(PARENT/'weights/dgp_v2.pth',expected_sha256=base['assets_sha256']['weights/dgp_v2.pth'],device='cpu')
        state_before=state_hash(original.net)
        assert state_before==p['original_DGP_state']
        identity=FixedObservedIdentity(PARENT/'weights/w600k_r50.onnx','cpu')
        identity_before=state_hash(identity)
        assert identity_before==p['frozen_recognizer_state']
        refs={ref['id']:ref for ref in base['references']}
        import ast
        from torch.nn import functional as F
        tree=ast.parse((PARENT/'cctv_dgp_feature_skips_v27.py').read_text(encoding='utf-8'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SpatialFeatureHead')
        definitions=[next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name) for name in ['blur','high']]
        namespace={'torch':torch,'F':F}
        exec(compile(ast.Module(body=definitions,type_ignores=[]),'<verified-original-filter-only>','exec'),namespace)
        class Filter:pass
        fixed=Filter();z=torch.arange(-6,7,dtype=torch.float32)
        kernel=torch.exp(-.5*(z/2).square());fixed.kernel=kernel/kernel.sum()
        fixed.reflect_indices=torch.cat((torch.arange(5,-1,-1),torch.arange(256),torch.arange(255,249,-1)))
        fixed.blur=MethodType(namespace['blur'],fixed);fixed.high=MethodType(namespace['high'],fixed)
        with torch.no_grad():
            for case in base['cases']:
                path=baseline_folder/(case['id']+'.npy')
                if not path.exists():continue
                assert time.monotonic()-start<600,'Local source/array/replay cap600 seconds'
                raw=np.load(path,allow_pickle=False)
                assert raw.shape==(256,256,3) and raw.dtype==np.float32 and np.isfinite(raw).all() and (raw>=0).all() and (raw<=1).all()
                with Image.open(PARENT/case['input']) as im:camera=np.asarray(im.convert('RGB')).copy()
                with Image.open(PARENT/case['observed']) as im:mask=np.asarray(im).copy()>0
                expected=original(as_tensor(camera))[0].permute(1,2,0).numpy().copy()
                expected=np.where(mask[...,None],expected,camera.astype(np.float32)/np.float32(255))
                error=float(np.abs(expected[mask]-raw[mask]).max())
                assert error<=p['CPU_CUDA_raw_replay_tolerance'],'CPU/CUDA baseline replay: '+case['id']
                with Image.open(baseline_folder/(case['id']+'.png')) as im:delivered=np.asarray(im.convert('RGB')).copy()
                assert np.array_equal(delivered,np.where(mask[...,None],np.floor(raw*np.float32(255)),camera).astype(np.uint8))
                truth=np.load(baseline_folder/(case['id']+'_target_embedding.npy'),allow_pickle=False)
                assert truth.dtype==np.float32 and truth.shape==(512,) and np.isfinite(truth).all() and abs(float(truth@truth)-1)<1e-5
                with Image.open(PARENT/case['target']) as im:target=np.asarray(im.convert('RGB')).copy()
                tensor_mask=torch.from_numpy(mask.astype(np.float32))[None,None]
                grid=torch.from_numpy(grid112(refs[case['source_person_or_reference']]['matrix112']))[None]
                replay_truth=identity.embedding(as_tensor(target),tensor_mask,grid)[0].numpy().copy()
                CPU_recognizer_forwards+=1
                truth_error=float(np.abs(replay_truth-truth).max())
                assert truth_error<=5e-5,'Fixed CPU/CUDA vector compatibility; no quality margin change'
                # Recompute the fixed baseline scalars from exact returned float32 raw.
                def erode(radius):
                    size=2*radius+1;padded=np.pad(mask.astype(np.int64),radius)
                    summed=np.pad(padded.cumsum(0).cumsum(1),((1,0),(1,0)))
                    return summed[size:,size:]-summed[:-size,size:]-summed[size:,:-size]+summed[:-size,:-size]==size*size
                interior=erode(6);feature=np.zeros((256,256),bool)
                for point in case['landmarks5_canvas_xy']:
                    xx,yy=np.floor(point).astype(int)
                    feature[max(0,yy-12):min(256,yy+12),max(0,xx-12):min(256,xx+12)]=True
                feature&=interior
                weights=torch.tensor([.299,.587,.114],dtype=torch.float32)[None,:,None,None]
                base_tensor=torch.from_numpy(raw.copy()).permute(2,0,1)[None]
                delta=fixed.high((base_tensor*weights).sum(1,keepdim=True))-fixed.high((as_tensor(target)*weights).sum(1,keepdim=True))
                fmask=torch.from_numpy(feature.astype(np.float32))[None,None]
                imask=torch.from_numpy(interior.astype(np.float32))[None,None]
                cohort_rows.append({'id':case['id'],'clear':case['profile']=='clear',
                    'feature_MSE':float((delta.square()*fmask).sum()/fmask.sum()),
                    'interior_MSE':float((delta.square()*imask).sum()/imask.sum())})
                replay_rows.append({'id':case['id'],'maximum_CPU_CUDA_raw_error':error,
                                    'maximum_CPU_CUDA_target_vector_error':truth_error,'exact_raw_to_PNG':True})
        assert state_hash(original.net)==state_before and all(not v.requires_grad and v.grad is None for v in original.parameters())
        assert state_hash(identity)==identity_before and all(not v.requires_grad and v.grad is None for v in identity.parameters())
        if complete_proof:assert len(replay_rows)==50
        if (OUT/'outputs/cohort_loss_setup.json').exists():
            cohort=read(OUT/'outputs/cohort_loss_setup.json')
            assert len(cohort_rows)==len(cohort['rows'])==50 and cohort['clear_controls']==10 and cohort['degraded_cases']==40
            assert cohort['normalizer_floor']==1e-6 and not cohort['clear_reward'] and not cohort['optimizer_constructed']
            assert cohort['degraded_weight']==1.25 and cohort['clear_baseline_anchor_weight']==.05 and cohort['fixed_high_pass_calls']==100
            for actual,saved in zip(cohort_rows,cohort['rows']):
                assert actual['id']==saved['id'] and actual['clear']==saved['clear']
                for key in ['feature_MSE','interior_MSE']:assert abs(actual[key]-saved[key])<=2e-9
            for key,field in [('feature_MSE','feature_normalizer'),('interior_MSE','interior_normalizer')]:
                scalar=np.float32(sum(row[key] for row in cohort['rows'] if not row['clear'])/40)
                assert float(max(scalar,np.float32(1e-6)))==cohort[field]
    audit={'complete':True,'protocol_sha256':PIN,'checker_sha256':sha(Path(__file__)),'archive_sha256':expected_sha,'archive_bytes':expected_bytes,
        'members_verified':imported['members'],'source_assets_verified':len(base['assets_sha256']),
        'VM_proof_complete':complete_proof,'VM_failure_retained':failure_path.exists(),
        'complete_batch_gradient_matrices':len(arrays),'saved_gradient_values_independently_checked':checked_values,
        'gradient_evidence_limit':'All saved values, source assertions, counts, states, dimensions, component/parameter statistics and zero initial preservation gradients checked; local derivatives are not rerun. Forward-only CPU replay is a separate numerical compatibility check.',
        'CPU_original_DGP_forwards':len(replay_rows),'CPU_fixed_recognizer_forwards':CPU_recognizer_forwards,
        'fixed_float32_cohort_baseline_rows_checked':len(cohort_rows),'CPU_baseline_replay':replay_rows,'CPU_DGP_state_before_after':state_before,
        'numerical_replay_bounds':'Original raw1e-5; fixed target-vector5e-5; baseline float32 scalar2e-9. Numerical compatibility only; all restoration quality thresholds unchanged.',
        'local_gradient_calls':0,'local_backward_calls':0,'local_optimizer_updates':0,
        'diagnostic_pass_is_not_restoration_or_training_acceptance':True,'next_training_recipe_created':False,
        'native_or_reserved_used':False,'independent_final_review':False,'app_promotion':False,'goal_complete':False,
        'seconds':time.monotonic()-start}
    write(ROOT/'outputs/cctv_dgp_original_decoder_gradient_v1_r2_independent_audit.json',audit)
    print(json.dumps({k:audit[k] for k in ['complete','VM_proof_complete','VM_failure_retained','complete_batch_gradient_matrices',
        'saved_gradient_values_independently_checked','CPU_original_DGP_forwards','seconds']},indent=2))
    return audit


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--expected-sha',required=True)
    parser.add_argument('--expected-bytes',type=int,required=True)
    a=parser.parse_args();audit(a.expected_sha,a.expected_bytes)


if __name__=='__main__':main()
