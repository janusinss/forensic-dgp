"""Prospective zero-update diagnostic return audit; no returned code or local gradients."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'outputs/cctv_dgp_v28_preservation_diagnostic_v1_vm'
PARENT=ROOT/'outputs/cctv_dgp_feature_skips_vm_v27'
V28=ROOT/'outputs/cctv_dgp_active_original_decoder_v28_return'
OUT=ROOT/'outputs/cctv_dgp_v28_preservation_diagnostic_v1_return'
PREFIX='cctv_dgp_v28_preservation_diagnostic_v1_return/'
STEM='cctv-dgp-v28-preservation-diagnostic-v1'


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def matrix(value):
    assert value.dtype==np.float64 and value.shape==(10,498627) and np.isfinite(value).all()
    return value


def allowed_names(p):
    names={'protocol.json','export_manifest.json','trainer.log','trainer_exit_code.txt','supervisor_receipt.json',*p['assets_sha256']}
    names|={'outputs/'+name for name in ['results.json','failure.json','cohort_loss_setup.json','gradient_summary.json','gradient_components.npy','parameter_displacement.npy']}
    names|={'outputs/gradients/batch'+str(i)+'.npy' for i in range(10)}
    for c in p['case_rows']:
        names|={'outputs/initial_baseline/'+c['id']+suffix for suffix in ['.npy','.png','_target_embedding.npy']}
        names|={'outputs/final_replay/'+c['id']+suffix for suffix in ['.npy','.png','_embedding.npy']}
    return names


def members_checked(members,allowed):
    rows=[];seen=set();total=0
    for m in members:
        assert m.isfile() and not m.issym() and not m.islnk()
        assert m.name.startswith(PREFIX) and '\\' not in m.name and ':' not in m.name
        name=m.name[len(PREFIX):];parts=PurePosixPath(name).parts
        assert parts and not PurePosixPath(name).is_absolute() and all(v not in ['.','..'] for v in parts)
        assert name in allowed and name.casefold() not in seen and 0<=m.size<=64*1024**2
        seen.add(name.casefold());total+=m.size;assert total<=600_000_000 and len(rows)<400
        rows.append((m,name))
    return rows,total


def import_return(expected_sha,expected_bytes,p):
    archive=ROOT/'outputs'/(STEM+'-results.tar.gz');export=read(ROOT/'outputs'/(STEM+'-export.json'))
    assert archive.stat().st_size==expected_bytes==export['bytes'] and sha(archive)==expected_sha==export['archive_sha256']
    assert Path(str(archive)+'.sha256').read_text(encoding='ascii').strip()==expected_sha+'  '+archive.name
    assert export['complete'] and export['optimizer_updates']==0 and export['training_success_not_implied']
    assert not OUT.exists(),'Preserve every earlier or partial import'
    with tarfile.open(archive,'r:gz') as tar:
        rows,total=members_checked(tar.getmembers(),allowed_names(p));OUT.mkdir()
        for m,name in rows:
            target=OUT/name;target.parent.mkdir(parents=True,exist_ok=True);stream=tar.extractfile(m);assert stream
            with stream,target.open('xb') as f:
                for block in iter(lambda:stream.read(1024**2),b''):f.write(block)
    manifest=read(OUT/'export_manifest.json');assert manifest['complete'] and manifest['protocol_sha256']==sha(BUNDLE/'protocol.json')
    names={name for _,name in rows};assert set(manifest['files_sha256'])==names-{'export_manifest.json'}
    bindings={name:sha(OUT/name) for name in names}
    for name,digest in manifest['files_sha256'].items():assert bindings[name]==digest
    assert bindings['protocol.json']==sha(BUNDLE/'protocol.json')
    for name,digest in p['assets_sha256'].items():assert bindings[name]==digest
    receipt={'complete':True,'archive_sha256':expected_sha,'archive_bytes':expected_bytes,'members':len(rows),'uncompressed_bytes':total,'files_sha256':bindings}
    write(ROOT/'outputs/cctv_dgp_v28_preservation_diagnostic_v1_return_import.json',receipt)
    return export,receipt


def audit(expected_sha,expected_bytes):
    start=time.monotonic();p=read(BUNDLE/'protocol.json');base=read(PARENT/'protocol.json')
    for name,digest in p['assets_sha256'].items():assert sha(BUNDLE/name)==digest
    for name,digest in p['local_basis_sha256'].items():assert sha(ROOT/name)==digest
    for name,digest in base['assets_sha256'].items():assert sha(PARENT/name)==digest
    for name,digest in p['closed_V28_evidence_sha256'].items():assert sha(V28/name)==digest
    export,imported=import_return(expected_sha,expected_bytes,p)
    result=OUT/'outputs/results.json';failure=OUT/'outputs/failure.json';assert result.exists()!=failure.exists()
    terminal=read(result if result.exists() else failure);pin=sha(BUNDLE/'protocol.json')
    assert terminal['protocol_sha256']==pin and terminal['optimizer_updates']==terminal['epochs']==0
    assert terminal['backwards']==0 and terminal['optimizer_constructed'] is False and terminal['new_checkpoint_created'] is False
    assert terminal['new_training_recipe_created'] is False and terminal['app_promotion'] is False and terminal['goal_complete'] is False
    assert export['run_results_present']==result.exists() and export['failure_present']==failure.exists()
    supervision=read(OUT/'supervisor_receipt.json');exit_code=int((OUT/'trainer_exit_code.txt').read_text())
    assert supervision['protocol_sha256']==pin and supervision['cap_seconds']==330 and supervision['kill_grace_seconds']==30
    assert supervision['trainer_exit_code']==exit_code and supervision['within_external_bound']==(supervision['seconds']<=360)
    assert (exit_code==0)==result.exists()
    original_path=PARENT/'weights/dgp_v2.pth';final_path=V28/'outputs/update800/dgp_candidate_v28.pth'
    sys.path.insert(0,str(PARENT));sys.path.insert(0,str(ROOT/'outputs/cctv_dgp_active_original_decoder_vm_v28'))
    import torch
    from PIL import Image
    from cctv_dgp_pilot import FixedObservedIdentity,grid112,state_hash
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_app_input_v28 import canonical_tensor
    torch.set_num_threads(4)
    original_state=torch.load(original_path,map_location='cpu',weights_only=True)
    final_state=torch.load(final_path,map_location='cpu',weights_only=True)
    assert set(original_state)==set(final_state) and sha(final_path)==p['final_V28_checkpoint_sha256']
    selected={row['name'] for row in p['parameter_layout']}
    for name,value in final_state.items():
        assert value.dtype==original_state[name].dtype and value.shape==original_state[name].shape and bool(torch.isfinite(value).all())
        if name not in selected:assert torch.equal(value,original_state[name])
    # Named-parameter order is the prospectively frozen layout, not state-dict aliases.
    displacement=np.concatenate([(final_state[row['name']].double()-original_state[row['name']].double()).numpy().reshape(-1) for row in p['parameter_layout']])
    if (OUT/'outputs/parameter_displacement.npy').exists():
        assert np.array_equal(displacement,np.load(OUT/'outputs/parameter_displacement.npy',allow_pickle=False))
    arrays=[];values_checked=0
    for path in sorted((OUT/'outputs/gradients').glob('batch*.npy')):
        assert int(path.stem[5:])==len(arrays)
        value=matrix(np.load(path,allow_pickle=False));arrays.append(value);values_checked+=value.size
    if (OUT/'outputs/gradient_summary.json').exists():
        summary=read(OUT/'outputs/gradient_summary.json');assert len(arrays)==10
        total=matrix(np.load(OUT/'outputs/gradient_components.npy',allow_pickle=False));values_checked+=total.size
        derived=np.zeros_like(total)
        for value in arrays:derived+=value
        assert np.array_equal(total,derived) and summary['terms']==p['terms'] and summary['parameter_layout']==p['parameter_layout']
        norm=np.linalg.norm(total,axis=1);gram=total@total.T;den=norm[:,None]*norm[None,:]
        cos=np.divide(gram,den,out=np.zeros_like(gram),where=den>0)
        assert np.array_equal(norm,np.asarray(summary['component_norms']))
        assert np.array_equal(gram,np.asarray(summary['component_gram'])) and np.array_equal(cos,np.asarray(summary['component_cosines']))
        assert np.array_equal(total@displacement,np.asarray(summary['component_dot_observed_displacement']))
        scalar_values=np.zeros(10,dtype=np.float64)
        for index,(row,value) in enumerate(zip(summary['batches'],arrays)):
            assert row['batch']==index and row['ids']==[c['id'] for c in p['case_rows'][index*5:index*5+5]]
            assert row['gradient_array_sha256']==sha(OUT/'outputs/gradients'/('batch'+str(index)+'.npy'))
            assert np.array_equal(np.linalg.norm(value,axis=1),np.asarray(row['component_norms']))
            scalar_values+=np.asarray(row['component_values'])
        assert np.array_equal(scalar_values,np.asarray(summary['component_values']))
        assert float(scalar_values[:7].sum())==summary['original_seven_term_objective']
        for row in p['parameter_layout']:
            v=total[:,row['start']:row['end']];delta=displacement[row['start']:row['end']];q=summary['per_parameter_gradients'][row['name']]
            assert np.array_equal(np.linalg.norm(v,axis=1),np.asarray(q['component_norms']))
            assert np.array_equal(v@delta,np.asarray(q['component_dot_observed_displacement']))
        assert summary['diagnostic_rows_are_not_a_new_training_objective']
    original,_=load_frozen_dgp_restorer(original_path,expected_sha256=sha(original_path),device='cpu')
    final,_=load_frozen_dgp_restorer(final_path,expected_sha256=sha(final_path),device='cpu')
    recognizer=FixedObservedIdentity(PARENT/'weights/w600k_r50.onnx','cpu')
    assert state_hash(original.net)==p['original_DGP_state'] and state_hash(final.net)==p['final_V28_DGP_state'] and state_hash(recognizer)==p['frozen_recognizer_state']
    references={r['id']:r for r in base['references']};counts={'original_DGP':0,'final_DGP':0,'recognizer':0}
    with torch.no_grad():
        for case in p['case_rows']:
            assert time.monotonic()-start<600
            cid=case['id']
            with Image.open(PARENT/case['input']) as im:camera=np.asarray(im.convert('RGB')).copy()
            with Image.open(PARENT/case['observed']) as im:mask=np.asarray(im).copy()>0
            support=torch.from_numpy(mask)[None,None]
            grid=torch.from_numpy(grid112(references[case['source_person_or_reference']]['matrix112']))[None]
            for folder,net,key in [('initial_baseline',original,'original_DGP'),('final_replay',final,'final_DGP')]:
                raw_path=OUT/'outputs'/folder/(cid+'.npy')
                if not raw_path.exists():continue
                raw=np.load(raw_path,allow_pickle=False);assert raw.shape==(256,256,3) and raw.dtype==np.float32 and np.isfinite(raw).all()
                x=canonical_tensor(camera,'cpu');expected=torch.where(support,net(x),x)[0].permute(1,2,0).numpy().copy();counts[key]+=1
                assert float(np.abs(raw-expected).max())<=p['CPU_return_raw_tolerance']
                with Image.open(raw_path.with_suffix('.png')) as im:png=np.asarray(im.convert('RGB')).copy()
                assert np.array_equal(png,np.where(mask[...,None],np.floor(raw*np.float32(255)),camera).astype(np.uint8))
                if folder=='initial_baseline':
                    with Image.open(PARENT/case['target']) as im:picture=np.asarray(im.convert('RGB')).copy()
                    vector_path=OUT/'outputs/initial_baseline'/(cid+'_target_embedding.npy')
                else:
                    picture=png;vector_path=OUT/'outputs/final_replay'/(cid+'_embedding.npy')
                    with Image.open(V28/'outputs/update800'/(cid+'.png')) as im:prior_png=np.asarray(im.convert('RGB')).copy()
                    assert np.array_equal(png,prior_png)
                vector=np.load(vector_path,allow_pickle=False);assert vector.shape==(512,) and vector.dtype==np.float32 and np.isfinite(vector).all()
                expected_vector=recognizer.embedding(canonical_tensor(picture,'cpu'),support.float(),grid)[0].numpy().copy();counts['recognizer']+=1
                assert float(np.abs(vector-expected_vector).max())<=p['CPU_return_vector_tolerance']
    if result.exists():
        assert len(arrays)==10 and values_checked==54848970 and terminal['component_gradient_calls']==100
        assert terminal['reference_DGP_forwards']==10 and terminal['candidate_DGP_forwards']==10 and terminal['recognizer_forwards']==30
        assert counts=={'original_DGP':50,'final_DGP':50,'recognizer':100}
        assert terminal['candidate_state_before_after']==p['final_V28_DGP_state'] and terminal['original_DGP_state_before_after']==p['original_DGP_state']
        assert terminal['recognizer_state_before_after']==p['frozen_recognizer_state']
        assert len(terminal['fresh_same_batch_replay_rows'])==50 and terminal['V28_failed_gates_retained']
        for c,row in zip(p['case_rows'],terminal['fresh_same_batch_replay_rows']):
            assert row['id']==c['id'] and row['final_PNG_exact']
            assert row['final_raw_maximum_error']<=2e-6 and row['initial_raw_maximum_error']<=2e-6
            assert row['final_vector_maximum_error']<=2e-6 and row['truth_vector_maximum_error']<=2e-6
        assert terminal['seconds']<=300 and terminal['peak_allocated_VRAM_bytes']<=20*1024**3 and supervision['within_external_bound']
    assert state_hash(original.net)==p['original_DGP_state'] and state_hash(final.net)==p['final_V28_DGP_state'] and state_hash(recognizer)==p['frozen_recognizer_state']
    assert all(not v.requires_grad and v.grad is None for net in [original,final,recognizer] for v in net.parameters())
    receipt={'complete':True,'checker_sha256':sha(Path(__file__)),'protocol_sha256':pin,'archive_sha256':expected_sha,'archive_bytes':expected_bytes,
             'members_verified':imported['members'],'diagnostic_complete':result.exists(),'VM_failure_retained':failure.exists(),
             'gradient_arrays':len(arrays),'saved_gradient_values_checked':values_checked,'CPU_forward_counts':counts,
             'optimizer_updates':0,'local_gradient_calls':0,'local_backward_calls':0,'new_training_recipe_created':False,
             'V28_failed_gates_retained':True,'native_or_reserved_used':False,'app_promotion':False,'goal_complete':False,'seconds':time.monotonic()-start}
    write(ROOT/'outputs/cctv_dgp_v28_preservation_diagnostic_v1_independent_audit.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--expected-sha',required=True);parser.add_argument('--expected-bytes',required=True,type=int)
    a=parser.parse_args();audit(a.expected_sha,a.expected_bytes)
