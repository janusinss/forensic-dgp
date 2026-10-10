"""Incoming archive, every saved raw/PNG metric and full-state boundary audit."""
from pathlib import Path,PurePosixPath
import hashlib
import sys
import tarfile
import time
import zipfile
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];PACKET=ROOT/'outputs/cctv_dgp_head4_capacity_vm_v1'
sys.path.insert(0,str(PACKET))
from cctv_dgp_head4_capacity_contract_v1 import NAME,STEM,STATE,BUDGETS,read,write,sha,validate
RETURN=ROOT/'outputs'/(NAME+'_return');RECEIPT=ROOT/'outputs/cctv_dgp_head4_capacity_v1_independent_audit.json'


def safe(member):
    path=PurePosixPath(member.name)
    assert member.isfile() and not member.issym() and not member.islnk() and member.size>=0
    assert path.parts and path.parts[0]==NAME+'_return' and '..' not in path.parts
    assert not path.is_absolute() and '\\' not in member.name and ':' not in member.name and member.name==path.as_posix()


def pixels(path,mode='RGB'):
    with Image.open(path) as im:assert im.size==(256,256);return np.asarray(im.convert(mode)).copy()
def close(a,b,tol=1e-12):assert abs(a-b)<=tol and np.isfinite(a) and np.isfinite(b),(a,b)


def pack_headers(path):
    expected={'planes.npy':((4,5*256*256*3),np.dtype('uint8')),'shape.npy':((4,),np.dtype('int64')),
        'xor.npy':((),np.dtype('bool')),'embeddings.npy':((5,3,512),np.dtype('float32'))}
    with zipfile.ZipFile(path) as z:
        info=z.infolist();assert len(info)==5 and {m.filename for m in info}==set(expected)|{'ids.npy'}
        assert sum(m.file_size for m in info)<8*1024**2
        for m in info:
            with z.open(m) as f:
                version=np.lib.format.read_magic(f)
                assert version in [(1,0),(2,0)]
                header=np.lib.format.read_array_header_1_0(f) if version==(1,0) else np.lib.format.read_array_header_2_0(f)
                shape,fortran,dtype=header;assert not fortran and not dtype.hasobject
                if m.filename=='ids.npy':assert shape==(5,) and dtype.kind=='U' and dtype.itemsize<=512
                else:assert (shape,dtype)==expected[m.filename]
                assert f.tell()+int(np.prod(shape,dtype=np.int64))*dtype.itemsize==m.file_size


def main():
    start=time.monotonic();assert not RECEIPT.exists() and not RETURN.exists()
    p=read(PACKET/'protocol.json');validate(p);pin=sha(PACKET/'protocol.json')
    for n,d in p['assets_sha256'].items():assert sha(PACKET/n)==d,n
    for n,d in p['local_sources'].items():assert sha(ROOT/n)==d,n
    archive=ROOT/'outputs'/(STEM+'-results.tar.gz');export=read(ROOT/'outputs'/(STEM+'-export.json'))
    assert export['complete'] and export['training_success_not_implied']
    assert archive.stat().st_size==export['bytes'] and sha(archive)==export['archive_sha256']
    assert Path(str(archive)+'.sha256').read_text().strip().split()==[export['archive_sha256'],archive.name]
    with tarfile.open(archive,'r:gz') as tar:
        members=[];names=set();total=0
        for m in tar:
            safe(m);assert m.name not in names;names.add(m.name);members.append(m);total+=m.size
            assert len(members)<=2500 and total<=BUDGETS['return_uncompressed_bytes']+1024**2
        prefix=NAME+'_return/'
        with tar.extractfile(prefix+'export_manifest.json') as f:manifest=__import__('json').load(f)
        assert manifest['complete'] and manifest['protocol_sha256']==pin
        assert manifest['training_success_not_implied'] and manifest['uncompressed_bytes']==total-next(m.size for m in members if m.name==prefix+'export_manifest.json')
        assert names=={prefix+n for n in manifest['files_sha256']}|{prefix+'export_manifest.json'}
        for m in members:
            if m.name==prefix+'export_manifest.json':continue
            with tar.extractfile(m) as f:
                h=hashlib.sha256()
                for b in iter(lambda:f.read(1024**2),b''):h.update(b)
            assert h.hexdigest()==manifest['files_sha256'][m.name[len(prefix):]],m.name
        RETURN.mkdir()
        for m in members:
            target=RETURN/str(PurePosixPath(m.name).relative_to(NAME+'_return'));assert target.resolve().is_relative_to(RETURN.resolve())
            target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(m) as f,target.open('xb') as g:
                for b in iter(lambda:f.read(1024**2),b''):g.write(b)
    assert sha(RETURN/'protocol.json')==pin
    for n,d in p['assets_sha256'].items():
        if n.endswith('.py'):assert sha(RETURN/n)==d,n
    result_path=RETURN/'outputs/results.json';failure_path=RETURN/'outputs/failure.json'
    receipt={'complete':True,'protocol_sha256':pin,'archive_sha256':export['archive_sha256'],'archive_members':len(members),
        'failure_preserved':failure_path.exists(),'checker_sha256':sha(Path(__file__)),'goal_complete':False,'app_promotion':False}
    if not result_path.exists():
        assert failure_path.exists();f=read(failure_path)
        updates=f['progress']['optimizer_updates']
        assert updates is None or type(updates) is int and 0<=updates<=50
        if updates is None:assert f['exact_completed_updates_unknown'] and f['supervisor_retained_after_child_exit']
        receipt.update({'training_completed':False,'optimizer_updates':updates,'exact_updates_unknown':updates is None,
            'full_case_metric_audit_completed':False,'seconds':time.monotonic()-start})
        write(RECEIPT,receipt);print(receipt);return
    r=read(result_path);assert r['complete'] and r['protocol_sha256']==pin
    assert r['optimizer_updates']==50 and r['completed_epochs']==0 and r['epoch_fraction']==50/781
    assert r['original_state']==STATE and r['original_checkpoint_unchanged']
    assert r['initial_full_TRAIN_parity_cases']==3905 and r['trainable_elements']==147456 and r['trainable_tensors']==3
    assert r['progress']=={'optimizer_updates':50,'backwards':50,'gradient_queries':6,'completed_epochs':0,'epoch_fraction':50/781,'optimizer_constructed':True}
    assert r['forward_counts']=={'original':781,'candidate':1614,'recognizer':1614}
    for k in ['native_DEV_or_reserved_final_used','app_promotion','model_qualification','goal_complete']:assert r[k] is False
    supervision=read(RETURN/'supervisor_receipt.json')
    assert supervision['optimizer_updates']==50 and supervision['external_cap_seconds']==2430
    assert supervision['trainer_exit_code']==int((RETURN/'trainer_exit_code.txt').read_text())
    assert supervision['within_external_bound']==(supervision['seconds']<=2465)
    assert supervision['within_external_bound'] and r['seconds']<2400
    gradient=read(RETURN/'outputs/gradient_preflight.json')
    assert gradient['complete'] and gradient['queries']==6 and gradient['optimizer_updates']==0 and len(gradient['rows'])==6
    for batch in range(2):
        for offset,component in enumerate([0,2,3]):
            row=gradient['rows'][3*batch+offset]
            assert row['component']==component and row['case_ids']==[p['cases'][i]['id'] for i in p['preflight_batches'][batch]]
            assert len(row['part_gradient_L2'])==3 and all(np.isfinite(n) and n>0 for n in row['part_gradient_L2'])
    steps=[__import__('json').loads(line) for line in (RETURN/'outputs/training_steps.jsonl').read_text().splitlines()]
    assert len(steps)==50
    for update,row in enumerate(steps,1):
        assert row['update']==update and row['case_ids']==[p['cases'][i]['id'] for i in p['schedule'][update-1]]
        assert len(row['normalized_components'])==4 and all(np.isfinite(v) for v in row['normalized_components'])
        assert np.isfinite(row['objective']) and np.isfinite(row['regression_guard']) and row['regression_guard']>=0
        assert np.isfinite(row['gradient_L2_before_clip']) and row['gradient_L2_before_clip']>=0
        close(row['objective'],sum(w*v for w,v in zip(p['reconstruction_weights'],row['normalized_components']))+row['regression_guard'],1e-9)
    from cctv_dgp_head4_lossless_v1 import unpack as decode_pack
    def unpack(path,ids,baseline=None):
        pack_headers(path);return decode_pack(path,ids,baseline)
    from frozen_capacity_contract import feature_support,exported_pixel_metrics,detail_metric,capacity
    from frozen_raw_metrics import pixel_metrics,detail_float,deliver,mean_only,review_groups
    cases=p['cases'];refs={q['id']:q for q in p['references']};snapshots={};checked=0
    for update in [0,50]:
        saved=read(RETURN/f'outputs/update{update}/metrics.json');assert saved['complete'] and saved['update']==update
        assert [v['id'] for v in saved['rows']]==[c['id'] for c in cases]
        for begin in range(0,3905,5):
            cs=cases[begin:begin+5];ids=[c['id'] for c in cs]
            base,base_em=unpack(RETURN/'outputs/update0/packs'/f'b{begin//5:04d}.npz',ids)
            raw,em=(base,base_em) if update==0 else unpack(RETURN/'outputs/update50/packs'/f'b{begin//5:04d}.npz',ids,baseline=base)
            for slot,c in enumerate(cs):
                row=saved['rows'][begin+slot];target=pixels(PACKET/refs[c['source_person_or_reference']]['target'])
                camera=pixels(PACKET/c['input']);mask=pixels(PACKET/refs[c['source_person_or_reference']]['observed'],'L')>0
                feature=feature_support(mask,c['landmarks5_canvas_xy']);a=raw[slot];png=deliver(a,camera,mask)
                assert row['source']==c['source'] and row['profile']==c['profile']
                assert hashlib.sha256(a.tobytes()).hexdigest()==row['raw_RGB_float32_sha256']
                assert hashlib.sha256(png.tobytes()).hexdigest()==row['PNG_RGB_sha256']
                assert np.array_equal(a[~mask],camera[~mask].astype(np.float32)/np.float32(255))
                mraw,mpng,shift=mean_only(a,base[slot],camera,mask)
                rm=pixel_metrics(a,target,mask);pm=exported_pixel_metrics(png,target,mask)
                rm.update({'ArcFace_observed_fixed':float(em[slot,0]@em[slot,2]),'landmark_high_frequency_MSE':detail_float(a,target,feature),
                    'constant_mean_shift_only_MSE':pixel_metrics(mraw,target,mask)['MSE']})
                pm.update({'ArcFace_observed_fixed':float(em[slot,1]@em[slot,2]),'landmark_high_frequency_MSE':detail_metric(png,target,feature),
                    'constant_mean_shift_only_MSE':exported_pixel_metrics(mpng,target,mask)['MSE']})
                assert set(rm)==set(row['raw']) and set(pm)==set(row['png'])
                for stage,values in [('raw',rm),('png',pm)]:
                    for key,val in values.items():
                        if val is None or isinstance(val,bool):assert val==row[stage][key]
                        else:close(val,row[stage][key])
                assert np.allclose(shift,row['postclip_mean_RGB_shift'],atol=1e-14,rtol=0)
                if c['id'] in p['preview_case_ids']:assert np.array_equal(png,pixels(RETURN/f'outputs/update{update}/previews'/(c['id']+'.png')))
                checked+=1
            if begin%500==0:print({'audited_snapshot':update,'cases':begin+5,'of':3905},flush=True)
        for stage in ['raw','png']:
            fresh=review_groups(saved['rows'],stage)
            assert set(fresh)==set(saved['groups'][stage])
            for key,values in fresh.items():
                for metric,value in values.items():
                    if metric=='cases':assert value==saved['groups'][stage][key][metric]
                    else:close(value,saved['groups'][stage][key][metric],1e-14)
        snapshots[update]=saved
    gate=read(RETURN/'outputs/early_gate.json');assert gate['update']==50
    recomputed={stage:capacity(snapshots[0]['groups'][stage],snapshots[50]['groups'][stage],.01) for stage in ['raw','png']}
    assert recomputed==r['gates']==gate['comparisons'];passed=all(v['pass'] for v in recomputed.values())
    assert passed==gate['pass']==r['gate_pass'] and failure_path.exists()==(not passed)
    assert supervision['trainer_exit_code']==(0 if passed else 1)
    # Full-state boundaries and two model replay cases, inference mode only.
    import torch
    from cctv_dgp_pilot import state_hash,buffer_hash,FixedObservedIdentity,grid112
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_head4_capacity_model_v1 import Head4CapacityDGP
    torch.set_num_threads(4)
    original,_=load_frozen_dgp_restorer(PACKET/'weights/dgp_v2.pth',expected_sha256=p['original_checkpoint_sha256'],device='cpu')
    initial=Head4CapacityDGP(original.net).eval().requires_grad_(False);learned=Head4CapacityDGP(original.net).eval().requires_grad_(False)
    weights=torch.load(RETURN/'outputs/update50/candidate.pth',map_location='cpu',weights_only=True);learned.load_state_dict(weights,strict=True)
    names=set(initial.learning_names());assert all(torch.equal(v,weights[n]) for n,v in initial.state_dict().items() if n not in names)
    assert state_hash({n:v for n,v in weights.items() if n not in names})==r['frozen_state_hash']
    assert buffer_hash(learned)==r['frozen_buffers_hash'] and state_hash(learned)==r['candidate_state']
    for update in [0,50]:
        state=torch.load(RETURN/f'outputs/update{update}/training_state.pt',map_location='cpu',weights_only=True)
        assert state['protocol_sha256']==pin and state['completed_updates']==state['next_schedule_index']==update
        assert state['optimizer_parameter_names']==initial.learning_names() and state['completed_epochs']==0
        assert state['epoch_fraction']==update/781 and state['automatic_resume'] is False
        assert state['failed_gate_must_not_resume'] and state['scheduler'] is not None and state['optimizer'] is not None
        assert state['original_checkpoint_sha256']==p['original_checkpoint_sha256'] and state['schedule_sha256']==pin
        group=state['optimizer']['param_groups'];assert len(group)==1 and group[0]['params']==[0,1,2]
        assert group[0]['lr']==1e-5 and list(group[0]['betas'])==[.9,.999] and group[0]['eps']==1e-8 and group[0]['weight_decay']==1e-5
        assert state['scheduler']['last_epoch']==update and state['scheduler']['step_size']==781 and state['scheduler']['gamma']==1.
        assert state_hash(state['model'])==snapshots[update]['candidate_state']
        assert state['torch_rng'].dtype==torch.uint8 and isinstance(state['cuda_rng'],list) and len(state['numpy_rng'])==5
        if update==50:
            assert state['gate_status']==('early_capacity_pass_pending_independent_review' if passed else 'failed_gate')
            assert len(state['optimizer']['state'])==3
            assert all(float(v['step'])==50 for v in state['optimizer']['state'].values())
        else:assert state['optimizer']['state']=={} and state['gate_status']=='not_run'
    identity=FixedObservedIdentity(PACKET/'weights/w600k_r50.onnx','cpu');replays=[]
    by_id={c['id']:i for i,c in enumerate(cases)}
    with torch.inference_mode():
        for cid in p['independent_replay_ids']:
            index=by_id[cid];begin=index//5*5;cs=cases[begin:begin+5];ids=[c['id'] for c in cs]
            base,_=unpack(RETURN/'outputs/update0/packs'/f'b{begin//5:04d}.npz',ids)
            raw,em=unpack(RETURN/'outputs/update50/packs'/f'b{begin//5:04d}.npz',ids,baseline=base)
            c=cases[index];camera=pixels(PACKET/c['input']);ref=refs[c['source_person_or_reference']];mask8=pixels(PACKET/ref['observed'],'L')>0
            x=torch.from_numpy(camera.astype(np.float32)/np.float32(255)).permute(2,0,1)[None]
            mask=torch.from_numpy(mask8.astype(np.float32))[None,None];pred=torch.where(mask.bool(),learned(x),x)
            error=float(np.abs(pred[0].permute(1,2,0).numpy()-raw[index-begin]).max());assert error<=3e-6
            target=torch.from_numpy(pixels(PACKET/ref['target']).astype(np.float32)/np.float32(255)).permute(2,0,1)[None]
            grid=torch.from_numpy(grid112(ref['matrix112']))[None]
            fresh=identity.embedding(torch.cat([pred,target]),torch.cat([mask]*2),torch.cat([grid]*2)).numpy()
            ee=float(np.abs(fresh-em[index-begin,[0,2]]).max());assert ee<=5e-5
            replays.append({'id':cid,'raw_max_abs':error,'embedding_max_abs':ee})
    for n,d in p['assets_sha256'].items():assert sha(PACKET/n)==d,n
    for n,d in p['local_sources'].items():assert sha(ROOT/n)==d,n
    receipt.update({'training_completed':True,'early_capacity_pass':passed,'full_raw_PNG_metric_records_verified':checked,
        'optimizer_updates':50,'completed_epochs':0,'full_state_and_frozen_partition_verified':True,'fresh_CPU_replays':replays,
        'local_gradient_queries':0,'local_optimizer_updates':0,'model_qualification':False,'visual_review_pending':True,
        'native_development_not_reviewed_here':True,'final800_requirement_not_tested':True,'seconds':time.monotonic()-start})
    write(RECEIPT,receipt);print({'complete':True,'early_capacity_pass':passed,'records_verified':checked,'trained_updates':50})


if __name__=='__main__':main()
