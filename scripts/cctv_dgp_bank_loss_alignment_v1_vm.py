"""Manual L4 loss-direction measurement; zero optimizer updates and checkpoints."""
import argparse
from datetime import datetime, timezone
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

NAME='cctv_dgp_bank_loss_alignment_v1_vm'
STEM='cctv-dgp-bank-loss-alignment-v1'
PARENT_NAME='cctv_dgp_bank_photometric_guard_v1_vm'
PARENT_PIN='f64154bd602070edaea61ccbf960ec3028e545c6540a4421ec41f9af21128a12'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def verify(root,parent,pin):
    assert sha(root/'protocol.json')==pin and sha(parent/'protocol.json')==PARENT_PIN
    p,old=read(root/'protocol.json'),read(parent/'protocol.json')
    assert p['name']==NAME and p['parent_protocol_sha256']==PARENT_PIN
    assert p['states']==['initial','joint50'] and p['case_ids']==old['preview_case_ids']
    assert p['paired_TRAIN_cases']==100 and p['native_pixels_used']==p['final_identity_pixels']==0
    assert p['maximum_gradient_queries']==360 and p['model_forward_cap']==80 and p['recognizer_forward_cap']==40
    assert p['optimizer_updates']==p['backwards']==p['full_epochs']==0
    assert p['worker_seconds']==900 and p['measurement_seconds']==600 and p['export_seconds']==300
    assert p['external_seconds']==930 and p['output_byte_cap']==1024**3
    assert p['minimum_free_bytes']==4*1024**3 and p['disk_reserve_bytes']==1024**3
    assert p['preserved_scientific_thresholds']==old['scientific_thresholds']
    assert not p['failed_run_resume_allowed'] and not p['app_promotion'] and not p['automatic_follow_on']
    for folder,files in [(root,p['assets_sha256']),(parent,p['parent_required_sha256'])]:
        for name,expected in files.items():
            path=folder/name
            assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(folder)
            assert sha(path)==expected,name
    assert not read(parent/'outputs/bank_comparison_v1/outcome_B.json')['model_qualified']
    return p,old


def require_vm(root,parent):
    assert platform.system()=='Linux' and platform.node().split('.')[0]=='forensic-dgp-thesis'
    assert root==(Path.home()/'forensic-dgp'/NAME).resolve()
    assert parent==(Path.home()/'forensic-dgp'/PARENT_NAME).resolve()
    assert os.environ.get('TMUX'),'Human manual tmux launch required'
    assert sys.prefix==str(Path.home()/'forensic-dgp/cctv_dgp_vm_bundle/.venv')
    request=urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/machine-type',headers={'Metadata-Flavor':'Google'})
    with urllib.request.urlopen(request,timeout=3) as response:
        assert response.read().decode().rstrip().endswith('/g2-standard-4')
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=10)
    assert not result.stdout.strip(),'Competing GPU workload; preserve it and do not launch'


def run(root,parent,p,old,pin):
    require_vm(root,parent)
    assert not (root/'outputs').exists(),'Immutable used diagnostic; no unchanged rerun'
    assert shutil.disk_usage(root).free>=p['minimum_free_bytes'],'Need4GiB free'
    sys.path.insert(0,str(root)); sys.path.insert(1,str(parent))
    import numpy as np
    from PIL import Image
    import torch
    from cctv_dgp_bank_loss_alignment_v1 import COMPONENTS,components,hypothetical_adam_direction,summarize
    from cctv_dgp_bank_comparison_v1_model import ConditionedBankDGP
    from cctv_dgp_generative_bank_v1 import load_prior
    from cctv_dgp_pilot import FixedObservedIdentity,grid112,state_hash,buffer_hash
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_bank_comparison_v1_contract import NORMALIZERS,ANCHORS
    from cctv_dgp_group_conflicts_v1_losses import pixel_and_structure_losses
    from cctv_dgp_bank_photometric_guard_v1_loss import brightness_guard
    from cctv_dgp_head4_lossless_v1 import unpack
    from frozen_capacity_contract import feature_support
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0)=='NVIDIA L4'
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    torch.manual_seed(20261010); torch.cuda.manual_seed_all(20261010)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.cuda.reset_peak_memory_stats(); start=time.monotonic()
    out=root/'outputs/loss_alignment_v1'; out.mkdir(parents=True); (out/'sampled_gradients').mkdir()
    counts=dict(B=0,recognizer=0,original=0,gradient_queries=0,backwards=0,optimizer_updates=0)
    progress=dict(stage='initialize',counts=counts); candidate=None
    def stop(_signal,_frame): raise TimeoutError('External finite diagnostic stop; retain evidence')
    signal.signal(signal.SIGTERM,stop)
    def clock():
        torch.cuda.synchronize()
        assert time.monotonic()-start<p['worker_seconds'],'Finite worker stop'
        assert shutil.disk_usage(root).free>=p['disk_reserve_bytes'],'Disk reserve stop'
        assert torch.cuda.max_memory_allocated()<=p['peak_VRAM_bytes'],'VRAM stop'
        assert sum(f.stat().st_size for f in out.rglob('*') if f.is_file())<=p['output_byte_cap'],'Output stop'
        assert counts['B']<=80 and counts['recognizer']<=40 and counts['gradient_queries']<=360
    def pixels(path,mode='RGB'):
        with Image.open(path) as im:
            assert im.size==(256,256); return np.asarray(im.convert(mode)).copy()
    def tensor(a): return torch.from_numpy(np.asarray(a).astype(np.float32)/np.float32(255)).permute(0,3,1,2).cuda()
    try:
        original,_=load_frozen_dgp_restorer(parent/'weights/dgp_v2.pth',expected_sha256=old['original_checkpoint_sha256'],device='cuda')
        identity=FixedObservedIdentity(parent/'weights/w600k_r50.onnx','cuda')
        prior=load_prior(parent/'prior_source',parent/'bank_implementation',old['standalone_prior_sha256']).cuda()
        candidate=ConditionedBankDGP(original.net,prior,torch.from_numpy(np.load(parent/'initializers/mean_style.npy',allow_pickle=False)).cuda()).cuda()
        initial=torch.load(parent/'initializers/initial_B.pth',map_location='cpu',weights_only=True)
        stored=torch.load(parent/'outputs/bank_comparison_v1/state_B50/full_state.pth',map_location='cpu',weights_only=True)
        assert stored['updates']==50 and not stored['manual_failed_resume_allowed']
        names=candidate.learning_names(); params=candidate.learning_parameters(); assert len(names)==37
        assert names==p['learning_names']
        projected=sum(a.numel()*a.element_size() for a in params)*10*4+16*1024**2
        assert projected==p['uncompressed_sampled_gradient_upper_bound_bytes'] and projected<p['output_byte_cap']
        candidate.enable_vm_learning(parent)
        # The retained model's manual gradient guard points at its hash-bound parent.
        # This worker also independently guards its new diagnostic root before imports.
        candidate.register_forward_hook(lambda *_:counts.__setitem__('B',counts['B']+1))
        identity.encoder.register_forward_hook(lambda *_:counts.__setitem__('recognizer',counts['recognizer']+1))
        write(out/'environment.json',dict(python=sys.version,torch=torch.__version__,CUDA=torch.version.cuda,GPU=torch.cuda.get_device_name(0),
            host=platform.node(),manual_tmux=True,optimizer_constructed=False,failed_state_used_for_diagnosis_only=True))
        refs={r['id']:r for r in old['references']}
        selected=[i for i in range(0,3905,5) if old['cases'][i]['id'] in p['case_ids']]; assert len(selected)==20
        sample_begins=p['sample_gradient_begins']; rows=[]; durations=[]; measurement_start=time.monotonic()
        for state_label in p['states']:
            state=initial if state_label=='initial' else stored['model']
            candidate.load_state_dict(state,strict=True)
            before,buffers=state_hash(candidate),buffer_hash(candidate); assert before==p['model_states'][state_label]
            for serial,begin in enumerate(selected):
                clock(); assert time.monotonic()-measurement_start<p['measurement_seconds'],'Measurement10-minute stop'
                stamp=time.monotonic(); progress.update(stage='component_gradients',state=state_label,begin=begin)
                cs=old['cases'][begin:begin+5]; ids=[c['id'] for c in cs]
                cameras=[pixels(parent/c['input']) for c in cs]
                targets=[pixels(parent/refs[c['source_person_or_reference']]['target']) for c in cs]
                masks=[pixels(parent/refs[c['source_person_or_reference']]['observed'],'L')>0 for c in cs]
                x,target=tensor(cameras),tensor(targets); mask=torch.from_numpy(np.stack(masks).astype(np.float32)).cuda()[:,None]
                feature=torch.from_numpy(np.stack([feature_support(m,c['landmarks5_canvas_xy']) for m,c in zip(masks,cs)]).astype(np.float32)).cuda()[:,None]
                grids=torch.from_numpy(np.stack([grid112(refs[c['source_person_or_reference']]['matrix112']) for c in cs])).cuda()
                baseline,base_emb=unpack(parent/'outputs/bank_comparison_v1/baseline/packs'/f'b{begin//5:03d}.npz',ids)
                baseline_tensor=torch.from_numpy(baseline).permute(0,3,1,2).cuda()
                expected=baseline if state_label=='initial' else unpack(parent/'outputs/bank_comparison_v1/B50/packs'/f'b{begin//5:03d}.npz',ids,baseline=baseline)[0]
                with torch.no_grad():
                    frozen=torch.where(mask.bool(),candidate(x),x)
                    assert np.array_equal(frozen.permute(0,2,3,1).cpu().numpy(),expected),'Same L4 frozen raw control differs'
                    old_terms=pixel_and_structure_losses(baseline_tensor,target,mask,feature)
                pred=torch.where(mask.bool(),candidate(x),x)
                error=float((pred.detach()-frozen).abs().max())
                assert error<=1e-6,'Gradient/frozen forward disagreement; retain diagnostic stop'
                terms=pixel_and_structure_losses(pred,target,mask,feature)
                vectors=identity.embedding(torch.cat([pred,baseline_tensor.detach(),target]),torch.cat([mask]*3),torch.cat([grids]*3))
                assert np.allclose(vectors[10:15].detach().cpu().numpy(),base_emb[:,2],rtol=1e-6,atol=1e-6)
                target_vec=vectors[10:15].detach().double()
                arc=1-(vectors[:5].double()*target_vec).sum(1); old_arc=1-(vectors[5:10].detach().double()*target_vec).sum(1)
                photo,photo_terms=brightness_guard(pred,baseline_tensor,target,mask,NORMALIZERS[0],10.,.2)
                values=components(terms,old_terms,arc,old_arc,photo,NORMALIZERS,ANCHORS,old['reconstruction_weights'])
                gradient_rows=[]
                for index,value in enumerate(values):
                    clock(); grads=torch.autograd.grad(value,params,retain_graph=index<8,allow_unused=False)
                    counts['gradient_queries']+=1
                    assert all(torch.isfinite(g).all() for g in grads)
                    gradient_rows.append(tuple(g.detach() for g in grads))
                directions,norm,factor=hypothetical_adam_direction(params,gradient_rows,
                    None if state_label=='initial' else stored['optimizer'])
                summaries=summarize(gradient_rows,directions,names)
                sample_file=None
                if begin in sample_begins:
                    sample_file=f'sampled_gradients/{state_label}_b{begin//5:03d}.npz'
                    arrays={f'g{j:02d}_p{i:02d}':g.cpu().numpy() for j,row in enumerate(gradient_rows) for i,g in enumerate(row)}
                    arrays.update({f'd_p{i:02d}':d.cpu().numpy() for i,d in enumerate(directions)})
                    np.savez_compressed(out/sample_file,**arrays); del arrays
                torch.cuda.synchronize(); elapsed=time.monotonic()-stamp; durations.append(elapsed)
                row=dict(state=state_label,begin=begin,reference=cs[0]['source_person_or_reference'],source=cs[0]['source'],case_ids=ids,
                    model_state=before,component_values=[float(v.detach()) for v in values],groups=summaries,
                    objective=sum(float(v.detach()) for v in values[:7]),total_gradient_norm=norm,clip_factor=factor,
                    gradient_forward_maximum_error=error,sampled_gradient_file=sample_file,
                    gradient_queries=9,seconds=elapsed,photometric_terms=photo_terms,
                    Adam_direction_is_hypothetical_single_reference_not_actual_accumulated_update=True)
                write(out/f'{state_label}_b{begin//5:03d}.json',row); rows.append(row)
                del gradient_rows,directions,grads,pred,frozen,values,vectors,terms,old_terms,photo,arc,old_arc,target_vec
                if state_label=='initial' and serial==1:
                    projection=max(durations)*40*1.5
                    write(out/'timing_projection.json',dict(projection_seconds=projection,cap_seconds=600,
                        measured_reference_batches=2,safety_factor=1.5,optimizer_updates=0))
                    assert projection<=600,'Measured gradient timing projection stop'
                print(dict(state=state_label,reference_batches=serial+1,of=20,gradient_queries=counts['gradient_queries'],seconds=time.monotonic()-start),flush=True)
            assert state_hash(candidate)==before and buffer_hash(candidate)==buffers
            assert all(a.grad is None for a in params)
        assert len(rows)==40 and counts==dict(B=80,recognizer=40,original=0,gradient_queries=360,backwards=0,optimizer_updates=0)
        assert state_hash(original.net)==old['original_state'] and state_hash(identity)==old['recognizer_state']
        assert all(not m.training for net in [candidate,original,identity] for m in net.modules())
        assert all(not a.requires_grad and a.grad is None for net in [original,identity] for a in net.parameters())
        assert all(not a.requires_grad and a.grad is None for n,a in candidate.named_parameters() if n not in names)
        clock(); verify(root,parent,pin)
        write(out/'run_results.json',dict(complete=True,counts=counts,component_names=COMPONENTS,learning_names=names,rows=rows,
            state_and_buffer_bytes_unchanged=True,new_checkpoint_created=False,scientific_quality_stops_retained=True,
            paired_TRAIN_cases=100,states_measured=2,native_pixels_used=0,final_identity_pixels=0,
            sampled_full_gradient_reference_batches=4,full_gradient_audit_scope_is_sampled_not_all40=True,
            raw_PNG_model_quality_not_requalified=True,optimizer_or_resume_not_run=True,
            seconds=time.monotonic()-start,peak_VRAM=torch.cuda.max_memory_allocated(),model_qualified=False))
    except BaseException as exc:
        write(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),progress=progress,
            seconds=time.monotonic()-start,optimizer_updates=0,quality_failure_retained=True))
        raise


def export(root,p,pin):
    assert platform.system()=='Linux' and root==(Path.home()/'forensic-dgp'/NAME).resolve()
    assert os.environ.get('TMUX') and sys.prefix==str(Path.home()/'forensic-dgp/cctv_dgp_vm_bundle/.venv')
    start=time.monotonic(); manifest=root/'export_manifest.json'; assert not manifest.exists()
    files=sorted(f for f in root.rglob('*') if f.is_file() and f.name not in {'export.log','export_exit_code.txt'})
    assert all(not f.is_symlink() and f.resolve().is_relative_to(root) for f in files)
    assert sum(f.stat().st_size for f in files)<p['output_byte_cap']+1024**2
    write(manifest,dict(protocol_sha256=pin,parent_protocol_sha256=PARENT_PIN,parent_required_sha256=p['parent_required_sha256'],
        files={f.relative_to(root).as_posix():sha(f) for f in files},optimizer_updates=0,
        complete_diagnostic=(root/'outputs/loss_alignment_v1/run_results.json').exists(),failure_present=(root/'outputs/loss_alignment_v1/failure.json').exists()))
    archive=Path.home()/(STEM+'-results.tar.gz'); assert not archive.exists()
    with tarfile.open(archive,'x:gz',compresslevel=1) as tar:
        for f in files+[manifest]:
            assert time.monotonic()-start<300,'Finite export stop'
            tar.add(f,arcname=NAME+'/'+f.relative_to(root).as_posix(),recursive=False)
    digest=sha(archive)
    with Path(str(archive)+'.sha256').open('x',encoding='ascii',newline='\n') as stream: stream.write(digest+'  '+archive.name+'\n')
    write(Path.home()/(STEM+'-export.json'),dict(complete=True,archive_sha256=digest,bytes=archive.stat().st_size,
        seconds=time.monotonic()-start,optimizer_updates=0,diagnostic_success_not_implied=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True); parser.add_argument('--parent-root',type=Path,required=True)
    parser.add_argument('--protocol-sha',required=True); action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--verify-transfer',action='store_true'); action.add_argument('--run',action='store_true'); action.add_argument('--export',action='store_true')
    a=parser.parse_args(); root,parent=a.root.resolve(),a.parent_root.resolve(); p,old=verify(root,parent,a.protocol_sha)
    if a.verify_transfer:
        if platform.system()=='Linux': require_vm(root,parent)
        print(dict(complete=True,packet_assets=len(p['assets_sha256']),parent_assets=len(p['parent_required_sha256']),neural_calls=0,gradient_queries=0,optimizer_updates=0))
    elif a.run: run(root,parent,p,old,a.protocol_sha)
    else: export(root,p,a.protocol_sha)
