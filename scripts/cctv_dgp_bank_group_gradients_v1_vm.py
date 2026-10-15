"""Manual L4 measurement of all frozen source/profile gradients; no updates."""
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

NAME = 'cctv_dgp_bank_group_gradients_v1_vm'
STEM = 'cctv-dgp-bank-group-gradients-v1'
PARENT_NAME = 'cctv_dgp_bank_photometric_guard_v1_vm'
PARENT_PIN = 'f64154bd602070edaea61ccbf960ec3028e545c6540a4421ec41f9af21128a12'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def verify(root, parent, pin):
    assert sha(root/'protocol.json') == pin and sha(parent/'protocol.json') == PARENT_PIN
    p, old = read(root/'protocol.json'), read(parent/'protocol.json')
    assert p['name'] == NAME and p['parent_protocol_sha256'] == PARENT_PIN
    assert p['case_ids'] == old['preview_case_ids'] and p['paired_TRAIN_cases'] == 100
    assert p['optimizer_updates'] == p['backwards'] == p['full_epochs'] == p['parameter_displacements'] == 0
    assert p['maximum_gradient_queries'] == 320 and p['model_forward_cap'] == 40 and p['recognizer_forward_cap'] == 20
    assert p['worker_seconds'] == 900 and p['measurement_seconds'] == 600 and p['export_seconds'] == 300
    assert p['external_seconds'] == 930 and p['export_external_seconds'] == 330 and p['kill_grace_seconds'] == 30
    assert p['minimum_free_bytes'] == 4*1024**3 and p['disk_reserve_bytes'] == 1024**3
    assert p['output_byte_cap'] == 1280*1024**2 and p['peak_VRAM_bytes'] == 20*1024**3 and p['peak_RSS_bytes'] == 12*1024**3
    assert p['preserved_scientific_thresholds'] == old['scientific_thresholds']
    assert not p['failed_run_resume_allowed'] and not p['app_promotion'] and not p['automatic_follow_on']
    assert p['native_pixels_used'] == p['final_identity_pixels'] == 0
    for folder, files in [(root,p['assets_sha256']), (parent,p['parent_required_sha256'])]:
        for name, expected in files.items():
            path = folder/name
            assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(folder)
            assert sha(path) == expected, name
    sys.path.insert(0,str(root))
    from cctv_dgp_bank_group_gradients_v1_contract import scopes, fixed_reference_matrices
    blocks, groups = scopes(old['cases'],p['case_ids'])
    assert blocks == p['reference_blocks'] and groups == p['gradient_groups']
    selected = [c for c in old['cases'] if c['id'] in p['case_ids']]
    refs = {r['id']:r for r in old['references']}
    assert len(fixed_reference_matrices(selected,refs)) == 100
    assert p['parameters'] == 3364643 and len(p['learning_names']) == 37
    direction = read(root/'direction_review.json')
    assert direction['direction_file_sha256'] == sha(root/'directions.npz')
    assert direction['names'] == p['learning_names'] and direction['initial_file_sha256'] == sha(parent/'initializers/initial_B.pth')
    return p, old


def require_vm(root, parent):
    assert platform.system() == 'Linux' and platform.node().split('.')[0] == 'forensic-dgp-thesis'
    assert root == (Path.home()/'forensic-dgp'/NAME).resolve() and parent == (Path.home()/'forensic-dgp'/PARENT_NAME).resolve()
    assert os.environ.get('TMUX'), 'Human manual tmux required'
    assert sys.prefix == str(Path.home()/'forensic-dgp/cctv_dgp_vm_bundle/.venv')
    request = urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/machine-type',headers={'Metadata-Flavor':'Google'})
    with urllib.request.urlopen(request,timeout=3) as response:
        assert response.read().decode().rstrip().endswith('/g2-standard-4')
    result = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=10)
    assert not result.stdout.strip(), 'Competing GPU workload; preserve it'


def run(root, parent, p, old, pin):
    require_vm(root,parent)
    assert not (root/'outputs').exists(), 'Immutable used diagnostic; no unchanged rerun'
    assert shutil.disk_usage(root).free >= p['minimum_free_bytes'], 'Need4GiB free after installation'
    sys.path.insert(0,str(root));sys.path.insert(1,str(parent))
    import resource
    import numpy as np
    from PIL import Image
    import torch
    from cctv_dgp_bank_group_gradients_v1_contract import batch_values, query_key, accumulate, projection_seconds, fixed_reference_matrices
    from cctv_dgp_bank_comparison_v1_model import ConditionedBankDGP
    from cctv_dgp_generative_bank_v1 import load_prior
    from cctv_dgp_pilot import FixedObservedIdentity, grid112, state_hash, buffer_hash
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_group_conflicts_v1_losses import pixel_and_structure_losses
    from cctv_dgp_head4_lossless_v1 import unpack
    from frozen_capacity_contract import feature_support
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == 'NVIDIA L4'
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.manual_seed(20261010);torch.cuda.manual_seed_all(20261010)
    torch.backends.cudnn.benchmark = False;torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats(); start = time.monotonic()
    out = root/'outputs/group_gradients_v1'
    out.mkdir(parents=True);(out/'controls').mkdir();(out/'aggregates').mkdir();(out/'samples').mkdir()
    counts = dict(B=0,recognizer=0,original=0,gradient_queries=0,backwards=0,optimizer_updates=0,parameter_displacements=0)
    progress = dict(stage='initialize',counts=counts)
    candidate = None
    def stop(_signal,_frame):
        raise TimeoutError('External finite diagnostic stop; preserve evidence')
    signal.signal(signal.SIGTERM,stop)
    def clock():
        torch.cuda.synchronize()
        assert time.monotonic()-start < p['worker_seconds'], 'Worker stop'
        assert shutil.disk_usage(root).free >= p['disk_reserve_bytes'], 'Disk reserve stop'
        assert torch.cuda.max_memory_allocated() <= p['peak_VRAM_bytes'], 'VRAM stop'
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= p['peak_RSS_bytes'], 'Host memory stop'
        assert sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) <= p['output_byte_cap'], 'Output stop'
        assert counts['B'] <= p['model_forward_cap'] and counts['recognizer'] <= p['recognizer_forward_cap'] and counts['gradient_queries'] <= p['maximum_gradient_queries']
    def pixels(path,mode='RGB'):
        with Image.open(path) as im:
            assert im.size == (256,256)
            return np.asarray(im.convert(mode)).copy()
    def tensor(a):
        return torch.from_numpy(np.asarray(a).astype(np.float32)/np.float32(255)).permute(0,3,1,2).contiguous().cuda()
    try:
        original, _ = load_frozen_dgp_restorer(parent/'weights/dgp_v2.pth',expected_sha256=old['original_checkpoint_sha256'],device='cuda')
        identity = FixedObservedIdentity(parent/'weights/w600k_r50.onnx','cuda')
        prior = load_prior(parent/'prior_source',parent/'bank_implementation',old['standalone_prior_sha256']).cuda()
        candidate = ConditionedBankDGP(original.net,prior,torch.from_numpy(np.load(parent/'initializers/mean_style.npy',allow_pickle=False)).cuda()).cuda()
        initial = torch.load(parent/'initializers/initial_B.pth',map_location='cpu',weights_only=True)
        candidate.load_state_dict(initial,strict=True)
        candidate.enable_vm_learning(parent)
        names, params = candidate.learning_names(), candidate.learning_parameters()
        assert names == p['learning_names'] and sum(v.numel() for v in params) == p['parameters']
        before, buffers = state_hash(candidate), buffer_hash(candidate)
        assert before == p['initial_model_state'] and state_hash(original.net) == old['original_state'] and state_hash(identity) == old['recognizer_state']
        candidate.register_forward_hook(lambda *_:counts.__setitem__('B',counts['B']+1))
        identity.encoder.register_forward_hook(lambda *_:counts.__setitem__('recognizer',counts['recognizer']+1))
        with np.load(root/'directions.npz',allow_pickle=False) as archive:
            directions = {label:np.concatenate([archive[f'{label}_p{i:02d}'].reshape(-1) for i in range(37)]).astype(np.float64)
                          for label in ['ordinary','constrained']}
        assert all(v.shape == (p['parameters'],) and np.isfinite(v).all() for v in directions.values())
        keys = [g['key'] for g in p['gradient_groups']]
        aggregates = {key:np.zeros(p['parameters'],np.float64) for key in keys}
        contributions = {key:[] for key in keys}
        write(out/'environment.json',dict(python=sys.version,torch=torch.__version__,CUDA=torch.version.cuda,GPU=torch.cuda.get_device_name(0),
            host=platform.node(),manual_tmux=True,optimizer_constructed=False,failed_B50_state_loaded=False,
            parameter_displacement_planned=False,metric_gradient_measurement_only=True))
        refs = {r['id']:r for r in old['references']}
        rows, durations, samples = [], [], []
        for serial,block in enumerate(p['reference_blocks']):
            clock(); batch_start = time.monotonic()
            progress.update(stage='group_gradients',begin=block['begin'],reference=block['reference'])
            begin, ids = block['begin'], block['case_ids']
            cs = old['cases'][begin:begin+5]
            cameras = [pixels(parent/c['input']) for c in cs]
            targets = [pixels(parent/refs[c['source_person_or_reference']]['target']) for c in cs]
            masks = [pixels(parent/refs[c['source_person_or_reference']]['observed'],'L')>0 for c in cs]
            x, target = tensor(cameras), tensor(targets)
            mask = torch.from_numpy(np.stack(masks).astype(np.float32)).cuda()[:,None]
            feature = torch.from_numpy(np.stack([feature_support(m,c['landmarks5_canvas_xy']) for m,c in zip(masks,cs)]).astype(np.float32)).cuda()[:,None]
            grids = torch.from_numpy(np.stack([grid112(m) for m in fixed_reference_matrices(cs,refs)])).cuda()
            baseline, base_emb = unpack(parent/f'outputs/bank_comparison_v1/baseline/packs/b{begin//5:03d}.npz',ids)
            baseline_tensor = torch.from_numpy(baseline).permute(0,3,1,2).contiguous().cuda()
            with torch.no_grad():
                frozen = torch.where(mask.bool(),candidate(x),x)
                raw = frozen.permute(0,2,3,1).cpu().numpy().copy()
                raw_path = out/'controls'/f'b{begin//5:03d}.npy'
                with raw_path.open('xb') as stream:
                    np.save(stream,raw,allow_pickle=False)
                assert np.array_equal(raw,baseline), 'Same L4 initial raw control differs; retain stop'
            pred = torch.where(mask.bool(),candidate(x),x)
            maximum_error = float((pred.detach()-frozen).abs().max())
            assert maximum_error <= 1e-6, 'Gradient/frozen forward disagreement; retain stop'
            terms = pixel_and_structure_losses(pred,target,mask,feature)
            vectors = identity.embedding(torch.cat([pred,baseline_tensor.detach(),target]),torch.cat([mask]*3),torch.cat([grids]*3))
            assert np.allclose(vectors[5:10].detach().cpu().numpy(),base_emb[:,0],rtol=1e-6,atol=1e-6)
            assert np.allclose(vectors[10:15].detach().cpu().numpy(),base_emb[:,2],rtol=1e-6,atol=1e-6)
            target_vector = vectors[10:15].detach().double()
            arc = 1-(vectors[:5].double()*target_vector).sum(1)
            values = batch_values(terms,arc)
            query_rows = []
            for query,value in enumerate(values):
                clock(); assert value.ndim == 0 and bool(torch.isfinite(value))
                grads = torch.autograd.grad(value,params,retain_graph=query<15,allow_unused=False)
                counts['gradient_queries'] += 1
                assert all(bool(torch.isfinite(g).all()) for g in grads)
                arrays = [g.detach().cpu().numpy().copy() for g in grads]
                flat = np.concatenate([g.reshape(-1) for g in arrays])
                assert flat.dtype == np.float32 and flat.shape == (p['parameters'],)
                key = query_key(block['source'],query)
                accumulate(aggregates[key],flat,10)
                dot = {label:float(np.dot(flat.astype(np.float64),direction)) for label,direction in directions.items()}
                contribution = dict(begin=begin,reference=block['reference'],query=query,value=float(value.detach()),
                    gradient_vector_sha256=hashlib.sha256(flat.tobytes()).hexdigest(),
                    gradient_norm=float(np.linalg.norm(flat.astype(np.float64))),saved_direction_dot=dot)
                contributions[key].append(contribution)
                query_rows.append(dict(key=key,**contribution))
                if begin in p['sample_gradient_begins'] and query in [0,2]:
                    path = out/'samples'/f'b{begin//5:03d}_q{query:02d}.npz'
                    np.savez_compressed(path,**{f'p{i:02d}':g for i,g in enumerate(arrays)})
                    samples.append(dict(file=path.relative_to(out).as_posix(),sha256=sha(path),key=key,**contribution))
                del grads,arrays,flat
            torch.cuda.synchronize(); duration = time.monotonic()-batch_start
            durations.append(duration)
            row = dict(**block,queries=query_rows,completed_batch_duration_seconds=duration,seconds=time.monotonic()-start,
                raw_control_file=raw_path.relative_to(out).as_posix(),raw_control_sha256=sha(raw_path),
                exact_control_RGB_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),maximum_gradient_forward_error=maximum_error,
                raw_control_exact=True,input_NCHW_contiguous=x.is_contiguous(),target_NCHW_contiguous=target.is_contiguous())
            write(out/f'b{begin//5:03d}.json',row);rows.append(row)
            del pred,frozen,terms,vectors,arc,target_vector,values,x,target,mask,feature,grids,baseline_tensor,raw
            if serial == 1:
                projection = projection_seconds(durations[:2])
                write(out/'timing_projection.json',dict(projection_seconds=projection,cap_seconds=p['measurement_seconds'],
                    explicit_completed_batch_duration_seconds=durations[:2],measured_reference_batches=2,safety_factor=1.5))
                assert projection <= p['measurement_seconds'], 'Explicit batch timing projection stop'
            assert time.monotonic()-start < p['measurement_seconds'], 'Measurement stop'
            print(dict(reference_batches=serial+1,of=20,gradient_queries=counts['gradient_queries'],seconds=row['seconds']),flush=True)
        group_records = []
        for index,key in enumerate(keys):
            clock(); assert len(contributions[key]) == 10
            vector = aggregates[key]
            path = out/'aggregates'/f'g{index:02d}.npy'
            with path.open('xb') as stream:
                np.save(stream,vector,allow_pickle=False)
            assert np.isfinite(vector).all()
            group_records.append(dict(**p['gradient_groups'][index],file=path.relative_to(out).as_posix(),sha256=sha(path),
                value=float(np.mean([v['value'] for v in contributions[key]])),gradient_norm=float(np.linalg.norm(vector)),
                saved_direction_dot={label:float(np.dot(vector,direction)) for label,direction in directions.items()},
                contribution_records=contributions[key],aggregate_dtype='float64',raw_neural_gradient_dtype='float32',mean_divisor=10))
        assert state_hash(candidate) == before and buffer_hash(candidate) == buffers
        assert state_hash(original.net) == old['original_state'] and state_hash(identity) == old['recognizer_state']
        assert all(a.grad is None for a in params)
        assert all(not m.training for net in [candidate,original,identity] for m in net.modules())
        assert all(not a.requires_grad and a.grad is None for net in [original,identity] for a in net.parameters())
        assert counts == dict(B=40,recognizer=20,original=0,gradient_queries=320,backwards=0,optimizer_updates=0,parameter_displacements=0)
        clock();verify(root,parent,pin)
        write(out/'run_results.json',dict(complete=True,counts=counts,rows=rows,groups=group_records,samples=samples,
            parameters=p['parameters'],learning_names=names,model_state=before,buffer_state=buffers,
            state_and_buffer_bytes_unchanged=True,original_and_recognizer_unchanged=True,
            groups_not_independent_final_identities=True,independent_saved_arithmetic_audit_pending=True,
            raw_and_delivered_PNG_gates_unchanged_and_not_requalified=True,new_checkpoint_created=False,
            native_pixels_used=0,final_identity_pixels=0,more_epochs_not_qualified=True,model_qualified=False,
            peak_VRAM=torch.cuda.max_memory_allocated(),peak_RSS=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            seconds=time.monotonic()-start))
    except BaseException as error:
        write(out/'failure.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),progress=progress,
            seconds=time.monotonic()-start,optimizer_updates=0,model_qualified=False,more_epochs_not_qualified=True,
            state_not_displaced_but_no_fresh_post_failure_state_hash_claimed=True))
        raise


def export(root,parent,p,pin):
    stamp = time.monotonic()
    assert platform.system() == 'Linux' and platform.node().split('.')[0] == 'forensic-dgp-thesis'
    assert root == (Path.home()/'forensic-dgp'/NAME).resolve() and os.environ.get('TMUX')
    assert sys.prefix == str(Path.home()/'forensic-dgp/cctv_dgp_vm_bundle/.venv')
    manifest = root/'export_manifest.json'
    assert not manifest.exists()
    files = sorted(f for f in root.rglob('*') if f.is_file() and '__pycache__' not in f.parts)
    assert not any(f.suffix == '.pth' for f in files)
    assert sum(f.stat().st_size for f in files) < p['output_byte_cap']+64*1024**2
    assert shutil.disk_usage(root).free >= p['disk_reserve_bytes']+sum(f.stat().st_size for f in files), 'Retain reserve before export'
    write(manifest,dict(protocol_sha256=pin,parent_protocol_sha256=PARENT_PIN,parent_required_sha256=p['parent_required_sha256'],
        files={f.relative_to(root).as_posix():sha(f) for f in files},optimizer_updates=0,
        complete_diagnostic=(root/'outputs/group_gradients_v1/run_results.json').exists(),failure_present=(root/'outputs/group_gradients_v1/failure.json').exists()))
    archive = Path.home()/(STEM+'-results.tar.gz')
    assert not archive.exists()
    with tarfile.open(archive,'x:gz',compresslevel=1) as tar:
        for file in files+[manifest]:
            assert time.monotonic()-stamp < p['export_seconds'], 'Finite export stop'
            assert shutil.disk_usage(root).free >= p['disk_reserve_bytes'], 'Export disk reserve stop'
            tar.add(file,arcname=NAME+'/'+file.relative_to(root).as_posix(),recursive=False)
    digest = sha(archive)
    with Path(str(archive)+'.sha256').open('x',encoding='ascii',newline='\n') as stream:
        stream.write(digest+'  '+archive.name+'\n')
    record = dict(complete=True,archive_sha256=digest,bytes=archive.stat().st_size,seconds=time.monotonic()-stamp,
                  optimizer_updates=0,diagnostic_success_not_implied=True)
    write(Path.home()/(STEM+'-export.json'),record);print(record,flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--parent-root',type=Path,required=True)
    parser.add_argument('--protocol-sha',required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--verify-transfer',action='store_true')
    action.add_argument('--run',action='store_true')
    action.add_argument('--export',action='store_true')
    a = parser.parse_args();root,parent = a.root.resolve(),a.parent_root.resolve()
    if a.run:
        require_vm(root,parent)
    p,old = verify(root,parent,a.protocol_sha)
    if a.verify_transfer:
        print(dict(complete=True,packet_assets=len(p['assets_sha256']),parent_assets=len(p['parent_required_sha256']),
            groups=32,planned_gradient_queries=320,optimizer_updates=0,actual_neural_or_gradient_calls=0),flush=True)
    elif a.run:
        run(root,parent,p,old,a.protocol_sha)
    else:
        export(root,parent,p,a.protocol_sha)


if __name__ == '__main__':
    main()
