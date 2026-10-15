"""Human-manual saved-path inference only. No optimizer or historical training."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cctv_dgp_spatial_path_replay_v1_contract import (
    NAME, STEM, SOURCE_NAME, PARENT_NAME, read, write, sha, array_sha,
    require_manual, verify_transfer, selected_vector, expected_state_hash, label,
)

STAGE = 'outputs/spatial_path_replay_v1'


def machine_guard(root, idle=False):
    require_manual(root)
    request = urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/machine-type',
                                     headers={'Metadata-Flavor':'Google'})
    with urllib.request.urlopen(request, timeout=3) as response:
        assert response.read().decode().rstrip().endswith('/g2-standard-4')
    if idle:
        result = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],
                                capture_output=True,text=True,check=True,timeout=10)
        assert not result.stdout.strip(), 'Competing GPU workload; preserve it'


@contextmanager
def source_cwd(source):
    """Keep the original inference factory's exact cwd guard, without modifying it."""
    before = Path.cwd()
    assert source == (Path.home()/'forensic-dgp'/SOURCE_NAME).resolve()
    os.chdir(source)
    try:
        yield
    finally:
        os.chdir(before)


def review(root, source, parent, p, old, data, pin):
    machine_guard(root, idle=True)
    out = root/STAGE
    assert not (root/'outputs').exists(), 'Used immutable diagnostic; retain it'
    out.mkdir(parents=True)
    start = time.monotonic()
    counts = dict(optimizer_updates=0,gradient_queries=0,backwards=0,
                  original_batch_calls=0,prior_feature_batch_calls=0,
                  decoder_batch_calls=0,recognizer_batch_calls=0)
    progress = dict(stage='pre-model resources', snapshot=None, reference=None)
    torch = None
    def clock():
        assert time.monotonic()-start <= p['budgets']['worker_seconds'], 'Finite replay worker stop'
        assert shutil.disk_usage(root).free >= p['budgets']['disk_reserve_bytes'], 'Disk reserve stop'
        assert sum(v.stat().st_size for v in out.rglob('*') if v.is_file()) <= p['budgets']['output_byte_cap'], 'Output cap stop'
        for key,value in counts.items():
            cap = p['budgets'].get('maximum_'+key,p['budgets'].get(key))
            assert value <= cap, 'Finite '+key+' stop'
        if torch is not None:
            torch.cuda.synchronize()
            import resource
            assert torch.cuda.max_memory_allocated() <= p['budgets']['peak_VRAM_bytes'], 'VRAM stop'
            assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= p['budgets']['peak_RSS_bytes'], 'RSS stop'
            assert not torch.is_grad_enabled()
    def interrupted(_signal,_frame):
        raise RuntimeError('Finite supervisor stop signal; retain partial evidence')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        mem = {line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
        free = shutil.disk_usage(root).free
        write(out/'storage_start.json',dict(free_bytes=free,MemAvailable_bytes=mem['MemAvailable'],neural_calls=0,optimizer_updates=0))
        assert free >= p['budgets']['minimum_free_bytes'], 'Need8GiB free after installation'
        assert mem['MemAvailable'] >= p['budgets']['available_RAM_before_import_bytes'], 'Need8GiB available RAM before imports'
        assert source == (root.parent/SOURCE_NAME).resolve() and parent == (root.parent/PARENT_NAME).resolve()
        with source_cwd(source):
            # The bound original loader/factory is used for inference only.
            # Never call its train, replay, export, supervise or learning methods.
            spec = importlib.util.spec_from_file_location('retained_spatial_inference_loader',source/'scripts/cctv_dgp_spatial_readout_v1_vm.py')
            loader = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(loader)
            np,torch,original,identity,provider,provenance,loading = loader.load_dependencies(source,parent,old)
            torch.set_grad_enabled(False)
            torch.cuda.reset_peak_memory_stats()
            from PIL import Image
            from cctv_dgp_pilot import grid112,state_hash,buffer_hash
            from cctv_dgp_head4_lossless_v1 import unpack
            from cctv_dgp_spatial_readout_v1_model import build_model
            from cctv_dgp_spatial_readout_v1_contract import stage_gate
            from frozen_capacity_contract import feature_support,groups,exported_pixel_metrics,detail_metric
            from frozen_raw_metrics import deliver,mean_only,pixel_metrics,detail_float
            assert not torch.is_grad_enabled() and all(not v.requires_grad for model in [original.net,identity,provider] for v in model.parameters())
            write(out/'environment.json',dict(host=os.uname().nodename,GPU=torch.cuda.get_device_name(0),
                torch=torch.__version__,CUDA=torch.version.cuda,DGP_provenance=provenance,
                original_state=state_hash(original.net),recognizer_state=state_hash(identity),
                provider_state=state_hash(provider),retained_guarded_factory_cwd=str(source),
                historical_training_called=False,gradient_mode=False))
            refs,cases = {r['id']:r for r in data['references']},{c['id']:c for c in data['cases']}
            previous_cache = read(source/'outputs/spatial_readout_v1/cache_receipt.json')['rows']
            caches,cache_rows = [],[]
            def pixels(path,mode='RGB'):
                with Image.open(path) as image:
                    assert image.size == (256,256)
                    return np.asarray(image.convert(mode)).copy()
            def tensor(images):
                return torch.from_numpy(np.stack(images).astype(np.float32)/np.float32(255)).permute(0,3,1,2).contiguous().cuda()
            def gpu(b):
                return {k:v.cuda() for k,v in b['CPU'].items()} | dict(
                    maps=[v.cuda() for v in b['maps']],features={k:v.cuda() for k,v in b['features'].items()})
            stamp = time.monotonic()
            for serial,block in enumerate(p['reference_blocks']):
                progress.update(stage='exact fresh feature cache',reference=block['reference'])
                cs,ref = [cases[c] for c in block['case_ids']],refs[block['reference']]
                cameras = [pixels(parent/c['input']) for c in cs]
                targets = [pixels(parent/ref['target']) for c in cs]
                masks = [pixels(parent/ref['observed'],'L')>0 for _ in cs]
                support = [feature_support(m,c['landmarks5_canvas_xy']) for m,c in zip(masks,cs)]
                values = dict(x=tensor(cameras),target=tensor(targets),
                    mask=torch.from_numpy(np.stack(masks).astype(np.float32))[:,None].cuda(),
                    grid=torch.from_numpy(np.stack([grid112(ref['matrix112'])]*5)).cuda())
                counts['original_batch_calls'] += 1
                baseline = torch.where(values['mask'].bool(),original(values['x']),values['x'])
                maps = provider.current.observe(values['x'])
                counts['prior_feature_batch_calls'] += 1
                features = provider.conditioned_features(maps)
                retained,embeddings = unpack(parent/'outputs/bank_comparison_v1/baseline/packs'/f"b{block['begin']//5:03d}.npz",block['case_ids'])
                raw = baseline.cpu().permute(0,2,3,1).numpy().copy()
                assert np.array_equal(raw,retained), 'Fresh original RAW differs; stop replay'
                values['baseline'] = baseline
                b = dict(cases=cs,camera=cameras,target8=targets,mask8=masks,support8=support,
                    baseline=raw,retained_embeddings=embeddings,target_vectors=embeddings[:,2].copy(),
                    CPU={k:v.detach().cpu().contiguous() for k,v in values.items()},
                    maps=[v.detach().cpu().contiguous() for v in maps],
                    features={k:v.detach().cpu().contiguous() for k,v in features.items()})
                proof = dict(reference=block['reference'],maps=[array_sha(v.numpy()) for v in b['maps']],
                    features={k:array_sha(v.numpy()) for k,v in b['features'].items()},source=block['source'])
                assert all(proof[k] == previous_cache[serial][k] for k in ['reference','maps','features','source']), 'Fresh feature cache differs'
                caches.append(b);cache_rows.append(proof)
                del values,baseline,maps,features,retained,embeddings,raw
                clock()
                elapsed = time.monotonic()-stamp
                if serial == 1:
                    projection = elapsed*10*1.5
                    write(out/'cache_projection.json',dict(references=2,seconds=elapsed,projected_seconds=projection,cap_seconds=240))
                    assert projection <= 240, 'Cache projection stop'
                assert elapsed <= p['budgets']['cache_seconds'], 'Cache time stop'
                print(json.dumps(dict(cache_reference=serial+1,of=20)),flush=True)
            write(out/'cache_receipt.json',dict(complete=True,rows=cache_rows,seconds=time.monotonic()-stamp,all_source_feature_hashes_exact=True))

            def snapshot(snapshot_label,model=None,state=None):
                progress.update(stage='saved-state inference',snapshot=snapshot_label)
                began = time.monotonic()
                folder = out/snapshot_label
                folder.mkdir();(folder/'packs').mkdir();(folder/'png').mkdir()
                rows = []
                old_label = 'baseline' if model is None or state['update'] == 0 else state['route']+'50' if state['update'] == 50 else None
                for index,b in enumerate(caches):
                    progress['reference'] = p['reference_blocks'][index]['reference']
                    g = gpu(b)
                    if model is None:
                        pred = g['baseline']
                    else:
                        counts['decoder_batch_calls'] += 1
                        pred = torch.where(g['mask'].bool(),model.forward_cached(g['x'],g['maps'],g['features']),g['x'])
                    raw = pred.cpu().permute(0,2,3,1).numpy().copy()
                    png = np.stack([deliver(a,c,m) for a,c,m in zip(raw,b['camera'],b['mask8'])])
                    if old_label:
                        with np.load(source/'outputs/spatial_readout_v1'/old_label/'packs'/f'r{index:02d}.npz',allow_pickle=False) as previous:
                            assert np.array_equal(raw,previous['raw']), 'Original/endpoint RAW parity differs'
                        for i,c in enumerate(b['cases']):
                            assert np.array_equal(png[i],pixels(source/'outputs/spatial_readout_v1'/old_label/'png'/(c['id']+'.png'))), 'Original/endpoint PNG parity differs'
                    zero_png = [deliver(z,c,m) for z,c,m in zip(b['baseline'],b['camera'],b['mask8'])]
                    counts['recognizer_batch_calls'] += 2
                    vr = identity.embedding(torch.cat([pred,g['baseline'],g['target']]),torch.cat([g['mask']]*3),torch.cat([g['grid']]*3)).cpu().numpy()
                    vp = identity.embedding(torch.cat([tensor(list(png)),tensor(zero_png),g['target']]),torch.cat([g['mask']]*3),torch.cat([g['grid']]*3)).cpu().numpy()
                    for values,slot in [(vr,0),(vp,1)]:
                        np.testing.assert_allclose(values[5:10],b['retained_embeddings'][:,slot],rtol=1e-6,atol=1e-6)
                        np.testing.assert_allclose(values[10:15],b['target_vectors'],rtol=1e-6,atol=1e-6)
                    np.savez_compressed(folder/'packs'/f'r{index:02d}.npz',ids=np.asarray([c['id'] for c in b['cases']]),
                        raw=raw,embeddings_raw=vr,embeddings_png=vp,target_embeddings=b['target_vectors'])
                    for i,c in enumerate(b['cases']):
                        actual,mask,target = raw[i],b['mask8'][i],b['target8'][i]
                        assert np.array_equal(actual[~mask],b['camera'][i][~mask].astype(np.float32)/np.float32(255))
                        assert np.array_equal(png[i][~mask],b['camera'][i][~mask])
                        control,control_png,_ = mean_only(actual,b['baseline'][i],b['camera'][i],mask)
                        rm,pm = pixel_metrics(actual,target,mask),exported_pixel_metrics(png[i],target,mask)
                        initial = model is None or state['update'] == 0
                        for mode,m,vectors,slot in [('raw',rm,vr,0),('png',pm,vp,1)]:
                            vec = b['retained_embeddings'][i,slot] if initial else vectors[i]
                            tv = b['target_vectors'][i] if initial else vectors[10+i]
                            m['ArcFace_observed_fixed'] = float(vec@tv)
                        rm.update(landmark_high_frequency_MSE=detail_float(actual,target,b['support8'][i]),constant_mean_shift_only_MSE=pixel_metrics(control,target,mask)['MSE'])
                        pm.update(landmark_high_frequency_MSE=detail_metric(png[i],target,b['support8'][i]),constant_mean_shift_only_MSE=exported_pixel_metrics(control_png,target,mask)['MSE'])
                        Image.fromarray(png[i]).save(folder/'png'/(c['id']+'.png'))
                        rows.append(dict(case=c['id'],source=c['source'],profile=c['profile'],raw=rm,png=pm))
                    del g,pred,raw,png,vr,vp,zero_png
                    clock()
                    assert time.monotonic()-began <= p['budgets']['snapshot_seconds'], 'Snapshot time stop'
                reductions = {mode:groups([dict(source=r['source'],profile=r['profile'],metrics=r[mode]) for r in rows]) for mode in ['raw','png']}
                record = dict(rows=rows,groups=reductions,seconds=time.monotonic()-began,old_endpoint_pixel_parity=old_label)
                write(folder/'metrics.json',record)
                return record

            baseline = snapshot('baseline')
            projected = time.monotonic()-start+14*baseline['seconds']*1.75+90
            write(out/'worker_projection.json',dict(projected_seconds=projected,cap_seconds=p['budgets']['worker_seconds'],before_candidate_replay=True))
            assert projected <= p['budgets']['worker_seconds'], 'Finite replay projection stop'
            # All RAW arrays, PNGs and selected vectors must fit before replay.
            estimated = 1500*(256*256*3*4+256*256*3)+14*9*1024**2+64*1024**2
            write(out/'output_projection.json',dict(projected_bytes=estimated,cap_bytes=p['budgets']['output_byte_cap']))
            assert estimated <= p['budgets']['output_byte_cap']
            assert shutil.disk_usage(root).free >= 2*estimated+p['budgets']['disk_reserve_bytes']
            outcomes=[]
            for route in ['A','B']:
                model = build_model(source,parent,provider,source/'readouts.npz',route)
                frozen = torch.load(source/('outputs/spatial_readout_v1/state_'+route+'0/full_state.pth'),map_location='cpu',weights_only=True)
                model.load_state_dict(frozen['model'],strict=True)
                model.eval().requires_grad_(False)
                names = old['trainable_parameter_names'][route]
                assert model.learning_names() == names and all(not v.requires_grad for v in model.parameters())
                layout = read(source/('outputs/spatial_readout_v1/'+route+'_parameter_layout.json'))
                initial = np.load(source/('outputs/spatial_readout_v1/'+route+'_initial_vector.npy'),allow_pickle=False)
                fixed = state_hash({n:v for n,v in model.state_dict().items() if n not in names})
                buffers = buffer_hash(model)
                for state in [v for v in p['states'] if v['route'] == route]:
                    vector = selected_vector(source,state,initial)
                    with torch.no_grad():
                        for row,value in zip(layout,model.learning_parameters()):
                            assert list(value.shape) == row['shape']
                            value.copy_(torch.from_numpy(vector[row['begin']:row['end']].reshape(row['shape'])).cuda())
                    observed = np.concatenate([v.detach().cpu().numpy().reshape(-1) for v in model.learning_parameters()])
                    assert np.array_equal(observed,vector)
                    current_hash = state_hash(model)
                    assert current_hash == expected_state_hash(frozen['model'],names,layout,vector)
                    assert state_hash({n:v for n,v in model.state_dict().items() if n not in names}) == fixed and buffer_hash(model) == buffers
                    key = label(state)
                    np.save(out/(key+'_vector.npy'),vector,allow_pickle=False)
                    write(out/(key+'_state.json'),dict(route=route,source_update=state['update'],model_state=current_hash,
                        fixed_state=fixed,buffer_state=buffers,vector_sha256=array_sha(vector),
                        installed_vector_exact=True,gradients_enabled=False,optimizer_updates=0))
                    metrics = snapshot(key,model,state)
                    assert state_hash(model) == current_hash and buffer_hash(model) == buffers
                    gates = {mode:stage_gate(baseline['groups'][mode],metrics['groups'][mode]) for mode in ['raw','png']}
                    outcome=dict(route=route,source_update=state['update'],snapshot=key,gates=gates,
                        numerical_exposed_TRAIN_gate_only=True,incumbent_selection=False,model_qualified=False,
                        new_optimizer_updates=0,epoch_extension=False)
                    outcomes.append(outcome)
                    write(out/(key+'_outcome.json'),outcome)
                    print(json.dumps(dict(snapshot=key,RAW_gain_percent=100*gates['raw']['global_structure_gain'],PNG_gain_percent=100*gates['png']['global_structure_gain'],
                        RAW_failures=len(gates['raw']['preservation_failures']),PNG_failures=len(gates['png']['preservation_failures']),optimizer_updates=0)),flush=True)
                    clock()
                del model,frozen
                torch.cuda.empty_cache()
            assert state_hash(original.net) == old['original_state'] and state_hash(identity) == old['recognizer_state'] and state_hash(provider) == old['feature_provider_initial_state']
            assert counts == dict(optimizer_updates=0,gradient_queries=0,backwards=0,original_batch_calls=20,prior_feature_batch_calls=20,decoder_batch_calls=280,recognizer_batch_calls=600)
            write(out/'run_results.json',dict(complete=True,counts=counts,outcomes=outcomes,seconds=time.monotonic()-start,
                snapshot_compositions=1400,fresh_original_controls=100,full_corpus_epochs=0,
                original_provider_recognizer_unchanged=True,all_original_and_endpoint_100_RAW_PNG_controls_exact=True,
                model_qualified=False,retrospective_exposed_TRAIN_only=True,automatic_follow_on=False))
        require_manual(root)
    except BaseException as exc:
        write(out/'failure.json',dict(complete=False,error_type=type(exc).__name__,message=str(exc),
            traceback=traceback.format_exc(),progress=progress,counts=counts,seconds=time.monotonic()-start,
            optimizer_updates=0,model_qualified=False,failed_recipe_training_resumed=False))
        raise


def export(root,p,pin):
    machine_guard(root)
    start=time.monotonic();out=root/STAGE
    assert out.is_dir(), 'Export reads existing evidence; it does not launch inference'
    archive=Path.home()/(STEM+'-results.tar.gz');partial=Path(str(archive)+'.partial')
    sidecar=Path(str(archive)+'.sha256');receipt=Path.home()/(STEM+'-export.json')
    assert not any(v.exists() for v in [archive,partial,sidecar,receipt])
    files=[root/name for name in p['assets_sha256']]+[root/'protocol.json']
    files += [v for v in out.rglob('*') if v.is_file()]
    files += [root/name for name in ['review.log','review_exit.json','saved_audit.log','saved_audit_exit.json'] if (root/name).is_file()]
    assert len(files)==len(set(files)) and all(not v.is_symlink() and v.resolve().is_relative_to(root) for v in files)
    total=sum(v.stat().st_size for v in files)
    assert total<=p['budgets']['output_byte_cap']+16*1024**2 and shutil.disk_usage(root).free>=total+p['budgets']['disk_reserve_bytes']
    manifest=root/'result_manifest.json'
    write(manifest,dict(protocol_sha256=pin,files={v.relative_to(root).as_posix():dict(bytes=v.stat().st_size,sha256=sha(v)) for v in sorted(files)},
        model_quality_not_implied=True,new_optimizer_updates=0))
    files.append(manifest)
    with partial.open('xb') as output:
        with tarfile.open(fileobj=output,mode='w:gz',compresslevel=1) as pack:
            for path in sorted(files):
                assert time.monotonic()-start<=p['budgets']['export_seconds'] and shutil.disk_usage(root).free>=p['budgets']['disk_reserve_bytes']
                pack.add(path,arcname=path.relative_to(root).as_posix(),recursive=False)
    assert time.monotonic()-start<=p['budgets']['export_seconds']
    partial.rename(archive);digest=sha(archive)
    with sidecar.open('x',encoding='utf-8',newline='\n') as s:s.write(digest+'  '+archive.name+'\n')
    value=dict(complete=True,archive_sha256=digest,bytes=archive.stat().st_size,protocol_sha256=pin,seconds=time.monotonic()-start,
        optimizer_updates=0,gradient_queries=0,training_success_not_implied=True,model_qualified=False,
        run_results_present=(out/'run_results.json').is_file(),failure_present=(out/'failure.json').is_file())
    write(receipt,value);print(json.dumps(value),flush=True)


def child(root,source,parent,pin,flag,seconds,log_name):
    import selectors
    command=[sys.executable,'-B','-u',str(root/'scripts/cctv_dgp_spatial_path_replay_v1_vm.py'),
        '--root',str(root),'--source-root',str(source),'--parent-root',str(parent),'--protocol-sha',pin,flag]
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,bufsize=0,start_new_session=True)
    print(json.dumps(dict(flag=flag,child_pid=process.pid,log=log_name,external_cap_seconds=seconds)),flush=True)
    start=time.monotonic();terminated=None;timeout=False;eof=False;killed=False
    selector=selectors.DefaultSelector();selector.register(process.stdout,selectors.EVENT_READ)
    with (root/log_name).open('xb') as log:
        while not (eof and process.poll() is not None):
            now=time.monotonic()
            if now-start>seconds and process.poll() is None and terminated is None:
                timeout=True;terminated=now;os.killpg(process.pid,signal.SIGTERM)
            if terminated is not None and now-terminated>30 and process.poll() is None and not killed:
                os.killpg(process.pid,signal.SIGKILL);killed=True
            for key,_ in selector.select(.2):
                chunk=os.read(key.fileobj.fileno(),65536)
                if not chunk:eof=True;selector.unregister(key.fileobj)
                else:log.write(chunk);log.flush();sys.stdout.buffer.write(chunk);sys.stdout.buffer.flush()
    selector.close();code=process.wait(timeout=30)
    return dict(flag=flag,exit_code=code,timeout=timeout,seconds=time.monotonic()-start,child_pid=process.pid,kill_sent=killed,maximum_seconds=seconds,kill_grace_seconds=30)


def supervise(root,source,parent,p,pin):
    machine_guard(root,idle=True)
    assert not (root/'outputs').exists() and not (root/'review.log').exists()
    reviewed=child(root,source,parent,pin,'--review',p['budgets']['external_seconds'],'review.log')
    write(root/'review_exit.json',reviewed)
    checked=None
    if (root/STAGE/'run_results.json').is_file():
        checked=child(root,source,parent,pin,'--audit-saved',p['saved_audit_external_seconds'],'saved_audit.log')
        write(root/'saved_audit_exit.json',checked)
    exported=child(root,source,parent,pin,'--export',p['export_external_seconds'],'export.log')
    write(root/'supervisor_receipt.json',dict(complete=True,review=reviewed,saved_audit=checked,export=exported,
        optimizer_updates=0,training_success_not_implied=True,model_qualified=False,automatic_follow_on=False))
    print(json.dumps(dict(review_exit=reviewed['exit_code'],audit_exit=None if checked is None else checked['exit_code'],export_exit=exported['exit_code'],optimizer_updates=0)),flush=True)
    return 0 if reviewed['exit_code']==0 and checked and checked['exit_code']==0 and exported['exit_code']==0 else 1


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--source-root',type=Path);parser.add_argument('--parent-root',type=Path)
    parser.add_argument('--protocol-sha',required=True)
    flags=parser.add_mutually_exclusive_group(required=True)
    for name in ['verify-transfer','review','audit-saved','export','supervise']:flags.add_argument('--'+name,action='store_true')
    a=parser.parse_args();root=a.root.resolve()
    source=a.source_root.resolve() if a.source_root else (root.parent/SOURCE_NAME).resolve()
    parent=a.parent_root.resolve() if a.parent_root else (root.parent/PARENT_NAME).resolve()
    if not a.verify_transfer:require_manual(root)
    p,old,data=verify_transfer(root,source,parent,a.protocol_sha)
    if a.verify_transfer:
        print(json.dumps(dict(complete=True,packet_assets=len(p['assets_sha256']),source_assets=len(p['source_required_sha256']),parent_assets=len(p['parent_required_sha256']),saved_states=14,optimizer_updates=0,neural_calls=0)))
    elif a.review:review(root,source,parent,p,old,data,a.protocol_sha)
    elif a.export:export(root,p,a.protocol_sha)
    elif a.audit_saved:
        from audit_cctv_dgp_spatial_path_replay_v1_return import audit_saved
        audit_saved(root,source,parent,p,old,data,root/STAGE/'independent_saved_audit_vm.json')
    else:sys.exit(supervise(root,source,parent,p,a.protocol_sha))


if __name__=='__main__':main()
