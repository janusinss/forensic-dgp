"""Human-manual fixed decoder feasibility comparison; no optimizer or gradients."""
import argparse
import importlib.metadata
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

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from cctv_dgp_decoder_feasibility_v1_contract import (
    NAME,STEM,STAGE,ACTIVATION_NAME,REPAIR_NAME,SOURCE_NAME,PARENT_NAME,REPLAY_NAME,
    read,write,sha,array_sha,require_manual,verify_transfer,
)
from cctv_dgp_decoder_feasibility_v1_context import FeasibilityContext,source_cwd,file_module

PROGRESS_CAP=2*1024**2


def machine_guard(root,idle=False):
    require_manual(root)
    request=urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/machine-type',headers={'Metadata-Flavor':'Google'})
    with urllib.request.urlopen(request,timeout=3) as response:
        assert response.read().decode().rstrip().endswith('/g2-standard-4')
    if idle:
        result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=10)
        assert not result.stdout.strip(),'Competing GPU workload; preserve it'



def run(root,activation,repair,source,parent,replay,p,ap,repaired,old,data,pin):
    machine_guard(root,idle=True)
    assert not (root/'outputs').exists(),'Immutable completed/partial comparison; retain evidence'
    out=root/STAGE;out.mkdir(parents=True)
    start,ctx=time.monotonic(),None
    counts=dict(optimizer_updates=0,gradient_queries=0,backwards=0,original_batch_calls=0,
                prior_feature_batch_calls=0,decoder_batch_calls=0,recognizer_batch_calls=0)
    progress=dict(stage='resource preflight',state=None,reference=None)
    snapshots=[]
    def size(): return sum(path.stat().st_size for path in out.rglob('*') if path.is_file())
    def reserve(additional):
        assert size()+additional<=p['budgets']['output_byte_cap']-p['budgets']['emergency_metadata_bytes']
        assert shutil.disk_usage(root).free>=additional+p['budgets']['disk_reserve_bytes']
    def clock():
        assert time.monotonic()-start<=p['budgets']['worker_seconds'],'Finite worker stop'
        assert shutil.disk_usage(root).free>=p['budgets']['disk_reserve_bytes'],'Disk reserve stop'
        assert size()<=p['budgets']['output_byte_cap']-p['budgets']['emergency_metadata_bytes'],'Output reserve stop'
        for key,value in counts.items():
            assert value<=p['budgets'].get('maximum_'+key,p['budgets'].get(key)),key+' finite call stop'
        if ctx is not None and hasattr(ctx,'torch'):
            import resource
            ctx.torch.cuda.synchronize()
            assert ctx.torch.cuda.max_memory_allocated()<=p['budgets']['peak_VRAM_bytes'],'VRAM stop'
            assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=p['budgets']['peak_RSS_bytes'],'RSS stop'
        else:
            import resource
            assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=p['budgets']['peak_RSS_bytes'],'CPU phase RSS stop'
        line=json.dumps(dict(seconds=time.monotonic()-start,progress=progress,counts=dict(counts)),allow_nan=False)+'\n'
        journal=out/'progress.jsonl'
        assert (journal.stat().st_size if journal.exists() else 0)+len(line.encode())<=PROGRESS_CAP
        with journal.open('a',encoding='utf-8',newline='\n') as stream:stream.write(line)
    def interrupted(_signal,_frame): raise RuntimeError('Finite supervisor stop; retain partial comparison')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
        full_state_bytes=(source/'outputs/spatial_readout_v1/state_A0/full_state.pth').stat().st_size
        estimate=400*256*256*3*5+5*full_state_bytes+16*609219*8+48*1024**2
        free=shutil.disk_usage(root).free
        write(out/'storage_start.json',dict(free_bytes=free,MemAvailable_bytes=mem['MemAvailable'],
            projected_output_bytes=estimate,output_cap_bytes=p['budgets']['output_byte_cap'],
            emergency_metadata_bytes=p['budgets']['emergency_metadata_bytes'],neural_calls=0,gradient_queries=0,optimizer_updates=0))
        assert free>=p['budgets']['minimum_free_bytes'] and mem['MemAvailable']>=p['budgets']['available_RAM_before_import_bytes']
        assert estimate<=p['budgets']['output_byte_cap']-p['budgets']['emergency_metadata_bytes'] and free>=2*estimate+p['budgets']['disk_reserve_bytes']
        import numpy as np
        from PIL import Image
        progress.update(stage='one saved-gradient direction solve')
        math_module=file_module(root/'cctv_dgp_decoder_feasibility_v1_math.py','frozen_feasibility_math',p['assets_sha256']['cctv_dgp_decoder_feasibility_v1_math.py'])
        lossless=file_module(parent/'cctv_dgp_head4_lossless_v1.py','feasibility_saved_array_unpack',p['parent_required_sha256']['cctv_dgp_head4_lossless_v1.py'])
        references={r['id']:r for r in data['references']}
        initial=np.load(repair/'outputs/decoder_hinge_repair_v1/initial_vector.npy',allow_pickle=False)
        matrices,measurements,photo_inputs=[],[],[]
        for index in ap['derivative_reference_indices']:
            folder=activation/'outputs/decoder_activation_v1/gradients'/f'r{index:02d}'
            matrices.append(np.load(folder/'gradients.npy',allow_pickle=False));measurements.append(read(folder/'measurement.json'))
            block=p['reference_blocks'][index];ref=references[block['reference']]
            baseline,_=lossless.unpack(parent/'outputs/bank_comparison_v1/baseline/packs'/f"b{block['begin']//5:03d}.npz",block['case_ids'])
            with Image.open(parent/ref['target']) as image:target8=np.asarray(image.convert('RGB')).copy()
            with Image.open(parent/ref['observed']) as image:mask=np.asarray(image.convert('L')).copy()>0
            target=np.stack([target8]*5).astype(np.float32)/np.float32(255)
            photo_inputs.append(dict(baseline=baseline.transpose(0,3,1,2).copy(),target=target.transpose(0,3,1,2).copy(),
                                     observed=np.stack([mask]*5).astype(np.float32)[:,None]))
        def save_proposal(direction,vectors,solver_receipt):
            progress.update(stage='persist one candidate before exact certificate')
            reserve(direction.nbytes+3*609219*4+2*1024**2)
            assert list(vectors)==['quarter','half','full']
            np.save(out/'selected_direction.npy',direction,allow_pickle=False)
            for label,vector in vectors.items():np.save(out/('proposed_'+label+'_vector.npy'),vector,allow_pickle=False)
            write(out/'solver_proposal_receipt.json',dict(complete=True,solver=solver_receipt,
                direction_sha256=array_sha(direction),placed_vector_sha256={label:array_sha(v) for label,v in vectors.items()},
                certificate_pending=True,neural_assignments=0,model_qualified=False))
            clock()
            progress.update(stage='exact endpoint certificate')
        def save_solver(solver_receipt):
            reserve(2*1024**2)
            write(out/'solver_terminal_receipt.json',dict(complete=True,solver=solver_receipt,
                certificate_pending=True,neural_assignments=0,model_qualified=False))
            clock()
        counts_before_solve=dict(counts)
        solved=math_module.select_direction(initial,matrices,measurements,photo_inputs,p,clock,root,
            on_proposal=save_proposal,on_solver=save_solver)
        assert counts==counts_before_solve,'Saved-gradient solver must not execute a neural call'
        write(out/'solver_receipt.json',solved['solver_receipt']);write(out/'direction_certificate.json',solved['certificate'])
        assert solved['admitted'] is solved['admitted_to_neural']
        # Retain rejected rays/rounded states too; their failure is evidence.
        if solved['direction64'] is not None:
            assert np.array_equal(np.load(out/'selected_direction.npy',allow_pickle=False),solved['direction64'])
            for label,vector in solved['vectors'].items():
                assert np.array_equal(np.load(out/('proposed_'+label+'_vector.npy'),allow_pickle=False),vector)
        del matrices,measurements,photo_inputs
        clock()
        if not solved['admitted']:
            write(out/'no_certified_direction.json',dict(complete=True,no_neural_placements=True,model_qualified=False,no_automatic_follow_on=True))
            write(out/'no_model_state_created.json',dict(complete=True,neural_models_constructed=0,parameter_assignments=0,
                model_state_restore_not_needed=True,source_checkpoints_unchanged=True))
            write(out/'run_results.json',dict(complete=True,stop_reason='no_certified_direction',counts=counts,
                state_labels=[],certified_direction=False,new_parameter_placements=0,full_corpus_epochs=0,
                scientific_experiment_qualified=False,source_full_states_not_modified=True,
                model_qualified=False,new_incumbent_selection=False,automatic_follow_on=False,seconds=time.monotonic()-start))
            return 0
        vectors=solved['vectors'];assert list(vectors)==['quarter','half','full']
        remaining_projection=time.monotonic()-start+p['budgets']['cache_seconds']+4*p['budgets']['snapshot_seconds']+30
        write(out/'pre_neural_projection.json',dict(elapsed_seconds=time.monotonic()-start,
            conservative_remaining_cache_seconds=p['budgets']['cache_seconds'],remaining_snapshots=4,
            per_snapshot_cap_seconds=p['budgets']['snapshot_seconds'],projected_worker_seconds=remaining_projection,
            worker_cap_seconds=p['budgets']['worker_seconds']))
        assert remaining_projection<=p['budgets']['worker_seconds'],'Preserve finite snapshot budget before models'
        ctx=FeasibilityContext.__new__(FeasibilityContext)
        ctx.__init__(root,activation,repair,source,parent,replay,p,ap,repaired,old,data,out,counts,clock)
        np,torch=ctx.np,ctx.torch
        assert not torch.is_grad_enabled() and all(not v.requires_grad for v in ctx.model.parameters())
        write(out/'environment.json',dict(host=os.uname().nodename,GPU=torch.cuda.get_device_name(0),torch=torch.__version__,
            CUDA=torch.version.cuda,python=sys.version,
            packages={name:importlib.metadata.version(name) for name in ['torch','torchvision','numpy','Pillow','scipy','scikit-image','onnx','onnx2torch']},
            DGP_provenance=ctx.provenance,frozen_states=ctx.frozen_start,corrected_helper_sha256=sha(ctx.corrected.__file__),
            optimizer_present=False,peak_includes_dependency_loading=True,historical_training_called=False,new_gradient_queries=0))
        write(out/'direction_admission.json',dict(complete=True,certified=True,one_fixed_comparison=True,
            fixed_placements=['quarter','half','full'],quality_failure_does_not_authorize_continuation=True,model_qualified=False))
        with source_cwd(source):
            from frozen_raw_metrics import mean_only,pixel_metrics,detail_float
            from frozen_capacity_contract import groups,exported_pixel_metrics,detail_metric
            from cctv_dgp_spatial_readout_v1_contract import stage_gate
            reserve(full_state_bytes+2*1024**2);ctx.save_state('before',pin)
            baseline_groups=None
            for label in ['initial','quarter','half','full']:
                progress.update(stage='fixed placement snapshot',state=label,reference=None)
                ctx.assign_vector(ctx.initial if label=='initial' else vectors[label],label)
                began=time.monotonic();folder=out/label;folder.mkdir();(folder/'packs').mkdir();(folder/'png').mkdir()
                rows,objective_rows=[],[]
                vector=ctx.flat();np.save(out/(label+'_vector.npy'),vector,allow_pickle=False)
                write(out/(label+'_state.json'),dict(model_state=ctx.state_hash(ctx.model),buffer_state=ctx.buffer_hash(ctx.model),
                    fixed_state=ctx.fixed_hash,vector_sha256=array_sha(vector),installed_vector_exact=True,gradients_enabled=False,
                    optimizer_updates=0,new_gradient_queries=0,model_qualified=False))
                with torch.no_grad():
                    for index,b in enumerate(ctx.caches):
                        block=p['reference_blocks'][index];progress['reference']=block['reference'];g=ctx.gpu(b)
                        pred=ctx.forward(g)
                        value,current,baseline,mean_shift,embedding,record=ctx.objective(pred,g,b)
                        raw=pred.cpu().permute(0,2,3,1).numpy().copy()
                        png=np.stack([ctx.deliver(a,c,m) for a,c,m in zip(raw,b['camera'],b['mask8'])])
                        zero_png=[ctx.deliver(a,c,m) for a,c,m in zip(b['baseline'],b['camera'],b['mask8'])]
                        counts['recognizer_batch_calls']+=2
                        er=ctx.identity.embedding(torch.cat([pred,g['baseline'],g['target']]),torch.cat([g['mask']]*3),torch.cat([g['grid']]*3)).cpu().numpy()
                        ep=ctx.identity.embedding(torch.cat([ctx.tensor(list(png)),ctx.tensor(zero_png),g['target']]),torch.cat([g['mask']]*3),torch.cat([g['grid']]*3)).cpu().numpy()
                        for values,slot in [(er,0),(ep,1)]:
                            np.testing.assert_allclose(values[5:10],b['retained_embeddings'][:,slot],rtol=1e-6,atol=1e-6)
                            np.testing.assert_allclose(values[10:15],b['target_vectors'],rtol=1e-6,atol=1e-6)
                        if label=='initial':
                            with np.load(activation/'outputs/decoder_activation_v1/initial/packs'/f'r{index:02d}.npz',allow_pickle=False) as saved:
                                assert np.array_equal(raw,saved['raw']) and np.array_equal(embedding.cpu().numpy(),saved['embedding_batch5'])
                            assert all(np.array_equal(v,ctx.pixels(activation/'outputs/decoder_activation_v1/initial/png'/(c['id']+'.png'))) for v,c in zip(png,b['cases']))
                        reserve(5*256*256*3*5+2*1024**2)
                        np.savez_compressed(folder/'packs'/f'r{index:02d}.npz',ids=np.asarray(block['case_ids']),raw=raw,
                            embedding_batch5=embedding.cpu().numpy(),embeddings_raw=er,embeddings_png=ep,target_embeddings=b['target_vectors'])
                        objective_rows.append(dict(reference=block['reference'],source=block['source'],case_ids=block['case_ids'],
                            **record,raw_sha256=array_sha(raw),delivered_uint8_sha256=array_sha(png),model_qualified=False))
                        for i,case in enumerate(b['cases']):
                            mask,target=b['mask8'][i],b['target8'][i]
                            assert np.array_equal(raw[i][~mask],b['camera'][i][~mask].astype(np.float32)/np.float32(255))
                            assert np.array_equal(png[i][~mask],b['camera'][i][~mask])
                            mean,mean_png,_=mean_only(raw[i],b['baseline'][i],b['camera'][i],mask)
                            rm,pm=pixel_metrics(raw[i],target,mask),exported_pixel_metrics(png[i],target,mask)
                            for metric,ev,slot in [(rm,er,0),(pm,ep,1)]:
                                a,t=(b['retained_embeddings'][i,slot],b['target_vectors'][i]) if label=='initial' else (ev[i],ev[10+i])
                                metric['ArcFace_observed_fixed']=float(a@t)
                            rm.update(landmark_high_frequency_MSE=detail_float(raw[i],target,b['support8'][i]),constant_mean_shift_only_MSE=pixel_metrics(mean,target,mask)['MSE'])
                            pm.update(landmark_high_frequency_MSE=detail_metric(png[i],target,b['support8'][i]),constant_mean_shift_only_MSE=exported_pixel_metrics(mean_png,target,mask)['MSE'])
                            ctx.Image.fromarray(png[i]).save(folder/'png'/(case['id']+'.png'))
                            rows.append(dict(case=case['id'],source=case['source'],profile=case['profile'],raw=rm,png=pm))
                        del g,pred,value,current,baseline,mean_shift,embedding,raw,png,er,ep,zero_png
                        clock();assert time.monotonic()-began<=p['budgets']['snapshot_seconds'],'Finite snapshot stop'
                reductions={mode:groups([dict(source=row['source'],profile=row['profile'],metrics=row[mode]) for row in rows]) for mode in ['raw','png']}
                write(folder/'metrics.json',dict(rows=rows,groups=reductions,seconds=time.monotonic()-began))
                write(folder/'objective_components.json',dict(schema='fixed_batch5_objective_v1',complete=True,rows=objective_rows,
                    recognizer_batch=5,scientific_RAW_PNG_embedding_batch=15,model_qualified=False))
                if label=='initial':baseline_groups=reductions
                outcome=dict(label=label,gates={mode:stage_gate(baseline_groups[mode],reductions[mode]) for mode in ['raw','png']},
                    diagnostic_fixed_placement=True,quality_does_not_authorize_follow_on=True,model_qualified=False)
                write(out/(label+'_outcome.json'),outcome);snapshots.append(label)
                if label!='initial':
                    reserve(full_state_bytes+2*1024**2);ctx.save_state(label,pin)
                if label=='initial':
                    elapsed=time.monotonic()-began;projection=time.monotonic()-start+3*elapsed*1.5+30
                    write(out/'snapshot_projection.json',dict(snapshot_seconds=elapsed,remaining_snapshots=3,multiplier=1.5,
                        projected_worker_seconds=projection,worker_cap_seconds=p['budgets']['worker_seconds']))
                    assert projection<=p['budgets']['worker_seconds'],'Finite snapshot projection stop'
            assert ctx.buffer_hash(ctx.model)==ctx.initial_buffer_hash
            assert ctx.state_hash({k:v for k,v in ctx.model.state_dict().items() if k not in ctx.names})==ctx.fixed_hash
            restored=ctx.restore();write(out/'restoration.json',restored)
            assert restored['RNG_unchanged_before_restore']
            reserve(full_state_bytes+2*1024**2);ctx.save_state('after',pin)
        assert counts==dict(optimizer_updates=0,gradient_queries=0,backwards=0,original_batch_calls=20,
                            prior_feature_batch_calls=20,decoder_batch_calls=80,recognizer_batch_calls=240)
        clock()
        write(out/'run_results.json',dict(complete=True,stop_reason='three_fixed_placements_complete',counts=counts,
            state_labels=snapshots,certified_direction=True,new_parameter_placements=3,full_corpus_epochs=0,
            original_and_frozen_models_unchanged=True,initial_model_and_RNG_restored_exact=True,
            model_qualified=False,new_incumbent_selection=False,automatic_follow_on=False,seconds=time.monotonic()-start))
        return 0
    except BaseException as exc:
        save_error=restore_error=None
        if ctx is not None and hasattr(ctx,'model'):
            try:
                selected=ctx.flat();reserve(selected.nbytes+2*1024**2)
                np=ctx.np;np.save(out/'technical_stop_selected_vector.npy',selected,allow_pickle=False)
                write(out/'technical_stop_selected_state.json',dict(array_sha256=array_sha(selected),model_state=ctx.state_hash(ctx.model),
                    buffer_state=ctx.buffer_hash(ctx.model),exact_selected_vector_saved=True,model_qualified=False))
            except BaseException as stopped:save_error=repr(stopped)
            try:
                emergency=full_state_bytes+2*1024**2
                if size()+emergency<=p['budgets']['output_byte_cap'] and shutil.disk_usage(root).free>=emergency+p['budgets']['disk_reserve_bytes']:
                    with source_cwd(source):ctx.save_state('technical_stop',pin)
                elif save_error is None:
                    save_error='Optional full emergency state omitted to preserve finite output/export reserve; exact selected vector retained when available'
            except BaseException as stopped:save_error=repr(stopped)
            if hasattr(ctx,'CPU_rng'):
                try:
                    with source_cwd(source):restored=ctx.restore()
                    write(out/'failure_restoration.json',restored)
                except BaseException as stopped:restore_error=repr(stopped)
        write(out/'failure.json',dict(complete=False,error_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),
            progress=progress,counts=counts,completed_snapshots=snapshots,failure_state_save_error=save_error,
            restoration_error=restore_error,model_qualified=False,failed_recipe_resume_allowed=False,seconds=time.monotonic()-start))
        raise


def export(root, p, pin):
    require_manual(root)
    began = time.monotonic()
    manifest = root/'result_manifest.json'
    archive = Path.home()/(STEM+'-results.tar.gz')
    sidecar, receipt, partial = Path(str(archive)+'.sha256'), Path.home()/(STEM+'-export.json'), Path(str(archive)+'.partial')
    assert all(not value.exists() for value in (manifest, archive, sidecar, receipt, partial)), 'Preserve existing complete/partial export'
    files = [root/name for name in p['assets_sha256']] + [root/'protocol.json']
    files += sorted(value for value in (root/STAGE).rglob('*') if value.is_file())
    files += [root/name for name in ('run.log','run_exit.json','saved_audit.log','saved_audit_exit.json') if (root/name).is_file()]
    assert len(files) == len(set(files)) and all(not v.is_symlink() and v.resolve().is_relative_to(root) for v in files)
    total = sum(value.stat().st_size for value in files)
    assert total <= p['budgets']['output_byte_cap']+16*1024**2
    assert shutil.disk_usage(root).free >= total+p['budgets']['disk_reserve_bytes']
    result_path, failure_path = root/STAGE/'run_results.json', root/STAGE/'failure.json'
    result = None
    receipt_read_errors = []
    for candidate in (result_path, failure_path):
        if not candidate.is_file():
            continue
        try:
            value = read(candidate)
            queries = value['counts']['gradient_queries']
            assert isinstance(queries, int) and not isinstance(queries, bool) and 0 <= queries <= p['budgets']['maximum_gradient_queries']
            result = value
            break
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError, AssertionError) as invalid:
            # SIGKILL can interrupt a JSON write as well as prevent it. Retain
            # the opaque partial file, report unknown exact counts, and export.
            receipt_read_errors.append(dict(path=candidate.relative_to(root).as_posix(), error=repr(invalid)))
    last_progress = None
    progress_path = root/STAGE/'progress.jsonl'
    if progress_path.is_file():
        for line in progress_path.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # A killed final write can be incomplete opaque evidence.
            if isinstance(row, dict) and isinstance(row.get('counts'), dict):
                last_progress = row
    terminal = read(root/'run_exit.json') if (root/'run_exit.json').is_file() else None
    assert result is not None or terminal is not None, 'Need a worker receipt or authoritative terminal supervisor receipt'
    exact_gradient_queries = result['counts']['gradient_queries'] if result is not None else None
    partial_without_receipt = result is None
    bindings = {}
    for path in files:
        bindings[path.relative_to(root).as_posix()] = dict(bytes=path.stat().st_size, sha256=sha(path))
        assert time.monotonic()-began <= p['budgets']['export_seconds']
    write(manifest, dict(complete=True, protocol_sha256=pin, files=bindings, optimizer_updates=0,
        gradient_queries=exact_gradient_queries, gradient_query_count_exact=result is not None,
        last_known_progress=last_progress, terminal_worker=terminal,
        partial_terminal_without_worker_receipt=partial_without_receipt,
        incomplete_worker_receipt_errors=receipt_read_errors,
        model_qualified=False, training_success_not_implied=True))
    with tarfile.open(partial, 'x:gz', compresslevel=1) as stream:
        for path in [manifest]+files:
            assert time.monotonic()-began <= p['budgets']['export_seconds']
            assert shutil.disk_usage(root).free >= p['budgets']['disk_reserve_bytes']
            stream.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    digest = sha(partial)
    assert time.monotonic()-began <= p['budgets']['export_seconds']
    assert shutil.disk_usage(root).free >= p['budgets']['disk_reserve_bytes'] and not archive.exists()
    partial.rename(archive)
    with sidecar.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(digest+'  '+archive.name+'\n')
    write(receipt, dict(complete=True, protocol_sha256=pin, archive_sha256=digest, bytes=archive.stat().st_size,
        seconds=time.monotonic()-began, optimizer_updates=0, gradient_queries=exact_gradient_queries,
        gradient_query_count_exact=result is not None, last_known_progress=last_progress,
        terminal_worker=terminal, partial_terminal_without_worker_receipt=partial_without_receipt,
        incomplete_worker_receipt_errors=receipt_read_errors,
        model_qualified=False, training_success_not_implied=True,
        run_results_present=result_path.is_file(), failure_present=failure_path.is_file()))
    print(json.dumps(read(receipt)), flush=True)


def child(root, activation, repair, source, parent, replay, pin, flag, seconds, log_name):
    import selectors
    args = [sys.executable, '-B', '-u', str(root/'scripts/cctv_dgp_decoder_feasibility_v1_vm.py'),
        '--root', str(root), '--activation-root', str(activation), '--repair-root', str(repair), '--source-root', str(source),
        '--parent-root', str(parent), '--replay-root', str(replay), '--protocol-sha', pin, flag]
    began, timeout, killed = time.monotonic(), False, False
    with (root/log_name).open('xb') as log, selectors.DefaultSelector() as selector:
        process = subprocess.Popen(args, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True, bufsize=0)
        print(json.dumps(dict(flag=flag, child_pid=process.pid, log=log_name, external_cap_seconds=seconds)), flush=True)
        selector.register(process.stdout, selectors.EVENT_READ)
        while selector.get_map() or process.poll() is None:
            elapsed = time.monotonic()-began
            if elapsed > seconds and not timeout and process.poll() is None:
                timeout = True
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            if elapsed > seconds+30 and process.poll() is None and not killed:
                killed = True
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            for key, _ in selector.select(.2):
                value = os.read(key.fileobj.fileno(), 65536)
                if value:
                    log.write(value)
                    log.flush()
                    sys.stdout.buffer.write(value)
                    sys.stdout.buffer.flush()
                else:
                    selector.unregister(key.fileobj)
        code = process.wait()
    return dict(flag=flag, exit_code=code, timeout=timeout, kill_sent=killed, seconds=time.monotonic()-began,
                child_pid=process.pid, maximum_seconds=seconds, kill_grace_seconds=30)


def supervise(root, activation, repair, source, parent, replay, p, ap, repaired, old, data, pin):
    machine_guard(root, idle=True)
    reserved = ['outputs','result_manifest.json','run.log','run_exit.json','saved_audit.log','saved_audit_exit.json','export.log','supervisor_receipt.json']
    reserved += [str(Path.home()/(STEM+'-results.tar.gz'+suffix)) for suffix in ('','.partial','.sha256')]
    reserved += [str(Path.home()/(STEM+'-export.json'))]
    assert all(not (root/name).exists() for name in reserved), 'Used run/export; preserve evidence'
    worked = child(root, activation, repair, source, parent, replay, pin, '--run', p['budgets']['external_seconds'], 'run.log')
    write(root/'run_exit.json', worked)
    checked = None
    if (root/STAGE/'run_results.json').is_file() or (root/STAGE/'failure.json').is_file():
        checked = child(root, activation, repair, source, parent, replay, pin, '--audit-saved', p['budgets']['saved_audit_external_seconds'], 'saved_audit.log')
        write(root/'saved_audit_exit.json', checked)
    exported = child(root, activation, repair, source, parent, replay, pin, '--export', p['budgets']['export_external_seconds'], 'export.log')
    write(root/'supervisor_receipt.json', dict(complete=True, run=worked, saved_audit=checked, export=exported,
        model_qualified=False, optimizer_updates=0, automatic_follow_on=False))
    return 0 if worked['exit_code'] == 0 and checked and checked['exit_code'] == 0 and exported['exit_code'] == 0 else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    for name in ['activation-root','repair-root','source-root','parent-root','replay-root']:
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--protocol-sha',required=True)
    flags=parser.add_mutually_exclusive_group(required=True)
    for name in ['verify-transfer','run','audit-saved','export','supervise']:
        flags.add_argument('--'+name,action='store_true')
    args=parser.parse_args();root=args.root.resolve()
    activation=args.activation_root.resolve() if args.activation_root else root.parent/ACTIVATION_NAME
    repair=args.repair_root.resolve() if args.repair_root else root.parent/REPAIR_NAME
    source=args.source_root.resolve() if args.source_root else root.parent/SOURCE_NAME
    parent=args.parent_root.resolve() if args.parent_root else root.parent/PARENT_NAME
    replay=args.replay_root.resolve() if args.replay_root else root.parent/REPLAY_NAME
    if not args.verify_transfer:require_manual(root)
    p,ap,rp,old,data=verify_transfer(root,activation,repair,source,parent,replay,args.protocol_sha)
    if args.verify_transfer:
        print(json.dumps(dict(complete=True,assets=len(p['assets_sha256']),activation_assets=len(p['activation_required_sha256']),
            local_neural_calls=0,gradient_queries=0,optimizer_updates=0)),flush=True)
    elif args.run:
        raise SystemExit(run(root,activation,repair,source,parent,replay,p,ap,rp,old,data,args.protocol_sha))
    elif args.export:export(root,p,args.protocol_sha)
    elif args.audit_saved:
        from audit_cctv_dgp_decoder_feasibility_v1_return import audit_saved
        audit_saved(root,activation,repair,source,parent,replay,p,ap,rp,old,data,root/STAGE/'independent_saved_audit_vm.json')
    else:raise SystemExit(supervise(root,activation,repair,source,parent,replay,p,ap,rp,old,data,args.protocol_sha))


if __name__=='__main__':main()
