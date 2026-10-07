"""Prospective independent return audit: local inference only, no returned code."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs/cctv_dgp_feature_fusion_vm_v32'
PARENT = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'
MIXED = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'
OUT = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_return'
PIN = 'FUSION_V32_PROTOCOL_PIN'
PREFIX = OUT.name + '/'


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def helpers():
    path = ROOT / 'scripts/audit_cctv_dgp_mean_centered_decoder_v29_return.py'
    assert sha(path) == '6704bf6390e87f823a8136c54e9105f1d3312fa5d910e45f99581be54f0096f8'
    spec = importlib.util.spec_from_file_location('pinned_local_v29_return_helpers', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.BUNDLE = BUNDLE; module.OUT = OUT; module.PIN = PIN
    old_matrix = module.matrix
    module.matrix = lambda a, shape=(7,978243): old_matrix(a,shape)
    return module


def allowed_names(p):
    names = {'protocol.json', 'export_manifest.json', 'trainer.log', 'trainer_exit_code.txt',
             'supervisor_receipt.json', *p['assets_sha256']}
    names.update('outputs/' + n for n in ['gradient_components.npy', 'gradient_summary.json',
                 'cohort_loss_setup.json', 'gradient_preflight.json', 'cache_timing.json', 'cache_receipt.json',
                 'results.json', 'failure.json', 'timing_update20.json', 'early_structure_stop.json',
                 'execution_receipt.json', 'stopped_dgp_candidate_v32.pth'])
    names.update('outputs/gradients/batch' + str(i) + '.npy' for i in range(10))
    for cid in p['preview_case_ids']:
        names.update('outputs/initial_baseline/' + cid + s for s in ['.npy', '.png', '_target_embedding.npy'])
    for update in [0, 50, 400, 800]:
        prefix = 'outputs/update' + str(update) + '/'
        names.update(prefix + n for n in ['metrics.json', 'dgp_candidate_v32.pth'])
        for c in p['case_rows']:
            names.update(prefix + c['id'] + s for s in ['.png', '_mean_only.png', '_embedding.npy'])
            if update == 0: names.add(prefix + c['id'] + '_target_embedding.npy')
            if c['id'] in p['preview_case_ids']: names.add(prefix + c['id'] + '.npy')
    return names


def validate_members(members, allowed, cap=3*1024**3):
    rows, seen, total = [], set(), 0
    for m in members:
        assert m.isfile() and not m.issym() and not m.islnk(), 'Regular files only'
        assert m.name.startswith(PREFIX) and '\\' not in m.name and ':' not in m.name
        name = m.name[len(PREFIX):]; parts = PurePosixPath(name).parts
        assert parts and not PurePosixPath(name).is_absolute() and all(x not in ['.', '..'] for x in parts)
        assert name in allowed and name.casefold() not in seen, 'Unexpected or colliding member: ' + name
        assert 0 <= m.size <= 64*1024**2
        total += m.size; seen.add(name.casefold()); rows.append((m, name))
        assert total <= cap and len(rows) <= 60000, 'Archive finite bounds'
    return rows, total


def import_return(expected_sha, expected_bytes, p):
    archive = ROOT / 'outputs/cctv-dgp-feature-fusion-v32-results.tar.gz'
    assert re.fullmatch('[a-f0-9]{64}', expected_sha) and 0 < expected_bytes <= 3*1024**3
    assert archive.stat().st_size == expected_bytes and sha(archive) == expected_sha
    assert Path(str(archive)+'.sha256').read_text().strip().split() == [expected_sha, archive.name]
    exported = read(ROOT / 'outputs/cctv-dgp-feature-fusion-v32-export.json')
    assert exported['complete'] and exported['archive_sha256'] == expected_sha and exported['bytes'] == expected_bytes
    assert exported['training_success_not_implied'] and not OUT.exists(), 'Preserve prior imports and partials'
    with tarfile.open(archive, 'r:gz') as tar:
        rows, total = validate_members(tar.getmembers(), allowed_names(p))
        OUT.mkdir()
        for member, name in rows:
            path = (OUT / name).resolve(); assert path.is_relative_to(OUT)
            path.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as f, path.open('xb') as g: shutil.copyfileobj(f, g)
            assert path.stat().st_size == member.size
    files = {name: sha(OUT/name) for _, name in rows}
    assert sha(OUT/'protocol.json') == PIN
    for name, digest in p['assets_sha256'].items(): assert files[name] == digest
    manifest = read(OUT/'export_manifest.json')
    assert manifest['complete'] and manifest['protocol_sha256'] == PIN and manifest['quality_acceptance_not_implied']
    assert manifest['files_sha256'] == {k:v for k,v in files.items() if k != 'export_manifest.json'}
    receipt = {'complete':True,'archive_sha256':expected_sha,'archive_bytes':expected_bytes,
               'members':len(rows),'uncompressed_bytes':total,'files_sha256':files,
               'local_neural_or_gradient_calls':0,'goal_complete':False}
    write(ROOT/'outputs/cctv_dgp_feature_fusion_v32_return_import.json', receipt)
    return exported, receipt


def gradients(p, base, h):
    arrays, checked = [], 0
    folder = OUT/'outputs/gradients'
    for path in sorted(folder.glob('batch*.npy')) if folder.exists() else []:
        assert int(path.stem[5:]) == len(arrays)
        a = h.matrix(np.load(path,allow_pickle=False)); assert np.count_nonzero(a[3:]) == 0
        arrays.append(a); checked += a.size
    summary_path = OUT/'outputs/gradient_summary.json'
    if summary_path.exists():
        r = read(summary_path); assert len(arrays) == len(r['batches']) == 10
        assert r['parameter_layout'] == p['parameter_layout'] and r['terms'] == p['terms']
        total = h.matrix(np.load(OUT/'outputs/gradient_components.npy',allow_pickle=False)); checked += total.size
        assert np.array_equal(total, sum(arrays,np.zeros_like(total)))
        assert r['gradient_array_sha256'] == sha(OUT/'outputs/gradient_components.npy')
        values = np.zeros(7,np.float64)
        for i, (row,a) in enumerate(zip(r['batches'], arrays)):
            assert row['batch'] == i and row['ids'] == [c['id'] for c in base['cases'][i*5:i*5+5]]
            assert row['gradient_array_sha256'] == sha(folder/('batch'+str(i)+'.npy'))
            assert row['initial_raw_and_PNG_parity_exact'] and row['initial_all23_preservation_gradients_exact_zero']
            assert row['component_values'][3:] == [0,0,0,0]
            assert np.allclose(np.linalg.norm(a,axis=1),row['component_norms'],rtol=2e-10,atol=1e-11)
            values += row['component_values']
        assert np.array_equal(values,np.asarray(r['component_values'])) and r['objective'] == float(values.sum())
        norms, gram, cosine, blocks = h.statistics(total,p['parameter_layout'])
        for actual,saved in [(norms,r['component_norms']),(gram,r['component_gram']),(cosine,r['component_cosines'])]:
            assert np.allclose(actual,saved,rtol=2e-10,atol=1e-11)
        for name,block in blocks.items():
            for key,value in block.items(): assert np.allclose(value,r['per_parameter_gradients'][name][key],rtol=2e-10,atol=1e-11)
    proof_path = OUT/'outputs/gradient_preflight.json'; proof = proof_path.exists()
    if proof:
        q = read(proof_path)
        assert len(arrays) == 10 and q['complete'] and q['protocol_sha256'] == PIN
        assert q['component_gradient_calls'] == 70 and q['parameter_layout'] == p['parameter_layout']
        assert q['selected_parameters'] == 978243 and q['selected_tensors'] == 23 and q['fusion_parameters'] == 479616
        assert q['decoder_parameters'] == 498627 and q['decoder_parameter_tensors'] == 12
        assert q['raw_and_PNG_parity_exact_cases'] == 50 and q['all_selected23_improvement_gradients_nonzero']
        assert q['initial_preservation_gradients_exact_zero_all_batches'] and all(b['improvement_gradient_norm'] > 0 for b in blocks.values())
        assert q['reference_DGP_forwards'] == q['candidate_DGP_forwards'] == 10 and q['recognizer_forwards'] == 20
        assert q['optimizer_updates'] == q['epochs'] == 0 and not q['optimizer_constructed'] and not q['new_checkpoint_created']
        assert q['DGP_state_before_after'] == p['original_DGP_state'] and q['recognizer_state_before_after'] == p['frozen_recognizer_state']
        assert 0 < q['seconds'] < 300 and q['peak_allocated_VRAM_bytes'] <= 20*1024**3
    return proof, checked


def verify_early_receipt(receipt, gain):
    assert receipt['update'] == 50 and receipt['minimum'] == .01
    assert np.isfinite(gain) and np.isfinite(receipt['relative_feature_error_gain'])
    assert abs(receipt['relative_feature_error_gain'] - gain) <= 1e-12
    assert receipt['pass'] == (gain >= .01) == (receipt['relative_feature_error_gain'] >= .01)


def audit(expected_sha, expected_bytes):
    started = time.monotonic(); assert sha(BUNDLE/'protocol.json') == PIN
    p = read(BUNDLE/'protocol.json'); base = read(PARENT/'protocol.json'); h = helpers()
    for name,digest in p['assets_sha256'].items(): assert sha(BUNDLE/name) == digest
    for name,digest in p['mixed_TRAIN_assets_sha256'].items(): assert sha(MIXED/name) == digest
    assert sha(MIXED/'mixed_protocol_v9.json') == p['mixed_data_protocol_sha256']
    for name,digest in p['local_basis_sha256'].items(): assert sha(ROOT/name) == digest
    for name,digest in base['assets_sha256'].items(): assert sha(PARENT/name) == digest
    assert p['initial_proof_case_rows'] == base['cases'] and p['retained_capacity_gates'] == base['prospective_gates']
    spec = importlib.util.spec_from_file_location('pinned_V32_metadata_schedule', BUNDLE/'cctv_dgp_profile_batches_v31_schedule.py')
    schedule_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(schedule_module)
    schedule_module.validate_profile_batches(p['case_rows'], read(BUNDLE/'schedule.json')['batches'])
    assert sorted(i for b in read(BUNDLE/'schedule.json')['batches'][:781] for i in b)==list(range(3905))
    old = ROOT/'outputs/cctv_dgp_broader_mean_vm_v30'
    returned_old = ROOT/'outputs/cctv_dgp_broader_mean_v30_return'
    oldp=read(old/'protocol.json')
    for name,digest in p['closed_V30_evidence_sha256'].items():
        path=old/name if name=='protocol.json' or name in oldp['assets_sha256'] else returned_old/name
        assert sha(path)==digest
    sampling=ROOT/'outputs/cctv_dgp_v30_sampling_gradient_v1_return'
    for name,digest in p['closed_sampling_evidence_sha256'].items(): assert sha(sampling/name)==digest
    closed31 = ROOT/'outputs/cctv_dgp_profile_batches_vm_v31'
    return31 = ROOT/'outputs/cctv_dgp_profile_batches_v31_return'
    old31p = read(closed31/'protocol.json')
    for name,digest in p['closed_V31_evidence_sha256'].items():
        path = closed31/name if name=='protocol.json' or name in old31p['assets_sha256'] else return31/name
        assert sha(path)==digest
    fusion_return = ROOT/'outputs/cctv_dgp_feature_fusion_gradient_v1_return'
    for name,digest in p['closed_fusion_diagnostic_evidence_sha256'].items(): assert sha(fusion_return/name)==digest
    spec = importlib.util.spec_from_file_location('pinned_V32_parameter_policy', BUNDLE/'cctv_dgp_feature_fusion_v32_policy.py')
    policy = importlib.util.module_from_spec(spec); spec.loader.exec_module(policy)
    policy.validate_layout(p['parameter_layout'])
    assert p['selected_parameters']==978243 and p['selected_tensors']==23
    exported, imported = import_return(expected_sha,expected_bytes,p)
    result_path, failure_path = OUT/'outputs/results.json', OUT/'outputs/failure.json'
    assert result_path.exists() != failure_path.exists()
    terminal = read(result_path if result_path.exists() else failure_path)
    assert terminal['protocol_sha256'] == PIN and 0 <= terminal['optimizer_updates'] <= 800
    assert not terminal['app_promotion'] and not terminal['goal_complete']
    assert exported['optimizer_updates'] == terminal['optimizer_updates'] and exported['failure_present'] == failure_path.exists()
    assert exported['run_results_present'] == result_path.exists()
    supervision = read(OUT/'supervisor_receipt.json'); exit_code = int((OUT/'trainer_exit_code.txt').read_text())
    assert supervision['protocol_sha256'] == PIN and supervision['trainer_exit_code'] == exit_code
    assert supervision['cap_seconds'] == 4800 and supervision['kill_grace_seconds'] == 30
    assert supervision['within_external_bound'] == (supervision['seconds'] <= 4830)
    assert (exit_code == 0) == result_path.exists()
    proof, gradient_values = gradients(p,base,h)
    sys.path.insert(0,str(ROOT/'outputs/cctv_dgp_active_original_decoder_vm_v28'))
    sys.path.insert(0,str(PARENT)); sys.path.insert(0,str(BUNDLE))
    replay,cohort,initial_recognizer,original_state = h.initial_CPU_replay(started,p,base,proof)
    import torch
    from PIL import Image
    from scipy.ndimage import convolve1d
    from cctv_dgp_pilot import FixedObservedIdentity, grid112, state_hash, exported_pixel_metrics
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_app_input_v28 import canonical_tensor
    from cctv_dgp_feature_fusion_v32_training import frozen_partition
    from cctv_dgp_mean_centered_decoder_v29 import center_observed_delta
    torch.set_num_threads(4)
    selected = {r['name'] for r in p['parameter_layout']}
    original = torch.load(PARENT/'weights/dgp_v2.pth',map_location='cpu',weights_only=True)
    def checkpoint(path,initial=False):
        state = torch.load(path,map_location='cpu',weights_only=True)
        assert isinstance(state,dict) and set(state) == set(original)
        for name,value in state.items():
            assert isinstance(value,torch.Tensor) and value.dtype == original[name].dtype and value.shape == original[name].shape and torch.isfinite(value).all()
            if name not in selected or initial: assert torch.equal(value,original[name]),name
        return state
    recognizer = FixedObservedIdentity(PARENT/'weights/w600k_r50.onnx','cpu')
    recog_state = state_hash(recognizer); assert recog_state == p['frozen_recognizer_state']
    baseline_net,_ = load_frozen_dgp_restorer(PARENT/'weights/dgp_v2.pth',expected_sha256=p['original_checkpoint_sha256'],device='cpu')
    baseline_state = state_hash(baseline_net.net); assert baseline_state == p['original_DGP_state']
    refs = {r['id']:r for r in p['training_references']}; ref_cache = {}
    z=np.arange(-6,7,dtype=np.float64);kernel=np.exp(-.5*(z/2)**2);kernel/=kernel.sum()
    high=lambda a:a-convolve1d(convolve1d(a,kernel,axis=0,mode='reflect'),kernel,axis=1,mode='reflect')
    def data(c):
        rid = c['source_person_or_reference']
        if rid not in ref_cache:
            with Image.open(MIXED/c['target']) as im: target=np.asarray(im.convert('RGB')).copy()
            with Image.open(MIXED/c['observed']) as im: mask=np.asarray(im).copy()>0
            padded=np.pad(mask.astype(np.int64),6);summed=np.pad(padded.cumsum(0).cumsum(1),((1,0),(1,0)))
            interior=summed[13:,13:]-summed[:-13,13:]-summed[13:,:-13]+summed[:-13,:-13]==169
            feature=np.zeros((256,256),bool)
            for point in c['landmarks5_canvas_xy']:
                xx,yy=np.floor(point).astype(int);feature[max(0,yy-12):min(256,yy+12),max(0,xx-12):min(256,xx+12)]=True
            feature &= interior; assert feature.any()
            ref_cache[rid]=(target,mask,feature,torch.from_numpy(grid112(refs[rid]['matrix112']))[None])
        with Image.open(MIXED/c['input']) as im: camera=np.asarray(im.convert('RGB')).copy()
        return camera,*ref_cache[rid]
    snapshots, partial_snapshots, CPU_original, CPU_candidate, CPU_recognizer = [],[],0,0,0
    for update in [0,50,400,800]:
        folder=OUT/('outputs/update'+str(update))
        if not folder.exists(): continue
        assert proof
        if not (folder/'metrics.json').exists():
            assert failure_path.exists(), 'Incomplete snapshot cannot be a completed result'
            checkpoint(folder/'dgp_candidate_v32.pth',initial=update==0)
            partial_snapshots.append({'update':update,'files_retained':sum(f.is_file() for f in folder.iterdir()),
                                      'all3905_outputs_or_metrics_not_implied':True})
            continue
        saved=read(folder/'metrics.json'); checkpoint(folder/'dgp_candidate_v32.pth',initial=update==0)
        net,_=load_frozen_dgp_restorer(folder/'dgp_candidate_v32.pth',expected_sha256=sha(folder/'dgp_candidate_v32.pth'),device='cpu')
        before=state_hash(net.net); frozen=frozen_partition(net.net,selected)
        assert before==saved['candidate_DGP_state'] and frozen==saved['frozen_partition_sha256']
        assert sha(folder/'dgp_candidate_v32.pth')==saved['candidate_checkpoint_sha256'] and saved['snapshot_batch_size']==5
        assert len(saved['rows'])==3905
        actual_rows=[]; worst_raw=0.;worst_byte=0;worst_vector=0.;replayed=0
        with torch.inference_mode():
            for c,reported in zip(p['case_rows'],saved['rows']):
                assert time.monotonic()-started<7200,'V32 independent inference audit7200s cap'
                assert reported['id']==c['id'] and reported['source']==c['source'] and reported['profile']==c['profile']
                cid=c['id'];camera,target,mask,feature,grid=data(c)
                with Image.open(folder/(cid+'.png')) as im: png=np.asarray(im.convert('RGB')).copy()
                with Image.open(folder/(cid+'_mean_only.png')) as im: mean_png=np.asarray(im.convert('RGB')).copy()
                for a in [png,mean_png]: assert a.shape==camera.shape==(256,256,3) and np.array_equal(a[~mask],camera[~mask])
                vector=np.load(folder/(cid+'_embedding.npy'),allow_pickle=False)
                truth=np.load(OUT/'outputs/update0'/(cid+'_target_embedding.npy'),allow_pickle=False)
                for a in [vector,truth]: assert a.dtype==np.float32 and a.shape==(512,) and np.isfinite(a).all() and abs(float(a@a)-1)<1e-5
                metrics=exported_pixel_metrics(png,target,mask);metrics['ArcFace_observed_fixed']=float(vector@truth)
                luma=np.array([.299,.587,.114])
                metrics['landmark_high_frequency_MSE']=float(np.square(high((png.astype(np.float64)/255*luma).sum(2))-high((target.astype(np.float64)/255*luma).sum(2)))[feature].mean())
                metrics['constant_mean_shift_only_MSE']=exported_pixel_metrics(mean_png,target,mask)['MSE']
                for key,value in metrics.items(): assert abs(value-reported['metrics'][key])<=1e-10,(update,cid,key)
                if update==800 or cid in p['preview_case_ids']:
                    x=canonical_tensor(camera,'cpu');support=torch.from_numpy(mask)[None,None]
                    fresh_base=torch.where(support,baseline_net(x),x);CPU_original+=1
                    raw=center_observed_delta(torch.where(support,net(x),x),fresh_base,x,support)[0].permute(1,2,0).numpy().copy();CPU_candidate+=1
                    expected=np.where(mask[...,None],np.floor(raw*np.float32(255)),camera).astype(np.uint8)
                    byte=int(np.abs(expected.astype(int)-png.astype(int)).max()); assert byte<=1,(update,cid,'CPU/CUDA PNG')
                    worst_byte=max(worst_byte,byte)
                    if cid in p['preview_case_ids']:
                        stored=np.load(folder/(cid+'.npy'),allow_pickle=False)
                        assert stored.dtype==np.float32 and stored.shape==(256,256,3) and np.isfinite(stored).all() and (stored>=0).all() and (stored<=1).all()
                        error=float(np.abs(raw-stored).max());assert error<=1e-5;worst_raw=max(worst_raw,error)
                        assert np.array_equal(png,np.where(mask[...,None],np.floor(stored*np.float32(255)),camera).astype(np.uint8))
                    fresh_vector=recognizer.embedding(canonical_tensor(png,'cpu'),support.float(),grid)[0].numpy();CPU_recognizer+=1
                    fresh_truth=recognizer.embedding(canonical_tensor(target,'cpu'),support.float(),grid)[0].numpy();CPU_recognizer+=1
                    error=max(float(np.abs(fresh_vector-vector).max()),float(np.abs(fresh_truth-truth).max()))
                    assert error<=5e-5;worst_vector=max(worst_vector,error)
                    base_raw=fresh_base[0].permute(1,2,0).numpy()
                    shift=(raw-base_raw)[mask].astype(np.float64).mean(0)
                    assert float(np.abs(shift-np.asarray(reported['postclip_mean_RGB_shift'])).max())<=2e-5
                    fresh_mean=np.clip(base_raw.astype(np.float64)+np.asarray(reported['postclip_mean_RGB_shift']),0,1).astype(np.float32)
                    fresh_mean_png=np.where(mask[...,None],np.floor(fresh_mean*np.float32(255)),camera).astype(np.uint8)
                    assert int(np.abs(fresh_mean_png.astype(int)-mean_png.astype(int)).max())<=1
                    replayed+=1
                actual_rows.append({'id':cid,'source':c['source'],'profile':c['profile'],'metrics':metrics})
                if len(actual_rows)%250==0: print({'V32_audit_snapshot':update,'cases':len(actual_rows),'of':3905,'seconds':time.monotonic()-started},flush=True)
        derived=h.groups(actual_rows);assert set(derived)==set(saved['groups']) and len(derived)==17
        for key,g in derived.items():
            for metric,value in g.items(): assert abs(value-saved['groups'][key][metric])<=1e-10
        assert state_hash(net.net)==before and all(not v.requires_grad and v.grad is None for v in net.parameters())
        snapshots.append({'update':update,'groups':derived,'neural_replayed_cases':replayed,
                          'CPU_raw_maximum_error_on50_previews':worst_raw,'CPU_PNG_maximum_byte_error':worst_byte,
                          'CPU_vector_maximum_error':worst_vector,'frozen_partition_sha256':frozen})
    assert state_hash(recognizer)==recog_state and state_hash(baseline_net.net)==baseline_state
    assert all(not v.requires_grad and v.grad is None for m in [recognizer,baseline_net] for v in m.parameters())
    updates=[r['update'] for r in snapshots];assert updates==[0,50,400,800][:len(updates)]
    stopped=OUT/'outputs/stopped_dgp_candidate_v32.pth'
    if stopped.exists(): checkpoint(stopped)
    timing_path=OUT/'outputs/timing_update20.json'
    if timing_path.exists():
        timing=read(timing_path);samples=timing['steady_sample_seconds'];initial=read(OUT/'outputs/update0/metrics.json')
        overhead=3*initial['snapshot_duration_seconds']*1.25
        assert timing['updates']==20 and len(samples)==19 and timing['remaining_updates']==780 and timing['cap_seconds']==3600
        assert timing['safety_factor']==1.25 and timing['overhead_seconds']==overhead
        assert timing['projected_seconds']==timing['seconds']+780*float(np.mean(samples))*1.25+overhead
    cache_path=OUT/'outputs/cache_timing.json'
    if cache_path.exists():
        c=read(cache_path);assert c['references']==781 and c['cases']==100 and len(c['steady_sample_seconds'])==19 and c['remaining_batches']==761
        assert c['cap_seconds']==900 and c['safety_factor']==1.25
        assert c['projected_seconds']==c['seconds']+761*float(np.mean(c['steady_sample_seconds']))*1.25
    early_path=OUT/'outputs/early_structure_stop.json'
    if early_path.exists():
        e=read(early_path);assert updates[:2]==[0,50]
        gain=1-snapshots[1]['groups']['degraded']['landmark_high_frequency_MSE']/snapshots[0]['groups']['degraded']['landmark_high_frequency_MSE']
        verify_early_receipt(e,gain)
    capacity=False
    ex_path=OUT/'outputs/execution_receipt.json'
    if ex_path.exists():
        ex=read(ex_path);assert ex['complete'] and ex['protocol_sha256']==PIN and ex['fit_cap_seconds']==3600 and ex['worker_cap_seconds']==4500
        assert len(ex['step_times_seconds'])==terminal['optimizer_updates'] and all(x>=0 and np.isfinite(x) for x in ex['step_times_seconds'])
        for key in ['optimizer_updates','epochs','backwards','optimizer_constructed','new_checkpoint_created']:assert ex[key]==terminal[key]
        assert ex['epochs']==ex['optimizer_updates']/781 and ex.get('completed_epochs',0)==ex['optimizer_updates']//781
    if result_path.exists():
        assert proof and not partial_snapshots and updates==[0,50,400,800] and terminal['complete'] and terminal['optimizer_updates']==terminal['backwards']==800
        assert terminal['epochs']==800/781 and terminal['training_cases']==3905 and terminal['training_references']==781
        failures,gain,source_gains,brightness,capacity=h.capacity(snapshots[0]['groups'],snapshots[-1]['groups'])
        assert failures==terminal['preservation_failures'] and gain==terminal['degraded_feature_MSE_relative_gain']
        assert source_gains==terminal['source_feature_gains'] and brightness==terminal['brightness_gain_fraction'] and capacity==terminal['necessary_capacity_pass']
        assert terminal['parameter_partition_changed_only'] and terminal['clear_and_degraded_views_paired_in_every_batch'] and terminal['mean_centered_path_used'] and terminal['original_seven_loss_weights_unchanged']
        assert ex['fit_seconds']<=3600 and ex['worker_seconds']<=4500 and ex['peak_allocated_VRAM_bytes']<=20*1024**3 and supervision['within_external_bound']
        assert ex['reference_DGP_forwards']==791 and ex['candidate_DGP_forwards']==3934 and ex['recognizer_forwards']==4725 and ex['component_gradient_calls']==70
    receipt={'complete':True,'checker_sha256':sha(Path(__file__)),'protocol_sha256':PIN,'archive_sha256':expected_sha,
             'archive_bytes':expected_bytes,'members_verified':imported['members'],'TRAIN_assets_verified':5467,
             'initial50_CPU_replay':len(replay),'cohort_rows_checked':len(cohort),'selected23_preflight_complete':proof,
             'saved_gradient_values_checked':gradient_values,'snapshots_audited':snapshots,'partial_snapshots_retained':partial_snapshots,
             'CPU_original_DGP_forwards':CPU_original+len(replay),'CPU_candidate_DGP_forwards':CPU_candidate,
             'CPU_recognizer_forwards':CPU_recognizer+initial_recognizer,
             'all_final3905_neural_replayed':800 in updates and snapshots[-1]['neural_replayed_cases']==3905,
             'completed_snapshot_PNG_metrics_vectors_and_mean_controls_audited':len(snapshots)*3905,
             'VM_failure_retained':failure_path.exists(),'training_completed800':result_path.exists(),'necessary_capacity_pass':capacity,
             'local_gradient_calls':0,'local_backward_calls':0,'local_optimizer_updates':0,
             'native_or_reserved_used':False,'app_promotion':False,'independent_final_review':False,
             'goal_complete':False,'seconds':time.monotonic()-started}
    write(ROOT/'outputs/cctv_dgp_feature_fusion_v32_independent_audit.json',receipt)
    print(json.dumps({k:receipt[k] for k in ['complete','VM_failure_retained','training_completed800','necessary_capacity_pass','seconds']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--expected-sha',required=True);parser.add_argument('--expected-bytes',required=True,type=int)
    args=parser.parse_args();audit(args.expected_sha,args.expected_bytes)
