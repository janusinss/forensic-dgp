"""Prospective V35 audit: saved arrays, finite measurements and frozen CPU replay."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import time
from types import MethodType

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'outputs/cctv_dgp_finite_clearance_probe_v36_vm'
OUT=ROOT/'outputs/cctv_dgp_finite_clearance_probe_v36_return'
PREFIX='cctv_dgp_finite_clearance_probe_v36_return/'
STEM='cctv-dgp-finite-clearance-probe-v36'
PARENT=ROOT/'outputs/cctv_dgp_feature_skips_vm_v27'
ACTIVE=ROOT/'outputs/cctv_dgp_active_original_decoder_vm_v28'
V32=ROOT/'outputs/cctv_dgp_feature_fusion_vm_v32_r2'
MIXED=ROOT/'outputs/cctv_dgp_mixed_vm_v9_r2'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    encoded=json.dumps(value,indent=2,allow_nan=False)+'\n'
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:stream.write(encoded)


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def verify_basis(p):
    for name,digest in p['assets_sha256'].items():assert sha(BUNDLE/name)==digest,name
    for name,digest in p['local_basis_sha256'].items():assert sha(ROOT/name)==digest,name
    audit=read(ROOT/'outputs/cctv_dgp_group_guard_grad_v34_independent_audit.json')
    assert audit['complete'] and audit['diagnostic_complete'] and not audit['failure_retained']
    analysis=read(ROOT/'outputs/cctv_dgp_group_guard_grad_v34_analysis_v1_r1/independent_analysis_audit.json')
    assert analysis['complete'] and analysis['group_rows_independently_reassembled']==102
    prior=module('pinned_V33_readonly_basis',ROOT/'scripts/audit_cctv_dgp_loss_cone_probe_v33_return_r2.py')
    basis=prior.verify_basis(read(ROOT/'outputs/cctv_dgp_loss_cone_probe_v33_vm/protocol.json'))
    assert basis['cohorts']==p['cohorts'] and basis['parameter_layout']==p['parameter_layout'] and basis['terms']==p['terms']
    guard=ROOT/'outputs/cctv_dgp_group_guard_grad_v34_return'
    for name,digest in p['guard_return_sha256'].items():assert sha(guard/name)==digest,name
    return basis


def safe_members(members,p):
    allowed=set(p['assets_sha256'])|{'protocol.json','export_manifest.json','probe.log','probe_exit_code.txt',
        'supervisor_receipt.json','outputs/results.json','outputs/failure.json','outputs/geometry_verification.json'}
    for cohort in p['cohorts']:
        start='outputs/state0_'+cohort['name']+'/'
        allowed.add(start+'theta_before.npy')
        for variant in ['before']+[v['name'] for v in p['variants']]:
            sub=start+variant+'/'
            allowed.add(sub+'receipt.json')
            if variant!='before':allowed.add(sub+'comparison.json')
            for case in cohort['cases']:
                for suffix in ['.npy','.png','_embedding.npy','_raw_embedding.npy']:
                    allowed.add(sub+case['id']+suffix)
                if variant=='before':allowed.add(sub+case['id']+'_target_embedding.npy')
    seen,result,total=set(),[],0
    for member in members:
        assert member.isfile() and not member.issym() and not member.islnk()
        assert member.name.startswith(PREFIX) and not any(c in member.name for c in ['\\',':','\x00'])
        name=member.name[len(PREFIX):];parts=PurePosixPath(name).parts
        assert parts and not PurePosixPath(name).is_absolute() and all(v not in ['.','..'] for v in parts)
        assert name in allowed and name.lower() not in seen
        # Shipped float64 displacement arrays need8MiB each; raw outputs<1MiB.
        assert 0<=member.size<=8*1024**2
        total+=member.size;seen.add(name.lower());result.append((member,name))
        assert total<=p['budgets']['export_uncompressed_bytes'] and len(seen)<=p['budgets']['return_files_maximum']
    return result,total


def import_return(p,pin,digest,size):
    archive=ROOT/'outputs'/(STEM+'-results.tar.gz')
    assert archive.stat().st_size==size and sha(archive)==digest
    assert Path(str(archive)+'.sha256').read_text().strip().split()==[digest,archive.name]
    exported=read(ROOT/'outputs'/(STEM+'-export.json'))
    assert exported['complete'] and exported['archive_sha256']==digest and exported['bytes']==size
    assert exported['training_success_not_implied'] and exported['optimizer_updates']==0
    assert not OUT.exists(),'Retain every preceding or partial return audit'
    with tarfile.open(archive,'r:gz') as tar:
        members,total=safe_members(tar.getmembers(),p);OUT.mkdir()
        for item,name in members:
            destination=(OUT/name).resolve();assert destination.is_relative_to(OUT)
            destination.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(item) as source,destination.open('xb') as stream:
                for block in iter(lambda:source.read(1024**2),b''):stream.write(block)
    hashes={name:sha(OUT/name) for _,name in members}
    assert hashes['protocol.json']==pin
    for name,digest in p['assets_sha256'].items():assert hashes[name]==digest,name
    manifest=read(OUT/'export_manifest.json')
    assert manifest['complete'] and manifest['protocol_sha256']==pin
    assert manifest['files_sha256']=={name:digest for name,digest in hashes.items() if name!='export_manifest.json'}
    receipt={'complete':True,'archive_sha256':digest,'archive_bytes':size,'members':len(members),
        'uncompressed_bytes':total,'files_sha256':hashes,'returned_code_executed':False}
    # Avoid comprehension variable shadowing of the input archive digest.
    receipt['archive_sha256']=sha(archive)
    write(ROOT/'outputs/cctv_dgp_finite_clearance_probe_v36_return_import.json',receipt)
    return exported,receipt


def proposals(p,state,cohort):
    import numpy as np
    assert state==0
    theta=np.load(BUNDLE/'theta_before.npy',allow_pickle=False)
    direction=np.load(BUNDLE/'projected_displacement.npy',allow_pickle=False)
    assert np.array_equal(theta,np.load(OUT/f'outputs/state0_{cohort["name"]}/theta_before.npy',allow_pickle=False))
    return theta,None,{'projected_restoration':direction},None


def finite_metrics(p):
    import numpy as np
    from PIL import Image
    metrics=module('independent_V35_delivered_metrics',ROOT/'scripts/cctv_dgp_loss_cone_probe_v33_metrics.py')
    reports=[];checked=0
    for cohort in p['cohorts']:
        label=cohort['name'];parent=OUT/f'outputs/state0_{label}'
        before=read(parent/'before/receipt.json')
        for variant in ['before']+[v['name'] for v in p['variants']]:
            folder=parent/variant;r=read(folder/'receipt.json')
            assert r['complete'] and r['state']==0 and r['cohort']==label and r['variant']==variant and r['cases']==50
            assert [row['id'] for row in r['rows']]==[c['id'] for c in cohort['cases']]
            fresh=[]
            for case,row in zip(cohort['cases'],r['rows']):
                cid=case['id']
                def png(path):
                    with Image.open(path) as im:return np.asarray(im.convert('RGB')).copy()
                raw=np.load(folder/(cid+'.npy'),allow_pickle=False)
                output=png(folder/(cid+'.png'));target=png(MIXED/case['target']);camera=png(MIXED/case['input'])
                with Image.open(MIXED/case['observed']) as im:mask=np.asarray(im).copy()>0
                feature=np.zeros((256,256),bool)
                for point in case['landmarks5_canvas_xy']:
                    xx,yy=np.floor(point).astype(int);feature[max(0,yy-12):min(256,yy+12),max(0,xx-12):min(256,xx+12)]=True
                feature&=metrics.erode(mask,6)
                baseline=np.load(parent/'before'/(cid+'.npy'),allow_pickle=False)
                vector=np.load(folder/(cid+'_embedding.npy'),allow_pickle=False)
                raw_vector=np.load(folder/(cid+'_raw_embedding.npy'),allow_pickle=False)
                truth=np.load(parent/'before'/(cid+'_target_embedding.npy'),allow_pickle=False)
                measured=metrics.case_metrics(raw,output,target,camera,mask,feature,baseline,vector,raw_vector,truth)
                for key,value in measured.items():assert np.allclose(value,row['metrics'][key],rtol=2e-10,atol=1e-11),key
                fresh.append({**row,'metrics':measured});checked+=1
            assert metrics.groups(r['rows'])==r['groups']
            fresh_groups=metrics.groups(fresh)
            for group in fresh_groups:
                for name,value in fresh_groups[group].items():assert np.isclose(value,r['groups'][group][name],rtol=2e-10,atol=1e-11)
            if variant!='before':
                comparison=read(folder/'comparison.json')
                exact=metrics.compare_groups(before['groups'],r['groups'])
                assert comparison['preservation_against_original']==exact
                independent=metrics.compare_groups(before['groups'],fresh_groups)
                keys=lambda d:{(r['group'],r['metric']) for r in d['failures']}
                assert keys(independent)==keys(exact) and independent['brightness_gate_pass']==exact['brightness_gate_pass']
                assert {k:np.sign(v) for k,v in independent['source_structure_gains'].items()}=={k:np.sign(v) for k,v in exact['source_structure_gains'].items()}
                expected_gain=1-r['groups']['degraded']['landmark_high_frequency_MSE']/before['groups']['degraded']['landmark_high_frequency_MSE']
                assert comparison['incremental_degraded_PNG_structure_gain']==expected_gain
                assert np.allclose(np.asarray(r['raw_component_means'])-before['raw_component_means'],comparison['finite_raw_component_change'],rtol=2e-10,atol=1e-11)
            reports.append({'cohort':label,'variant':variant,'cases':50,'groups':17})
    assert checked==500
    return reports


def CPU_replay(p, basis, started):
    import numpy as np
    from PIL import Image
    import torch
    from torch.nn import functional as F
    for path in [PARENT, ACTIVE, V32]: sys.path.insert(0, str(path))
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_pilot import FixedObservedIdentity, grid112, state_hash
    from cctv_dgp_mean_centered_decoder_v29 import center_observed_delta
    from cctv_dgp_app_input_v28 import canonical_tensor
    from cctv_dgp_batchmatched_identity_v26 import objective_terms
    metrics = module('pinned_probe_erode_only', ROOT / 'scripts/cctv_dgp_loss_cone_probe_v33_metrics.py')
    helper = module('pinned_original_loss_AST_only', V32 / 'scripts/cctv_dgp_feature_fusion_v32_vm.py')
    _, definitions, filters = helper.original_functions(PARENT)
    ns = {'torch': torch, 'F': F}; exec(compile(ast.Module(body=filters, type_ignores=[]), '<pinned-fixed-filter-CPU>', 'exec'), ns)
    class Fixed: pass
    fixed = Fixed(); z = torch.arange(-6, 7, dtype=torch.float32); k = torch.exp(-.5 * (z / 2).square()); fixed.kernel = k / k.sum()
    fixed.reflect_indices = torch.cat((torch.arange(5, -1, -1), torch.arange(256), torch.arange(255, 249, -1)))
    fixed.blur = MethodType(ns['blur'], fixed); fixed.high = MethodType(ns['high'], fixed)
    original, _ = load_frozen_dgp_restorer(PARENT / 'weights/dgp_v2.pth', expected_sha256=basis['original_checkpoint_sha256'], device='cpu')
    candidate, _ = load_frozen_dgp_restorer(PARENT / 'weights/dgp_v2.pth', expected_sha256=basis['original_checkpoint_sha256'], device='cpu')
    identity = FixedObservedIdentity(PARENT / 'weights/w600k_r50.onnx', 'cpu').eval().requires_grad_(False)
    ns.update({'head': fixed, 'identity': identity}); exec(compile(ast.Module(body=definitions, type_ignores=[]), '<pinned-seven-loss-CPU>', 'exec'), ns)
    normalizers = tuple(torch.tensor(v, dtype=torch.float32) for v in basis['normalizers'])
    initial = {name: value.detach().clone() for name, value in candidate.net.state_dict().items()}
    baseline_states = [state_hash(original.net), state_hash(identity)]; torch.set_num_threads(4)
    assert baseline_states == [basis['original_DGP_state'], basis['recognizer_state']]
    references = {r['id']: r for r in read(V32 / 'protocol.json')['training_references']}
    keys = ['x', 'base', 'target', 'mask', 'feature', 'interior', 'valid7', 'grid', 'truth', 'degraded_weight', 'clear_weight']
    counts = {'original_DGP_forwards': 0, 'candidate_DGP_forwards': 0, 'recognizer_forwards': 0}
    for model, key in [(original.net, 'original_DGP_forwards'), (candidate.net, 'candidate_DGP_forwards'), (identity.encoder, 'recognizer_forwards')]:
        model.register_forward_hook(lambda *_args, key=key: counts.__setitem__(key, counts[key] + 1))
    worst_raw, worst_png, worst_vector, worst_term, checked = 0., 0, 0., 0., 0
    with torch.inference_mode():
        for cohort in p['cohorts']:
            label, items = cohort['name'], []
            for case in cohort['cases']:
                with Image.open(MIXED / case['input']) as im: camera = np.asarray(im.convert('RGB')).copy()
                with Image.open(MIXED / case['target']) as im: target = np.asarray(im.convert('RGB')).copy()
                with Image.open(MIXED / case['observed']) as im: mask = np.asarray(im).copy() > 0
                interior = metrics.erode(mask, 6); feature = np.zeros((256, 256), bool)
                for point in case['landmarks5_canvas_xy']:
                    xx, yy = np.floor(point).astype(int); feature[max(0, yy - 12):min(256, yy + 12), max(0, xx - 12):min(256, xx + 12)] = True
                feature &= interior
                items.append({'case': case, 'camera': camera, 'mask8': mask, 'x': canonical_tensor(camera, 'cpu'), 'target': canonical_tensor(target, 'cpu'),
                              'mask': torch.from_numpy(mask.astype(np.float32))[None, None], 'interior': torch.from_numpy(interior.astype(np.float32))[None, None],
                              'feature': torch.from_numpy(feature.astype(np.float32))[None, None], 'valid7': torch.from_numpy(metrics.erode(mask, 3).astype(np.float32))[None, None],
                              'grid': torch.from_numpy(grid112(references[case['source_person_or_reference']]['matrix112']))[None],
                              'degraded_weight': torch.tensor([0. if case['profile'] == 'clear' else 1.25]), 'clear_weight': torch.tensor([1. if case['profile'] == 'clear' else 0.])})
            for begin in range(0, 50, 5):
                group = items[begin:begin + 5]; x = torch.cat([i['x'] for i in group]); mask = torch.cat([i['mask'] for i in group]); grid = torch.cat([i['grid'] for i in group])
                baseline = torch.where(mask.bool(), original(x), x); truth = identity.embedding(torch.cat([i['target'] for i in group]), mask, grid)
                for index, item in enumerate(group): item['base'] = baseline[index:index + 1]; item['truth'] = truth[index:index + 1]
            for state, weights in [(0, initial)]:
                theta, _, deltas, _ = proposals(p, state, cohort); folder = OUT / f'outputs/state{state}_{label}'
                candidate.net.load_state_dict(weights, strict=True)
                selected = dict(candidate.net.named_parameters())
                actual_before = torch.cat([selected[r['name']].reshape(-1) for r in p['parameter_layout']]).numpy()
                assert np.array_equal(actual_before, theta)
                for variant in ['before'] + [v['name'] for v in p['variants']]:
                    assert time.monotonic() - started < p['budgets']['local_audit_seconds']
                    candidate.net.load_state_dict(weights, strict=True)
                    spec = None if variant == 'before' else next(v for v in p['variants'] if v['name'] == variant)
                    if spec:
                        array = (theta.astype(np.float64) - spec['scale'] * deltas[spec['proposal']]).astype(np.float32)
                        for row in p['parameter_layout']: selected[row['name']].copy_(torch.from_numpy(array[row['start']:row['end']].copy()).reshape(row['shape']))
                    path = folder / variant; receipt = read(path / 'receipt.json'); assert state_hash(candidate.net) == receipt['candidate_state']
                    for begin in range(0, 50, 5):
                        group = items[begin:begin + 5]
                        if not all(i['case']['id'] in p['CPU_replay_case_ids'][label] for i in group): continue
                        b = {key: torch.cat([i[key] for i in group]) for key in keys}
                        pred = center_observed_delta(torch.where(b['mask'].bool(), candidate(b['x']), b['x']), b['base'], b['x'], b['mask'])
                        raw = pred.permute(0, 2, 3, 1).numpy().copy(); delivered = []
                        for item, fresh in zip(group, raw):
                            cid = item['case']['id']; saved = np.load(path / (cid + '.npy'), allow_pickle=False)
                            error = float(np.abs(saved - fresh).max()); assert error <= p['CPU_raw_absolute_tolerance']; worst_raw = max(worst_raw, error)
                            with Image.open(path / (cid + '.png')) as im: png = np.asarray(im.convert('RGB')).copy()
                            expected = np.where(item['mask8'][..., None], np.floor(fresh * np.float32(255)), item['camera']).astype(np.uint8)
                            byte = int(np.abs(png.astype(int) - expected.astype(int)).max()); assert byte <= p['CPU_PNG_byte_tolerance']; worst_png = max(worst_png, byte)
                            delivered.append(png); checked += 1
                        vectors = identity.embedding(torch.cat([canonical_tensor(a, 'cpu') for a in delivered]), b['mask'], b['grid']).numpy()
                        raw_vectors = identity.embedding(pred, b['mask'], b['grid']).numpy()
                        for item, vector, raw_vector in zip(group, vectors, raw_vectors):
                            cid = item['case']['id']; truth = np.load(OUT / f'outputs/state0_{label}/before/{cid}_target_embedding.npy', allow_pickle=False)
                            error = max(float(np.abs(vector - np.load(path / (cid + '_embedding.npy'), allow_pickle=False)).max()),
                                        float(np.abs(raw_vector - np.load(path / (cid + '_raw_embedding.npy'), allow_pickle=False)).max()), float(np.abs(truth - item['truth'][0].numpy()).max()))
                            assert error <= p['CPU_vector_absolute_tolerance']; worst_vector = max(worst_vector, error)
                        terms = objective_terms(b, pred, identity, ns['mean'], ns['feature_errors'], ns['ssim'], normalizers)
                        values = torch.stack([terms[name] for name in p['terms']], 1).numpy()
                        expected = np.asarray([r['raw_terms'] for r in receipt['rows'][begin:begin + 5]])
                        error = float(np.abs(values - expected).max()); assert error <= p['CPU_component_value_tolerance']; worst_term = max(worst_term, error)
                    candidate.net.load_state_dict(weights, strict=True)
            candidate.net.load_state_dict(initial, strict=True)
    assert checked == p['CPU_replay_outputs'] == 100
    assert [state_hash(original.net), state_hash(identity)] == baseline_states and state_hash(candidate.net) == basis['original_DGP_state']
    assert all(not v.requires_grad and v.grad is None for model in [original, candidate, identity] for v in model.parameters())
    return {'outputs': checked, **counts, 'raw_maximum_error': worst_raw, 'PNG_maximum_byte_error': worst_png,
            'vector_maximum_error': worst_vector, 'component_value_maximum_error': worst_term, 'all_states_restored': True}


def audit(digest,size):
    started=time.monotonic();p=read(BUNDLE/'protocol.json');pin=sha(BUNDLE/'protocol.json')
    basis=verify_basis(p);exported,imported=import_return(p,pin,digest,size)
    success,failure=OUT/'outputs/results.json',OUT/'outputs/failure.json'
    assert success.exists()!=failure.exists();result=read(success if success.exists() else failure)
    assert result['protocol_sha256']==pin and result['optimizer_updates']==result['committed_trajectory_updates']==result['new_gradient_queries']==result['backwards']==result['epochs']==0
    assert not result['new_checkpoint_created'] and not result['app_promotion'] and not result['goal_complete']
    assert exported['run_results_present']==success.exists() and exported['failure_present']==failure.exists()
    code=int((OUT/'probe_exit_code.txt').read_text());supervisor=read(OUT/'supervisor_receipt.json')
    assert supervisor['protocol_sha256']==pin and supervisor['probe_exit_code']==code and (code==0)==success.exists()
    assert supervisor['cap_seconds']==930 and supervisor['kill_grace_seconds']==30
    assert supervisor['within_external_bound']==(supervisor['seconds']<=960)
    reports=[];replay=None
    if success.exists():
        import numpy as np
        assert result['raw_outputs']==500 and result['candidate_displacement_trials']==4 and result['seconds']<=900
        assert result['peak_allocated_VRAM_bytes']<=p['budgets']['peak_vram_bytes']
        assert {k:result[k] for k in p['forward_call_limits']}==p['forward_call_limits']
        assert result['original_DGP_state']==basis['original_DGP_state'] and result['recognizer_state']==basis['recognizer_state'] and result['all_trial_states_reset']
        geometry=read(OUT/'outputs/geometry_verification.json')
        assert geometry['complete'] and geometry['preservation_guard_rows']==102 and geometry['existing_restoration_rows']==6
        assert geometry['KKT_stationarity_error']<=1e-12
        matrix=np.load(ROOT/'outputs/cctv_dgp_group_guard_grad_v34_analysis_v1_r1/group_guard_matrix.npy',allow_pickle=False)
        component=[]
        for cohort in p['cohorts']:
            g=np.load(ROOT/f'outputs/cctv_dgp_v32_loss_gradient_v1_return/outputs/state0_{cohort["name"]}/gradient_components.npy',allow_pickle=False)
            assert np.count_nonzero(g[3:])==0;component.append(g[:3])
        guards=np.concatenate([matrix,*component],axis=0);theta=np.load(BUNDLE/'theta_before.npy',allow_pickle=False);direction=np.load(BUNDLE/'projected_displacement.npy',allow_pickle=False)
        for cohort in p['cohorts']:
            for variant in p['variants']:
                folder=OUT/f'outputs/state0_{cohort["name"]}'/variant['name'];comparison=read(folder/'comparison.json')
                delta=theta.astype(np.float64)-(theta.astype(np.float64)-variant['scale']*direction).astype(np.float32).astype(np.float64)
                assert np.allclose(-(guards@delta),comparison['all108_linear_function_changes'],rtol=2e-10,atol=1e-11)
        reports=finite_metrics(p);replay=CPU_replay(p,basis,started)
    else:
        assert not result['resume_permitted'] and 0<=result['candidate_displacement_trials']<=4
    assert time.monotonic()-started<=1200
    receipt={'complete':True,'finite_probe_complete':success.exists(),'failure_retained':failure.exists(),
        'archive_sha256':digest,'archive_bytes':size,'protocol_sha256':pin,'checker_sha256':sha(Path(__file__)),
        'members_verified':imported['members'],'finite_output_arithmetic':reports,'CPU_replay':replay,
        'local_gradient_calls':0,'local_optimizer_updates':0,'retained_checkpoints_modified':False,
        'training_capacity_pass':False,'app_promotion':False,'goal_complete':False,'seconds':time.monotonic()-started}
    write(ROOT/'outputs/cctv_dgp_finite_clearance_probe_v36_independent_audit.json',receipt)
    print(json.dumps({'complete':True,'finite_probe_complete':success.exists(),'members_verified':imported['members'],'seconds':receipt['seconds']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--expected-sha',required=True);parser.add_argument('--expected-bytes',type=int,required=True)
    a=parser.parse_args();audit(a.expected_sha,a.expected_bytes)
