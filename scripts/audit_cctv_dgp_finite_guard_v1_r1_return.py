"""Prospective independent return audit: saved arithmetic and CPU inference only."""
import argparse
import math
from pathlib import Path
import shutil
import sys
import tarfile
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'outputs/cctv_dgp_finite_guard_v1_r1_vm'
OUT = ROOT/'outputs/cctv_dgp_finite_guard_v1_r1_vm_return'
sys.path.insert(0, str(BUNDLE))
from cctv_dgp_finite_guard_v1_r1_contract import NAME, STEM, STATE, BUDGETS, FRACTIONS, read, write, sha, verified_assets
from frozen_capacity_contract import capacity, feature_support, exported_pixel_metrics, detail_metric
from frozen_raw_metrics import pixel_metrics, detail_float, deliver, mean_only, review_groups
from cctv_dgp_finite_guard_v1_r1_training import mechanics_accept
# This immutable prior checker supplies arithmetic roundoff checks, not a worker.
from audit_cctv_dgp_group_conflicts_v1_return import pixels, exact_structure, compare_capacity_receipt, BRIGHTNESS_ARITHMETIC_ROWS


def certificate_check(matrix, receipt, folder, p):
    assert receipt['complete'] and receipt['global_infeasibility_proven'] is False
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, receipt['gradient_norms'], rtol=1e-12, atol=1e-12)
    found = receipt['common_direction_found']; assert isinstance(found, bool)
    if np.any(norms == 0):
        assert not found and receipt['dual_solver_iterations'] == 0
        assert not (folder/'common_direction.npy').exists()
        return None
    unit = matrix/norms[:, None]; gram = unit@unit.T
    assert np.allclose(gram, receipt['normalized_Gram'], rtol=0, atol=1e-12)
    weights = np.asarray(receipt['simplex_weights'], np.float64)
    assert weights.shape == (64,) and np.isfinite(weights).all() and (weights >= 0).all()
    assert abs(weights.sum()-1.) <= 1e-12
    combined = weights@unit; norm = float(np.linalg.norm(combined))
    assert abs(norm-receipt['combined_gradient_norm']) <= 1e-12
    direction = -combined/norm if norm > 0 else np.zeros(1996035, np.float64)
    cosines = -(unit@direction)
    assert np.allclose(cosines, receipt['normalized_descent_cosines'], rtol=0, atol=1e-12)
    assert np.allclose(matrix@direction, receipt['raw_directional_derivatives'], rtol=1e-12, atol=1e-12)
    assert abs(float(cosines.min())-receipt['minimum_normalized_descent_cosine']) <= 1e-12
    assert receipt['minimum_descent_cosine_requirement'] == 1e-7
    assert found == bool(receipt['dual_solver_success'] and norm > 0 and cosines.min() >= 1e-7)
    assert receipt['dual_solver_iterations'] <= 500 and receipt['seconds'] <= 120
    gap = 2.*(float(weights@gram@weights)-float(np.min(gram@weights)))
    assert abs(gap-receipt['duality_gap']) <= 1e-12
    if found:
        saved = np.load(folder/'common_direction.npy', allow_pickle=False)
        assert saved.dtype == np.float64 and saved.shape == (1996035,) and np.isfinite(saved).all()
        assert np.allclose(saved, direction, rtol=0, atol=p['direction_reconstruction_arithmetic_atol'])
        assert abs(float(np.linalg.norm(saved))-1.) <= 1e-12
        return saved
    assert not (folder/'common_direction.npy').exists()
    return None


def recompute(label, folder, p, deadline):
    saved = read(folder/label/'metrics.json')
    assert saved['complete'] and saved['variant'] == label and not saved['model_qualification']
    assert [r['id'] for r in saved['rows']] == [c['id'] for c in p['cases']]
    previous_label = saved['previous_reference_variant']
    rows = []
    for c, report in zip(p['cases'], saved['rows']):
        assert time.monotonic() <= deadline, 'Local independent audit deadline'
        cid = c['id']; camera = pixels(BUNDLE/c['input']); target = pixels(BUNDLE/c['target'])
        mask = pixels(BUNDLE/c['observed'], 'L') > 0
        feature = feature_support(mask, c['landmarks5_canvas_xy'])
        raw = np.load(folder/label/(cid+'.npy'), allow_pickle=False)
        base = np.load(folder/'baseline'/(cid+'.npy'), allow_pickle=False)
        previous = np.load(folder/previous_label/(cid+'.npy'), allow_pickle=False)
        png = pixels(folder/label/(cid+'.png'))
        assert raw.dtype == np.float32 and raw.shape == (256,256,3) and np.isfinite(raw).all()
        assert np.array_equal(raw[~mask], camera[~mask].astype(np.float32)/np.float32(255))
        assert np.array_equal(deliver(raw, camera, mask), png)
        mraw, mpng, shift = mean_only(raw, base, camera, mask)
        praw, ppng, pshift = mean_only(raw, previous, camera, mask)
        assert np.array_equal(mpng, pixels(folder/label/(cid+'_mean_only.png')))
        vectors = np.load(folder/label/(cid+'_embeddings.npy'), allow_pickle=False)
        assert vectors.dtype == np.float32 and vectors.shape == (3,512) and np.isfinite(vectors).all()
        assert np.allclose(np.sum(vectors.astype(np.float64)**2, axis=1), 1, rtol=0, atol=1e-5)
        if label != 'baseline':
            assert np.array_equal(vectors[2], np.load(folder/'baseline'/(cid+'_embeddings.npy'), allow_pickle=False)[2])
        rm = pixel_metrics(raw, target, mask); pm = exported_pixel_metrics(png, target, mask)
        rm.update({'ArcFace_observed_fixed': float(vectors[0]@vectors[2]),
            'landmark_high_frequency_MSE': detail_float(raw,target,feature),
            'constant_mean_shift_only_MSE': pixel_metrics(mraw,target,mask)['MSE']})
        pm.update({'ArcFace_observed_fixed': float(vectors[1]@vectors[2]),
            'landmark_high_frequency_MSE': detail_metric(png,target,feature),
            'constant_mean_shift_only_MSE': exported_pixel_metrics(mpng,target,mask)['MSE']})
        fresh = {'id': cid, 'source': c['source'], 'profile': c['profile'], 'raw': rm, 'png': pm,
            'postclip_mean_RGB_shift': shift.tolist(), 'previous_postclip_mean_RGB_shift': pshift.tolist(),
            'previous_constant_mean_shift_only_MSE': {'raw': pixel_metrics(praw,target,mask)['MSE'],
                'png': exported_pixel_metrics(ppng,target,mask)['MSE']}}
        exact_structure(report, fresh); rows.append(fresh)
    def grouped(values):
        return {co['name']: {s: review_groups([r for r in values if r['id'] in co['case_ids']], s)
            for s in ['raw','png']} for co in p['cohorts']}
    def previous_rows(values):
        return [{**r, **{s: {**r[s], 'constant_mean_shift_only_MSE':
            r['previous_constant_mean_shift_only_MSE'][s]} for s in ['raw','png']}} for r in values]
    groups, pgroups = grouped(rows), grouped(previous_rows(rows))
    assert saved['groups'] == grouped(saved['rows'])
    assert saved['previous_anchor_groups'] == grouped(previous_rows(saved['rows']))
    exact_structure(saved['groups'], groups); exact_structure(saved['previous_anchor_groups'], pgroups)
    return saved, groups, pgroups


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--expected-sha', required=True); a = ap.parse_args()
    start = time.monotonic(); deadline = start + 2400
    pin = sha(BUNDLE/'protocol.json'); p = verified_assets(BUNDLE,pin)
    for name,digest in p['local_sources_sha256'].items(): assert sha(ROOT/name) == digest, name
    archive = ROOT/'outputs'/(STEM+'-results.tar.gz')
    assert sha(archive) == a.expected_sha
    assert Path(str(archive)+'.sha256').read_text().split() == [a.expected_sha, archive.name]
    export = read(ROOT/'outputs'/(STEM+'-export.json'))
    assert export['complete'] and export['archive_sha256'] == a.expected_sha and export['bytes'] == archive.stat().st_size
    assert not OUT.exists(), 'Retain every previous import or partial failure'
    with tarfile.open(archive,'r:gz') as tar:
        members = tar.getmembers(); seen = set(); size = 0
        for m in members:
            assert m.isfile() and not m.issym() and not m.islnk()
            parts = m.name.split('/')
            assert parts[0] == NAME+'_return' and all(part not in ['', '.', '..'] for part in parts)
            assert '\\' not in m.name and ':' not in m.name
            name = '/'.join(parts[1:]); assert name not in seen and name not in p['assets_sha256']; seen.add(name)
            assert name.startswith('outputs/') or name in ['protocol.json','export_manifest.json','supervisor_receipt.json','diagnostic.log','diagnostic_exit_code.txt']
            size += m.size; assert size <= BUDGETS['return_uncompressed_bytes'] + 32*1024**2
        assert len(seen) <= 7000
        assert {'protocol.json','export_manifest.json','supervisor_receipt.json','diagnostic.log','diagnostic_exit_code.txt'} <= seen
        OUT.mkdir()
        for m in members:
            destination = OUT/'/'.join(m.name.split('/')[1:])
            assert destination.resolve().is_relative_to(OUT.resolve())
            destination.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(m) as incoming, destination.open('xb') as target: shutil.copyfileobj(incoming,target)
        assert {q.relative_to(OUT).as_posix() for q in OUT.rglob('*') if q.is_file()} == seen
        assert all((OUT/'/'.join(m.name.split('/')[1:])).stat().st_size == m.size for m in members)
    assert sha(OUT/'protocol.json') == pin
    manifest = read(OUT/'export_manifest.json'); assert manifest['complete'] and manifest['protocol_sha256'] == pin
    assert set(manifest['files_sha256']) == seen-{'export_manifest.json'}
    for name,digest in manifest['files_sha256'].items(): assert sha(OUT/name) == digest
    success = (OUT/'outputs/results.json').is_file()
    assert success != (OUT/'outputs/failure.json').is_file()
    result = read(OUT/('outputs/results.json' if success else 'outputs/failure.json'))
    assert result['protocol_sha256'] == pin and 0 <= result['optimizer_updates'] <= 3
    assert result['committed_trajectory_updates'] == result['training_parameter_updates'] == result['optimizer_updates']
    assert result['epochs'] == result['backwards'] == 0 and not result['optimizer_constructed']
    assert export['optimizer_updates'] == result['optimizer_updates'] and export['accepted_changes_are_actual_training']
    outer = read(OUT/'supervisor_receipt.json')
    assert outer['protocol_sha256'] == pin and outer['cap_seconds'] == BUDGETS['external_seconds'] and outer['kill_grace_seconds'] == 30
    assert outer['worker_exit_code'] == int((OUT/'diagnostic_exit_code.txt').read_text())
    if not success:
        write(ROOT/'outputs/cctv_dgp_finite_guard_v1_r1_failure_import.json', {'complete': True,
            'archive_sha256': a.expected_sha, 'protocol_sha256': pin, 'members_verified': len(members),
            'failure': result, 'partial_neural_and_learning_evidence_not_audited': True,
            'model_qualification': False, 'goal_complete': False})
        print({'complete': True, 'failure_imported': True, 'model_qualification': False}); return
    assert outer['worker_exit_code'] == 0 and result['complete']
    assert result['seconds'] <= BUDGETS['worker_seconds'] and result['peak_allocated_VRAM_bytes'] <= BUDGETS['peak_vram_bytes']
    assert result['accepted_changes_are_actual_training'] and result['torch_optimizer_steps'] == 0
    for key in ['native_or_DEV_or_final_used','model_qualification','app_promotion','automatic_follow_on','resume_permitted','goal_complete']:
        assert result[key] is False
    assert result['former_cross_cohort_used_for_fitting'] and result['full_TRAIN_capacity_not_tested']
    assert result['states_before'] == result['states_after_restore']
    assert result['states_before']['original'] == result['states_before']['candidate'] == STATE
    assert result['states_before']['recognizer'] == p['recognizer_state']
    folder = OUT/'outputs'; steps, trials = result['steps'], result['trial_summaries']
    assert 1 <= len(steps) <= 3 and len(trials) <= 9
    assert result['gradient_queries'] == 320*len(steps)
    assert result['forward_counts'] == {'original':20,'candidate':20+20*len(steps)+20*len(trials),
        'recognizer':40+20*len(steps)+20*len(trials)}
    labels = ['baseline']+[t['variant'] for t in trials]
    assert result['trials_completed'] == labels[1:] and result['raw_and_PNG_cases'] == 100*len(labels)
    initial = np.load(folder/'initial_parameters.npy', allow_pickle=False)
    assert initial.dtype == np.float32 and initial.shape == (1996035,)
    assert np.array_equal(initial,np.load(BUNDLE/'evidence/initial_parameters.npy',allow_pickle=False))
    metric_data = {label: recompute(label,folder,p,deadline) for label in labels}
    assert metric_data['baseline'][0]['candidate_state'] == STATE
    current, current_label, accepted = initial.copy(), 'baseline', []
    used_labels = []
    for index, step in enumerate(steps,1):
        assert time.monotonic() <= deadline
        step_dir = folder/f'step_{index:02d}'
        assert step['iteration'] == index and step == read(step_dir/'step_receipt.json')
        assert step['current_state'] == metric_data[current_label][0]['candidate_state']
        expected_file = 'initial_parameters.npy' if current_label == 'baseline' else current_label+'_parameters.npy'
        assert step['current_vector_file'] == expected_file
        assert np.array_equal(current,np.load(step_dir/'current_parameters.npy',allow_pickle=False))
        with np.load(step_dir/'group_gradients.npz',allow_pickle=False) as saved:
            assert saved.files == ['gradients']; matrix = saved['gradients'].copy()
        assert matrix.dtype == np.float64 and matrix.shape == (64,1996035) and np.isfinite(matrix).all()
        grad = read(step_dir/'gradient_receipt.json')
        assert grad['complete'] and grad['groups'] == p['group_losses'] and grad['candidate_state'] == step['current_state']
        assert grad['seconds'] <= BUDGETS['gradient_seconds_per_state'] and grad['gradient_queries'] == 320
        assert grad['aggregate_vectors'] == 64 and len(grad['rows']) == 320
        assert grad['individual_query_vectors_exported'] is False and grad['local_autograd_replay_permitted'] is False
        query_counts = [0]*64; seen_queries = set()
        for row in grad['rows']:
            gi,bi = row['group_index'],row['batch']
            assert 0 <= gi < 64 and 0 <= bi < 20 and (gi,bi) not in seen_queries
            seen_queries.add((gi,bi)); query_counts[gi] += 1
            assert row['group'] == p['group_losses'][gi]
            assert row['case_ids'] == [c['id'] for c in p['cases'][bi*5:bi*5+5]]
            assert set(row['case_ids']) & set(row['group']['case_ids'])
            assert len(row['query_vector_sha256']) == 64 and set(row['query_vector_sha256']) <= set('0123456789abcdef')
            assert math.isfinite(row['query_L2']) and row['query_L2'] >= 0 and math.isfinite(row['value'])
            assert len(row['parameters']) == 158
            total = 0.
            for desc,value in zip(p['parameter_layout'],row['parameters']):
                assert value['name'] == desc['name'] and isinstance(value['graph_connected'],bool)
                assert math.isfinite(value['L2']) and value['L2'] >= 0 and 0 <= value['nonzero_values'] <= desc['elements']
                assert (value['nonzero_values'] == 0) == (value['L2'] == 0)
                if not value['graph_connected']: assert value['L2'] == 0
                total += value['L2']**2
            assert abs(math.sqrt(total)-row['query_L2']) <= 1e-10
            group = row['group']; ids = [cid for cid in row['case_ids'] if cid in group['case_ids']]
            rows = {r['id']: r for r in metric_data[current_label][0]['rows']}
            key = {'MSE':'MSE','SSIM_loss':'SSIM','ArcFace_loss':'ArcFace_observed_fixed',
                'landmark_structure':'landmark_high_frequency_MSE'}[group['metric']]
            value = float(np.mean([rows[cid]['raw'][key] for cid in ids]))
            if group['metric'] in ['SSIM_loss','ArcFace_loss']: value = 1.-value
            tolerance = 3e-5 if group['metric'] == 'SSIM_loss' else 1e-6 if group['metric'] == 'ArcFace_loss' else 1e-10
            assert abs(row['value']-value) <= tolerance
        assert query_counts == [5]*64
        direction = certificate_check(matrix,read(step_dir/'direction_certificate.json'),step_dir,p)
        assert step['common_direction_found'] == (direction is not None)
        chosen = step['proposal_variants']
        if direction is None:
            assert not chosen and step['accepted_variant'] is None and index == len(steps)
            break
        assert 1 <= len(chosen) <= 3 and step['seconds_for_proposals'] <= BUDGETS['trial_seconds_per_state']
        expected_labels = [f'proposal_s{index:02d}_'+format(f,'.0e').replace('-','m') for f in FRACTIONS[:len(chosen)]]
        assert chosen == expected_labels
        step_accept = None
        for fraction,label in zip(FRACTIONS,chosen):
            used_labels.append(label)
            trial = next(t for t in trials if t['variant'] == label)
            assert trial == read(folder/(label+'_decision.json'))
            assert trial['iteration'] == index and trial['starting_vector_file'] == expected_file
            assert trial['relative_fraction'] == fraction and trial['selected_weight_L2'] == p['initial_decoder_weight_L2']
            actual = np.load(folder/(label+'_parameters.npy'),allow_pickle=False)
            expected = (current.astype(np.float64)+fraction*p['initial_decoder_weight_L2']*direction).astype(np.float32)
            assert np.array_equal(actual,expected)
            delta = actual.astype(np.float64)-current.astype(np.float64); dots = matrix@delta
            assert abs(float(np.linalg.norm(delta))-trial['actual_displacement_L2']) <= 1e-12
            assert np.allclose(dots,trial['gradient_dot_actual_displacement'],rtol=1e-12,atol=1e-12)
            assert trial['all_actual_group_derivatives_negative'] == bool(np.all(dots < 0))
            assert trial['candidate_state'] == metric_data[label][0]['candidate_state']
            assert metric_data[label][0]['previous_reference_variant'] == current_label
            gates = []
            for comparison,anchor,groups_index in [('comparisons','baseline',1),('previous_state_comparisons',current_label,2)]:
                for co in p['cohorts']:
                    name = co['name']
                    for stage in ['raw','png']:
                        bs = metric_data[anchor][0]['groups'][name][stage]
                        cs = metric_data[label][0]['groups' if groups_index == 1 else 'previous_anchor_groups'][name][stage]
                        bf = metric_data[anchor][1][name][stage]; cf = metric_data[label][groups_index][name][stage]
                        fresh = capacity(bf,cf,.01); saved = trial[comparison][name][stage]
                        compare_capacity_receipt(saved,fresh,bs,cs,bf,cf,label+'/'+comparison+'/'+name+'/'+stage)
                        gates.append(fresh)
            decision = bool(np.all(dots < 0) and np.any(actual != current) and mechanics_accept(gates))
            assert decision == trial['accepted_for_finite_mechanics_only']
            assert trial['model_qualification'] is False and trial['new_epoch_or_full_capacity_pass'] is False
            if decision:
                assert label == chosen[-1] and step['accepted_variant'] == label
                step_accept = label; accepted.append(label)
        if step_accept:
            current = np.load(folder/(step_accept+'_parameters.npy'),allow_pickle=False); current_label = step_accept
        else:
            assert step['accepted_variant'] is None and len(chosen) == 3 and index == len(steps)
        del matrix, direction
    assert used_labels == labels[1:]
    assert result['optimizer_updates'] == len(accepted) and result['new_trained_checkpoint'] == bool(accepted)
    assert [s['variant'] for s in result['accepted_states']] == accepted
    assert result['final_vector_file'] == ('initial_parameters.npy' if current_label == 'baseline' else current_label+'_parameters.npy')
    assert np.array_equal(current,np.load(folder/'final_parameters.npy',allow_pickle=False))
    if len(accepted) == 3: assert result['stop_reason'] == 'Three accepted changes reached; no automatic continuation'
    elif not steps[-1]['common_direction_found']: assert result['stop_reason'] == 'No certified direction at current state; stop without further changes'
    else: assert result['stop_reason'] == 'All three finite proposals rejected; retain gates and stop'
    parity = read(folder/'loss_surrogate_parity.json')
    assert parity['complete'] and parity['passed'] and parity['cases'] == 100 and parity['gradient_queries'] == 0
    assert parity['maximum_errors'] == {key:max(r['errors'][key] for r in parity['rows']) for key in ['MSE','SSIM','structure']}
    assert parity['maximum_errors']['MSE'] <= 1e-12 and parity['maximum_errors']['SSIM'] <= 3e-5 and parity['maximum_errors']['structure'] <= 1e-10
    cache = read(folder/'cache_receipt.json'); assert cache['complete'] and cache['cases'] == 100 and cache['seconds'] <= 120
    for receipt_name in ['storage_projection.json','trial_timing_projection.json']:
        projection = read(folder/receipt_name)
        if receipt_name.startswith('storage'): assert projection['projected_uncompressed_bytes'] <= BUDGETS['return_uncompressed_bytes']
        else: assert projection['projected_trial_seconds'] <= BUDGETS['trial_seconds_per_state']
    # Loading saved trained parameters for inference is permitted; no local learning/autograd.
    import torch
    from cctv_dgp_pilot import FixedObservedIdentity, grid112, state_hash
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_finite_guard_v1_r1_candidate import OriginalFeatureProbe
    torch.set_num_threads(4)
    original,_ = load_frozen_dgp_restorer(BUNDLE/'weights/dgp_v2.pth',expected_sha256=p['original_checkpoint_sha256'],device='cpu')
    model = OriginalFeatureProbe(original.net,p['parameter_layout'])
    identity = FixedObservedIdentity(BUNDLE/'weights/w600k_r50.onnx','cpu')
    fixed = (state_hash(original.net),state_hash(identity))
    cases = {c['id']:c for c in p['cases']}; refs = {r['id']:r for r in p['references']}
    replay_ids = [cid for co in p['cohorts'] for source in sorted({c['source'] for c in p['cases']})
        for cid in [x for x in co['case_ids'] if cases[x]['source'] == source][:5]]
    worst_raw = 0.; worst_png = 0; worst_vector = 0.
    with torch.no_grad():
        for label in labels:
            values = initial if label == 'baseline' else np.load(folder/(label+'_parameters.npy'),allow_pickle=False)
            for desc,(_,value) in zip(p['parameter_layout'],model.selected):
                value.copy_(torch.from_numpy(values[desc['start']:desc['end']].reshape(desc['shape'])))
            assert state_hash(model.net) == metric_data[label][0]['candidate_state']
            if label in accepted:
                iteration = next(t['iteration'] for t in trials if t['variant'] == label)
                snapshot = torch.load(folder/f'trained_copy_step_{iteration:02d}.pth',map_location='cpu',weights_only=True)
                assert snapshot['protocol_sha256'] == pin and snapshot['original_checkpoint_sha256'] == p['original_checkpoint_sha256']
                assert snapshot['accepted_parameter_changes'] == accepted.index(label)+1
                assert snapshot['optimizer'] is snapshot['scheduler'] is None
                assert snapshot['resume_permitted'] is snapshot['automatic_follow_on'] is False
                assert snapshot['forward_policy'] == p['forward_policy']
                assert set(snapshot['model']) == set(model.net.state_dict())
                assert all(torch.equal(snapshot['model'][key],value) for key,value in model.net.state_dict().items())
                assert snapshot['torch_cpu_rng'].dtype == torch.uint8 and len(snapshot['torch_cuda_rng_all']) == 1
            for cid in replay_ids:
                assert time.monotonic() <= deadline
                c = cases[cid]; camera = pixels(BUNDLE/c['input']); target = pixels(BUNDLE/c['target'])
                mask = pixels(BUNDLE/c['observed'],'L') > 0
                rgb = lambda v: torch.from_numpy(v.astype(np.float32)/np.float32(255)).permute(2,0,1)[None]
                x = rgb(camera); support = torch.from_numpy(mask.astype(np.float32))[None,None]
                base = original(x).detach().clone(); fresh = model(x,support,base)[0].permute(1,2,0).numpy().copy()
                raw = np.load(folder/label/(cid+'.npy'),allow_pickle=False); png = pixels(folder/label/(cid+'.png'))
                worst_raw = max(worst_raw,float(np.abs(fresh-raw).max()))
                worst_png = max(worst_png,int(np.abs(deliver(fresh,camera,mask).astype(np.int16)-png.astype(np.int16)).max()))
                grid = torch.from_numpy(grid112(refs[c['source_person_or_reference']]['matrix112']))[None]
                rt = torch.from_numpy(raw).permute(2,0,1)[None]
                vectors = identity.embedding(torch.cat([rt,rgb(png),rgb(target)]),support.repeat(3,1,1,1),grid.repeat(3,1,1,1)).numpy().copy()
                stored = np.load(folder/label/(cid+'_embeddings.npy'),allow_pickle=False)
                worst_vector = max(worst_vector,float(np.abs(vectors-stored).max()))
        for desc,(_,value) in zip(p['parameter_layout'],model.selected):
            value.copy_(torch.from_numpy(initial[desc['start']:desc['end']].reshape(desc['shape'])))
        assert state_hash(model.net) == state_hash(original.net) == STATE
    assert fixed == (state_hash(original.net),state_hash(identity))
    tol = p['CPU_replay_tolerances']
    assert worst_raw <= tol['raw_max_abs'] and worst_png <= tol['PNG_byte_max'] and worst_vector <= tol['embedding_max_abs']
    assert all(not value.requires_grad and value.grad is None for value in model.parameters())
    assert {q.name for q in folder.glob('trained_copy_step_*.pth')} == {
        f"trained_copy_step_{t['iteration']:02d}.pth" for t in trials if t['variant'] in accepted}
    verified_assets(BUNDLE,pin)
    write(ROOT/'outputs/cctv_dgp_finite_guard_v1_r1_independent_audit.json', {'complete': True,
        'archive_sha256':a.expected_sha,'protocol_sha256':pin,'checker_sha256':sha(Path(__file__)),
        'members_verified':len(members),'saved_group_vectors_per_state':64,'recertified_states_verified':len(steps),
        'proposals_verified':len(trials),'accepted_training_changes_verified':len(accepted),
        'all_raw_and_PNG_cases_recomputed':100*len(labels),'comparisons_verified':8*len(trials),
        'all_stored_row_comparisons_and_failure_decisions_exact':True,'scientific_thresholds_changed':False,
        'previous_and_original_mean_shift_anchors_verified':True,'individual_query_vector_aggregation_independently_recomputed':False,
        'gradient_autograd_replay_local':False,'local_gradient_queries':0,'local_optimizer_updates':0,
        'CPU_replay_cases':20*len(labels),'maximum_raw_error':worst_raw,'maximum_PNG_byte_error':worst_png,
        'maximum_embedding_error':worst_vector,'brightness_arithmetic_propagation':BRIGHTNESS_ARITHMETIC_ROWS,
        'seconds':time.monotonic()-start,'visual_review_pending':True,'independent_final_review':False,
        'full_TRAIN_capacity_not_tested':True,'model_qualification':False,'goal_complete':False})
    print({'complete':True,'accepted_training_changes_verified':len(accepted),'model_qualification':False})


if __name__ == '__main__': main()
