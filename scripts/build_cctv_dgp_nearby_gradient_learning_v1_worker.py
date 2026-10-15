"""Derive a new worker from retained RAW R1; never execute or overwrite historical code."""
import ast
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'scripts/cctv_dgp_raw_conditioning_learning_v1_r1_vm.py'
    text = source.read_text(encoding='utf-8')
    original = text
    def replace(before, after):
        nonlocal text
        assert text.count(before) == 1, before[:100]
        text = text.replace(before, after)
    text = text.replace('cctv_dgp_raw_conditioning_learning_v1_r1_vm', 'cctv_dgp_nearby_gradient_learning_v1_vm')
    text = text.replace('cctv-dgp-raw-conditioning-learning-v1-r1', 'cctv-dgp-nearby-gradient-learning-v1')
    text = text.replace('outputs/raw_conditioning_learning_v1_r1', 'outputs/nearby_gradient_learning_v1')
    replace("STAGE='outputs/nearby_gradient_learning_v1'", "STAGE='outputs/nearby_gradient_learning_v1'\nRESPONSE_NAME='cctv_dgp_conditioning_response_v1_vm'\nRESPONSE_PIN='e72d76e3f2330420ee93b43cf528d0b49da132f356a30c336a1ec578c8f82f44'")
    replace("p['maximum_accepted_updates']==2", "p['maximum_accepted_updates']==1")
    replace("p['maximum_gradient_queries']==640 and p['model_forward_cap']==180 and p['recognizer_forward_cap']==320", "p['maximum_gradient_queries']==960 and p['model_forward_cap']==200 and p['recognizer_forward_cap']==340")
    replace("p['optimizer_constraints']=='32 RAW groups; PNG measured by actual unchanged acceptance gates'", "p['optimizer_constraints']=='96 nearby-state RAW constraints; actual original RAW/PNG gates'\n    assert p['sampling_sites']==['initial','plus02','minus02'] and p['direction_solves']==1")
    replace("for folder,bindings in [(root,p['assets_sha256']),(parent,p['parent_required_sha256'])]:", "response=parent.parent/RESPONSE_NAME\n    assert sha(response/'protocol.json')==RESPONSE_PIN==p['response_protocol_sha256']\n    for folder,bindings in [(root,p['assets_sha256']),(parent,p['parent_required_sha256']),(response,p['response_required_sha256'])]:")
    replace("assert platform.system()=='Linux' and platform.node().split('.')[0]=='forensic-dgp-thesis'", "assert platform.system()=='Linux','Manual Linux L4 nearby-state training only'\n    assert platform.node().split('.')[0]=='forensic-dgp-thesis'")
    replace("assert counts['gradient_queries']<=p['maximum_gradient_queries'] and counts['accepted_updates']<=2 and counts['parameter_proposals']<=6", "assert counts['gradient_queries']<=p['maximum_gradient_queries'] and counts['accepted_updates']<=1 and counts['parameter_proposals']<=3\n        assert counts['derivative_sampling_displacements']<=2")
    replace("counts=dict(B=0,recognizer=0,gradient_queries=0,backwards=0,accepted_updates=0,parameter_proposals=0)", "counts=dict(B=0,recognizer=0,gradient_queries=0,backwards=0,accepted_updates=0,parameter_proposals=0,derivative_sampling_displacements=0)")
    replace("candidate=None;accepted=None;params=None;frozen_before=None", "candidate=None;accepted=None;params=None;frozen_before=None;initial_RNG=None;initial=None")
    replace("method='RAW32 full-cohort constrained descent with actual unchanged raw/PNG backtracking'", "method='Three-site96 RAW constrained descent with actual original RAW/PNG acceptance'")
    replace("original_state=original_before,recognizer_state=identity_before,full_epochs=0))", "original_state=original_before,recognizer_state=identity_before,full_epochs=0,\n            expected_torch=p['expected_torch'],precision=dict(matmul_tf32=False,cudnn_tf32=False,cudnn_benchmark=False,cudnn_deterministic=True)))\n        assert str(torch.__version__)==p['expected_torch']\n        def rng_values():return dict(torch_CPU_RNG=torch.get_rng_state().numpy(),**{f'torch_CUDA_RNG_{i}':s.cpu().numpy() for i,s in enumerate(torch.cuda.get_rng_state_all())})\n        def restore_rng(v):\n            torch.set_rng_state(torch.from_numpy(v['torch_CPU_RNG']))\n            torch.cuda.set_rng_state_all([torch.from_numpy(v[f'torch_CUDA_RNG_{i}']) for i in range(torch.cuda.device_count())])\n        initial_RNG=rng_values();np.savez_compressed(out/'initial_RNG.npz',**initial_RNG)")
    replace("if exact_controls:assert np.array_equal(raw,baseline),'Initial DGP parity differs'", "if exact_controls:assert np.array_equal(raw,baseline),'Initial DGP parity differs'\n                    if label in p['sampling_sites']:\n                        expected_raw=np.load(parent.parent/RESPONSE_NAME/STAGE.replace('nearby_gradient_learning_v1','conditioning_response_v1')/label/(stem+'.npy'),allow_pickle=False)\n                        assert np.array_equal(raw,expected_raw),'Sampling-state RAW forward differs from retained measurement'")
    start = text.index("        base_snapshot=evaluate('initial',True);current_snapshot=base_snapshot")
    stop = text.index("        assert state_hash({n:v for n,v in candidate.state_dict().items() if n not in names})==frozen_before", text.index("        final=decision(", start))
    gradient_start = original.index('            for serial,(block,cs,x8,target8,mask8,support8,baseline,expected) in enumerate(blocks):', original.index('        for round_index in range(2):'))
    gradient_end = original.index('            direction,certificate=solve_direction(', gradient_start)
    gradient = original[gradient_start:gradient_end]
    # Preserve the already validated full-cohort RAW differentiation loop exactly.
    gradient = gradient.replace("round=round_index", "site=site")
    gradient = gradient.replace("records.append(dict(g,file=", "records.append(dict(g,site=site,file=")
    new_loop = '''        base_snapshot=evaluate('initial',True);current_snapshot=base_snapshot
        response=parent.parent/RESPONSE_NAME/'outputs/conditioning_response_v1'
        raw_groups=p['raw_gradient_groups']
        all_groups=[dict(g,key='RAW/'+g['key'],metric_stage='RAW') for g in raw_groups]
        sampling_records=[];trial_records=[];qualified=False
        for site in p['sampling_sites']:
            clock();progress.update(stage='nearby_group_gradients',site=site)
            sample=initial if site=='initial' else np.load(response/site/'conditioning.npy',allow_pickle=False)
            if site!='initial':
                load_values(sample);counts['derivative_sampling_displacements']+=1
                current_snapshot=evaluate(site)
            else:current_snapshot=base_snapshot
            save_array(out/site/'conditioning.npy',sample)
            folder=out/('gradients_'+site);folder.mkdir();(folder/'aggregates').mkdir();(folder/'samples').mkdir()
            grad_start=time.monotonic()
            aggregates={g['key']:np.zeros(int(edges[-1]),np.float64) for g in all_groups}
            contributions={g['key']:[] for g in all_groups};durations=[]
            snapshot_rows={r['id']:r for r in current_snapshot['rows']}
            round_state=state_hash(candidate)
'''+gradient+'''            site_record=dict(site=site,model_state=round_state,snapshot=site,groups=records,seconds=time.monotonic()-grad_start)
            write(folder/'gradients.json',site_record);sampling_records.append(site_record)
            del aggregates,contributions
        load_values(initial);assert state_hash(candidate)==p['initial_model_state'];current_snapshot=base_snapshot
        vectors=[np.load(out/r['file'],allow_pickle=False,mmap_mode='r') for site in sampling_records for r in site['groups']]
        rows=[r for site in sampling_records for r in site['groups']]
        direction,certificate=solve_direction(vectors,rows);del vectors
        save_array(out/'direction.npy',direction)
        write(out/'direction_certificate.json',dict(complete=True,groups=rows,certificate=certificate,direction_sha256=sha(out/'direction.npy')))
        ceiling=bounded_rate(initial,initial,direction,initial_all37_norm)
        for attempt,factor in enumerate(p['backtrack_factors']):
            clock();label=f'trial{attempt:02d}';rate=np.float32(ceiling*factor)
            proposal=initial+rate*direction
            relative=float(np.linalg.norm(proposal.astype(np.float64)-initial.astype(np.float64))/initial_all37_norm)
            assert relative<=.001,'Rounded cumulative trust-region distance exceeds retained limit'
            load_values(proposal);counts['parameter_proposals']+=1
            progress.update(stage='actual_proposal_acceptance',trial=label)
            snapshot=evaluate(label);save_array(out/label/'conditioning.npy',proposal)
            checked=decision(base_snapshot['groups'],base_snapshot['groups'],snapshot['groups'])
            trial=dict(label=label,attempt=attempt,rate=float(rate),rate_ceiling=ceiling,
                before_snapshot='initial',direction_sha256=sha(out/'direction.npy'),snapshot_sha256=sha(out/label/'snapshot.json'),
                conditioning_sha256=sha(out/label/'conditioning.npy'),cumulative_relative_parameter_distance=relative,decision=checked)
            write(out/label/'decision.json',trial);trial_records.append(trial)
            print(dict(trial=label,accepted=checked['accept_step'],one_percent=checked['one_percent_checkpoint_pass'],
                raw_gain=checked['stages']['raw']['relative_feature_gain'],PNG_gain=checked['stages']['png']['relative_feature_gain']),flush=True)
            if checked['accept_step']:
                accepted=proposal.copy();current_snapshot=snapshot;counts['accepted_updates']+=1
                full_state={n:v.detach().cpu() for n,v in candidate.state_dict().items()}
                torch.save(full_state,out/'accepted_component_full_state.pth');del full_state
                np.savez_compressed(out/'accepted01.npz',conditioning=accepted,**rng_values())
                write(out/'accepted01.json',dict(model_state=state_hash(candidate),full_state_sha256=sha(out/'accepted_component_full_state.pth'),
                    checkpoint_sha256=sha(out/'accepted01.npz'),conditioning_names=names,accepted_update=1,trial=label,
                    frozen_state=frozen_before,buffer_state=buffers,original_state=original_before,recognizer_state=identity_before,
                    stateless_optimizer='Three-site96 constrained descent; no momentum/scheduler',protocol_sha256=pin,
                    exact_continuation_needs_code_environment_data_and_rng_bindings=True,research_component_not_app_model=True))
                qualified=checked['one_percent_checkpoint_pass'];break
            load_values(initial);assert state_hash(candidate)==p['initial_model_state']
        load_values(initial);restore_rng(initial_RNG)
        rollback=evaluate('rollback',True);save_array(out/'rollback/conditioning.npy',initial)
        assert rollback['groups']==base_snapshot['groups'] and state_hash(candidate)==p['initial_model_state']
        restore_rng(initial_RNG);np.savez_compressed(out/'restored_RNG.npz',**rng_values())
        write(out/'rollback.json',dict(complete=True,initial_model_state=p['initial_model_state'],fresh_state=state_hash(candidate),
            frozen_state=state_hash({n:v for n,v in candidate.state_dict().items() if n not in names}),buffer_state=buffer_hash(candidate),
            original_state=state_hash(original.net),recognizer_state=state_hash(identity),RNG_restored=True,
            rollback_outputs_exact=True,accepted_state_preserved_separately=counts['accepted_updates']>0))
        final=decision(base_snapshot['groups'],base_snapshot['groups'],current_snapshot['groups'])
        assert counts['accepted_updates']==1,'No actual raw/PNG preserving nearby-state proposal; retain stop'
        assert qualified and final['one_percent_checkpoint_pass'],'One-update budget ended below retained1% structure/preservation gate'
'''
    text = text[:start] + new_loop + text[stop:]
    replace("rounds=round_records", "sampling_sites=sampling_records,direction_certificate_sha256=sha(out/'direction_certificate.json')")
    replace("final_model_state=state_hash(candidate)", "final_model_state=current_snapshot['model_state'],fresh_restored_model_state=state_hash(candidate),RNG_restored=True")
    replace("            try:load_values(accepted)\n            except BaseException as failure:restore_error=repr(failure)", "            try:\n                load_values(initial)\n                if initial_RNG is not None:restore_rng(initial_RNG)\n                write(out/'failure_restoration.json',dict(complete=True,fresh_state=state_hash(candidate),expected_initial=p['initial_model_state'],\n                    frozen_state=state_hash({n:v for n,v in candidate.state_dict().items() if n not in names}),buffer_state=buffer_hash(candidate),RNG_restored=initial_RNG is not None))\n                assert state_hash(candidate)==p['initial_model_state']\n            except BaseException as failure:restore_error=repr(failure)")
    replace("no_post_failure_fresh_state_hash_claimed=True", "fresh_failure_restoration_receipt=(out/'failure_restoration.json').exists()")
    text = text.replace('rollback_to_last_accepted_component_error', 'rollback_to_initial_component_error')
    ast.parse(text, feature_version=(3, 10))
    target = ROOT / 'scripts/cctv_dgp_nearby_gradient_learning_v1_vm.py'
    with target.open('x', encoding='utf-8', newline='\n') as f:f.write(text)
    receipt = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_worker_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        new_worker_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        full_cohort_RAW_derivative_loop_retained_except_site_labels=True,original_worker_unchanged=True,
        neural_or_training_calls=0,manual_worker_not_run=True)
    with (ROOT/'outputs/cctv_dgp_nearby_gradient_learning_v1_worker_source.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(receipt)


if __name__ == '__main__':main()
