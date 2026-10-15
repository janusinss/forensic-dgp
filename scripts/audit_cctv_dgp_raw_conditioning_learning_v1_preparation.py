"""Archive, state, local rejection and genuine retained-gate checks."""
import ast
import copy
from datetime import datetime,timezone
import hashlib
import importlib.util
from pathlib import Path,PurePosixPath
import subprocess
import sys
import tarfile

import numpy as np
from import_cctv_dgp_live_conditioning_v1_r1_return import ROOT,OUT,ANALYSIS,PARENT,read,sha,write
from audit_cctv_dgp_raw_conditioning_learning_v1_return import schema,state_hash,image,support,metrics,delivered,grouped
from cctv_dgp_raw_conditioning_learning_v1_contract import decision,solve_direction,bounded_rate


def main():
    packet=OUT/'cctv_dgp_raw_conditioning_learning_v1_vm'
    prepared=read(OUT/'cctv_dgp_raw_conditioning_learning_v1_preparation.json')
    p=read(packet/'protocol.json');pin=sha(packet/'protocol.json')
    assert prepared['complete'] and prepared['manual_pilot_unrun'] and pin==prepared['protocol_sha256']
    archive=OUT/'cctv-dgp-raw-conditioning-learning-v1-execution.tar.gz'
    assert sha(archive)==prepared['archive_sha256'] and archive.stat().st_size==prepared['archive_bytes']
    assert Path(str(archive)+'.sha256').read_bytes()==f"{prepared['archive_sha256']}  {archive.name}\n".encode('ascii')
    bindings={**p['assets_sha256'],'protocol.json':pin}
    with tarfile.open(archive,'r:gz') as tar:
        names=set()
        for member in tar.getmembers():
            path=PurePosixPath(member.name)
            assert member.isfile() and not member.islnk() and not member.issym()
            assert path.parts[0]==packet.name and '..' not in path.parts and not path.is_absolute()
            name=PurePosixPath(*path.parts[1:]).as_posix();assert name not in names and name in bindings
            with tar.extractfile(member) as stream:assert hashlib.sha256(stream.read()).hexdigest()==bindings[name]
            assert sha(packet/name)==bindings[name] and member.size==(packet/name).stat().st_size
            names.add(name)
        assert names==set(bindings)
    assert not (packet/'outputs').exists() and not (packet/'review.log').exists()
    assert len(p['parent_required_sha256'])==5852
    assert all(sha(PARENT/n)==h for n,h in p['parent_required_sha256'].items())
    virtual=schema(packet);assert state_hash(virtual)==p['initial_model_state']
    assert len(p['conditioning_names'])==12 and sum(virtual[n].numel() for n in p['conditioning_names'])==2580864
    assert all(n.startswith(('live.style_conditioner.','live.spatial_conditioners.')) for n in p['conditioning_names'])
    old=read(PARENT/'protocol.json');assert p['preserved_scientific_thresholds']==old['scientific_thresholds']
    selected=[c for c in old['cases'] if c['id'] in p['case_ids']]
    assert len(selected)==100 and all(c['role']=='train' for c in selected)
    assert len({c['source_person_or_reference'] for c in selected})==20 and len({c['source'] for c in selected})==2
    for file in packet.rglob('*.py'):ast.parse(file.read_text(encoding='utf-8'),feature_version=(3,10))
    for file in [ROOT/'scripts/import_cctv_dgp_raw_conditioning_learning_v1_return.py',ROOT/'scripts/audit_cctv_dgp_raw_conditioning_learning_v1_return.py']:
        ast.parse(file.read_text(encoding='utf-8'),feature_version=(3,10))
    code=(packet/'scripts/cctv_dgp_raw_conditioning_learning_v1_vm.py').read_text(encoding='utf-8')
    assert "param.requires_grad_(name in names)" in code and 'candidate.load_state_dict' not in code
    assert "m['ArcFace_observed_fixed']=float(emb[i]@emb[10+i])" in code
    assert "m['ArcFace_observed_fixed']=float(expected[i,j]@expected[i,2])" in code
    assert 'float(np.dot(emb[i].astype(np.float64)' not in code
    assert "assert time.monotonic()-start<p['measurement_seconds'],'Learning measurement stop'" in code
    assert 'clock();verify(root,parent,pin);clock()' in code
    assert "counts['accepted_updates']+=1" in code and 'load_values(accepted)' in code
    assert "for round_index in range(2)" in code and "for attempt,factor in enumerate([1.,.5,.25])" in code
    assert "complete_pilot=result.exists()" in code and "accepted_updates=count" in code
    worker=packet/'scripts/cctv_dgp_raw_conditioning_learning_v1_vm.py'
    transfer=subprocess.run([sys.executable,'-B',str(worker),'--root',str(packet),'--parent-root',str(PARENT),
        '--protocol-sha',pin,'--verify-transfer'],capture_output=True,text=True,timeout=180,check=True)
    verified=ast.literal_eval(transfer.stdout.strip())
    assert verified['complete'] and verified['actual_neural_calls']==0 and verified['parent_assets']==5852
    guard="import runpy,sys,json;sys.argv="+repr([str(worker),'--root',str(packet),'--parent-root',str(PARENT),'--protocol-sha',pin,'--run'])+"\ntry: runpy.run_path("+repr(str(worker))+",run_name='__main__')\nexcept AssertionError: print(json.dumps({'rejected_before_Torch':'torch' not in sys.modules}))\nelse: raise RuntimeError('Windows run not rejected')"
    rejected=subprocess.run([sys.executable,'-B','-c',guard],capture_output=True,text=True,timeout=30,check=True)
    assert __import__('json').loads(rejected.stdout)['rejected_before_Torch']
    # Replay independent metrics against all100 real saved controls, without a
    # model. This tests the future checker instead of inventing returned data.
    sys.path.insert(0,str(PARENT))
    from cctv_dgp_head4_lossless_v1 import unpack
    source={r['id']:r for r in old['references']}
    expected={r['id']:r for r in read(PARENT/'outputs/bank_comparison_v1/baseline/metrics.json')['rows']}
    rows=[]
    for block in p['reference_blocks']:
        begin=block['begin'];cs=old['cases'][begin:begin+5]
        raw,emb=unpack(PARENT/f'outputs/bank_comparison_v1/baseline/packs/b{begin//5:03d}.npz',block['case_ids'])
        for i,c in enumerate(cs):
            ref=source[c['source_person_or_reference']]
            camera,target,mask=image(PARENT/c['input']),image(PARENT/ref['target']),image(PARENT/ref['observed'],'L')>0
            patch=support(mask,c['landmarks5_canvas_xy']);png=delivered(raw[i],camera,mask)
            row=dict(id=c['id'],source=c['source'],profile=c['profile'])
            for stage,index,actual in [('raw',0,raw[i]),('png',1,png.astype(np.float32)/np.float32(255))]:
                row[stage]=metrics(actual,target,mask,patch,emb[i,index],emb[i,2],actual,prediction8=png if stage=='png' else None)
                for key,value in row[stage].items():
                    other=expected[c['id']][stage][key]
                    if value is None or isinstance(value,bool):assert value==other
                    else:assert np.isclose(value,other,rtol=0,atol=1e-13),(c['id'],stage,key,value,other)
            rows.append(row)
    assert len(rows)==100
    base={s:grouped(rows,s) for s in ['raw','png']}
    fixture=copy.deepcopy(base)
    for stage in fixture.values():
        for row in stage.values():
            row['MSE']*=.99;row['constant_mean_shift_only_MSE']=row['MSE']+.01
            row['landmark_high_frequency_MSE']*=.98
    boundaries=[]
    for label,edit,accepted,qualified in [
        ('declared_positive_synthetic_fixture',lambda f:None,True,True),
        ('clear_MSE_regression',lambda f:f['png']['clear'].update(MSE=base['png']['clear']['MSE']+1e-10),False,False),
        ('raw_SSIM_regression',lambda f:f['raw']['all'].update(SSIM=base['raw']['all']['SSIM']-2e-6),False,False),
        ('PNG_ArcFace_regression',lambda f:f['png']['degraded'].update(ArcFace_observed_fixed=base['png']['degraded']['ArcFace_observed_fixed']-2e-6),False,False),
        ('one_source_below_one_percent',lambda f:f['png']['dataset/asian_faces/degraded'].update(landmark_high_frequency_MSE=base['png']['dataset/asian_faces/degraded']['landmark_high_frequency_MSE']*.995),True,False),
        ('excess_brightness_gain',lambda f:f['png']['degraded'].update(constant_mean_shift_only_MSE=base['png']['degraded']['MSE']*.99),False,False),
    ]:
        after=copy.deepcopy(fixture);edit(after);result=decision(base,base,after)
        assert result['accept_step']==accepted and result['one_percent_checkpoint_pass']==qualified,label
        boundaries.append(dict(label=label,accept_step=accepted,one_percent_pass=qualified,synthetic_fixture_not_model_result=True))
    for bad in [float('nan'),float('inf'),-float('inf')]:
        after=copy.deepcopy(fixture);after['png']['clear']['SSIM']=bad
        try:decision(base,base,after)
        except AssertionError:pass
        else:raise AssertionError('Nonfinite metric accepted')
    # Actual earlier rejected proposal groups stay rejected by the new
    # acceptance gate. Training more cannot bypass their delivered failures.
    prior=read(OUT/'cctv_dgp_joint_raw_png_forward_v1_analysis/finite_metrics.json')
    previous_rejected=[]
    for label,proposal in prior['proposals'].items():
        after={s:proposal['groups'][s] for s in ['raw','png']}
        result=decision(prior['baseline_groups'],prior['baseline_groups'],after)
        assert not result['accept_step'] and not result['one_percent_checkpoint_pass']
        previous_rejected.append(label)
    r4=OUT/'cctv_dgp_conditioning_learning_v1_r4_return/outputs/conditioning_learning_v1_r4'
    current_base=read(r4/'initial/snapshot.json')['groups']
    r4_rejected=[]
    for label in ['r00_trial00','r00_trial01','r00_trial02']:
        after=read(r4/label/'snapshot.json')['groups']
        checked=decision(current_base,current_base,after)
        assert checked==read(r4/label/'decision.json')['decision']
        assert not checked['accept_step'] and not checked['one_percent_checkpoint_pass']
        r4_rejected.append(label)
    records=[dict(value=.01,metric='feature_MSE' if i in [30,31] else 'MSE') for i in range(32)]
    vectors=[row.copy() for row in np.eye(32)]
    direction,certificate=solve_direction(vectors,records)
    assert certificate['all32_RAW_loss_dimensions_retained'] and np.all(direction<0)
    rate=bounded_rate(np.zeros(32,np.float32),np.zeros(32,np.float32),direction,10.)
    assert np.linalg.norm(rate*direction.astype(np.float64))<.01
    assert rate<=.02
    from audit_cctv_dgp_raw_conditioning_learning_v1_design import audit as design_audit
    repair_proof=design_audit(packet,p)
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    record=dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),protocol_sha256=pin,
        archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,archive_files=len(bindings),
        RAW_optimizer_and_unchanged_quality_gate_design_audit=repair_proof,
        LF_ASCII_checksum_exact=True,parent_assets_verified=5852,initial_full_virtual_state_reconstructed=True,
        conditioning_names=12,conditioning_parameters=2580864,paired_TRAIN_cases=100,
        all100_saved_raw_and_PNG_checker_baselines_replayed=True,prospective_checker_metric_rows=200,
        retained_gate_boundary_tests=boundaries,NaN_and_infinity_metrics_rejected=True,
        actual_prior_rejected_joint_proposals_still_rejected=previous_rejected,
        all_three_actual_R4_trial_decisions_replayed_and_still_rejected=r4_rejected,
        declared_synthetic32_RAW_constraint_solver_and_cumulative_distance_fixture_passed=True,
        synthetic_fixture_is_not_neural_training_or_restoration_evidence=True,
        windows_run_rejected_before_Torch_import=True,hash_only_transfer_preflight=verified,
        future_importer_sha256=sha(ROOT/'scripts/import_cctv_dgp_raw_conditioning_learning_v1_return.py'),
        future_auditor_sha256=sha(ROOT/'scripts/audit_cctv_dgp_raw_conditioning_learning_v1_return.py'),
        future_full_return_audit_not_run=True,new_CUDA_training_recipe_unrun=True,
        local_neural_gradient_optimizer_or_parameter_displacement_calls=0,
        original_app_and_research_unchanged=True,model_qualified=False,more_epochs_qualified=False,goal_complete=False)
    write(OUT/'cctv_dgp_raw_conditioning_learning_v1_transfer_audit.json',record)
    print(dict(complete=True,archive_files=len(bindings),parent_assets=5852,actual_baseline_cases=100,
        training_unrun=True,local_neural_calls=0))


if __name__=='__main__':main()
