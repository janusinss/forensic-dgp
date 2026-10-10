"""Independent packet verification. CPU baseline inference only; no learning."""
import ast
import copy
import hashlib
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'outputs/cctv_dgp_finite_guard_v1_r1_vm'
PREP = ROOT/'outputs/cctv_dgp_finite_guard_v1_r1_preparation'
sys.path.insert(0,str(BUNDLE))
from cctv_dgp_finite_guard_v1_r1_contract import NAME,STEM,BUDGETS,read,write,sha,verified_assets,definitions


def main():
    started = time.monotonic(); pin = sha(BUNDLE/'protocol.json'); p = verified_assets(BUNDLE,pin)
    for n,d in p['local_sources_sha256'].items(): assert sha(ROOT/n) == d,n
    for n,origin in p['copied_source_mapping'].items(): assert sha(BUNDLE/n) == sha(ROOT/origin),n
    archive = ROOT/'outputs'/(STEM+'-execution.tar.gz')
    assert Path(str(archive)+'.sha256').read_text().split() == [sha(archive),archive.name]
    with tarfile.open(archive,'r:gz') as tar:
        members = tar.getmembers(); names = set()
        for m in members:
            assert m.isfile() and not m.issym() and not m.islnk()
            parts = m.name.split('/')
            assert parts[0] == NAME and all(x not in ['', '.', '..'] for x in parts)
            n = '/'.join(parts[1:]); assert n not in names; names.add(n)
            assert hashlib.sha256(tar.extractfile(m).read()).hexdigest() == sha(BUNDLE/n)
        assert names == set(p['assets_sha256'])|{'protocol.json'}
    parent = read(BUNDLE/'evidence/parent_protocol.json')
    assert sha(BUNDLE/'evidence/parent_protocol.json') == p['parent_protocol_sha256']
    for key in ['cases','references','cohorts','parameter_layout','terms_and_overlap','scientific_thresholds','CPU_replay_tolerances','forward_policy']:
        assert p[key] == parent[key],key
    assert p['group_losses'] == definitions(p['cases'],p['cohorts'])
    assert len(p['group_losses']) == 64 and {r['cohort'] for r in p['group_losses']} == {c['name'] for c in p['cohorts']}
    for co in p['cohorts']:
        assert all(r['case_ids'] == [cid for cid in co['case_ids'] if cid in r['case_ids']]
            for r in p['group_losses'] if r['cohort'] == co['name'])
    audit,visual = read(BUNDLE/'evidence/parent_independent_audit.json'),read(BUNDLE/'evidence/parent_visual_review.json')
    assert audit['complete'] and visual['complete'] and visual['unique_model_outputs_reviewed'] == 400
    assert audit['archive_sha256'] == p['parent_archive_sha256'] and not visual['independent_final_reviewer']
    assert read(BUNDLE/'evidence/parent_gallery_audit.json')['complete']
    parent_result = read(BUNDLE/'evidence/parent_results.json')
    assert parent_result['optimizer_updates'] == 0 and parent_result['gradient_queries'] == 160
    assert all(not stage['pass'] for trial in parent_result['trial_summaries'] for stages in trial['comparisons'].values() for stage in stages.values())
    overhead = int(6*p['prior_32_group_compressed_bytes']*1.15)+256*1024**2
    assert overhead == p['projected_gradient_and_snapshot_bytes']
    projection = int(10*(p['prior_baseline_bytes']+2*1024**2)*1.25+overhead)
    assert projection < BUDGETS['return_uncompressed_bytes']
    assert 2*BUDGETS['return_uncompressed_bytes']+BUDGETS['disk_reserve_bytes'] < BUDGETS['minimum_free_disk_bytes']
    trees = {f.relative_to(BUNDLE).as_posix():ast.parse(f.read_text(),feature_version=(3,10)) for f in BUNDLE.rglob('*.py')}
    worker = BUNDLE/'scripts/cctv_dgp_finite_guard_v1_r1_vm.py'
    training = (BUNDLE/'cctv_dgp_finite_guard_v1_r1_training.py').read_text()
    for code in [worker.read_text(),training]: assert 'torch.optim' not in code and '.backward(' not in code
    assert training.count('torch.autograd.grad(') == 1 and 'range(1, 4)' in training and 'range(0, 100, 5)' in training
    assert "progress['optimizer_updates'] += 1" in training and 'torch.save(snapshot' in training
    assert "candidate.assign_trial(current)" in training and 'candidate.assign_trial(initial)' in training
    assert "receipt['previous_anchor_groups']" in training and "evaluate(label, current_receipt['variant'])" in training
    assert "previous_constant_mean_shift_only_MSE" in worker.read_text() and "previous_reference_variant" in worker.read_text()
    source = (BUNDLE/'cctv_dgp_finite_guard_v1_r1_candidate.py').read_text()
    assert source == (ROOT/'scripts/cctv_dgp_group_conflicts_v1_candidate.py').read_text().replace('group_conflicts_v1','finite_guard_v1_r1')
    shell = (BUNDLE/'scripts/run_finite_guard.sh').read_text()
    assert '\r' not in shell and NAME in shell and '1830s' in shell and '390s' in shell and 'PIPESTATUS[0]' in shell
    assert 'run_group_conflicts' not in shell and not (BUNDLE/'scripts/run_group_conflicts.sh').exists()
    denied = []
    for mode in ['verify-transfer','run','export','record-supervision']:
        child = subprocess.run([sys.executable,'-B',str(worker),'--root',str(BUNDLE),'--protocol-sha',pin,'--'+mode],capture_output=True,text=True,timeout=15)
        assert child.returncode != 0 and 'AssertionError' in child.stderr
        denied.append(mode)
    from cctv_dgp_group_conflicts_v1_math import common_direction
    fixtures = [(np.array([[1.,0.],[0.,1.]],np.float64),True),
        (np.array([[1.,0.],[-1.,0.]],np.float64),False),
        (np.zeros((2,2),np.float64),False),
        (np.tile(np.array([[1.,0.],[0.,1.]],np.float64),(32,1)),True)]
    for matrix,expected in fixtures:
        direction,cert = common_direction(matrix)
        assert cert['common_direction_found'] == expected and (direction is not None) == expected
        if expected: assert np.all(matrix@direction < 0)
    from cctv_dgp_finite_guard_v1_r1_training import mechanics_accept
    good = {'preservation_failures':[],'relative_feature_gain':.0001,
        'source_feature_gains':{'a':.0001,'b':.0001},'brightness_gain_fraction':.1,'pass':False}
    assert mechanics_accept([good]*8)  # Investigation only; original1percent pass remains false.
    for change in [{'preservation_failures':[{'metric':'MSE'}]}, {'relative_feature_gain':0.},
        {'source_feature_gains':{'a':-.000001,'b':.0001}}, {'brightness_gain_fraction':.200001}]:
        assert not mechanics_accept([good]*7+[{**good,**change}])
    # Actual prior pixels distinguish original and preceding-state mean-only anchors.
    from frozen_raw_metrics import mean_only,pixel_metrics
    c = p['cases'][0]; mask = np.asarray(Image.open(BUNDLE/c['observed']).convert('L')) > 0
    camera = np.asarray(Image.open(BUNDLE/c['input']).convert('RGB')).copy()
    prev = ROOT/'outputs/cctv_dgp_group_conflicts_v1_vm_return/outputs'
    baseline = np.load(prev/'baseline'/(c['id']+'.npy'),allow_pickle=False)
    middle = np.load(prev/'common_1em05'/(c['id']+'.npy'),allow_pickle=False)
    proposal = np.load(prev/'common_1em04'/(c['id']+'.npy'),allow_pickle=False)
    m1 = mean_only(proposal,baseline,camera,mask)[0]; m2 = mean_only(proposal,middle,camera,mask)[0]
    assert not np.array_equal(m1,m2)
    import torch
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_finite_guard_v1_r1_candidate import OriginalFeatureProbe
    from cctv_dgp_pilot import state_hash,buffer_hash
    torch.set_num_threads(4)
    original,_ = load_frozen_dgp_restorer(BUNDLE/'weights/dgp_v2.pth',expected_sha256=p['original_checkpoint_sha256'],device='cpu')
    candidate = OriginalFeatureProbe(original.net,p['parameter_layout'])
    initial = np.load(BUNDLE/'evidence/initial_parameters.npy',allow_pickle=False)
    assert np.array_equal(candidate.vector(),initial)
    before = (state_hash(original.net),state_hash(candidate.net),buffer_hash(candidate.net))
    denied_candidate = []
    for name,call in [('gradient_enable',lambda:candidate.enable_diagnostic_gradients(BUNDLE)),
            ('trial_assignment',lambda:candidate.assign_trial(initial))]:
        try: call()
        except AssertionError: denied_candidate.append(name)
        else: raise AssertionError('Local VM-only training operation allowed')
    chosen = [c for source in sorted({c['source'] for c in p['cases']}) for c in [x for x in p['cases'] if x['source'] == source][:2]]
    worst = 0.
    for case in chosen:
        assert time.monotonic()-started <= 240
        rgb = np.asarray(Image.open(BUNDLE/case['input']).convert('RGB')).astype(np.float32)/np.float32(255)
        x = torch.from_numpy(rgb).permute(2,0,1)[None]
        support = np.asarray(Image.open(BUNDLE/case['observed']).convert('L')) > 0
        mask_tensor = torch.from_numpy(support.astype(np.float32))[None,None]
        with torch.no_grad():
            base = original(x).detach().clone(); output = candidate(x,mask_tensor,base)
            assert torch.equal(output,torch.where(mask_tensor.bool(),base,x))
        prior = np.load(prev/'baseline'/(case['id']+'.npy'),allow_pickle=False)
        worst = max(worst,float(np.abs(output[0].permute(1,2,0).numpy()-prior).max()))
    try:
        with torch.enable_grad(): candidate(x,mask_tensor,base)
    except AssertionError: denied_candidate.append('gradient_forward')
    else: raise AssertionError('Local differentiable forward allowed')
    after = (state_hash(original.net),state_hash(candidate.net),buffer_hash(candidate.net))
    assert before == after and before[0] == before[1] == p['original_state']
    assert worst <= 3e-6 and len(chosen) == 4
    assert all(v.grad is None and not v.requires_grad for v in candidate.parameters())
    verified_assets(BUNDLE,pin)
    write(PREP/'independent_packet_audit.json', {'complete':True,'protocol_sha256':pin,
        'archive_sha256':sha(archive),'archive_members_verified':len(members),'source_bindings_verified':len(p['local_sources_sha256']),
        'Python310_AST_files':len(trees),'pure_math_fixtures':4,'finite_acceptance_fixtures':5,
        'different_previous_mean_shift_anchor_verified':True,'local_worker_guards_denied':denied,
        'local_candidate_guards_denied':denied_candidate,'initial_CPU_parity_cases':4,
        'maximum_parent_raw_error':worst,'states_before':before,'states_after':after,
        'local_neural_forward_calls':8,'local_gradient_queries':0,'local_optimizer_updates':0,'local_trial_assignments':0,
        'VM_training_launched':False,'model_qualification':False,'goal_complete':False,
        'projected_return_bytes':projection,'checker_sha256':sha(Path(__file__)),'seconds':time.monotonic()-started})
    print({'complete':True,'initial_parity_cases':4,'raw_error':worst,'VM_training_launched':False})


if __name__ == '__main__': main()
