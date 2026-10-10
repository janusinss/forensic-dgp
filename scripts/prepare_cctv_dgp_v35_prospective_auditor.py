"""Write a prospective V35 checker before freezing its manual packet."""
import hashlib
from pathlib import Path
import ast

ROOT=Path(__file__).resolve().parents[1]

HEAD='''"""Prospective V35 audit: saved arrays, finite measurements and frozen CPU replay."""
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
BUNDLE=ROOT/'outputs/cctv_dgp_group_guard_probe_v35_vm'
OUT=ROOT/'outputs/cctv_dgp_group_guard_probe_v35_return'
PREFIX='cctv_dgp_group_guard_probe_v35_return/'
STEM='cctv-dgp-group-guard-probe-v35'
PARENT=ROOT/'outputs/cctv_dgp_face_code_fit_vm_v12_r2'
ACTIVE=ROOT/'outputs/cctv_dgp_batchmatched_identity_vm_v26'
V32=ROOT/'outputs/cctv_dgp_feature_fusion_vm_v32_r2'
MIXED=ROOT/'outputs/cctv_dgp_mixed_vm_v9_r2'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    encoded=json.dumps(value,indent=2,allow_nan=False)+'\\n'
    with Path(path).open('x',encoding='utf-8',newline='\\n') as stream:stream.write(encoded)


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
        assert member.name.startswith(PREFIX) and not any(c in member.name for c in ['\\\\',':','\\x00'])
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
    write(ROOT/'outputs/cctv_dgp_group_guard_probe_v35_return_import.json',receipt)
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


'''


TAIL='''def audit(digest,size):
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
    write(ROOT/'outputs/cctv_dgp_group_guard_probe_v35_independent_audit.json',receipt)
    print(json.dumps({'complete':True,'finite_probe_complete':success.exists(),'members_verified':imported['members'],'seconds':receipt['seconds']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--expected-sha',required=True);parser.add_argument('--expected-bytes',type=int,required=True)
    a=parser.parse_args();audit(a.expected_sha,a.expected_bytes)
'''


def main():
    source=ROOT/'scripts/audit_cctv_dgp_loss_cone_probe_v33_return_r2.py'
    with source.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    assert digest=='cbfad9e794c735cdf154027c4046d685183fb19f38f84bb9766331645b5382be'
    text=source.read_text(encoding='utf-8')
    replay=text[text.index('def CPU_replay(p, basis, started):'):text.index('def audit(digest, size):')]
    stopped="    stopped = torch.load(CLOSED / 'outputs/update50/dgp_candidate_v32.pth', map_location='cpu', weights_only=True)\n"
    assert replay.count(stopped)==1
    replay=replay.replace(stopped,'').replace('for state, weights in [(0, initial), (50, stopped)]:','for state, weights in [(0, initial)]:')
    replay=replay.replace("p['CPU_replay_outputs'] == 280","p['CPU_replay_outputs'] == 100")
    result=HEAD+replay+TAIL
    ast.parse(result,feature_version=(3,10))
    destination=ROOT/'scripts/audit_cctv_dgp_group_guard_probe_v35_return.py'
    assert not destination.exists()
    with destination.open('x',encoding='utf-8',newline='\n') as stream:stream.write(result)
    print({'complete':True,'prospective_auditor':destination.name,'source_replay_sha256':digest,'neural_calls':0})


if __name__=='__main__':main()
