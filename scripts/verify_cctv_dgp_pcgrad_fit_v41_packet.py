"""Independent treatment scope, finite packet and saved-step corruption checks."""
import ast
import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace
import numpy as np
from PIL import Image
from cctv_dgp_spatial_fit_v40_contract import read, write, sha, validate_schedule, groups, capacity
from render_cctv_dgp_pcgrad_v41 import NAME, worker, decoder, auditor
from cctv_dgp_pcgrad_v41 import combine, task_orders
from cctv_dgp_pcgrad_v41_evidence import projected_gradient, verify_step, read_arrays

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'/NAME
PREP=ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_preparation'
BASE=ROOT/'outputs/cctv_dgp_spatial_fit_vm_v40'


def reject(call):
    try: call()
    except (AssertionError, ValueError): return
    raise AssertionError('Corruption was accepted')


def main():
    start=time.monotonic(); p=read(OUT/'protocol.json'); old=read(BASE/'protocol.json'); prepared=read(PREP/'prepared.json')
    assert prepared['complete'] and prepared['protocol_sha256']==sha(OUT/'protocol.json')
    for name,digest in p['assets_sha256'].items(): assert sha(OUT/name)==digest,name
    for name,digest in p['sources_sha256'].items(): assert sha(ROOT/name)==digest,name
    for name,original in p['copied_source_mapping'].items(): assert sha(OUT/name)==sha(ROOT/original),name
    for key in ['cases','references','preview_case_ids','initial_states','parameter_layout','optimizer','terms','normalizers','snapshots',
                'retained_capacity_gates','budgets','updates','timing_update','timing_safety_factor','raw_storage']:
        assert p[key]==old[key],key
    assert p['gradient_combination']['queries_bound']==5600 and p['gradient_combination']['loss_weights_changed'] is False
    assert p['gradient_orders']==[task_orders(i) for i in range(1,801)]
    validate_schedule(p['cases'],read(OUT/'schedule.json')['batches'])
    assert sha(OUT/'schedule.json')==sha(BASE/'schedule.json')
    assert (OUT/'scripts/cctv_dgp_pcgrad_fit_v41_vm.py').read_text()==worker((ROOT/'scripts/cctv_dgp_spatial_fit_v40_vm.py').read_text())
    assert (ROOT/'scripts/audit_cctv_dgp_pcgrad_fit_v41_return.py').read_text()==auditor((ROOT/'scripts/audit_cctv_dgp_spatial_fit_v40_return.py').read_text())
    assert (OUT/'cctv_dgp_spatial_decoder_v41.py').read_text()==decoder((BASE/'cctv_dgp_spatial_decoder_v40.py').read_text())
    for name in ['frozen_definitions.py','untrained_initial_decoder.pth','cctv_dgp_spatial_fit_v40_contract.py',
                 'cctv_dgp_batchmatched_identity_v26.py','cctv_dgp_degraded_objective_v24.py','weights/dgp_v2.pth','weights/w600k_r50.onnx']:
        assert sha(OUT/name)==sha(BASE/name),name
    old_audit=read(ROOT/'outputs/cctv_dgp_spatial_fit_v40_independent_audit.json')
    assert old_audit['failure_retained'] and not old_audit['necessary_capacity_pass'] and not old_audit['gates'][0]['pass']
    observed=set(); targets=set()
    for c in p['cases']:
        with Image.open(OUT/c['input']) as im: assert im.size==(256,256) and im.mode=='RGB'
        observed.add(c['observed']); targets.add(c['target']); assert c['role']=='train'
        assert time.monotonic()-start<300
    assert len(observed)==len(targets)==781
    archive=ROOT/'outputs/cctv-dgp-pcgrad-fit-v41-execution.tar.gz'
    assert sha(archive)==prepared['archive_sha256'] and archive.stat().st_size==prepared['archive_bytes']
    seen=set(); total=0
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        for member in members:
            assert member.isfile() and not member.issym() and not member.islnk() and member.name.startswith(NAME+'/')
            name=member.name[len(NAME)+1:]; assert name not in seen and name in set(p['assets_sha256'])|{'protocol.json'}
            assert not any(v in name for v in ['\\',':','..']) and (OUT/name).resolve().is_relative_to(OUT)
            seen.add(name); total+=member.size
            import hashlib
            with tar.extractfile(member) as stream:
                digest=hashlib.sha256()
                for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
            assert digest.hexdigest()==sha(OUT/name)
    assert seen==set(p['assets_sha256'])|{'protocol.json'}
    # Independent normative gradient fixtures, including zero-hinge and opposite directions.
    aligned=np.zeros((7,3),np.float64); aligned[:3]=[[1,0,0],[2,0,0],[0,1,0]]
    merged, record=combine(aligned,1); assert np.array_equal(merged,np.asarray([3,1,0],np.float32)) and record['conflict_projections']==0
    opposite=np.zeros((7,2),np.float64); opposite[:2]=[[1,0],[-1,0]]
    assert np.count_nonzero(combine(opposite,1)[0])==0
    assert np.count_nonzero(combine(np.zeros((7,3),np.float64),1)[0])==0
    for invalid in [np.zeros((6,3)),np.zeros((7,0)),np.full((7,3),np.nan),np.zeros((7,3),np.int32)]:reject(lambda invalid=invalid:combine(invalid,1))
    rng=np.random.default_rng(121); components=rng.normal(size=(7,17952)).astype(np.float32)*np.float32(.001)
    merged, projection=combine(components,1); expected,orders,conflicts=projected_gradient(components,1)
    assert np.array_equal(merged,expected) and orders==projection['orders'] and conflicts==projection['conflict_projections']
    scale=min(1.,1./(np.linalg.norm(merged.astype(np.float64))+1e-6)); applied=(merged.astype(np.float64)*scale).astype(np.float32)
    before=rng.normal(size=17952).astype(np.float32)*np.float32(.1); first=np.zeros(17952,np.float32); second=first.copy()
    m=(.1*applied.astype(np.float64)).astype(np.float32); v=(.001*np.square(applied.astype(np.float64))).astype(np.float32)
    after=(before.astype(np.float64)*(1-.0003*.01)-(.0003/(1-.9))*m.astype(np.float64)/(np.sqrt(v.astype(np.float64))/np.sqrt(1-.999)+1e-8)).astype(np.float32)
    fixture={'components':components,'combined':merged,'applied':applied,'before':before,'after':after,'exp_avg':m,'exp_avg_sq':v}
    receipt={'update':1,'component_queries':7,'projection':projection,'optimizer_steps_all_equal_update':True,'optimizer_type':'AdamW',
             'loss_weights_or_definitions_changed':False,'backward_calls':0}
    verify_step(fixture,receipt,1,before,first,second)
    corruption=[]
    for key in ['components','combined','applied','before','after','exp_avg','exp_avg_sq']:
        bad={k:v.copy() for k,v in fixture.items()}; bad[key].flat[0]+=np.float32(.2)
        reject(lambda bad=bad:verify_step(bad,receipt,1,before,first,second));corruption.append(key)
    fixture_path=PREP/'synthetic_arithmetic_only_step_fixture.npz'
    with fixture_path.open('xb') as stream:np.savez(stream,**fixture)
    assert all(np.array_equal(fixture[k],read_arrays(fixture_path)[k]) for k in fixture)
    assert 'torch' not in sys.modules
    spec=importlib.util.spec_from_file_location('prospective_v41_return_checker',ROOT/'scripts/audit_cctv_dgp_pcgrad_fit_v41_return.py')
    checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
    def member(name,size=1,kind=tarfile.REGTYPE):
        v=tarfile.TarInfo(checker.OUT.name+'/'+name);v.size=size;v.type=kind;return v
    for name in ['../escape','/absolute','C:/escape','outputs\\escape','outputs/steps/step0801.npz']:
        reject(lambda name=name:checker.safe_members([member(name)],p))
    reject(lambda:checker.safe_members([member('protocol.json',kind=tarfile.SYMTYPE)],p))
    reject(lambda:checker.safe_members([member('protocol.json',kind=tarfile.LNKTYPE)],p))
    reject(lambda:checker.safe_members([member('protocol.json'),member('protocol.json')],p))
    reject(lambda:checker.safe_members([member('protocol.json',size=16*1024**2+1)],p))
    tree=ast.parse((OUT/'scripts/cctv_dgp_pcgrad_fit_v41_vm.py').read_text())
    guard=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='scope')
    ns={'sys':SimpleNamespace(platform='linux'),'platform':SimpleNamespace(node=lambda:'forensic-dgp-thesis'),'Path':Path,'NAME':NAME}
    exec(compile(ast.Module(body=[guard],type_ignores=[]),'<scope-only>', 'exec'),ns)
    allowed=Path.home()/'forensic-dgp'/NAME;ns['scope'](allowed)
    reject(lambda:ns['scope'](allowed.with_name('cctv_dgp_spatial_fit_vm_v40')))
    ns['sys'].platform='win32';reject(lambda:ns['scope'](allowed));ns['sys'].platform='linux'
    ns['platform'].node=lambda:'another-VM';reject(lambda:ns['scope'](allowed))
    blocked=subprocess.run([sys.executable,'-B',str(OUT/'scripts/cctv_dgp_pcgrad_fit_v41_vm.py'),'--root',str(OUT),'--protocol-sha',sha(OUT/'protocol.json'),'--verify-transfer'],capture_output=True,text=True,timeout=10)
    assert blocked.returncode!=0 and 'stop before neural imports' in blocked.stderr
    write(PREP/'Windows_pre_neural_rejection.json',{'complete':True,'exit_code':blocked.returncode,'stderr':blocked.stderr,'VM_calls':0})
    fake=[{'id':c['id'],'source':c['source'],'profile':c['profile'],'metrics':{k:.1 for k in ['MSE','SSIM','ArcFace_observed_fixed','landmark_high_frequency_MSE','constant_mean_shift_only_MSE']}} for c in p['cases']]
    initial=groups(fake); better=copy.deepcopy(initial)
    for row in better.values():row['landmark_high_frequency_MSE']=.098
    assert capacity(initial,better,.01)['pass'] and not capacity(initial,better,.1)['pass']
    gate_failures=[]
    for name,value in [('MSE',.1+2e-12),('SSIM',.1-1.1e-6),('ArcFace_observed_fixed',.1-1.1e-6)]:
        bad=copy.deepcopy(better);bad['all'][name]=value;assert not capacity(initial,bad,.01)['pass'];gate_failures.append(name)
    bad=copy.deepcopy(better);bad['dataset/asian_faces/degraded']['landmark_high_frequency_MSE']=.101;assert not capacity(initial,bad,.01)['pass'];gate_failures.append('source')
    bad=copy.deepcopy(better);bad['degraded']['MSE']=.09;bad['degraded']['constant_mean_shift_only_MSE']=.097;assert not capacity(initial,bad,.01)['pass'];gate_failures.append('mean_only')
    syntax=0
    for path in OUT.rglob('*.py'):ast.parse(path.read_text(),feature_version=(3,10));syntax+=1
    guide=(ROOT/'CCTV_DGP_PCGRAD_FIT_V41_VM.md').read_text()
    commands=[line for line in guide.splitlines() if line.startswith('gcloud compute scp ')]
    assert len(commands)==5 and all(line.count('janusdominic0@forensic-dgp-thesis:')==1 for line in commands)
    assert 'tmux new-session -A -s dgp_pcgrad_fit_v41' in guide and sha(OUT/'protocol.json') in guide and prepared['archive_sha256'] in guide
    bash=read(PREP/'Bash_readonly_syntax.json');assert bash['complete'] and bash['script_sha256']==sha(OUT/'scripts/run_v41.sh')
    app=read(ROOT/'outputs/completion_feature_fusion_off_v1/protocol.json')['app_preservation_sha256'];assert len(app)==14
    for name,digest in app.items():assert sha(ROOT/name)==digest,name
    assert not (OUT/'outputs').exists() and not (ROOT/'outputs/cctv-dgp-pcgrad-fit-v41-results.tar.gz').exists() and 'torch' not in sys.modules
    assert time.monotonic()-start<300
    write(PREP/'independent_packet_audit.json',{'complete':True,'protocol_sha256':sha(OUT/'protocol.json'),'checker_sha256':sha(Path(__file__)),
        'archive_sha256':prepared['archive_sha256'],'archive_bytes':archive.stat().st_size,'members_checked':len(members),'uncompressed_bytes':total,
        'all3905_TRAIN_inputs_and781_references_and800_batches_fixed':True,'V40_architecture_initialization_losses_and_all_gates_unchanged':True,
        'exact_declared_worker_and_prospective_auditor_render':True,'independent_PCGrad_zero_aligned_opposite_random_fixtures_pass':True,
        'saved_step_corruption_rejections':corruption,'unsafe_return_archive_rejections':9,'scope_rejections':3,'Windows_pre_neural_rejection':True,
        'Python310_packet_sources':syntax,'Bash_readonly_syntax':True,'single_remote_source_gcloud_commands':5,
        'all14_DGP_primary_app_bindings_unchanged':True,'local_neural_calls':0,'local_gradient_calls':0,'local_optimizer_updates':0,'VM_calls':0,
        'prepared_only':True,'training_capacity_pass':False,'app_promotion':False,'goal_complete':False,'seconds':time.monotonic()-start,'cap_seconds':300})
    print({'complete':True,'cases':3905,'step_corruptions_rejected':7,'seconds':time.monotonic()-start},flush=True)


if __name__=='__main__':main()
