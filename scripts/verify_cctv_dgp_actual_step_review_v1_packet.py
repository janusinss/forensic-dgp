"""Independent frozen-input, archive and unsafe-execution regressions; no models."""
import argparse
import ast
import copy
import hashlib
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace
import numpy as np
from cctv_dgp_actual_step_review_v1_contract import NAME,STEM,RETURN_PREFIX,PROPOSALS,UPDATES,ROLES,BUDGETS,read,write,sha,check_proposal,verify,safe_members,allowed_return_names
from cctv_dgp_actual_step_review_v1_metrics import parameter_state_hash,review_groups,finite_comparison
from audit_cctv_dgp_actual_step_review_v1_return import derived_equal

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'outputs'/NAME
PREP=ROOT/'outputs/cctv_dgp_actual_step_review_v1_preparation'


def reject(action):
    try:action()
    except (AssertionError,ValueError,KeyError):return
    raise AssertionError('Unsafe or corrupted fixture was accepted')


def contract_checks(p,worker,supervisor):
    checks=[]
    for probe in p['probes']:
        with np.load(BUNDLE/probe['step_arrays'],allow_pickle=False) as z:step={k:z[k].copy() for k in z}
        with np.load(BUNDLE/probe['proposal_arrays'],allow_pickle=False) as z:proposal={k:z[k].copy() for k in z}
        check_proposal(step,proposal)
        assert all(len(parameter_state_hash(v,p['parameter_layout']))==64 for v in proposal['values'])
    probe=p['probes'][0]
    with np.load(BUNDLE/probe['step_arrays'],allow_pickle=False) as z:step={k:z[k].copy() for k in z}
    with np.load(BUNDLE/probe['proposal_arrays'],allow_pickle=False) as z:proposal={k:z[k].copy() for k in z}
    for label in ['recorded_parameters','rounded_cone','negative_dual','intended_delta','NaN_components','wrong_vector_shape']:
        a=copy.deepcopy(step);b=copy.deepcopy(proposal)
        if label=='recorded_parameters':b['values'][1,0]+=np.float32(.01)
        elif label=='rounded_cone':b['values'][2,0]+=np.float32(.01)
        elif label=='negative_dual':b['dual'][0]=-1
        elif label=='intended_delta':b['intended_delta'][0]+=.01
        elif label=='NaN_components':a['components'][0,0]=np.nan
        else:b['values']=b['values'][:,:-1]
        reject(lambda a=a,b=b:check_proposal(a,b));checks.append(label)
    def member(name,prefix=RETURN_PREFIX,kind=tarfile.REGTYPE,size=1):
        m=tarfile.TarInfo(prefix+'/'+name);m.type=kind;m.size=size;return m
    accepted,total=safe_members([member('protocol.json')],p);assert len(accepted)==total==1
    unsafe=[('old_V40_export_prefix',[member('protocol.json',prefix='cctv_dgp_spatial_fit_vm_v40_return')]),
      ('traversal',[member('../escape')]),('absolute',[member('/absolute')]),('drive',[member('C:/escape')]),
      ('backslash',[member('outputs\\escape')]),('symlink',[member('protocol.json',kind=tarfile.SYMTYPE)]),
      ('hardlink',[member('protocol.json',kind=tarfile.LNKTYPE)]),('duplicate',[member('protocol.json'),member('protocol.json')]),
      ('overlarge_member',[member('protocol.json',size=16*1024**2+1)]),('payload_code',[member('scripts/train.py')]),
      ('wrong_proposal',[member('outputs/probes/update0010/other/metrics.json')]),
      ('total_size_limit',[member(n,size=16*1024**2) for n in sorted(allowed_return_names(p))[:193]])]
    for label,members in unsafe:reject(lambda members=members:safe_members(members,p));checks.append(label)
    tree=ast.parse((ROOT/'scripts/cctv_dgp_actual_step_review_v1_contract.py').read_text())
    guard=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='vm_scope')
    ns={'sys':SimpleNamespace(platform='linux'),'platform':SimpleNamespace(node=lambda:'forensic-dgp-thesis'),'Path':Path,'NAME':NAME}
    exec(compile(ast.Module(body=[guard],type_ignores=[]),'<scope-only>','exec'),ns)
    expected=Path.home()/'forensic-dgp'/NAME;ns['vm_scope'](expected)
    reject(lambda:ns['vm_scope'](expected.with_name('cctv_dgp_pcgrad_fit_vm_v41')));checks.append('historical_VM_root')
    ns['platform'].node=lambda:'another-VM';reject(lambda:ns['vm_scope'](expected));checks.append('other_VM')
    ns['platform'].node=lambda:'forensic-dgp-thesis';ns['sys'].platform='win32';reject(lambda:ns['vm_scope'](expected));checks.append('Windows_scope')
    trees=[ast.parse(path.read_text(),feature_version=(3,10)) for path in [worker,supervisor]]
    forbidden={'backward','grad','Adam','AdamW','SGD','enable_vm_learning','enable_learning','train'}
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                assert node.func.attr not in forbidden,node.func.attr
                if node.func.attr in {'set_grad_enabled','requires_grad_'}:assert len(node.args)==1 and isinstance(node.args[0],ast.Constant) and node.args[0].value is False
    export_function=next(n for n in trees[0].body if isinstance(n,ast.FunctionDef) and n.name=='export')
    assert any(isinstance(n,ast.Name) and n.id=='RETURN_PREFIX' for n in ast.walk(export_function))
    child_receipts=[]
    for path,flag in [(worker,'--review'),(supervisor,None)]:
        code="import runpy,sys;from pathlib import Path;p=Path(sys.argv[1]);sys.path.insert(0,str(p.parent));a=sys.argv[2:];sys.argv=[str(p)]+a\ntry:runpy.run_path(str(p),run_name='__main__')\nexcept AssertionError as e:\n assert 'no neural imports occurred' in str(e);assert 'torch' not in sys.modules;print('Windows stopped before torch imports; zero model calls')\nelse:raise AssertionError('Windows execution was accepted')"
        args=[sys.executable,'-B','-c',code,str(path),'--root',str(BUNDLE),'--protocol-sha','0'*64]
        if flag:args.append(flag)
        result=subprocess.run(args,capture_output=True,text=True,timeout=20)
        assert result.returncode==0 and 'Windows stopped before torch imports' in result.stdout,result.stderr
        child_receipts.append({'script_sha256':sha(path),'exit_code':result.returncode,'stdout':result.stdout,'timeout_seconds':20})
    fake=[{'id':c['id'],'source':c['source'],'profile':c['profile'],**{s:{k:.1 for k in ['MSE','SSIM','ArcFace_observed_fixed','landmark_high_frequency_MSE','constant_mean_shift_only_MSE']} for s in ['raw','PNG']}} for c in p['cases']]
    groups=review_groups(fake,'PNG');assert len(groups)==17 and finite_comparison(groups,groups)['finite_preservation_pass']
    one=[r for r in fake if r['source']==fake[0]['source']];assert len(review_groups(one,'raw'))==10
    for metric,value in [('MSE',.1+2e-12),('SSIM',.1-1.1e-6),('ArcFace_observed_fixed',.1-1.1e-6)]:
        bad=copy.deepcopy(groups);bad['all'][metric]=value;comparison=finite_comparison(groups,bad)
        assert not comparison['finite_preservation_pass'] and any(f['metric']==metric for f in comparison['preservation_failures'])
        checks.append('unchanged_'+metric+'_preservation_allowance')
    derived_equal({'score':.1,'pass':False,'cases':5},{'score':.1+1e-12,'pass':False,'cases':5})
    for bad in [{'score':.1,'pass':True,'cases':5},{'score':.1,'pass':False,'cases':6},{'score':.11,'pass':False,'cases':5}]:
        reject(lambda bad=bad:derived_equal({'score':.1,'pass':False,'cases':5},bad))
    checks.append('derived_float_roundoff_only_with_exact_decisions')
    assert not (BUNDLE/'outputs').exists() and 'torch' not in sys.modules
    return checks,child_receipts


def main(draft):
    started=time.monotonic();PREP.mkdir(exist_ok=True)
    p=read(BUNDLE/('protocol.draft.json' if draft else 'protocol.json'))
    worker=ROOT/'scripts/review_cctv_dgp_actual_steps_v1_vm.py' if draft else BUNDLE/'scripts/review_cctv_dgp_actual_steps_v1_vm.py'
    supervisor=ROOT/'scripts/supervise_cctv_dgp_actual_steps_v1.py' if draft else BUNDLE/'scripts/supervise_cctv_dgp_actual_steps_v1.py'
    checks,child=contract_checks(p,worker,supervisor)
    if draft:
        write(PREP/'contract_regressions_before_freeze.json',{'complete':True,'checks':checks,'Windows_pre_neural_rejections':child,
             'script_sha256':sha(Path(__file__)),'worker_sha256':sha(worker),'supervisor_sha256':sha(supervisor),
             'neural_calls':0,'optimizer_updates':0,'gradient_queries':0,'VM_calls':0,'seconds':time.monotonic()-started})
        print({'complete':True,'contract_regressions':len(checks),'Windows_pre_neural_rejections':len(child),'neural_calls':0},flush=True);return
    prep=read(ROOT/'outputs/cctv_dgp_actual_step_review_v1_packet_preparation.json');pin=sha(BUNDLE/'protocol.json');verify(BUNDLE,pin)
    assert prep['complete'] and prep['protocol_sha256']==pin and prep['VM_calls']==prep['neural_calls_here']==0 and not prep['training_started']
    for name,digest in p['source_evidence_sha256'].items():assert sha(ROOT/name)==digest,name
    parent=read(ROOT/'outputs/cctv_dgp_pcgrad_fit_vm_v41/protocol.json');analysis=read(ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_analysis/analysis.json')
    assert p['cohorts']==analysis['cohorts'] and p['parameter_layout']==parent['parameter_layout'] and p['initial_states']==parent['initial_states']
    assert p['retained_capacity_gates']==parent['retained_capacity_gates'] and p['terms']==parent['terms']
    old_cases={c['id']:c for c in parent['cases']};old_refs={r['id']:r for r in parent['references']}
    for c in p['cases']:assert c==old_cases[c['id']]
    for r in p['references']:assert r==old_refs[r['id']]
    imported=read(ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_return_import.json')
    source=ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_return'
    for probe in p['probes']:
        record=read(source/f"outputs/steps/step{probe['update']:04d}.json")
        assert probe['case_ids']==record['case_ids'] and read(BUNDLE/probe['step_record'])==record
        original=source/f"outputs/steps/step{probe['update']:04d}.npz"
        assert sha(original)==imported['files_sha256'][original.relative_to(source).as_posix()]==sha(BUNDLE/probe['step_arrays'])
        with np.load(BUNDLE/probe['proposal_arrays'],allow_pickle=False) as proposed,np.load(original,allow_pickle=False) as saved:
            assert np.array_equal(proposed['values'][0],saved['before']) and np.array_equal(proposed['values'][1],saved['after'])
            assert np.array_equal(proposed['dual'],analysis['step_rows'][probe['update']-1]['one_fixed_saved_array_cone_proposal']['dual'])
            assert [hashlib.sha256(v.tobytes()).hexdigest() for v in proposed['values']]==probe['proposal_flat_float32_sha256']
    archive=ROOT/'outputs'/(STEM+'-execution.tar.gz');assert sha(archive)==prep['archive_sha256'] and archive.stat().st_size==prep['archive_bytes']
    assert Path(str(archive)+'.sha256').read_text().split()==[prep['archive_sha256'],archive.name]
    expected=set(p['assets_sha256'])|{'protocol.json'};seen=set();total=0
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        for member in members:
            assert member.isfile() and not member.issym() and not member.islnk() and member.name.startswith(NAME+'/')
            name=member.name[len(NAME)+1:];assert name in expected and name.casefold() not in seen and not any(c in name for c in ['\\',':'])
            seen.add(name.casefold());digest=hashlib.sha256()
            with tar.extractfile(member) as stream:
                for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
            assert digest.hexdigest()==sha(BUNDLE/name),name;total+=member.size
            assert time.monotonic()-started<300
    assert seen=={n.casefold() for n in expected}
    assert not expected.intersection(p['excluded_unused_historical_training_files'])
    syntax=0
    for name in expected:
        if name.endswith('.py'):ast.parse((BUNDLE/name).read_text(),feature_version=(3,10));syntax+=1
    guide=(ROOT/'CCTV_DGP_ACTUAL_STEP_REVIEW_V1_VM.md').read_text()
    commands=[line for line in guide.splitlines() if line.startswith('gcloud compute scp ')]
    assert len(commands)==5 and all(line.count('janusdominic0@forensic-dgp-thesis:')==1 for line in commands)
    assert 'tmux new-session -A -s dgp_actual_step_review_v1' in guide and pin in guide and prep['archive_sha256'] in guide
    bash=read(PREP/'Bash_readonly_syntax.json');assert bash['complete'] and bash['script_sha256']==sha(BUNDLE/'scripts/run_actual_step_review.sh')
    calibration=read(PREP/'fixed_filter_arithmetic.json');assert calibration['complete'] and calibration['archived_float_cases']==100 and calibration['new_model_forwards']==0
    app=read(ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_audit_recovery_r1/app_preservation_readback.json')['files_sha256']
    assert len(app)==14
    for name,digest in app.items():assert sha(ROOT/name)==digest,name
    assert not (BUNDLE/'outputs').exists() and not (ROOT/'outputs'/(STEM+'-results.tar.gz')).exists() and 'torch' not in sys.modules
    assert time.monotonic()-started<300
    write(PREP/'independent_packet_audit.json',{'complete':True,'protocol_sha256':pin,'archive_sha256':prep['archive_sha256'],
          'members_verified':len(members),'uncompressed_bytes':total,'cases':145,'references':29,'proposals':30,'review_forward_slots':3150,
          'original_frozen_TRAIN_cases_and_metadata_and_cohorts_and_saved_steps_exact':True,'original_architecture_and_loss_and_capacity_gates_unchanged':True,
          'historical_training_workers_excluded':True,'predeclared_return_checker_bound':True,'regressions':checks,'Windows_pre_neural_rejections':child,
          'Python310_sources':syntax,'Bash_readonly_syntax':True,'single_remote_source_gcloud_commands':5,'app_bindings_verified':14,
          'neural_calls':0,'VM_calls':0,'optimizer_updates':0,'gradient_queries':0,'training_started':False,'app_promotion':False,
          'checker_sha256':sha(Path(__file__)),'seconds':time.monotonic()-started,'cap_seconds':300})
    print({'complete':True,'members':len(members),'regressions':len(checks),'cases':145,'proposals':30,'VM_calls':0,'seconds':time.monotonic()-started},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--draft-contract-only',action='store_true');a=parser.parse_args();main(a.draft_contract_only)
