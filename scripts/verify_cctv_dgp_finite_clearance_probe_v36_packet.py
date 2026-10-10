"""Independent manual-packet audit; array checks and syntax, no local neural work."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from unittest.mock import patch

os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'outputs/cctv_dgp_finite_clearance_probe_v36_vm'
PREP=ROOT/'outputs/cctv_dgp_finite_clearance_probe_v36_preparation'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    started=time.monotonic()
    import numpy as np
    p=read(BUNDLE/'protocol.json');prep=read(PREP/'preparation_receipt.json')
    assert sha(BUNDLE/'protocol.json')==prep['protocol_sha256']
    archive=ROOT/'outputs/cctv-dgp-finite-clearance-probe-v36-execution.tar.gz'
    assert archive.stat().st_size==prep['archive_bytes'] and sha(archive)==prep['archive_sha256']
    assert Path(str(archive)+'.sha256').read_text().strip().split()==[prep['archive_sha256'],archive.name]
    expected={'protocol.json':prep['protocol_sha256'],**p['assets_sha256']}
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();assert len(members)==9
        for item in members:
            assert item.isfile() and not item.issym() and not item.islnk()
            name=item.name.removeprefix(BUNDLE.name+'/')
            assert name in expected and hashlib.sha256(tar.extractfile(item).read()).hexdigest()==expected[name]
    for name,digest in p['assets_sha256'].items():assert sha(BUNDLE/name)==digest,name
    for name,digest in p['local_basis_sha256'].items():assert sha(ROOT/name)==digest,name
    guard=ROOT/'outputs/cctv_dgp_group_guard_grad_v34_return'
    for name,digest in p['guard_return_sha256'].items():assert sha(guard/name)==digest,name
    assert len(p['guard_return_sha256'])==230
    assert p['retained_capacity_gates']==read(guard/'protocol.json')['retained_capacity_gates']
    assert p['optimizer_updates']==p['committed_trajectory_updates']==p['new_gradient_queries']==p['epochs']==0
    assert p['candidate_displacement_trials']==4 and p['states']==[0]
    assert [v['scale'] for v in p['variants']]==[1.,.5,.25,.125]
    assert p['before_outputs']+p['trial_outputs']==500 and p['CPU_replay_outputs']==100
    assert p['forward_call_limits']=={'reference_DGP_forwards':20,'candidate_DGP_forwards':100,'recognizer_forwards':320}
    checked_sources=0
    for source in [*BUNDLE.rglob('*.py'),ROOT/'scripts/audit_cctv_dgp_finite_clearance_probe_v36_return.py']:
        text=source.read_text(encoding='utf-8');ast.parse(text,feature_version=(3,10));checked_sources+=1
        assert 'torch.optim' not in text and '.backward(' not in text and 'autograd.grad' not in text
    worker_path=BUNDLE/'scripts/cctv_dgp_finite_clearance_probe_v36_vm.py'
    worker=module('new_V35_r1_scope_only',worker_path)
    assert worker.NAME==BUNDLE.name and worker.STEM=='cctv-dgp-finite-clearance-probe-v36'
    rejected=0
    for platform_name,hostname,folder in [('win32','forensic-dgp-thesis',BUNDLE),
        ('linux','another-vm',Path.home()/'forensic-dgp'/BUNDLE.name),
        ('linux','forensic-dgp-thesis',Path.home()/'forensic-dgp'/'wrong-folder')]:
        with patch.object(worker.sys,'platform',platform_name),patch.object(worker.platform,'node',return_value=hostname):
            try:worker.scope(folder.resolve())
            except AssertionError:rejected+=1
            else:raise AssertionError('Wrong host/platform/root must be rejected')
    with patch.object(worker.sys,'platform','linux'),patch.object(worker.platform,'node',return_value='forensic-dgp-thesis'):
        worker.scope((Path.home()/'forensic-dgp'/BUNDLE.name).resolve())
    before=set(q.relative_to(BUNDLE).as_posix() for q in BUNDLE.rglob('*') if q.is_file())
    windows=subprocess.run([sys.executable,'-B',str(worker_path),'--root',str(BUNDLE),'--protocol-sha',prep['protocol_sha256'],'--verify-transfer'],capture_output=True,text=True,timeout=15)
    assert windows.returncode!=0 and 'Existing Linux VM only' in windows.stderr
    assert before==set(q.relative_to(BUNDLE).as_posix() for q in BUNDLE.rglob('*') if q.is_file())
    (PREP/'Windows_pre_neural_rejection.txt').write_text(windows.stdout+windows.stderr,encoding='utf-8')
    auditor=module('prospective_V35_r1_archive_boundary',ROOT/'scripts/audit_cctv_dgp_finite_clearance_probe_v36_return.py')
    prior=module('pinned_V33_directory_readback',ROOT/'scripts/audit_cctv_dgp_loss_cone_probe_v33_return_r2.py')
    assert [auditor.PARENT,auditor.ACTIVE,auditor.V32,auditor.MIXED]==[prior.PARENT,prior.ACTIVE,prior.V32,prior.MIXED]
    for folder in [auditor.PARENT,auditor.ACTIVE,auditor.V32,auditor.MIXED]:assert folder.is_dir()
    assert auditor.BUNDLE==BUNDLE and auditor.OUT.name=='cctv_dgp_finite_clearance_probe_v36_return'
    def item(name,size=0,kind=tarfile.REGTYPE):
        value=tarfile.TarInfo(name);value.size=size;value.type=kind;return value
    prefix=auditor.PREFIX
    auditor.safe_members([item(prefix+'protocol.json')],p)
    bad=[[item(prefix+'../protocol.json')],[item('/'+prefix+'protocol.json')],
        [item(prefix+'protocol.json',kind=tarfile.SYMTYPE)],[item(prefix+'protocol.json',kind=tarfile.LNKTYPE)],
        [item(prefix+'protocol.json'),item(prefix+'protocol.json')],[item(prefix+'C:protocol.json')],
        [item(prefix+'unknown.npy')],[item(prefix+'protocol.json',8*1024**2+1)]]
    unsafe=0
    for members in bad:
        try:auditor.safe_members(members,p)
        except AssertionError:unsafe+=1
        else:raise AssertionError('Unsafe member must stop')
    geometry=module('shipped_V35_r1_array_verifier',BUNDLE/'cctv_dgp_finite_clearance_geometry_v36.py')
    actual_load=np.load
    mapping=[(BUNDLE.parent/'cctv_dgp_group_guard_grad_v34_vm',guard),
        (BUNDLE.parent/'cctv_dgp_v32_loss_gradient_v1_vm',ROOT/'outputs/cctv_dgp_v32_loss_gradient_v1_return')]
    def local(path):
        path=Path(path)
        for source,destination in mapping:
            if path.is_relative_to(source):return destination/path.relative_to(source)
        return path
    with patch.object(np,'load',side_effect=lambda path,**kw:actual_load(local(path),**kw)):
        theta,direction,rows,certificate=geometry.verify_geometry(BUNDLE,p,lambda q:sha(local(q)),read)
    assert certificate['complete'] and rows.shape==(108,978243)
    assert certificate['nonzero_clearance_rows']==65 and certificate['empirical_clearance_not_a_validated_loss_bound']
    proof=read(BUNDLE/'projection.json')
    assert proof['clearance_targets']==p['clearance_targets']
    # Calibrate again from the saved numerical receipts, without using the
    # analysis's per-row residual values or trusting an asserted feasibility flag.
    measured=read(ROOT/'outputs/cctv_dgp_group_guard_probe_v35_r1_analysis_v1_r1/analysis.json')
    targets=np.zeros(108,np.float64)
    for index,label in enumerate(p['constraint_labels'][:102]):
        if label['metric']=='one_minus_raw_SSIM':continue
        base=ROOT/f'outputs/cctv_dgp_group_guard_probe_v35_r1_return/outputs/state0_{label["cohort"]}'
        before=read(base/'before/receipt.json');after=read(base/'joint_1/receipt.json')
        group=label['group']
        def selected(row):
            if group=='all':return True
            if group=='clear':return row['profile']=='clear'
            if group=='degraded':return row['profile']!='clear'
            source,kind=group.rsplit('/',1)
            return row['source']==source and (kind=='all' or (kind=='degraded' and row['profile']!='clear') or row['profile']==kind)
        metric='raw_MSE' if label['metric']=='raw_MSE' else 'raw_ArcFace'
        old=float(np.mean([r['metrics'][metric] for r in before['rows'] if selected(r)]))
        new=float(np.mean([r['metrics'][metric] for r in after['rows'] if selected(r)]))
        raw_change=new-old if metric=='raw_MSE' else old-new
        png_key='MSE' if metric=='raw_MSE' else 'ArcFace_observed_fixed'
        old,new=before['groups'][group][png_key],after['groups'][group][png_key]
        png_change=new-old if png_key=='MSE' else old-new
        linear=read(base/'joint_1/comparison.json')['all108_linear_function_changes'][index]
        targets[index]=2*max(0.,raw_change-linear,png_change-linear)
    assert np.array_equal(targets,np.asarray(p['clearance_targets'])) and np.count_nonzero(targets)==65
    assert proof['magnitude_ratio']<=2 and np.count_nonzero(targets[102:])==0
    # The original failure arose from mismatched archive prefixes. Check the
    # concrete exporter AST literal against this prospective checker.
    tree=ast.parse(worker_path.read_text())
    export=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='export')
    export_prefixes=[node.value for node in ast.walk(export) if isinstance(node,ast.Constant)
        and isinstance(node.value,str) and node.value.endswith('_return/')]
    assert export_prefixes==[auditor.PREFIX]
    original_tree=ast.parse((ROOT/'outputs/cctv_dgp_group_guard_probe_v35_r1_vm/scripts/cctv_dgp_group_guard_probe_v35_r1_vm.py').read_text())
    original_run=next(n for n in original_tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    revised_run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    find_snapshot=lambda node:next(n for n in node.body if isinstance(n,ast.Try) for n in ast.walk(n)
        if isinstance(n,ast.FunctionDef) and n.name=='snapshot')
    old_snapshot,new_snapshot=find_snapshot(original_run),find_snapshot(revised_run)
    # Only its printed version label differs; image preparation and measurement
    # code are the already audited V35 definitions.
    old_text=ast.unparse(old_snapshot).replace('V35_state','V36_state')
    assert old_text==ast.unparse(new_snapshot)
    a=read(ROOT/'outputs/cctv_dgp_group_guard_grad_v34_analysis_v1_r1/analysis.json')
    assert np.array_equal(direction,actual_load(ROOT/'outputs/cctv_dgp_v35_finite_clearance_v1/candidate_displacement.npy',allow_pickle=False))
    syntax=read(PREP/'Bash_readonly_syntax.json')
    assert syntax['exit_code']==0 and syntax['script_sha256']==sha(BUNDLE/'scripts/run_v36_probe.sh')
    receipt={'complete':True,'checker_sha256':sha(Path(__file__)),'protocol_sha256':prep['protocol_sha256'],
        'archive_sha256':prep['archive_sha256'],'archive_bytes':prep['archive_bytes'],'archive_files':9,
        'V34_return_bindings_verified':230,'local_bindings_verified':len(p['local_basis_sha256']),
        'all108_projection_rows_verified':True,'geometry_certificate':certificate,
        'all65_empirical_clearances_independently_recalibrated':True,
        'exporter_and_prospective_checker_prefix_match':True,'frozen_snapshot_AST_preserved':True,
        'Python310_sources_checked':checked_sources,'Bash_readonly_syntax_verified':True,
        'scope_rejections':rejected,'Windows_pre_neural_rejection_passed':True,'unsafe_member_rejections':unsafe,
        'valid_V36_scope_and_all_replay_directories_verified':True,'original_gates_retained':True,
        'neural_calls':0,'gradient_calls':0,'optimizer_updates':0,'VM_launches':0,
        'VM_execution_started':False,'app_promotion':False,'goal_complete':False,'seconds':time.monotonic()-started}
    assert receipt['seconds']<300
    with (PREP/'independent_packet_audit.json').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'complete':True,'files':9,'constraints':108,'seconds':receipt['seconds']}))


if __name__=='__main__':main()
