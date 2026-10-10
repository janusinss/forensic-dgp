"""Independent transfer/data/guard audit. No gradients, fitting or VM action."""
import ast
import hashlib
import io
from pathlib import Path,PurePosixPath
import random
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace
import zipfile
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];PACKET=ROOT/'outputs/cctv_dgp_head4_capacity_vm_v1'
PREP=ROOT/'outputs/cctv_dgp_head4_capacity_v1_preparation'
sys.path.insert(0,str(PACKET))
from cctv_dgp_head4_capacity_contract_v1 import NAME,STEM,BUDGETS,read,write,sha,validate


def main():
    start=time.monotonic();assert not (PREP/'independent_packet_audit.json').exists()
    prep=read(PREP/'preparation.json');p=read(PACKET/'protocol.json');validate(p);pin=sha(PACKET/'protocol.json')
    archive=ROOT/'outputs'/(STEM+'-execution.tar.gz')
    assert pin==prep['protocol_sha256'] and sha(archive)==prep['packet_sha256'] and archive.stat().st_size==prep['packet_bytes']
    assert Path(str(archive)+'.sha256').read_text().strip().split()==[prep['packet_sha256'],archive.name]
    for n,d in p['assets_sha256'].items():assert sha(PACKET/n)==d,n
    for n,d in p['local_sources'].items():assert sha(ROOT/n)==d,n
    for n,origin in p['source_mapping'].items():assert sha(PACKET/n)==sha(ROOT/origin),n
    expected={NAME+'/'+n:d for n,d in p['assets_sha256'].items()};expected[NAME+'/protocol.json']=pin;seen=set()
    with tarfile.open(archive,'r:gz') as tar:
        for m in tar:
            assert m.isfile() and not m.issym() and not m.islnk() and m.name in expected and m.name not in seen
            assert '..' not in PurePosixPath(m.name).parts and m.name==PurePosixPath(m.name).as_posix()
            seen.add(m.name);h=hashlib.sha256()
            with tar.extractfile(m) as f:
                for b in iter(lambda:f.read(1024**2),b''):h.update(b)
            assert h.hexdigest()==expected[m.name],m.name
    assert seen==set(expected)
    parses=[]
    for f in sorted(PACKET.rglob('*.py')):ast.parse(f.read_text(),feature_version=(3,10));parses.append(f.relative_to(PACKET).as_posix())
    for n in p['local_sources']:
        if n.endswith('.py'):ast.parse((ROOT/n).read_text(),feature_version=(3,10))
    corpus=p['mixed_TRAIN_assets_sha256'];assert len(corpus)==5467 and all(p['assets_sha256'][n]==d for n,d in corpus.items())
    cases=p['cases'];refs={r['id']:r for r in p['references']};assert len(refs)==781
    assert all(c['role']=='train' and c['source_person_or_reference'] in refs for c in cases)
    assert all(r['role']=='train' for r in refs.values())
    profiles=['clear','blur_lr24','lowlight_lr32','motion_lr48','compound_lr24'];clears=0
    for begin in range(0,3905,5):
        batch=cases[begin:begin+5];ref=refs[batch[0]['source_person_or_reference']]
        assert len({c['source_person_or_reference'] for c in batch})==1 and [c['profile'] for c in batch]==profiles
        for c in batch:
            assert c['input'] in corpus and c['target']==ref['target'] and c['observed']==ref['observed']
        with Image.open(PACKET/ref['target']) as im:assert im.size==(256,256);target=np.asarray(im.convert('RGB')).copy()
        assert hashlib.sha256(target.tobytes()).hexdigest()==p['canonical_target_RGB_sha256'][ref['id']]
        with Image.open(PACKET/batch[0]['input']) as im:assert np.array_equal(target,np.asarray(im.convert('RGB')))
        clears+=1
    shuffled=list(range(781));random.Random(501050).shuffle(shuffled)
    assert p['full_epoch_reference_order']==shuffled and p['schedule']==[list(range(i*5,i*5+5)) for i in shuffled[:50]]
    assert len({cases[q[0]]['source_person_or_reference'] for q in p['schedule']})==50
    assert all(i in {c['id'] for c in cases} for i in p['preview_case_ids']+p['independent_replay_ids'])
    parent=read(ROOT/'outputs/cctv_dgp_head4_reactivation_vm_v1/protocol.json')
    assert p['preview_case_ids']==[c['id'] for c in parent['cases']]
    parity_path=ROOT/'outputs/cctv_dgp_head4_capacity_v1_parity';parity=read(parity_path/'results.json')
    assert parity['complete'] and parity['exact_initial_parity_cases']==100 and parity['maximum_initial_error']==0
    assert parity['entire_original_fusion_tensor_unchanged'] and parity['unchanged_full_net_state_entries']==620
    assert parity['local_gradient_queries']==parity['optimizer_updates']==0
    for n,d in read(parity_path/'plan.json')['source_bindings'].items():assert sha(ROOT/n)==d
    assert read(parity_path/'codec_proof.json')['complete']
    worker=PACKET/'scripts/cctv_dgp_head4_capacity_vm_v1.py';commands=[]
    for flag in ['verify-transfer','run']:
        cp=subprocess.run([sys.executable,'-B',str(worker),'--root',str(PACKET),'--protocol-sha',pin,'--'+flag],capture_output=True,text=True,timeout=60)
        if flag=='verify-transfer':assert cp.returncode==0 and "'neural_or_training_calls': 0" in cp.stdout
        else:assert cp.returncode!=0 and 'Actual training requires the manual existing Linux VM' in cp.stderr
        commands.append({'flag':flag,'exit_code':cp.returncode,'stdout':cp.stdout,'stderr':cp.stderr})
    assert not (PACKET/'outputs').exists()
    checker=ROOT/'scripts/audit_cctv_dgp_head4_capacity_return_v1.py';tree=ast.parse(checker.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['safe','pack_headers']]
    ns={'PurePosixPath':PurePosixPath,'NAME':NAME,'np':np,'zipfile':zipfile}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<incoming-guards-only>','exec'),ns)
    good=tarfile.TarInfo(NAME+'_return/outputs/results.json');good.size=10;ns['safe'](good);rejected=0
    bad=[NAME+'_return/../../escape',NAME+'_return/outputs/../escape','/absolute',NAME+'_return/C:/escape',
        NAME+'_return/outputs\\escape',NAME+'_return/outputs//alias',NAME+'_return/outputs/./alias']
    for n in bad:
        try:ns['safe'](tarfile.TarInfo(n))
        except AssertionError:rejected+=1
        else:raise AssertionError('Unsafe incoming path accepted')
    for kind in [tarfile.SYMTYPE,tarfile.LNKTYPE,tarfile.DIRTYPE]:
        m=tarfile.TarInfo(NAME+'_return/outputs/link');m.type=kind
        try:ns['safe'](m)
        except AssertionError:rejected+=1
        else:raise AssertionError('Unsafe incoming member type accepted')
    for n in ['codec_baseline_test.npz','codec_xor_test.npz']:ns['pack_headers'](parity_path/n)
    malformed=io.BytesIO()
    with zipfile.ZipFile(malformed,'w') as z:
        for n in ['planes.npy','shape.npy','xor.npy','embeddings.npy','ids.npy']:
            header=io.BytesIO();np.lib.format.write_array_header_1_0(header,{'shape':(2**35,), 'fortran_order':False,'descr':'|u1'})
            z.writestr(n,header.getvalue())
    malformed.seek(0)
    try:ns['pack_headers'](malformed)
    except AssertionError:rejected+=1
    else:raise AssertionError('Oversized array header accepted before allocation')
    # Retain a synthetic killed-worker receipt without executing any subprocess.
    probe=ROOT/'scratch/head4_capacity_supervisor_guard_v1';assert not probe.exists()
    fake_home=probe/'home';stage=fake_home/'forensic-dgp'/NAME;stage.mkdir(parents=True)
    class ProbePath(type(Path())):
        @classmethod
        def home(cls):return fake_home.resolve()
    class Parser:
        def add_argument(self,*a,**k):pass
        def parse_args(self):return SimpleNamespace(root=stage,protocol_sha=pin)
    calls=[]
    def fake_child(root,pin,mode,log,seconds):
        calls.append({'mode':mode,'cap':seconds});return (124,2432.) if mode=='run' else (0,1.)
    sup=ast.parse((PACKET/'scripts/supervise_cctv_dgp_head4_capacity_v1.py').read_text())
    mainnode=next(n for n in sup.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    sn={'argparse':SimpleNamespace(ArgumentParser=Parser),'Path':ProbePath,'NAME':NAME,'BUDGETS':BUDGETS,
        'sys':SimpleNamespace(platform='linux',executable=str(fake_home.resolve()/'forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python')),
        'platform':SimpleNamespace(node=lambda:'forensic-dgp-thesis'),'os':SimpleNamespace(environ={'TMUX':'synthetic-test'}),
        'verify':lambda *_:None,'child':fake_child,'read':read,'write':write}
    exec(compile(ast.Module(body=[mainnode],type_ignores=[]),'<no-process-deadline-retention-test>','exec'),sn)
    try:sn['main']()
    except SystemExit as e:assert e.code==124
    else:raise AssertionError('Killed-worker status was lost')
    f=read(stage/'outputs/failure.json');assert f['exact_completed_updates_unknown'] and f['progress']['optimizer_updates'] is None
    assert calls==[{'mode':'run','cap':2430},{'mode':'export','cap':630}]
    text=(ROOT/'CCTV_DGP_HEAD4_CAPACITY_V1_VM.md').read_text();assert sha(ROOT/'CCTV_DGP_HEAD4_CAPACITY_V1_VM.md')==prep['manual_guide_sha256']
    assert pin in text and prep['packet_sha256'] in text and '14GiB' in text
    scps=[line for line in text.splitlines() if line.startswith('gcloud compute scp ')];assert len(scps)==5
    for line in scps:
        assert '--project=forensic-dgp-thesis' in line and '--zone=us-central1-a' in line
        assert line.count('janusdominic0@forensic-dgp-thesis:')==1
    assert sum(line.endswith('"."') for line in scps)==3 and 'tmux new-session -A -s dgp_head4_capacity_v1' in text
    for n,d in p['assets_sha256'].items():assert sha(PACKET/n)==d
    for n,d in p['local_sources'].items():assert sha(ROOT/n)==d
    receipt={'complete':True,'protocol_sha256':pin,'packet_sha256':prep['packet_sha256'],'archive_members_verified':len(seen),
        'TRAIN_assets_preserved':5467,'canonical_TRAIN_targets_verified':781,'clear_inputs_exact_to_targets':clears,
        'fitting_schedule_updates':50,'fitting_exposures':250,'snapshot_cases':3905,'completed_epochs_planned':0,
        'source_bindings_verified':len(p['local_sources']),'python310_files_parsed':parses,'black_box_windows_checks':commands,
        'unsafe_boundary_rejections':rejected,'bounded_array_headers_before_allocation':True,'initializer_parity_verified':True,
        'full_state_export_and_independent_checker_source_bound':True,'external_supervisor_partial_failure_retention_verified':True,
        'real_subprocesses_in_deadline_retention_test':0,'manual_scp_commands':5,'one_remote_source_per_download':True,
        'local_neural_calls':0,'local_gradient_queries':0,'optimizer_updates':0,'VM_launched':False,
        'CUDA_actual_training_unverified':True,'model_qualification':False,'goal_complete':False,'seconds':time.monotonic()-start}
    write(PREP/'independent_packet_audit.json',receipt)
    print({'complete':True,'members':len(seen),'TRAIN_targets':781,'source_bindings':len(p['local_sources']),
        'unsafe_rejections':rejected,'VM_launched':False},flush=True)


if __name__=='__main__':main()
