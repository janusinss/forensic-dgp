"""Independent packet, resource and grouping regressions; no neural/gradient calls."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs/cctv_dgp_group_guard_grad_v34_vm'
PREP = ROOT / 'outputs/cctv_dgp_group_guard_grad_v34_preparation'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def reject(call):
    try: call()
    except (AssertionError,ValueError): return
    raise AssertionError('Expected boundary rejection')


def main():
    started=time.monotonic();p=read(BUNDLE/'protocol.json');pin=sha(BUNDLE/'protocol.json');prep=read(PREP/'preparation_receipt.json')
    assert prep['complete'] and prep['protocol_sha256']==pin and prep['packet_files']==3
    checker=module('prospective_V34_hash_and_group_checks',ROOT/'scripts/audit_cctv_dgp_group_guard_grad_v34_return.py')
    _,old,basis=checker.verify_basis(p)
    assert p['retained_capacity_gates']==old['retained_capacity_gates']
    assert p['cohorts']==basis['cohorts'] and p['parameter_layout']==basis['parameter_layout']
    assert sum(len(c['cases']) for c in p['cohorts'])==100 and all(r['role']=='train' for c in p['cohorts'] for r in c['cases'])
    assert p['gradient_queries']==3*100==300 and p['optimizer_updates']==p['parameter_updates']==p['epochs']==0
    assert p['states']==[0] and p['budgets']['minimum_free_disk_bytes']==6*1024**3
    files={q.relative_to(BUNDLE).as_posix() for q in BUNDLE.rglob('*') if q.is_file()}
    assert files==set(p['assets_sha256'])|{'protocol.json'} and len(files)==3
    archive=ROOT/'outputs/cctv-dgp-group-guard-grad-v34-execution.tar.gz'
    assert sha(archive)==prep['archive_sha256'] and archive.stat().st_size==prep['archive_bytes']
    assert Path(str(archive)+'.sha256').read_text().strip().split()==[prep['archive_sha256'],archive.name]
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();assert len(members)==3
        for item in members:
            assert item.isfile() and not item.issym() and not item.islnk() and item.name.startswith(BUNDLE.name+'/')
            n=item.name[len(BUNDLE.name)+1:];assert n in files
            assert hashlib.sha256(tar.extractfile(item).read()).hexdigest()==sha(BUNDLE/n)
    worker=BUNDLE/'scripts/cctv_dgp_group_guard_grad_v34_vm.py'
    for path in [worker,ROOT/'scripts/audit_cctv_dgp_group_guard_grad_v34_return.py',Path(__file__),ROOT/'scripts/prepare_cctv_dgp_group_guard_grad_v34.py']:
        tree=ast.parse(path.read_text(encoding='utf-8'),feature_version=(3,10))
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                assert node.func.attr not in ['backward','AdamW','Adam','SGD','step'],(path.name,node.func.attr)
                if node.func.attr=='grad': assert path==worker and isinstance(node.func.value,ast.Attribute) and node.func.value.attr=='autograd'
                assert not (node.func.attr=='save' and isinstance(node.func.value,ast.Name) and node.func.value.id=='torch')
    guard=subprocess.run([sys.executable,'-B',str(worker),'--root',str(BUNDLE),'--protocol-sha',pin,'--run'],
        cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert guard.returncode!=0 and 'Existing Linux VM only; no local gradients' in guard.stderr and not (BUNDLE/'outputs').exists()
    with (PREP/'Windows_gradient_guard.txt').open('x',encoding='utf-8') as stream:stream.write(guard.stderr)
    helper=module('V34_asset_verifier_only',worker);assert helper.verify(BUNDLE,pin)==p
    reject(lambda:helper.verify(BUNDLE,'0'*64))
    # Root simulation executes the exact isolated guard, not the VM worker.
    tree=ast.parse(worker.read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='scope')
    namespace={'sys':SimpleNamespace(platform='linux'),'platform':SimpleNamespace(node=lambda:'forensic-dgp-thesis'),
        'Path':Path,'NAME':BUNDLE.name}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<V34-root-only-regression>','exec'),namespace)
    expected=(Path.home()/'forensic-dgp'/BUNDLE.name).resolve();namespace['scope'](expected)
    reject(lambda:namespace['scope'](expected.parent/'cctv_dgp_feature_fusion_vm_v32_r2'))
    reject(lambda:namespace['scope'](expected.parent/'cctv_dgp_group_guard_grad_v34_vm_r1'))
    namespace['platform']=SimpleNamespace(node=lambda:'different-vm');reject(lambda:namespace['scope'](expected))
    member=tarfile.TarInfo(checker.PREFIX+'protocol.json');member.size=1
    assert checker.safe_members([member],p)[1]==1
    adversaries=[]
    for name in ['../outside','unapproved.pth','outputs\\outside','C:/outside']:
        bad=copy.copy(member);bad.name=checker.PREFIX+name;adversaries.append([bad])
    for kind in [tarfile.SYMTYPE,tarfile.LNKTYPE]:
        bad=copy.copy(member);bad.type=kind;bad.linkname='/outside';adversaries.append([bad])
    bad=copy.copy(member);bad.size=65*1024**2;adversaries.append([bad]);adversaries.append([member,member])
    for value in adversaries:reject(lambda value=value:checker.safe_members(value,p))
    import numpy as np
    checks=0
    for cohort in p['cohorts']:
        cases=cohort['cases'];groups=checker.group_indices(cases)
        assert len(groups)==17 and len(groups['all'])==50 and len(groups['clear'])==10 and len(groups['degraded'])==40
        assert all(len(ids)==5 for key,ids in groups.items() if key.endswith('/blur_lr24') or key.endswith('/clear'))
        # An average can conceal opposing source-specific identity derivatives.
        sources=sorted({c['source'] for c in cases});case_g=np.zeros((50,3,2),np.float64)
        for index,c in enumerate(cases):
            if c['profile']=='blur_lr24':case_g[index,2,0]=1. if c['source']==sources[0] else -1.
        assert case_g.mean(0)[2,0]==0
        assert case_g[groups[sources[0]+'/blur_lr24']].mean(0)[2,0]==1
        assert case_g[groups[sources[1]+'/blur_lr24']].mean(0)[2,0]==-1
        assert case_g[groups['degraded']].mean(0)[2,0]==0
        order=np.arange(50)[::-1];reordered=[cases[i] for i in order];other=checker.group_indices(reordered)
        for key,ids in groups.items():assert np.array_equal(case_g[ids].mean(0),case_g[order][other[key]].mean(0))
        checks+=3
        reject(lambda:checker.group_indices([c for c in cases if c['profile']!='clear']))
    assert checks==6
    assert 15*978243*4+128 < 64*1024**2
    expected_arrays=20*(15*978243*4+128)+100*(256*256*3*4+128)
    assert expected_arrays+100*256*256*3+8*1024**2 < p['budgets']['export_uncompressed_bytes']
    bash=Path('C:/Program Files/Git/bin/bash.exe');assert bash.is_file()
    syntax=subprocess.run([str(bash),'--noprofile','--norc','-n',str(BUNDLE/'scripts/run_v34_guard.sh')],capture_output=True,text=True,timeout=30)
    assert syntax.returncode==0,syntax.stderr
    guide=(ROOT/'CCTV_DGP_GROUP_GUARD_GRAD_V34_VM.md').read_text();assert pin in guide and prep['archive_sha256'] in guide
    lines=[line for line in guide.splitlines() if line.startswith('gcloud compute scp')]
    assert len(lines)==5 and all(line.count('janusdominic0@forensic-dgp-thesis:')==1 for line in lines)
    receipt={'complete':True,'protocol_sha256':pin,'archive_sha256':prep['archive_sha256'],'archive_members_verified':3,
        'checker_sha256':sha(Path(__file__)),'basis_VM_bindings_verified':len(p['basis_VM_sha256']),
        'V33_original_bindings_verified':len(p['V33_original_readback_sha256']),'local_basis_bindings_verified':len(p['local_basis_sha256']),
        'Python310_syntax_verified':True,'Bash_readonly_syntax_verified':True,'Windows_rejected_before_neural_imports_or_writes':True,
        'wrong_root_and_instance_rejections':3,'unsafe_archive_boundary_regressions':8,'group_cancellation_reorder_count_regressions':6,
        'original_17_group_and_capacity_gates_retained':True,'expected_raw_array_bytes':expected_arrays,
        'gradient_queries_prepared':300,'actual_gradient_queries':0,'actual_neural_calls':0,'actual_optimizer_updates':0,
        'parameter_assignments':0,'VM_launches':0,'new_checkpoint_created':False,'goal_complete':False,'seconds':time.monotonic()-started}
    with (PREP/'independent_packet_audit.json').open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
