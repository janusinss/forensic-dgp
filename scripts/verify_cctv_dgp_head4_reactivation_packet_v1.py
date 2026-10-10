"""Independent archive/source/guard/command checks before any manual VM run."""
import ast
import io
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]
PACKET=ROOT/'outputs/cctv_dgp_head4_reactivation_vm_v1'
PREP=ROOT/'outputs/cctv_dgp_head4_reactivation_v1_preparation'
sys.path.insert(0,str(PACKET))
from cctv_dgp_head4_reactivation_contract_v1 import NAME,STEM,COMPONENTS,BUDGETS,read,write,sha,validate


def main():
    start=time.monotonic();receipt=read(PREP/'preparation.json');p=read(PACKET/'protocol.json');validate(p)
    assert not (PREP/'independent_packet_audit.json').exists()
    pin=sha(PACKET/'protocol.json');archive=ROOT/'outputs'/(STEM+'-execution.tar.gz')
    assert pin==receipt['protocol_sha256'] and sha(archive)==receipt['packet_sha256']
    assert archive.stat().st_size==receipt['packet_bytes']
    assert Path(str(archive)+'.sha256').read_text(encoding='ascii').strip().split()==[receipt['packet_sha256'],archive.name]
    for name,digest in p['assets_sha256'].items():assert sha(PACKET/name)==digest,name
    for name,digest in p['local_source_bindings'].items():assert sha(ROOT/name)==digest,name
    assert sha(PACKET/'evidence/parent_protocol.json')==p['source_protocol_sha256']
    assert p['provenance_terms_overlap_from_parent_unchanged']
    for name,origin in p['source_mapping'].items():assert sha(PACKET/name)==sha(ROOT/origin),name
    expected={NAME+'/'+n:d for n,d in p['assets_sha256'].items()};expected[NAME+'/protocol.json']=pin
    names=set()
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            assert member.isfile() and not member.issym() and not member.islnk() and member.name not in names
            assert member.name in expected and '..' not in member.name and '\\' not in member.name
            names.add(member.name)
            import hashlib
            h=hashlib.sha256()
            with tar.extractfile(member) as stream:
                for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
            assert h.hexdigest()==expected[member.name],member.name
    assert names==set(expected)
    parses=[]
    for file in sorted(PACKET.rglob('*.py')):
        ast.parse(file.read_text(encoding='utf-8'),feature_version=(3,10));parses.append(file.relative_to(PACKET).as_posix())
    checker=ROOT/'scripts/audit_cctv_dgp_head4_reactivation_return_v1.py'
    tree=ast.parse(checker.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='safe_member')
    from pathlib import PurePosixPath
    ns={'PurePosixPath':PurePosixPath,'NAME':NAME};exec(compile(ast.Module(body=[node],type_ignores=[]),'<incoming-boundary-only>','exec'),ns)
    valid=tarfile.TarInfo(NAME+'_return/outputs/results.json');valid.size=10;ns['safe_member'](valid)
    bad=[NAME+'_return/../../escape',NAME+'_return/outputs/../escape','/absolute',NAME+'_return/C:/escape',
        NAME+'_return/outputs\\escape',NAME+'_return/outputs//alias',NAME+'_return/outputs/./alias']
    rejected=0
    for name in bad:
        member=tarfile.TarInfo(name)
        try:ns['safe_member'](member)
        except AssertionError:rejected+=1
        else:raise AssertionError('Incoming unsafe path accepted: '+name)
    for kind in [tarfile.SYMTYPE,tarfile.LNKTYPE,tarfile.DIRTYPE]:
        member=tarfile.TarInfo(NAME+'_return/outputs/link');member.type=kind
        try:ns['safe_member'](member)
        except AssertionError:rejected+=1
        else:raise AssertionError('Incoming unsafe member type accepted')
    worker=PACKET/'scripts/cctv_dgp_head4_reactivation_vm_v1.py'
    commands=[]
    for mode in ['verify-transfer','run']:
        cp=subprocess.run([sys.executable,'-B',str(worker),'--root',str(PACKET),'--protocol-sha',pin,'--'+mode],
            capture_output=True,text=True,timeout=30)
        if mode=='verify-transfer':assert cp.returncode==0 and "'neural_or_gradient_calls': 0" in cp.stdout
        else:assert cp.returncode!=0 and 'Existing Linux VM only; no local gradients' in cp.stderr
        commands.append({'mode':mode,'exit_code':cp.returncode,'stdout':cp.stdout,'stderr':cp.stderr})
    assert not (PACKET/'outputs').exists()
    wt=ast.parse(worker.read_text());assert not any(isinstance(n,ast.Attribute) and n.attr in ['step','backward'] for n in ast.walk(wt))
    assert not any(isinstance(n,ast.Attribute) and n.attr in ['Adam','AdamW','SGD','optim'] for n in ast.walk(wt))
    guide=ROOT/'CCTV_DGP_HEAD4_REACTIVATION_V1_VM.md';text=guide.read_text(encoding='utf-8')
    assert sha(guide)==receipt['manual_guide_sha256'] and pin in text and receipt['packet_sha256'] in text
    scps=[line for line in text.splitlines() if line.startswith('gcloud compute scp ')]
    assert len(scps)==5
    for line in scps:
        assert '--project=forensic-dgp-thesis' in line and '--zone=us-central1-a' in line
        assert line.count('janusdominic0@forensic-dgp-thesis:')==1,'PuTTY permits one remote source'
    assert len([line for line in scps if line.endswith('"."')])==3
    assert 'tmux new-session -A -s dgp_head4_reactivation_v1' in text
    assert 'source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate' in text
    assert 'python -B -u scripts/supervise_cctv_dgp_head4_reactivation_v1.py --root . --protocol-sha '+pin in text
    assert '3GiB' in text and 'zero optimizer' in text
    parity=read(PACKET/'evidence/parity_independent_audit.json');assert parity['complete'] and parity['local_optimizer_updates']==0
    outcome={'complete':True,'protocol_sha256':pin,'packet_sha256':receipt['packet_sha256'],'archive_members_verified':len(names),
        'assets_verified':len(p['assets_sha256']),'source_bindings_verified':len(p['local_source_bindings']),
        'python310_files_parsed':parses,'unsafe_return_boundaries_rejected':rejected,'black_box_windows_checks':commands,
        'incoming_guard_accepts_regular_evidence':True,'manual_scp_lines_verified':5,'one_remote_source_per_download':True,
        'independent_initializer_parity_precondition_verified':True,'no_optimizer_or_backward_calls_in_worker':True,
        'all_neural_or_gradient_calls':0,'VM_launched':False,'goal_complete':False,'seconds':time.monotonic()-start}
    write(PREP/'independent_packet_audit.json',outcome)
    print({'complete':True,'members':len(names),'source_bindings':len(p['local_source_bindings']),
        'boundary_rejections':rejected,'manual_scp_commands':5,'VM_launched':False},flush=True)


if __name__=='__main__':main()
