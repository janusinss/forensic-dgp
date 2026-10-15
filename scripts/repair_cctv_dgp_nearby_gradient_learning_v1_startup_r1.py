"""Preserve sealed V1 and build an explicit response-root correction in R1."""
import ast
import copy
from datetime import datetime,timezone
import shutil
import subprocess
import sys
import tarfile

from import_cctv_dgp_nearby_gradient_learning_v1_return import ROOT,OUT,NAME,STEM,PIN,read,sha,write
from prepare_cctv_dgp_nearby_gradient_learning_v1 import text_file

RNAME='cctv_dgp_nearby_gradient_learning_v1_r1_vm'
RSTEM='cctv-dgp-nearby-gradient-learning-v1-r1'


def main():
    packet=OUT/NAME;p=read(packet/'protocol.json');prep=read(OUT/'cctv_dgp_nearby_gradient_learning_v1_preparation.json')
    original={**p['assets_sha256'],'protocol.json':PIN}
    assert sha(packet/'protocol.json')==PIN and all(sha(packet/n)==h for n,h in original.items())
    assert sha(OUT/(STEM+'-execution.tar.gz'))==prep['archive_sha256']
    worker=ROOT/'scripts/cctv_dgp_nearby_gradient_learning_v1_vm.py';text=worker.read_text(encoding='utf-8')
    assert sha(worker)==sha(packet/'scripts/cctv_dgp_nearby_gradient_learning_v1_vm.py')
    text=text.replace(NAME,RNAME).replace(STEM,RSTEM)
    changes=[('def verify(root,parent,pin):','def verify(root,parent,response_root,pin):'),
        ('response=parent.parent/RESPONSE_NAME\n    assert sha(response/', 'assert sha(response_root/'),
        ("(response,p['response_required_sha256'])", "(response_root,p['response_required_sha256'])"),
        ('def require_vm(root,parent,check_idle=True):','def require_vm(root,parent,response_root,check_idle=True):'),
        ("assert Path.cwd().resolve()==root and os.environ.get('TMUX')", "assert response_root==(Path.home()/'forensic-dgp'/RESPONSE_NAME).resolve()\n    assert Path.cwd().resolve()==root and os.environ.get('TMUX')"),
        ('def run(root,parent,p,old,pin):','def run(root,parent,response_root,p,old,pin):'),
        ('parent.parent/RESPONSE_NAME/STAGE.replace(\'nearby_gradient_learning_v1\',\'conditioning_response_v1\')', "response_root/'outputs/conditioning_response_v1'"),
        ("response=parent.parent/RESPONSE_NAME/'outputs/conditioning_response_v1'", "response=response_root/'outputs/conditioning_response_v1'"),
        ('def export(root,parent,p,pin):','def export(root,parent,response_root,p,pin):'),
        ('require_vm(root,parent,check_idle=False)','require_vm(root,parent,response_root,check_idle=False)'),
        ("parser.add_argument('--parent-root',required=True,type=Path);parser.add_argument('--protocol-sha',required=True)", "parser.add_argument('--parent-root',required=True,type=Path);parser.add_argument('--response-root',required=True,type=Path)\n    parser.add_argument('--protocol-sha',required=True)"),
        ('root,parent=a.root.resolve(),a.parent_root.resolve()', 'root,parent,response_root=a.root.resolve(),a.parent_root.resolve(),a.response_root.resolve()'),
        ('verify(root,parent,a.protocol_sha)','verify(root,parent,response_root,a.protocol_sha)'),
        ('run(root,parent,p,old,a.protocol_sha)','run(root,parent,response_root,p,old,a.protocol_sha)'),
        ('export(root,parent,p,a.protocol_sha)','export(root,parent,response_root,p,a.protocol_sha)')]
    for before,after in changes:
        assert text.count(before)==1,before;text=text.replace(before,after)
    assert text.count('require_vm(root,parent)')==2 and text.count('verify(root,parent,pin)')==1
    text=text.replace('require_vm(root,parent)','require_vm(root,parent,response_root)').replace('verify(root,parent,pin)','verify(root,parent,response_root,pin)')
    ast.parse(text,feature_version=(3,10))
    rworker=ROOT/'scripts/cctv_dgp_nearby_gradient_learning_v1_r1_vm.py';text_file(rworker,text)
    importer=ROOT/'scripts/import_cctv_dgp_nearby_gradient_learning_v1_return.py'
    rimporter=ROOT/'scripts/import_cctv_dgp_nearby_gradient_learning_v1_r1_return.py'
    itext=importer.read_text().replace('cctv_dgp_nearby_gradient_learning_v1','cctv_dgp_nearby_gradient_learning_v1_r1').replace(STEM,RSTEM).replace(PIN,'__PIN__')
    text_file(rimporter,itext)
    auditor=ROOT/'scripts/audit_cctv_dgp_nearby_gradient_learning_v1_return.py'
    rauditor=ROOT/'scripts/audit_cctv_dgp_nearby_gradient_learning_v1_r1_return.py'
    atext=auditor.read_text().replace('from import_cctv_dgp_nearby_gradient_learning_v1_return import','from import_cctv_dgp_nearby_gradient_learning_v1_r1_return import')
    text_file(rauditor,atext)
    guard=ROOT/'scripts/check_cctv_dgp_nearby_gradient_learning_v1_guard.py'
    rguard=ROOT/'scripts/check_cctv_dgp_nearby_gradient_learning_v1_r1_guard.py'
    gtext=guard.read_text().replace('cctv_dgp_nearby_gradient_learning_v1_vm.py','cctv_dgp_nearby_gradient_learning_v1_r1_vm.py')
    assert gtext.count("'--parent-root', '.',")==1
    text_file(rguard,gtext.replace("'--parent-root', '.',","'--parent-root', '.', '--response-root', '.',"))
    check=subprocess.run([sys.executable,'-B',str(rguard)],capture_output=True,text=True,timeout=30)
    assert check.returncode==0,check.stdout+check.stderr
    rpacket=OUT/RNAME;assert not rpacket.exists();rpacket.mkdir()
    for name in p['assets_sha256']:
        renamed=name.replace('scripts/cctv_dgp_nearby_gradient_learning_v1_vm.py','scripts/cctv_dgp_nearby_gradient_learning_v1_r1_vm.py')
        target=rpacket/renamed;target.parent.mkdir(parents=True,exist_ok=True)
        if name=='scripts/cctv_dgp_nearby_gradient_learning_v1_vm.py':shutil.copyfile(rworker,target)
        elif name=='conditioning_model.py':text_file(target,(packet/name).read_text().replace(NAME,RNAME))
        elif name=='scripts/run_learning.sh':
            shell=(packet/name).read_text().replace(NAME,RNAME)
            marker='--parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha'
            assert shell.count(marker)==2
            text_file(target,shell.replace(marker,'--parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --response-root ../cctv_dgp_conditioning_response_v1_vm --protocol-sha'))
        else:shutil.copyfile(packet/name,target)
    review=dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),failed_local_audit_tool_chunk_id='e46c21',
        location='V1 hash-only transfer preflight',cause='Derived response VM name selects unrun packet when local downloaded return has a different directory name',
        fix='Explicit required response-root argument; Linux guard binds its exact original VM path',
        original_protocol_sha256=PIN,original_archive_sha256=prep['archive_sha256'],original_packet_preserved=True,
        original_worker_sha256=sha(worker),corrected_worker_sha256=sha(rworker),
        scientific_or_learning_recipe_changed=False,manual_guard_output=check.stdout,
        original_return_auditor_sha256=sha(auditor),corrected_return_auditor_sha256=sha(rauditor),
        auditor_only_import_namespace_changed=True,
        original_saved_array_fixture_sha256=sha(OUT/'cctv_dgp_nearby_gradient_learning_v1_saved_fixture_r1.json'),
        unchanged_saved_array_fixture_reused=True,local_neural_or_training_calls=0,VM_launch=False,manual_UNRUN=True)
    assert rauditor.read_text().replace('from import_cctv_dgp_nearby_gradient_learning_v1_r1_return import','from import_cctv_dgp_nearby_gradient_learning_v1_return import')==auditor.read_text()
    write(rpacket/'evidence/startup_root_review.json',review)
    rp=copy.deepcopy(p);rp['name']=RNAME;rp['UTC']=datetime.now(timezone.utc).isoformat()
    rp['startup_root_repair_of_protocol_sha256']=PIN;rp['response_root_required_and_VM_path_exact']=True
    rp['startup_root_only_changed_no_scientific_or_recipe_change']=True
    rp['source_origins'].pop('scripts/cctv_dgp_nearby_gradient_learning_v1_vm.py')
    rp['source_origins']['scripts/cctv_dgp_nearby_gradient_learning_v1_r1_vm.py']=dict(source_path=rworker.relative_to(ROOT).as_posix(),source_sha256=sha(rworker),exact_copy=True)
    rp['assets_sha256']={f.relative_to(rpacket).as_posix():sha(f) for f in rpacket.rglob('*') if f.is_file()}
    write(rpacket/'protocol.json',rp);rpin=sha(rpacket/'protocol.json')
    text_file(OUT/'cctv_dgp_nearby_gradient_learning_v1_r1_importer_template.py',itext)
    rimporter.write_text(itext.replace("PIN='__PIN__'","PIN='"+rpin+"'"),encoding='utf-8',newline='\n')
    archive=OUT/(RSTEM+'-execution.tar.gz')
    with tarfile.open(archive,'x:gz',compresslevel=6) as tar:
        for f in sorted(rpacket.rglob('*')):
            if f.is_file():tar.add(f,arcname=RNAME+'/'+f.relative_to(rpacket).as_posix(),recursive=False)
    digest=sha(archive);text_file(OUT/(archive.name+'.sha256'),digest+'  '+archive.name+'\n','ascii')
    write(OUT/'cctv_dgp_nearby_gradient_learning_v1_withdrawn.json',dict(complete=True,status='WITHDRAWN before transfer; do not upload or run',
        protocol_sha256=PIN,archive_sha256=prep['archive_sha256'],original_packet_preserved=True,replacement=RSTEM,manual_training_UNRUN=True))
    write(OUT/'cctv_dgp_nearby_gradient_learning_v1_r1_preparation.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
        preparer_sha256=sha(__file__),protocol_sha256=rpin,archive_sha256=digest,archive_bytes=archive.stat().st_size,packet_bindings=len(rp['assets_sha256'])+1,
        dependency_bindings=prep['dependency_bindings'],original_source_bindings=prep['original_source_bindings'],immutable_importer_sha256=sha(rimporter),
        preparation_evidence_sha256=prep['preparation_evidence_sha256'],startup_root_review_sha256=sha(rpacket/'evidence/startup_root_review.json'),
        reused_saved_fixture_and_namespace_only_auditor_proof=review,local_neural_or_training_calls=0,VM_launch=False,manual_UNRUN=True,
        model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,protocol_sha256=rpin,archive_sha256=digest,packet_bindings=len(rp['assets_sha256'])+1,manual_UNRUN=True))


if __name__=='__main__':main()
