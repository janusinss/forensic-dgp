"""Independent sealed transfer and pre-neural Windows rejection checks."""
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
DEST=OUT/'cctv_dgp_bank_loss_alignment_v1_vm'
PARENT=OUT/'cctv_dgp_bank_photometric_guard_v1_return'
ANALYSIS=OUT/'cctv_dgp_bank_path_ablation_v1_analysis'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    receipt=OUT/'cctv_dgp_bank_loss_alignment_v1_transfer_audit.json';assert not receipt.exists()
    prep_path=OUT/'cctv_dgp_bank_loss_alignment_v1_preparation.json';prep=read(prep_path);assert prep['complete']
    p=read(DEST/'protocol.json');old=read(PARENT/'protocol.json');assert sha(DEST/'protocol.json')==prep['protocol_sha256']
    assert sha(PARENT/'protocol.json')==p['parent_protocol_sha256']
    assert p['case_ids']==old['preview_case_ids'] and len(p['case_ids'])==100
    selected=[i for i in range(0,3905,5) if old['cases'][i]['id'] in p['case_ids']];assert len(selected)==20
    assert len({old['cases'][i]['source_person_or_reference'] for i in selected})==20
    assert {old['cases'][i]['source'] for i in p['sample_gradient_begins']}=={'dataset/asian_faces','dataset/thumbnails128x128'}
    assert p['maximum_gradient_queries']==2*20*9==360 and p['model_forward_cap']==2*2*20==80 and p['recognizer_forward_cap']==40
    assert p['optimizer_updates']==p['backwards']==p['full_epochs']==p['native_pixels_used']==p['final_identity_pixels']==0
    assert p['preserved_scientific_thresholds']==old['scientific_thresholds']
    assert p['minimum_free_bytes']>=2*p['output_byte_cap']+p['disk_reserve_bytes']+1024**2
    assert p['uncompressed_sampled_gradient_upper_bound_bytes']<p['output_byte_cap']
    assert not p['failed_run_resume_allowed'] and not p['automatic_follow_on'] and not p['app_promotion']
    for root,files in [(DEST,p['assets_sha256']),(PARENT,p['parent_required_sha256'])]:
        for name,expected in files.items():
            path=root/name;assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root)
            assert sha(path)==expected,name
    for name,expected in p['evidence_sha256'].items():assert sha(ROOT/name)==expected and read(ROOT/name)['complete']
    math=read(OUT/'cctv_dgp_bank_loss_alignment_v1_math_audit.json')
    assert p['learning_names']==math['learning_names'] and math['local_gradient_queries']==0
    assert sha(DEST/'cctv_dgp_bank_loss_alignment_v1.py')==math['helper_sha256']
    worker=DEST/'scripts/cctv_dgp_bank_loss_alignment_v1_vm.py';source=worker.read_text(encoding='utf-8');assert 'file_digest' not in source
    tree=ast.parse(source);functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    run=functions['run'];assert run.body[0].value.func.id=='require_vm'
    assert not any(isinstance(n,(ast.Import,ast.ImportFrom)) and 'torch' in ast.unparse(n) for n in tree.body)
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['step','backward','Adam','AdamW','SGD'] for n in ast.walk(tree))
    assert not any(isinstance(n,(ast.Import,ast.ImportFrom)) and ('optim' in ast.unparse(n) or 'scripts.' in ast.unparse(n)) for n in ast.walk(tree))
    grads=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='grad']
    assert len(grads)==1 and all(any(n is g for n in ast.walk(run)) for g in grads)
    assert 'candidate.enable_vm_learning(parent)' in source and 'candidate.load_state_dict(state,strict=True)' in source
    assert 'state_hash(candidate)==before and buffer_hash(candidate)==buffers' in source
    helper=ast.parse((DEST/'cctv_dgp_bank_loss_alignment_v1.py').read_text(encoding='utf-8'))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['grad','step','backward','copy_','add_','Adam'] for n in ast.walk(helper))
    shell=DEST/'scripts/run_alignment.sh';text=shell.read_text(encoding='utf-8');assert b'\r' not in shell.read_bytes()
    assert '930 python' in text and '330 python' in text and '--kill-after=30' in text and 'Immutable used diagnostic' in text
    bash=Path('C:/Program Files/Git/bin/bash.exe');assert bash.is_file()
    syntax=subprocess.run([str(bash),'-n',str(shell)],capture_output=True,text=True,timeout=30)
    assert syntax.returncode==0,syntax.stderr
    archive=OUT/(prep['stem']+'-execution.tar.gz');assert sha(archive)==prep['archive_sha256']
    assert Path(str(archive)+'.sha256').read_bytes()==(prep['archive_sha256']+'  '+archive.name+'\n').encode('ascii')
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();expected={prep['name']+'/'+n:h for n,h in {**p['assets_sha256'],'protocol.json':prep['protocol_sha256']}.items()}
        assert len(members)==len({m.name for m in members})==len(expected)==4
        for member in members:
            assert member.isfile() and member.name in expected
            assert member.size==(DEST/Path(member.name).relative_to(prep['name'])).stat().st_size
            assert hashlib.sha256(tar.extractfile(member).read()).hexdigest()==expected[member.name]
    assert not (DEST/'outputs').exists()
    negative=subprocess.run([sys.executable,'-B',str(worker),'--root',str(DEST),'--parent-root',str(PARENT),
        '--protocol-sha',prep['protocol_sha256'],'--run'],capture_output=True,text=True,timeout=120)
    assert negative.returncode!=0 and "platform.system()=='Linux'" in negative.stderr
    assert not (DEST/'outputs').exists()
    with (OUT/'cctv_dgp_bank_loss_alignment_v1_windows_rejection.json').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(dict(returncode=negative.returncode,stdout=negative.stdout,stderr=negative.stderr,before_neural_and_gradient_imports=True),indent=2)+'\n')
    for name,expected in read(ANALYSIS/'local_protected_before.json')['files_sha256'].items():assert sha(ROOT/name)==expected
    record=dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),preparation_sha256=sha(prep_path),
        protocol_sha256=prep['protocol_sha256'],archive_sha256=sha(archive),archive_members_checked=4,packet_assets_checked=3,parent_assets_checked=5852,
        source_data_archive_and_LF_checksum_checked=True,Bash_readonly_syntax_pass=True,Python310_hashing_compatible=True,
        Windows_rejected_before_neural_gradients_or_outputs=True,scientific_gates_case_lists_and_parent_stop_unchanged=True,
        helper_arithmetic_audit_sha256=sha(OUT/'cctv_dgp_bank_loss_alignment_v1_math_audit.json'),
        local_neural_calls=0,local_gradient_queries=0,local_optimizer_updates=0,
        actual_manual_L4_gradients_timing_and_results_unrun=True,manual_VM_gradient_diagnostic_pending=True,
        model_qualified=False,app_promotion=False,goal_complete=False)
    with receipt.open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(dict(complete=True,archive_members=4,parent_assets=5852,local_gradient_queries=0,manual_VM_diagnostic_pending=True))


if __name__=='__main__':main()
