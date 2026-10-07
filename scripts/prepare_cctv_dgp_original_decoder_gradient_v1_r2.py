"""Release only the bounded-token shell correction; retain both unissued drafts."""
import ast
import hashlib
import json
from pathlib import Path
import re
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'outputs/cctv_dgp_original_decoder_gradient_v1_r1_vm'
BUNDLE=ROOT/'outputs/cctv_dgp_original_decoder_gradient_v1_r2_vm'
OUT=ROOT/'outputs/cctv_dgp_original_decoder_gradient_v1_r2_preparation'


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    start=time.monotonic();assert not BUNDLE.exists() and not OUT.exists()
    p=read(OLD/'protocol.json');prep=read(ROOT/'outputs/cctv_dgp_original_decoder_gradient_v1_r1_preparation/preparation.json')
    assert sha(OLD/'protocol.json')==prep['protocol_sha256']
    old_bindings={(OLD/'protocol.json').relative_to(ROOT).as_posix():sha(OLD/'protocol.json')}
    for name,digest in p['assets_sha256'].items():
        assert sha(OLD/name)==digest
        old_bindings[(OLD/name).relative_to(ROOT).as_posix()]=digest
    for name in ['outputs/cctv_dgp_original_decoder_gradient_v1_r1_preparation/preparation.json',
                 'outputs/cctv-dgp-original-decoder-gradient-v1-r1-execution.tar.gz',
                 'outputs/cctv-dgp-original-decoder-gradient-v1-r1-execution.tar.gz.sha256',
                 'scripts/verify_cctv_dgp_original_decoder_gradient_v1_r1.py']:
        old_bindings[name]=sha(ROOT/name)
    BUNDLE.mkdir();(BUNDLE/'scripts').mkdir()
    with (BUNDLE/'cctv_dgp_original_decoder_candidate_v1.py').open('xb') as f:f.write((OLD/'cctv_dgp_original_decoder_candidate_v1.py').read_bytes())
    worker=(OLD/'scripts/cctv_dgp_original_decoder_gradient_v1_r1_vm.py').read_text(encoding='utf-8')
    worker=worker.replace('gradient-proof-v1-r1','gradient-proof-v1-r2').replace('gradient_v1_r1','gradient_v1_r2').replace('gradient-v1-r1','gradient-v1-r2')
    ast.parse(worker,feature_version=(3,10))
    with (BUNDLE/'scripts/cctv_dgp_original_decoder_gradient_v1_r2_vm.py').open('x',encoding='utf-8',newline='\n') as f:f.write(worker)
    shell=(ROOT/'outputs/cctv_dgp_v26_gradient_diagnostic_v1_vm/scripts/run_gradient.sh').read_bytes()
    assert shell.count(b' 480s ')==1 and shell.count(b' 60s ')==1
    shell=shell.replace(b'cctv_dgp_v26_gradient_diagnostic_v1_vm.py',b'cctv_dgp_original_decoder_gradient_v1_r2_vm.py')
    shell=shell.replace(b' 480s ',b' 660s ').replace(b' 60s ',b' 120s ')
    assert re.findall(rb'timeout --signal=TERM --kill-after=\d+s (\d+)s ',shell)==[b'660',b'120']
    with (BUNDLE/'scripts/run_decoder_gradient.sh').open('xb') as f:f.write(shell)
    p['format']='own-DGP-original-decoder-zero-update-gradient-proof-v1-r2'
    p['unissued_R1_protocol_sha256']=prep['protocol_sha256']
    p['unissued_R1_source_check_failure']='Independent shell equality check found6120s after a substring replacement also matched within660s. R2 rebuilds from the original shell with whole-token480s/60s replacements. No draft or historical recipe ran; original600s worker and declared660s external budget retained.'
    p['local_basis_sha256'].update({**old_bindings,'scripts/prepare_cctv_dgp_original_decoder_gradient_v1_r2.py':sha(Path(__file__))})
    p['assets_sha256']={name:sha(BUNDLE/name) for name in ['cctv_dgp_original_decoder_candidate_v1.py',
                       'scripts/cctv_dgp_original_decoder_gradient_v1_r2_vm.py','scripts/run_decoder_gradient.sh']}
    write(BUNDLE/'protocol.json',p)
    pin=sha(BUNDLE/'protocol.json')
    archive=ROOT/'outputs/cctv-dgp-original-decoder-gradient-v1-r2-execution.tar.gz';assert not archive.exists()
    with tarfile.open(archive,'x:gz',compresslevel=6) as tar:
        for path in sorted(BUNDLE.rglob('*')):
            if path.is_file():tar.add(path,arcname=BUNDLE.name+'/'+path.relative_to(BUNDLE).as_posix(),recursive=False)
    digest=sha(archive)
    with Path(str(archive)+'.sha256').open('x',encoding='ascii',newline='\n') as f:f.write(digest+'  '+archive.name+'\n')
    OUT.mkdir()
    record={'complete':True,'protocol_sha256':pin,'archive_sha256':digest,'archive_bytes':archive.stat().st_size,
            'packet_regular_files':4,'uploaded_data_or_weights_bytes':0,'unissued_R1_bindings_preserved':old_bindings,
            'source_change':'Whole-token shell deadline correction and release names; unchanged R1 model/gradient/loss worker',
            'local_neural_or_gradient_calls':0,'optimizer_updates':0,'new_training_recipe_created':False,
            'VM_gradient_proof_pending':True,'VM_actions':False,'app_promotion':False,'goal_complete':False,
            'seconds':time.monotonic()-start}
    write(OUT/'preparation.json',record)
    # Prepare a separate prospective importer/auditor, never execute it on a fabricated return.
    old_audit=ROOT/'scripts/audit_cctv_dgp_original_decoder_gradient_v1_r1_return.py'
    source=old_audit.read_text(encoding='utf-8').replace('gradient_v1_r1','gradient_v1_r2').replace('gradient-v1-r1','gradient-v1-r2')
    source=source.replace(prep['protocol_sha256'],pin)
    target=ROOT/'scripts/audit_cctv_dgp_original_decoder_gradient_v1_r2_return.py'
    ast.parse(source,feature_version=(3,10))
    with target.open('x',encoding='utf-8',newline='\n') as f:f.write(source)
    tests=(ROOT/'tests/test_cctv_dgp_original_decoder_gradient_v1_r1_audit.py').read_text(encoding='utf-8').replace('gradient_v1_r1','gradient_v1_r2')
    with (ROOT/'tests/test_cctv_dgp_original_decoder_gradient_v1_r2_audit.py').open('x',encoding='utf-8',newline='\n') as f:f.write(tests)
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
