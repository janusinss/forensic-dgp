"""Correct A-only execution declarations; keep sealed r2 and B packets intact."""
import ast
import copy
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(v,indent=2,allow_nan=False)+'\n')
def replace(s,a,b):assert s.count(a)==1,a;return s.replace(a,b)

def main():
    prep=read(OUT/'cctv_dgp_bank_comparison_v1_r2_preparation.json');oldA,B=prep['routes']
    prior=OUT/oldA['name'];name='cctv_dgp_bank_comparison_v1_r3_A_vm';stem='cctv-dgp-bank-comparison-v1-r3-A'
    dest=OUT/name;archive=OUT/(stem+'-execution.tar.gz');assert not dest.exists() and not archive.exists();dest.mkdir()
    p=read(prior/'protocol.json');assert sha(prior/'protocol.json')==oldA['protocol_sha256']
    for n,h in p['assets_sha256'].items():
        assert sha(prior/n)==h
        d=dest/n;d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(prior/n,d)
    for n in ['cctv_dgp_bank_comparison_v1_contract.py','cctv_dgp_bank_comparison_v1_model.py']:
        d=dest/n;s=d.read_text();s=replace(s,repr(oldA['name']),repr(name))
        if n.endswith('contract.py'):
            s=replace(s,repr(oldA['stem']),repr(stem))
            s=replace(s,"assert p['format'] == 'own256-conditioned-frozen-generative-bank-comparison-v1-r2-split-route'",
                "assert p['format'] == 'own256-conditioned-frozen-generative-bank-comparison-v1-r3-A-execution-declaration'\n    assert p['preflight_generated_seed_test_uses_full512_prior_once'] is False")
        d.write_text(s,encoding='utf-8',newline='\n')
    worker=dest/'scripts/cctv_dgp_bank_comparison_v1_vm.py';s=worker.read_text()
    before='Fresh CUDA model output parity and saved pure-PyTorch CPU prior fixture agreement verified; upstream custom CUDA kernel equivalence is not asserted.'
    after='Selected A CUDA model output parity verified; no prior CUDA fixture is executed in A. Prior assets are retained for the separately audited B comparison. Upstream custom CUDA kernel equivalence is not asserted.'
    s=replace(s,before,after);worker.write_text(s,encoding='utf-8',newline='\n')
    evidence=dest/'provenance/retained_r2_A_protocol.json';shutil.copyfile(prior/'protocol.json',evidence)
    p=copy.deepcopy(p);p.update(format='own256-conditioned-frozen-generative-bank-comparison-v1-r3-A-execution-declaration',UTC=datetime.now(timezone.utc).isoformat(),preflight_generated_seed_test_uses_full512_prior_once=False,resource_metadata_correction='Selected A executes no prior CUDA fixture; retain the original two-route template and r2 packet as history.')
    p['assets_sha256']={n:sha(dest/n) for n in p['assets_sha256']};p['assets_sha256'][evidence.relative_to(dest).as_posix()]=sha(evidence)
    write(dest/'protocol.json',p);pin=sha(dest/'protocol.json')
    spec=importlib.util.spec_from_file_location('bank_A3_contract',dest/'cctv_dgp_bank_comparison_v1_contract.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.verify(dest,pin)
    for n in p['assets_sha256']:
        if n.endswith('.py'):ast.parse((dest/n).read_text(encoding='utf-8-sig'),feature_version=(3,10))
    with archive.open('xb') as f:
        with tarfile.open(fileobj=f,mode='w:gz',compresslevel=1) as tar:
            for d in sorted(dest.rglob('*')):
                if d.is_file():tar.add(d,arcname=name+'/'+d.relative_to(dest).as_posix(),recursive=False)
    digest=sha(archive);Path(str(archive)+'.sha256').write_text(digest+'  '+archive.name+'\n',encoding='ascii',newline='\n')
    A=dict(oldA,name=name,stem=stem,protocol_sha256=pin,archive_sha256=digest,archive_bytes=archive.stat().st_size,prepared_packet_assets=len(p['assets_sha256']),supersedes_unrun_r2_A_protocol_sha256=oldA['protocol_sha256'],selected_A_prior_CUDA_fixture_executed=False)
    write(OUT/'cctv_dgp_bank_comparison_v1_r3_preparation.json',dict(complete=True,routes=[A,B],resource_revision_only=True,execution_declaration_correction=True,original_r2_A_and_B_packets_preserved=True,original_failure_retained=True,local_neural_gradient_or_optimizer_calls=0,model_qualified=False,comparison_execution_pending=True,goal_complete=False))
    print(A,flush=True)

if __name__=='__main__':main()
