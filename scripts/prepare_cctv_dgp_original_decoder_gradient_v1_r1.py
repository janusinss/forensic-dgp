"""Retain the unissued draft and freeze the original CPU-built fixed filter."""
import ast
import hashlib
import json
from pathlib import Path
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'outputs/cctv_dgp_original_decoder_gradient_v1_vm'
BUNDLE = ROOT / 'outputs/cctv_dgp_original_decoder_gradient_v1_r1_vm'
OUT = ROOT / 'outputs/cctv_dgp_original_decoder_gradient_v1_r1_preparation'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2),b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    start=time.monotonic()
    assert not BUNDLE.exists() and not OUT.exists()
    p=read(OLD/'protocol.json')
    old=read(ROOT/'outputs/cctv_dgp_original_decoder_gradient_v1_preparation/preparation.json')
    assert sha(OLD/'protocol.json')==old['protocol_sha256']
    old_bindings={(OLD/'protocol.json').relative_to(ROOT).as_posix():sha(OLD/'protocol.json')}
    for name,digest in p['assets_sha256'].items():
        assert sha(OLD/name)==digest
        old_bindings[(OLD/name).relative_to(ROOT).as_posix()]=digest
    for name in ['outputs/cctv_dgp_original_decoder_gradient_v1_preparation/preparation.json',
                 'outputs/cctv-dgp-original-decoder-gradient-v1-execution.tar.gz',
                 'outputs/cctv-dgp-original-decoder-gradient-v1-execution.tar.gz.sha256']:
        old_bindings[name]=sha(ROOT/name)
    BUNDLE.mkdir();(BUNDLE/'scripts').mkdir()
    candidate=(OLD/'cctv_dgp_original_decoder_candidate_v1.py').read_bytes()
    with (BUNDLE/'cctv_dgp_original_decoder_candidate_v1.py').open('xb') as f:f.write(candidate)
    worker=(OLD/'scripts/cctv_dgp_original_decoder_gradient_v1_vm.py').read_text(encoding='utf-8')
    worker=worker.replace('own-DGP-original-decoder-zero-update-gradient-proof-v1','own-DGP-original-decoder-zero-update-gradient-proof-v1-r1')
    worker=worker.replace('cctv_dgp_original_decoder_gradient_v1_vm','cctv_dgp_original_decoder_gradient_v1_r1_vm')
    worker=worker.replace('cctv-dgp-original-decoder-gradient-v1-results','cctv-dgp-original-decoder-gradient-v1-r1-results')
    worker=worker.replace('cctv_dgp_original_decoder_gradient_v1_return','cctv_dgp_original_decoder_gradient_v1_r1_return')
    worker=worker.replace('cctv-dgp-original-decoder-gradient-v1-export','cctv-dgp-original-decoder-gradient-v1-r1-export')
    before="z = torch.arange(-6,7,dtype=torch.float32,device='cuda')\n        fixed.kernel = torch.exp(-.5*(z/2).square()); fixed.kernel /= fixed.kernel.sum()"
    after="z = torch.arange(-6,7,dtype=torch.float32)\n        kernel = torch.exp(-.5*(z/2).square()); kernel = kernel / kernel.sum()\n        fixed.kernel = kernel.cuda()"
    assert worker.count(before)==1
    worker=worker.replace(before,after)
    ast.parse(worker,feature_version=(3,10))
    with (BUNDLE/'scripts/cctv_dgp_original_decoder_gradient_v1_r1_vm.py').open('x',encoding='utf-8',newline='\n') as f:f.write(worker)
    shell=(OLD/'scripts/run_decoder_gradient.sh').read_bytes()
    shell=shell.replace(b'cctv_dgp_original_decoder_gradient_v1_vm.py',b'cctv_dgp_original_decoder_gradient_v1_r1_vm.py')
    with (BUNDLE/'scripts/run_decoder_gradient.sh').open('xb') as f:f.write(shell)
    p['format']='own-DGP-original-decoder-zero-update-gradient-proof-v1-r1'
    p['fixed_filter_kernel']='Exact original CPU arange/exp/sum/division initializer, then CUDA copy. No GPU-side kernel recomputation; original blur/high functions unchanged.'
    p['unissued_draft_protocol_sha256']=old['protocol_sha256']
    p['unissued_draft_change']='Source review caught CUDA construction of a formerly CPU-built fixed Gaussian kernel. R1 retains the original initializer before transfer; all measured gradients must use it. No draft was launched and no returned result or failed gate is altered.'
    p['local_basis_sha256'].update({**old_bindings,'scripts/prepare_cctv_dgp_original_decoder_gradient_v1_r1.py':sha(Path(__file__))})
    p['assets_sha256']={name:sha(BUNDLE/name) for name in ['cctv_dgp_original_decoder_candidate_v1.py',
                         'scripts/cctv_dgp_original_decoder_gradient_v1_r1_vm.py','scripts/run_decoder_gradient.sh']}
    write(BUNDLE/'protocol.json',p)
    pin=sha(BUNDLE/'protocol.json')
    archive=ROOT/'outputs/cctv-dgp-original-decoder-gradient-v1-r1-execution.tar.gz'
    assert not archive.exists()
    files=[path for path in BUNDLE.rglob('*') if path.is_file()]
    with tarfile.open(archive,'x:gz',compresslevel=6) as tar:
        for path in sorted(files):tar.add(path,arcname=BUNDLE.name+'/'+path.relative_to(BUNDLE).as_posix(),recursive=False)
    digest=sha(archive)
    with Path(str(archive)+'.sha256').open('x',encoding='ascii',newline='\n') as f:f.write(digest+'  '+archive.name+'\n')
    OUT.mkdir()
    record={'complete':True,'protocol_sha256':pin,'archive_sha256':digest,'archive_bytes':archive.stat().st_size,
        'packet_regular_files':4,'uploaded_data_or_weights_bytes':0,'unissued_draft_bindings_preserved':old_bindings,
        'source_change':'Restore original CPU kernel initializer before CUDA copy; names only otherwise',
        'local_neural_or_gradient_calls':0,'optimizer_updates':0,'new_training_recipe_created':False,
        'VM_gradient_proof_pending':True,'VM_actions':False,'app_promotion':False,'goal_complete':False,
        'seconds':time.monotonic()-start}
    write(OUT/'preparation.json',record)
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
