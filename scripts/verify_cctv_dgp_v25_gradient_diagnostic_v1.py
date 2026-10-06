"""Independently inspect the thin packet and Windows guards without model imports."""
import ast
import hashlib
import json
from pathlib import Path,PurePosixPath
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'outputs/cctv_dgp_v25_gradient_diagnostic_v1_vm'
OUT=ROOT/'outputs/cctv_dgp_v25_gradient_diagnostic_v1_preparation'
NAME='cctv-dgp-v25-gradient-diagnostic-v1'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def fn(t,n):return next(v for v in t.body if isinstance(v,ast.FunctionDef) and v.name==n)


def inspect_archive(archive):
    expected=['protocol.json','protocol.sha256','scripts/cctv_dgp_v25_gradient_diagnostic_v1_vm.py','scripts/run_gradient.sh']
    payload={}
    with tarfile.open(archive,'r:gz') as tar:
        for m in tar.getmembers():
            path=PurePosixPath(m.name)
            assert m.isreg() and not m.issym() and not m.islnk(),'Only regular diagnostic files'
            assert not path.is_absolute() and '..' not in path.parts and '\\' not in m.name and ':' not in m.name,'Unsafe diagnostic path'
            assert path.parts[0]==BUNDLE.name and path.as_posix()==m.name,'Diagnostic prefix/canonical path'
            relative='/'.join(path.parts[1:])
            assert relative in expected and relative not in payload,'Unexpected/duplicate diagnostic member'
            assert m.size<=128*1024,'Diagnostic member cap128KiB'
            payload[relative]=tar.extractfile(m).read()
    assert sorted(payload)==sorted(expected),'Missing diagnostic member'
    return payload


def source_contract(source):
    tree=ast.parse(source,feature_version=(3,10));run=fn(tree,'run')
    attributes={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not attributes.intersection({'backward','step'}),'No backward/optimizer operation'
    assert 'torch.save' not in {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)},'No new checkpoint'
    assert 'optim' not in {n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)},'No optimizer import/construction'
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='grad']
    assert len(calls)==1 and ast.unparse(calls[0].func)=='torch.autograd.grad','Only the declared finite gradient call'
    args={v.arg:ast.unparse(v.value) for v in calls[0].keywords}
    assert args=={'retain_graph':'i < 6','create_graph':'False','allow_unused':'False'}
    windows=next(n for n in run.body if isinstance(n,ast.Assert) and ast.unparse(n.test)=="sys.platform == 'linux'")
    first_try=next(n for n in run.body if isinstance(n,ast.Try))
    assert windows.lineno<first_try.lineno and 'outputs' not in source.splitlines()[windows.lineno-1],'Host guard before imports/gradients/output creation'
    assert source.index("assert sys.platform=='linux'")<source.index('out.mkdir()')
    assert 'range(0,50,5)' in source and 'for update in [0,50]' in source and 'scalar=terms[name].mean()/10' in source
    assert "assert progress=={'head_batches':20,'component_gradient_calls':140,'recognizer_forwards':120,'optimizer_updates':0}" in source
    assert source.count('parent_check(parent,p)')==3 and 'state_hash(head)==initial' in source
    assert 'state_hash(identity)==identity_before' in source and 'all(v.grad is None for v in parameters)' in source
    assert "p['head_states'][str(update)]" in source and 'clock()' in source
    assert 'time.monotonic()-start<420' in source and '20*1024**3' in source and 'shutil.disk_usage(root).free>=1024**3' in source
    assert "error<=2e-6" in source and "write(out/'failure.json'" in source
    assert "guard(root,idle=True)" in source and "namespace['SpatialFeatureHead']" in source
    assert "['mean','feature_errors','ssim']" in source and "['assemble_terms','objective_terms']" in source
    assert "n.name=='require_vm'" in source and "if a.export:export(root,p,a.protocol_sha);return" in source
    return tree


def expected_layout():
    convolutions=[('projections.'+str(i),16,c,1) for i,c in enumerate([64,128,128,128,128])]
    convolutions += [('decode3',24,32,3),('decode2',24,40,3),('decode1',24,40,3),('decode0',24,40,3),
        ('camera',16,13,3),('fuse',24,40,3),('tail',3,24,3),('direct',3,13,3)]
    rows=[];end=0
    for name,output,input_channels,size in convolutions:
        for suffix,shape in [('weight',[output,input_channels,size,size]),('bias',[output])]:
            count=1
            for dimension in shape:count*=dimension
            rows.append({'name':name+'.'+suffix,'shape':shape,'start':end,'end':end+count});end+=count
    assert end==53781 and len(rows)==26
    return rows


def gradient_arithmetic(array,layout):
    """Prospective returned-matrix checks with no gradient or model operation."""
    import numpy as np
    assert array.shape==(7,53781) and array.dtype==np.float64 and np.isfinite(array).all(),'Finite7x53781 float64 gradients'
    assert layout==expected_layout(),'Exact26 parameter names/shapes/offsets'
    end=0
    for row in layout:
        assert row['start']==end and row['end']-row['start']==int(np.prod(row['shape'])),'Contiguous parameter layout'
        end=row['end']
    norms=np.linalg.norm(array,axis=1);gram=array@array.T;den=norms[:,None]*norms[None,:]
    cosine=np.divide(gram,den,out=np.zeros_like(gram),where=den>0)
    assert np.all(np.abs(cosine)<=1+1e-12),'Gradient cosine range'
    return {'norms':norms.tolist(),'gram':gram.tolist(),'cosines':cosine.tolist(),'total_norm':float(np.linalg.norm(array.sum(0)))}


def main():
    start=time.monotonic();prep=read(OUT/'preparation.json');archive=ROOT/'outputs'/(NAME+'-execution.tar.gz')
    assert sha(archive)==prep['archive_sha256'] and archive.stat().st_size==prep['archive_bytes']
    assert Path(str(archive)+'.sha256').read_text().strip()==sha(archive)+'  '+archive.name
    payload=inspect_archive(archive)
    for name,data in payload.items():assert data==(BUNDLE/name).read_bytes(),name
    p=read(BUNDLE/'protocol.json');pin=sha(BUNDLE/'protocol.json');assert pin==prep['protocol_sha256']
    assert (BUNDLE/'protocol.sha256').read_text().strip()==pin+'  protocol.json'
    assert p['optimizer_updates']==0 and p['snapshots']==[0,50] and not p['new_training_recipe']
    assert p['head_parameters']==53781 and p['head_batches']==20 and p['component_gradient_calls']==140 and p['recognizer_forwards']==120
    assert p['timing_limits']=={'worker_seconds':420,'external_seconds':480,'external_kill_grace_seconds':30,
        'export_seconds':30,'external_export_seconds':60,'export_kill_grace_seconds':10,'free_disk_bytes':1024**3,'allocated_VRAM_bytes':20*1024**3}
    for name,digest in p['assets_sha256'].items():assert sha(BUNDLE/name)==digest,name
    for name,digest in p['local_basis_sha256'].items():assert sha(ROOT/name)==digest,name
    parent=ROOT/'outputs/cctv_dgp_spatial_features_vm_v25';returned=ROOT/'outputs/cctv_dgp_spatial_features_v25_return'
    assert sha(parent/'protocol.json')==p['closed_V25_protocol_sha256']==sha(returned/'protocol.json')
    original=read(parent/'protocol.json')
    for name,digest in original['assets_sha256'].items():assert sha(parent/name)==digest,name
    for name,digest in p['closed_V25_inputs_sha256'].items():assert sha(returned/name)==digest,name
    cache=read(returned/'outputs/frozen_DGP_features.json')
    for name,digest in cache['files_sha256'].items():assert sha(returned/'outputs/frozen_DGP_features'/name)==digest,name
    assert len(cache['files_sha256'])==250
    source=(BUNDLE/'scripts/cctv_dgp_v25_gradient_diagnostic_v1_vm.py').read_text()
    source_contract(source)
    shell=(BUNDLE/'scripts/run_gradient.sh').read_text()
    assert '480s' in shell and '--kill-after=30s' in shell and '60s' in shell and '--kill-after=10s' in shell
    assert 'PIPESTATUS[0]' in shell and 'test ! -e "$diagnostic_root/outputs"' in shell and 'test ! -e "$diagnostic_root/trainer.log"' in shell
    assert '--record-supervision' in shell and '--export' in shell and '--preflight' not in shell
    transfer=subprocess.run([sys.executable,'-B',str(BUNDLE/'scripts/cctv_dgp_v25_gradient_diagnostic_v1_vm.py'),
        '--root',str(BUNDLE),'--protocol-sha',pin,'--verify-transfer'],capture_output=True,text=True,timeout=20)
    assert transfer.returncode==0,transfer.stderr
    guard=subprocess.run([sys.executable,'-B',str(BUNDLE/'scripts/cctv_dgp_v25_gradient_diagnostic_v1_vm.py'),
        '--root',str(BUNDLE),'--protocol-sha',pin,'--run'],capture_output=True,text=True,timeout=20)
    assert sys.platform=='win32' and guard.returncode!=0 and 'Existing Linux VM only' in guard.stderr
    assert not (BUNDLE/'outputs').exists(),'Windows rejected before output creation'
    result={'complete':True,'date':'2026-10-06','checker_sha256':sha(Path(__file__)),
        'protocol_sha256':pin,'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'regular_members':4,
        'original_V25_assets_verified':len(original['assets_sha256']),'closed_saved_inputs_verified':len(p['closed_V25_inputs_sha256']),
        'frozen_feature_arrays_verified':250,'Windows_transfer_guard_passed':True,'Windows_neural_gradient_guard_passed':True,
        'exact_original_source_selected':True,'frozen_quality_gates_retained':True,'new_training_recipe':False,
        'neural_calls':0,'gradient_calls':0,'optimizer_updates':0,'VM_actions':False,'app_promotion':False,
        'actual_L4_diagnostic_pending':True,'goal_complete':False,'seconds':time.monotonic()-start}
    with (OUT/'independent_packet_audit.json').open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
