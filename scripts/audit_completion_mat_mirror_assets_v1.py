"""Independent original-byte validation and exact adapted-source readback, no model."""
import ast
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'outputs/completion_mat_mirror_assets_v1'
SOURCE=ROOT/'outputs/completion_mat_mirror_review_v1_r1'
PREFIX='cpu_implementation/source/libs/spandrel_extra_arches/spandrel_extra_arches/architectures/MAT/__arch/'


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    start=time.monotonic();p=json.loads((ASSETS/'preparation.json').read_text(encoding='utf-8'))
    assert p['complete'] and p['all_generator_and_helper_math_unchanged'] and not p['model_constructed']
    adapted=(ASSETS/'vendor/MAT.py').read_text(encoding='utf-8')
    assert adapted.replace('from .metadata_only import store_hyperparameters','from spandrel.util import store_hyperparameters')==(SOURCE/PREFIX/'MAT.py').read_text(encoding='utf-8')
    assert (ASSETS/'vendor/utils.py').read_bytes()==(SOURCE/PREFIX/'utils.py').read_bytes()
    for name,digest in p['vendor_sha256'].items():assert sha(ASSETS/name)==digest
    assert sha(ROOT/'mat_mirror_completion_v1.py')==p['adapter_sha256']
    weights=ASSETS/'MAT_FFHQ_512_fp16.safetensors'
    assert weights.stat().st_size==125280246 and sha(weights)==p['original_weight_sha256']=='eedb8504aef8a07feda7e89ef34e53344eaf3039cb1543615bf1092439ce3d98'
    with weights.open('rb') as stream:
        length=struct.unpack('<Q',stream.read(8))[0];assert length==54872
        header=json.loads(stream.read(length));assert len(header)==465
        cursor=0;values=0;layout=[]
        for name,entry in sorted(header.items(),key=lambda row:(row[1]['data_offsets'][0],row[1]['data_offsets'][1])):
            assert entry['dtype']=='F16' and set(entry)=={'dtype','shape','data_offsets'}
            begin,end=entry['data_offsets'];count=math.prod(entry['shape'])
            assert begin==cursor and end-begin==2*count
            stream.seek(8+length+begin);data=stream.read(end-begin)
            assert len(data)==end-begin and np.isfinite(np.frombuffer(data,dtype='<f2')).all()
            cursor=end;values+=count
            layout.append({'name':name,'shape':entry['shape'],'begin':begin,'end':end,'elements':count})
        assert cursor==weights.stat().st_size-8-length and values==62612683
    assert sorted(layout,key=lambda row:row['name'])==sorted(p['weight_layout'],key=lambda row:row['name'])
    result=subprocess.run([sys.executable,'-X','utf8','-B','-m','unittest','discover','-s','tests','-p','test_mat_mirror_completion_v1.py','-q'],
                           cwd=ROOT,capture_output=True,text=True,timeout=60)
    with (ASSETS/'adapter_regressions.txt').open('x',encoding='utf-8') as stream:stream.write(result.stdout+result.stderr)
    assert result.returncode==0 and 'Ran 11 tests' in result.stderr and 'OK' in result.stderr
    adapter=ast.parse((ROOT/'mat_mirror_completion_v1.py').read_text(encoding='utf-8'))
    assert not any(isinstance(n,ast.Call) and ast.unparse(n.func) in ['torch.load','pickle.load','eval','exec'] for n in ast.walk(adapter))
    receipt={'complete':True,'checker_sha256':sha(Path(__file__)),'original_weight_sha256':sha(weights),
             'source_routing_is_only_MAT_source_difference':True,'all_helpers_byte_identical':True,
             'original_tensor_count':465,'all62612683_weight_values_finite':True,
             'tensor_layout_independently_validated':True,'adapter_regressions_passed':11,
             'adapter_test_source_sha256':sha(ROOT/'tests/test_mat_mirror_completion_v1.py'),
             'adapter_test_output_sha256':sha(ASSETS/'adapter_regressions.txt'),
             'original_author_parity_verified':False,'model_constructed':False,
             'pretrained_model_forwards':0,'test_generator_is_mock':True,'gradient_calls':0,'optimizer_updates':0,
             'app_changes':False,'goal_complete':False,'seconds':time.monotonic()-start}
    with (ASSETS/'independent_assets_audit.json').open('x',encoding='utf-8',newline='\n') as stream:json.dump(receipt,stream,indent=2)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
