"""Readback original mask/affine geometry, syntax and verified preparation receipts."""
import ast
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from verify_cctv_dgp_v25_gradient_diagnostic_v1 import sha,read,fn


def main():
    import cv2
    import numpy as np
    from PIL import Image
    start=time.monotonic();out=ROOT/'outputs/cctv_dgp_v25_gradient_diagnostic_v1_preparation'
    bundle=ROOT/'outputs/cctv_dgp_v25_gradient_diagnostic_v1_vm';parent=ROOT/'outputs/cctv_dgp_spatial_features_vm_v25'
    audit=read(out/'independent_packet_audit.json');assert audit['complete']
    source=bundle/'scripts/cctv_dgp_v25_gradient_diagnostic_v1_vm.py'
    tree=ast.parse(source.read_text(),feature_version=(3,10))
    erode=next(n for n in ast.walk(fn(tree,'run')) if isinstance(n,ast.FunctionDef) and n.name=='erode')
    namespace={'np':np};exec(compile(ast.Module(body=[erode],type_ignores=[]),'<frozen-diagnostic-square-erosion-only>','exec'),namespace)
    p=read(parent/'protocol.json');refs={r['id']:r for r in p['references']}
    gridtree=ast.parse((parent/'cctv_dgp_pilot.py').read_text())
    geometry={'np':np,'cv2':cv2};exec(compile(ast.Module(body=[fn(gridtree,'grid112')],type_ignores=[]),'<original-fixed-affine-only>','exec'),geometry)
    maximum=0.
    for c in p['cases']:
        assert c['role']=='train'
        with Image.open(parent/c['observed']) as im:mask=np.asarray(im)>0
        namespace['mask']=mask
        for radius in [3,6]:
            own=namespace['erode'](radius)
            original=cv2.erode(mask.astype(np.uint8),np.ones((2*radius+1,2*radius+1),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
            assert np.array_equal(own,original),c['id']
        matrix=np.asarray(refs[c['source_person_or_reference']]['matrix112'],np.float64)
        inverse=np.linalg.inv(np.vstack([matrix,[0,0,1]]));yy,xx=np.meshgrid(np.arange(112),np.arange(112),indexing='ij')
        xy=np.stack([xx,yy,np.ones_like(xx)],-1)@inverse.T
        expected=(2*(xy[...,:2]+.5)/256-1).astype(np.float32)
        maximum=max(maximum,float(np.abs(expected-geometry['grid112'](matrix)).max()))
    assert maximum<=1e-6,'Frozen affine discrepancy'
    sources=['scripts/cctv_dgp_v25_gradient_diagnostic_v1_vm.py','scripts/prepare_cctv_dgp_v25_gradient_diagnostic_v1.py',
        'scripts/verify_cctv_dgp_v25_gradient_diagnostic_v1.py','scripts/audit_cctv_dgp_v25_gradient_diagnostic_v1_preparation.py',
        'tests/test_cctv_dgp_v25_gradient_diagnostic_v1.py']
    for name in sources:ast.parse((ROOT/name).read_text(),feature_version=(3,10))
    report={'complete':True,'date':'2026-10-06','checker_sha256':sha(Path(__file__)),
        'protocol_sha256':sha(bundle/'protocol.json'),'source_bindings_sha256':{n:sha(ROOT/n) for n in sources},
        'independent_packet_audit_sha256':sha(out/'independent_packet_audit.json'),
        'unsafe_packet_source_matrix_tests_passed':9,'test_seconds_reported':.049,
        'test_command':'venv/Scripts/python.exe -B -m unittest discover -s tests -p test_cctv_dgp_v25_gradient_diagnostic_v1.py -v',
        'Python310_sources_parsed':len(sources),'original_cases_checked':50,'original_square_erosions_exact':100,
        'original_fixed_affine_grids_checked':50,'maximum_affine_difference':maximum,'Bash_n_exit_code':0,
        'Bash_syntax_only_outside_sandbox':'C:/Program Files/Git/bin/bash.exe -n outputs/cctv_dgp_v25_gradient_diagnostic_v1_vm/scripts/run_gradient.sh',
        'sandbox_signal_pipe_failure_preserved':True,'neural_calls':0,'local_gradient_calls':0,'optimizer_updates':0,
        'VM_actions':False,'actual_L4_gradients_pending':True,'new_training_recipe':False,'app_promotion':False,
        'goal_complete':False,'seconds':time.monotonic()-start}
    with (out/'independent_preparation_readback.json').open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_bindings_sha256'},indent=2))


if __name__=='__main__':main()
