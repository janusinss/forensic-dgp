"""Real worker CLI on Windows must reject before data, Torch or networking."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--receipt',required=True,type=Path);a=parser.parse_args()
    root=Path(__file__).resolve().parents[1];worker=root/'scripts/cctv_dgp_conditioning_response_v1_vm.py'
    record=dict(complete=False,local_neural_or_training_calls=0)
    try:
        assert worker.is_file(),'Manual response worker not implemented'
        tree=ast.parse(worker.read_text(encoding='utf-8'),feature_version=(3,10))
        forbidden=[]
        for n in ast.walk(tree):
            if isinstance(n,ast.Call):
                name=n.func.id if isinstance(n.func,ast.Name) else n.func.attr if isinstance(n.func,ast.Attribute) else ''
                if name in ['solve_direction','minimize','nnls','Adam','AdamW','SGD','step']:forbidden.append(name)
        assert not forbidden,forbidden
        code='''import runpy,sys
sys.argv=[sys.argv[1],'--root','unprepared','--parent-root','missing-parent','--protocol-sha','0'*64,'--diagnose']
try: runpy.run_path(sys.argv[0],run_name='__main__')
except AssertionError as error:
    assert str(error)=='Manual Linux L4 diagnostic only',str(error)
    assert 'torch' not in sys.modules
    print('Manual guard rejected before Torch or protocol reads')
else: raise AssertionError('Windows diagnostic launch was not rejected')
'''
        r=subprocess.run([sys.executable,'-B','-c',code,str(worker)],cwd=root,capture_output=True,text=True,timeout=15)
        assert r.returncode==0,(r.stdout,r.stderr)
        record.update(complete=True,actual_worker_guard_output=r.stdout.strip(),no_optimizer_or_new_solve_calls_in_worker=True)
    except Exception as e:
        record['failure']=str(e)
        with a.receipt.open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(record,indent=2)+'\n')
        raise
    with a.receipt.open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(record,indent=2)+'\n')
    print(record)


if __name__=='__main__':main()
