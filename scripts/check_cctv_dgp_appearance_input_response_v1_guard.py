"""Actual Windows CLI guard plus static gradient-target and optimizer checks."""
import ast
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    path=ROOT/'scripts/cctv_dgp_appearance_input_response_v1_vm.py'
    assert path.is_file(),'Manual-only image-input diagnostic worker is not implemented'
    source=path.read_text(encoding='utf-8');tree=ast.parse(source,feature_version=(3,10))
    gradients=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='grad']
    assert len(gradients)==1 and len(gradients[0].args)==2 and isinstance(gradients[0].args[1],ast.Name) and gradients[0].args[1].id=='rgb'
    assert not any(isinstance(n,ast.Attribute) and n.attr in ['optim','optimizer','backward','step','add_','sub_','copy_'] for n in ast.walk(tree))
    assert 'ConditionedBankDGP' not in source and 'load_frozen_dgp_restorer' not in source and 'load_prior' not in source
    result=subprocess.run([sys.executable,'-B',str(path),'--root','.', '--response-root','.', '--bank-root','.',
                           '--protocol-sha','0'*64,'--measure'],cwd=ROOT,capture_output=True,text=True,timeout=15)
    assert result.returncode!=0 and 'Manual Linux L4 input diagnostic only' in result.stderr
    assert 'urllib' not in result.stderr and 'torch' not in result.stderr
    print(dict(complete=True,real_Windows_measurement_rejected_before_neural_or_metadata_calls=True,
               gradients_target_RGB_leaf_only=True,no_optimizer_or_restorer_construction=True,local_neural_or_gradient_calls=0))


if __name__=='__main__':main()
