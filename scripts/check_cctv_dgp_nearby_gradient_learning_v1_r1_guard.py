"""Real Windows CLI and static manual-only training boundary checks."""
import ast
from pathlib import Path
import subprocess
import sys


def main():
    path = Path(__file__).with_name('cctv_dgp_nearby_gradient_learning_v1_r1_vm.py')
    assert path.is_file(), 'Nearby-state manual worker is not implemented'
    result = subprocess.run([sys.executable, '-B', str(path), '--root', '.', '--parent-root', '.', '--response-root', '.',
        '--protocol-sha', '0' * 64, '--run'], capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and 'Manual Linux L4 nearby-state training only' in result.stderr
    assert 'metadata.google' not in result.stderr and 'import torch' not in result.stderr
    text = path.read_text(encoding='utf-8'); tree = ast.parse(text, feature_version=(3, 10))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert sum(isinstance(n.func, ast.Name) and n.func.id == 'solve_direction' for n in calls) == 1
    assert not any(isinstance(n.func, ast.Attribute) and n.func.attr in ['backward', 'step'] for n in calls)
    assert "p['sampling_sites']==['initial','plus02','minus02']" in text
    assert "counts['accepted_updates']<=1" in text and "counts['parameter_proposals']<=3" in text
    assert 'torch_optimizer_constructed=False' in text and 'stateless_gradient_descent_is_actual_training=True' in text
    body = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main').body
    guard_index = next(i for i, n in enumerate(body) if isinstance(n, ast.If) and ast.unparse(n.test) == 'a.run')
    verify_index = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign) and 'verify(' in ast.unparse(n))
    assert guard_index < verify_index and 'require_vm' in ast.unparse(body[guard_index])
    print(dict(complete=True, real_local_training_rejected=True, one_solve_three_proposals_one_acceptance_bound=True,
               neural_or_training_calls=0))


if __name__ == '__main__':
    main()
