"""Real CLI regression: local measurement must reject before neural loading."""
import ast
from pathlib import Path
import subprocess
import sys


def main():
    worker = Path(__file__).with_name('cctv_dgp_recognizer_precision_v1_vm.py')
    assert worker.is_file(), 'Manual-only precision worker is not implemented'
    result = subprocess.run([sys.executable, '-B', str(worker), '--root', '.', '--input-root', '.',
        '--response-root', '.', '--bank-root', '.', '--protocol-sha', '0' * 64, '--measure'],
        capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and 'Manual Linux L4 precision diagnostic only' in result.stderr
    assert 'import torch' not in result.stderr and 'metadata.google' not in result.stderr
    tree = ast.parse(worker.read_text(encoding='utf-8'), feature_version=(3, 10))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert not any(isinstance(n.func, ast.Attribute) and n.func.attr in ['backward', 'step', 'copy_', 'add_'] for n in calls)
    gradients = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == 'grad']
    assert len(gradients) == 1 and isinstance(gradients[0].args[1], ast.Name) and gradients[0].args[1].id == 'rgb'
    main_body = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main').body
    first = next(i for i, n in enumerate(main_body) if isinstance(n, ast.If) and ast.unparse(n.test) == 'a.measure')
    verify = next(i for i, n in enumerate(main_body) if isinstance(n, ast.Assign) and 'verify(' in ast.unparse(n))
    assert first < verify and 'require_vm' in ast.unparse(main_body[first])
    print(dict(complete=True, local_measurement_rejected=True, neural_or_training_calls=0,
               gradient_target_RGB_only=True, optimizer_or_neural_weight_updates_absent=True))


if __name__ == '__main__':
    main()
