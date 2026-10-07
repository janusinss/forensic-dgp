"""Execute actual pre-import path guards with simulated VM metadata only."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
WORKER = 'scripts/cctv_dgp_feature_fusion_v32_vm.py'
CACHE = 'cctv_dgp_feature_fusion_v32_cache.py'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def functions(path):
    return {n.name: n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}


def run_checks(bundle):
    home = (ROOT / 'scratch/metadata_only_simulated_VM_home').resolve()
    class HomePath:
        @staticmethod
        def home():
            return home
    fake_sys = SimpleNamespace(platform='linux')
    cache = functions(bundle / CACHE)
    load = cache['load_items']
    # Execute the actual guard, stopping before its first image/tensor import.
    if 'require_cache_scope' in cache:
        guard = cache['require_cache_scope']
        assert len(guard.body) == 2 and all(isinstance(n, ast.Assert) for n in guard.body)
        assert isinstance(load.body[0], ast.Expr) and ast.unparse(load.body[0]) == 'require_cache_scope(root)'
        assert isinstance(load.body[1], ast.Import) and ast.unparse(load.body[1]) == 'import numpy as np'
        guard_body = [guard]
    else:
        assert all(isinstance(n, ast.Assert) for n in load.body[:2])
        assert isinstance(load.body[2], ast.Import) and ast.unparse(load.body[2]) == 'import numpy as np'
        guard = ast.FunctionDef(name='require_cache_scope', args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='root')],
                                kwonlyargs=[], kw_defaults=[], defaults=[]), body=load.body[:2], decorator_list=[])
        guard_body = [guard]
    scope = functions(bundle / WORKER)['vm_scope']
    assert len(scope.body) in (4, 6)
    namespace = {'Path': HomePath, 'sys': fake_sys}
    exec(compile(ast.fix_missing_locations(ast.Module(body=guard_body, type_ignores=[])), '<actual-cache-guard-only>', 'exec'), namespace)
    def restricted_import(name, *args, **kwargs):
        assert name == 'cctv_dgp_feature_fusion_v32_cache', 'No neural or other module import allowed'
        return SimpleNamespace(require_cache_scope=namespace['require_cache_scope'])
    namespace['__builtins__'] = dict(vars(__import__('builtins')), __import__=restricted_import)
    exec(compile(ast.Module(body=[scope], type_ignores=[]), '<actual-worker-guard-only>', 'exec'), namespace)
    approved = home / 'forensic-dgp' / bundle.name
    parent = home / 'forensic-dgp/cctv_dgp_feature_skips_vm_v27'
    rows = []
    def check(label, path, platform, expected):
        fake_sys.platform = platform
        passed = True
        try:
            namespace['vm_scope'](path, parent)
            namespace['require_cache_scope'](path)
        except AssertionError:
            passed = False
        rows.append({'case': label, 'expected': expected, 'actual': passed, 'pass': passed == expected})
    check('approved Linux packet root', approved, 'linux', True)
    check('Windows local training rejected', approved, 'win32', False)
    check('base V32 root rejected', home / 'forensic-dgp/cctv_dgp_feature_fusion_vm_v32', 'linux', False)
    check('old R1 root rejected', home / 'forensic-dgp/cctv_dgp_feature_fusion_vm_v32_r1', 'linux', bundle.name.endswith('_r1'))
    check('nested same-name root rejected', home / 'forensic-dgp/nested' / bundle.name, 'linux', False)
    check('outside research root rejected', home / 'elsewhere' / bundle.name, 'linux', False)
    check('unknown sibling root rejected', approved.with_name(bundle.name + '_other'), 'linux', False)
    main = functions(bundle / WORKER)['main']
    transfer = next(n for n in main.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'a.verify_transfer')
    before_dependencies = ast.unparse(transfer.body[0]) == 'vm_scope(root, parent)'
    cache_called_from_vm_scope = any(isinstance(n, ast.Expr) and ast.unparse(n) == 'require_cache_scope(root)' for n in scope.body)
    if bundle.name.endswith('_r2'):
        rows.append({'case': 'transfer checks actual cache guard before data/model work', 'pass': before_dependencies and cache_called_from_vm_scope})
    return {'complete': all(row['pass'] for row in rows), 'rows': rows,
            'actual_cache_source_sha256': sha(bundle / CACHE), 'actual_worker_source_sha256': sha(bundle / WORKER),
            'transfer_cache_guard_before_dependencies': before_dependencies and cache_called_from_vm_scope,
            'metadata_simulation_only': True, 'neural_imports': 0, 'gradient_calls': 0, 'optimizer_updates': 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--receipt', required=True, type=Path)
    args = parser.parse_args()
    report = run_checks(args.bundle.resolve())
    report['checker_sha256'] = sha(Path(__file__))
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['complete'] else 1)


if __name__ == '__main__':
    main()
