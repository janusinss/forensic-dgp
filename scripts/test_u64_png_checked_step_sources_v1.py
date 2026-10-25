"""Run bounded pure-source fixtures; forbid model frameworks and worker entry."""

import ast
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.abc
import io
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/codeformer_u64_png_checked_step_v1_preparation'
TEST_MODULES = (
    'tests.test_u64_png_checked_step_selection_v1',
    'tests.test_u64_png_checked_step_contract_v1',
    'tests.test_u64_png_checked_step_quality_v1',
    'tests.test_u64_png_checked_step_worker_v1',
    'tests.test_u64_png_checked_step_transport_v1',
)
FORBIDDEN = {'torch', 'tensorflow', 'jax', 'scipy'}


class BlockNeural(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN:
            raise RuntimeError('Local source fixtures cannot import neural frameworks: ' + fullname)


def main():
    sys.path.insert(0, str(ROOT))
    sys.dont_write_bytecode = True
    sys.meta_path.insert(0, BlockNeural())
    builder = importlib.import_module('scripts.build_u64_png_checked_step_v1')
    paths = list(builder.SOURCES)
    test_paths = [name.replace('.', '/') + '.py' for name in TEST_MODULES]
    paths.extend(test_paths)
    paths.append('scripts/test_u64_png_checked_step_sources_v1.py')
    paths = list(dict.fromkeys(paths))
    def pin(name):
        path = ROOT / name
        data = path.read_bytes()
        return {'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    before = [pin(name) for name in paths]
    for name in paths:
        if name.endswith('.py'):
            ast.parse((ROOT / name).read_bytes(), filename=name, feature_version=(3, 10))
    # A fixture accidentally calling run() is a failed test, even if an early
    # VM check would stop before Torch. Only pure APIs/arrays may be exercised.
    worker = importlib.import_module('research.codeformer_u64_png_checked_step_v1.worker')
    def forbidden_run(*args, **kwargs):
        raise RuntimeError('Actual worker execution prohibited in local source fixtures')
    worker.run = forbidden_run
    builder.build = forbidden_run
    output = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromNames(TEST_MODULES)
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    after = [pin(name) for name in paths]
    neural = [name for name in sys.modules if name.split('.')[0] in FORBIDDEN]
    complete = result.wasSuccessful() and before == after and not neural
    receipt = {
        'complete': True, 'created_UTC': datetime.now(timezone.utc).isoformat(),
        'all_tests_pass': complete, 'test_count': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors),
        'sources': [row for row in before if row['path'] in builder.SOURCES],
        'tests_and_runner': [row for row in before if row['path'] not in builder.SOURCES],
        'all_sources_tests_unchanged_after': before == after,
        'AST_Python310_pass': True, 'neural_modules': neural,
        'test_output': output.getvalue(),
        'actual_worker_and_builder_entry_blocked': True,
        'neural_gradient_optimizer_training_VM_calls': 0,
        'finite_VM_behavior_proved': False, 'model_qualified': False,
    }
    PREP.mkdir(exist_ok=True)
    destination = PREP / 'source_tests_v1.json'
    with destination.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(output.getvalue())
    print(json.dumps({'all_tests_pass': complete, 'test_count': result.testsRun, 'receipt': str(destination)}))
    return 0 if complete else 1


if __name__ == '__main__':
    raise SystemExit(main())
