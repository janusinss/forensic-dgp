"""Pure/inert comparison guards; no model, server, browser or Torch execution."""
import ast
import builtins
import importlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
original_import = builtins.__import__


def blocked(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('Torch is forbidden in pure comparison fixtures')
    return original_import(name, *args, **kwargs)


builtins.__import__ = blocked
try:
    module = importlib.import_module('research_postfusion_restoration_v1')
finally:
    builtins.__import__ = original_import


class ComparisonGuards(unittest.TestCase):
    def test_01_inert_status_holds_both_models(self):
        builtins.__import__ = blocked
        try:
            row = module.status()
            self.assertFalse(any(item['available'] for item in row['models'].values()))
            self.assertEqual(row['neural_forward_attempts'], 0)
            self.assertEqual(row['maximum_forwards_per_request'], 2)
            self.assertFalse(row['main_app_default_changed'])
            self.assertNotIn('torch', sys.modules)
        finally: builtins.__import__ = original_import

    def test_02_manual_refusal_precedes_decode_and_manifest(self):
        engine = module.Comparison()
        builtins.__import__ = blocked
        try:
            for review in ('', 'needs_clearer', 'out_of_scope'):
                with self.assertRaises(ValueError): engine.generate(b'not an image', review)
            self.assertEqual(engine.measured_status()['operator_review_refusals'], 3)
            self.assertEqual(engine.measured_status()['neural_forward_attempts'], 0)
            self.assertEqual(engine.models, {})
        finally: builtins.__import__ = original_import

    def test_03_existing_engine_is_not_shared(self):
        first, second = module.Comparison(), module.Comparison()
        self.assertIsNot(first.incumbent_loader, module.spatial.engine)
        self.assertIsNot(first.incumbent_loader, second.incumbent_loader)
        self.assertIsNot(first.models, second.models)

    def test_04_nonfinite_and_duplicate_metadata_refused(self):
        for value in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":[{"y":1e999}]}'):
            with self.assertRaises(ValueError): module.strict_json(value)
        self.assertEqual(module.strict_json('{"x":[1,0.25]}'), {'x': [1, .25]})

    def test_05_noncanonical_bound_paths_refused_before_filesystem(self):
        for name in ('../outside', './research.py', 'a//b', 'a/./b', 'C:/outside', 'a\\b', '/outside'):
            with self.assertRaises(ValueError): module.bound({'path': name, 'bytes': 1, 'sha256': '0' * 64})
        with self.assertRaises(ValueError): module.bound({'path': 'research.py', 'bytes': True, 'sha256': '0' * 64})

    def test_06_support_composition_and_preparation_are_exact(self):
        np = module.np
        rgb = np.zeros((20, 12, 3), dtype=np.uint8); rgb[..., 1] = 103
        canvas, observed, mask, _ = module.prepare_crop(rgb)
        raw = np.full((256, 256, 3), .731, dtype=np.float32)
        delivered, pixels = module.delivered_views(raw, canvas, observed)
        self.assertTrue(np.array_equal(pixels[~observed], canvas[~observed]))
        self.assertTrue(np.array_equal(delivered[~observed], canvas[~observed].astype(np.float32) / np.float32(255)))
        self.assertFalse(mask.any())
        self.assertTrue(np.all(raw == np.float32(.731)))
        self.assertTrue(np.array_equal(canvas, module.prepare_crop(rgb)[0]))

    def test_07_sources_python310_and_no_training_calls(self):
        for path in module.OWN_SOURCES[:2]:
            tree = ast.parse((ROOT / path).read_text(encoding='utf-8'), feature_version=(3, 10))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    self.assertNotIn(node.func.attr, ('backward', 'step', 'zero_grad'))
        tree = ast.parse((ROOT / module.OWN_SOURCES[0]).read_text(encoding='utf-8'))
        imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any('torch' in ast.unparse(n) for n in imports))

    def test_08_template_and_script_use_same_two_arms_and_no_context_processing(self):
        html = (ROOT / module.OWN_SOURCES[2]).read_text(encoding='utf-8')
        script = (ROOT / module.OWN_SOURCES[3]).read_text(encoding='utf-8')
        for arm in module.ARMS:
            self.assertIn('id="result-' + arm + '"', html)
            self.assertIn('id="download-' + arm + '"', html)
            self.assertIn(arm, script)
        self.assertIn('new FormData(event.target)', script)
        self.assertLess(script.index('new FormData(event.target)'), script.index('control.disabled = true'))
        self.assertIn(module.ROUTE + '/compare', script)
        self.assertIn('No extra blur, noise', html)
        self.assertNotIn('selection', html)

    def test_09_launcher_refuses_receipt_dot_escape_before_configuration(self):
        tree = ast.parse((ROOT / module.OWN_SOURCES[1]).read_text(encoding='utf-8'))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        guard = next(node for node in main.body if isinstance(node, ast.If) and 'receipt.parts' in ast.unparse(node.test))
        import types
        for path, reject in ((ROOT / 'scratch' / '..' / 'outside.json', True), (ROOT / 'scratch' / 'fresh.json', False)):
            actual = eval(compile(ast.Expression(guard.test), '<actual receipt guard>', 'eval'),
                          {'args': types.SimpleNamespace(receipt=path), 'ROOT': ROOT})
            self.assertEqual(actual, reject)


if __name__ == '__main__': unittest.main()
