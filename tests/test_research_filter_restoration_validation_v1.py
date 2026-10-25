"""Pure input, export-lineage and delivered-pixel checks; no neural calls.

The actual legacy validation functions are compiled from their AST solely to
check geometry equivalence without importing their unrelated Torch modules.
This is functional evidence, never qualification of the stopped model.
"""
import ast
import base64
import builtins
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
import zipfile
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import research_filter_restoration_web_v1 as web


def legacy(path, names, namespace):
    tree = ast.parse(path.read_bytes())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return types.SimpleNamespace(**{name: namespace[name] for name in names})


old_input = legacy(ROOT / 'dgp_face_workflow_v3.py', ('decode_crop', 'review_input'),
    {'np': np, 'Image': Image, 'ImageOps': ImageOps, 'io': io, 'UPLOAD_LIMIT': web.LIMIT})
old_geometry = legacy(ROOT / 'dgp_face_restoration.py', ('prepare_crop',),
    {'np': np, 'Image': Image, 'hashlib': hashlib, 'POLICY': 'dgp256-observed-canvas-v1'})


def image_bytes(size=(29, 47), mode='RGB', fmt='PNG'):
    stream = io.BytesIO()
    Image.new(mode, size, 128).save(stream, format=fmt)
    return stream.getvalue()


def export_fixture():
    return {'complete': True, 'model_qualified': False, 'protocol_sha256': web.PROTOCOL_SHA,
        'archive_sha256': web.ARCHIVE_SHA, 'source_checkpoint': web.CHECKPOINT_PATH,
        'source_checkpoint_sha256': web.CHECKPOINT_SHA, 'source_checkpoint_bytes': 461955077,
        'model_tensor_sha256': web.STATE_SHA, 'source_training_updates': 50,
        'all515_tensor_values_exact': True, 'app_default_changed': False,
        'review_pins': {**{name: {'bytes': 1, 'sha256': digest}
                            for name, (_, digest) in web.REVIEW_PINS.items()},
                        'visual': {'bytes': 1, 'sha256': 'a' * 64}},
        'preservation_pass': False, 'parent_dark_retention_pass': True,
        'R2_dark_retention_pass': False, 'additional_interim_stop_pass': False,
        'params_ema_key_is_loader_compatibility_not_new_EMA': True,
        'original_training_optimizer_RNG_archive_retained': True,
        'neural_gradient_optimizer_VM_calls': 0, 'export_sha256': 'b' * 64,
        'export_bytes': 12, 'model_path': str(web.MODEL_DIR / web.MODEL_NAME)}


class FilterResearchValidation(unittest.TestCase):
    def test_module_and_status_import_no_torch(self):
        self.assertFalse(any(k == 'torch' or k.startswith('torch.') for k in sys.modules))
        self.assertEqual(web.status()['neural_forward_attempts'], 0)
        self.assertFalse(web.status()['candidate_qualified'])

    def test_input_review_exact_legacy_messages(self):
        for value in ('', 'usable', 'needs_clearer', 'out_of_scope', 'untrusted'):
            def outcome(function):
                try:
                    return function(value)
                except ValueError as exc:
                    return str(exc)
            self.assertEqual(outcome(web.review_input), outcome(old_input.review_input))

    def test_decode_matches_actual_legacy_png_jpeg_grayscale_and_alpha(self):
        for mode, fmt in (('RGB', 'PNG'), ('RGB', 'JPEG'), ('L', 'PNG'), ('RGBA', 'PNG')):
            data = image_bytes(mode=mode, fmt=fmt)
            self.assertTrue(np.array_equal(web.decode_crop(data), old_input.decode_crop(data)))

    def test_geometry_matches_actual_legacy_for_square_wide_tall_small_and_odd_crops(self):
        generator = np.random.default_rng(20261019)
        for h, w in ((256, 256), (47, 29), (29, 47), (1, 1), (5, 17), (17, 5), (138, 165), (443, 467)):
            with self.subTest(h=h, w=w):
                rgb = generator.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
                current, prior = web.prepare_crop(rgb), old_geometry.prepare_crop(rgb)
                for index in (0, 1, 2):
                    self.assertTrue(np.array_equal(current[index], prior[index]))
                    self.assertEqual(current[index].dtype, prior[index].dtype)
                self.assertEqual(current[3], prior[3])

    def test_empty_support_is_clean_refusal_instead_of_index_error(self):
        for h, w in ((1, 4096), (4096, 1)):
            with self.assertRaisesRegex(ValueError, 'readable facial width'):
                web.prepare_crop(np.zeros((h, w, 3), np.uint8))

    def test_malformed_unsupported_oversize_or_overdimension_input_refused(self):
        for data in (b'', b'not an image', image_bytes(fmt='GIF'),
                     image_bytes(size=(4097, 1)), b'x' * (web.LIMIT + 1)):
            with self.assertRaises(ValueError):
                web.decode_crop(data)

    def test_operator_refusal_happens_before_any_export_or_neural_load(self):
        engine = web.Comparison()
        with (patch.object(web, 'fusion_file', side_effect=AssertionError('No weights')),
              patch.object(engine, 'load', side_effect=AssertionError('No neural loader'))):
            for review in ('', 'needs_clearer', 'out_of_scope', 'untrusted'):
                with self.assertRaises(ValueError):
                    engine.generate(image_bytes(), review, 'all')
        self.assertEqual(engine.measured_status()['operator_review_refusals'], 4)
        self.assertEqual(engine.measured_status()['neural_forward_attempts'], 0)
        self.assertEqual(engine.models, {})

    def test_unknown_selection_and_bad_decode_happen_before_neural_load(self):
        engine = web.Comparison()
        with patch.object(engine, 'load', side_effect=AssertionError('No neural loader')):
            for selection in ('', '../../model.pth', 'AUTO', 'both', 'original,r2'):
                with self.assertRaises(ValueError):
                    engine.generate(image_bytes(), 'usable', selection)
            with self.assertRaises(ValueError):
                engine.generate(b'bad upload', 'usable', 'all')
        self.assertEqual(engine.measured_status()['neural_forward_attempts'], 0)

    def test_export_fixture_passes_metadata_only_with_no_physical_load(self):
        fixture = export_fixture()
        self.assertIs(web.validate_export_receipt(fixture), fixture)

    def test_export_forged_lineage_quality_claim_or_missing_reviews_refused(self):
        replacements = {'model_qualified': True, 'complete': False, 'source_training_updates': 51,
            'protocol_sha256': '0' * 64, 'archive_sha256': '0' * 64,
            'source_checkpoint_sha256': '0' * 64, 'source_checkpoint_bytes': 1,
            'model_tensor_sha256': '0' * 64, 'review_pins': {}, 'preservation_pass': True,
            'R2_dark_retention_pass': True, 'app_default_changed': True,
            'export_sha256': 'bad', 'export_bytes': -1, 'model_path': '../../arbitrary.pth',
            'params_ema_key_is_loader_compatibility_not_new_EMA': False,
            'neural_gradient_optimizer_VM_calls': 1}
        for key, value in replacements.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                web.validate_export_receipt({**export_fixture(), key: value})
        for name in ('import', 'state', 'pixels', 'visual'):
            row = export_fixture(); row['review_pins'][name]['sha256'] = 'invalid'
            with self.subTest(name=name), self.assertRaises(ValueError):
                web.validate_export_receipt(row)

    def test_duplicate_nonfinite_or_oversize_receipt_refused_without_torch(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'scratch') as temporary:
            path = Path(temporary) / 'fixture.json'
            for payload in ('{"complete":true,"complete":false}', '{"x":NaN}', 'x' * (1024**2 + 1)):
                path.write_text(payload, encoding='utf-8')
                with self.assertRaises(ValueError):
                    web.read_receipt(path)

    def test_delivered_RAW_and_PNG_match_input_exactly_outside_support(self):
        rgb = np.full((138, 165, 3), 49, np.uint8)
        canvas, observed, _, _ = web.prepare_crop(rgb)
        native = np.full((256, 256, 3), .499, np.float32)
        raw, pixels = web.delivered_views(native, canvas, observed)
        self.assertTrue(np.array_equal(raw[~observed], canvas[~observed].astype(np.float32) / np.float32(255)))
        self.assertTrue(np.array_equal(pixels[~observed], canvas[~observed]))
        self.assertTrue(np.array_equal(raw[observed], native[observed]))
        self.assertTrue(np.all(pixels[observed] == 127))
        self.assertTrue(np.all(native == np.float32(.499)))

    def test_delivered_views_reject_invalid_RAW_types_range_and_support(self):
        raw = np.zeros((256, 256, 3), np.float32)
        canvas = np.zeros((256, 256, 3), np.uint8); observed = np.ones((256, 256), bool)
        for bad in (raw.astype(np.float64), raw[:255], raw + np.float32(1.1), raw + np.float32(float('nan'))):
            with self.assertRaises(ValueError):
                web.delivered_views(bad, canvas, observed)
        with self.assertRaises(ValueError):
            web.delivered_views(raw, canvas, observed.astype(np.uint8))

    def test_actual_bundle_serialization_keeps_RAW_PNG_and_unique_upload_names(self):
        module = ast.parse((ROOT / 'research_filter_restoration_web_v1.py').read_bytes())
        cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'Comparison')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'generate')
        protected = [n for n in method.body if isinstance(n, ast.Try)][-1]
        start = next(i for i, n in enumerate(protected.body) if isinstance(n, ast.Assign) and
                     any(isinstance(t, ast.Name) and t.id == 'bundle' for t in n.targets))
        serialization = protected.body[start:]
        rgb = np.full((19, 13, 3), 27, np.uint8)
        canvas, observed, _, _ = web.prepare_crop(rgb)
        for arms in (('original',), ('r2',), ('fusion50',), web.ARMS):
            outputs = {}
            for index, arm in enumerate(arms):
                raw, pixels = web.delivered_views(np.full((256, 256, 3), .25 + index * .1, np.float32), canvas, observed)
                outputs[arm] = {'pixels': pixels, 'raw': raw, 'PNG': web.base.png(pixels)}
            metadata = {'selection': list(arms), 'model_qualified': False}
            engine = web.Comparison()
            namespace = {'io': io, 'zipfile': zipfile, 'json': json, 'np': np, 'base': web.base,
                         'rgb': rgb, 'canvas': canvas, 'observed': observed, 'outputs': outputs,
                         'metadata': metadata, 'self': engine}
            function = ast.FunctionDef(name='actual_serialization', args=ast.arguments(posonlyargs=[], args=[],
                kwonlyargs=[], kw_defaults=[], defaults=[]), body=serialization, decorator_list=[])
            exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), '<actual_bundle>', 'exec'), namespace)
            response = namespace['actual_serialization']()
            bundle = base64.b64decode(response['bundle'].split(',', 1)[1])
            with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
                names = archive.namelist(); self.assertEqual(len(names), len(set(names)))
                self.assertEqual(set(names), {'original-upload.png', 'resize-control.png', 'observed-support.png', 'processing.json'} |
                    {arm + suffix for arm in arms for suffix in ('.png', '-raw.npy')})
                self.assertTrue(np.array_equal(np.asarray(Image.open(archive.open('original-upload.png'))), rgb))
                for arm in arms:
                    self.assertTrue(np.array_equal(np.asarray(Image.open(archive.open(arm + '.png'))), outputs[arm]['pixels']))
                    self.assertTrue(np.array_equal(np.load(io.BytesIO(archive.read(arm + '-raw.npy')), allow_pickle=False), outputs[arm]['raw']))
                self.assertFalse(json.loads(archive.read('processing.json'))['model_qualified'])
            self.assertEqual(engine.measured_status()['completed_comparisons'], 1)
            self.assertEqual(engine.measured_status()['neural_forward_attempts'], 0)


if __name__ == '__main__':
    (ROOT / 'scratch').mkdir(exist_ok=True)
    original_import = builtins.__import__
    def forbid_neural_import(name, *args, **kwargs):
        if name == 'torch' or name.startswith('torch.') or name.startswith('third_party.codeformer'):
            raise AssertionError('Pure validation must not import a neural module: ' + name)
        return original_import(name, *args, **kwargs)
    with patch.object(builtins, '__import__', forbid_neural_import):
        unittest.main()
