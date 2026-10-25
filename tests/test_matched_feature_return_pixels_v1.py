"""Synthetic saved-array negatives; no model, worker or returned payload load."""
import ast
import copy
import importlib.abc
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import unittest
from unittest import mock


class Blocker(importlib.abc.MetaPathFinder):
    attempted = []
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','torchvision','tensorflow','jax','jaxlib','keras'} or fullname.endswith(('.model','.worker')):
            self.attempted.append(fullname); raise RuntimeError('Forbidden return-pixel test import: ' + fullname)
        return None


BLOCKER = Blocker(); sys.meta_path.insert(0, BLOCKER)
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'scripts/audit_matched_feature_return_pixels_v1.py'
spec = importlib.util.spec_from_file_location('matched_pixel_audit_test_module', SOURCE)
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)


class PixelPrimitives(unittest.TestCase):
    def test_bounded_NPY_valid_actual_shapes_read_without_pickle(self):
        for shape in P.ARRAY_SHAPES:
            original=np.full(shape,.125,np.float32);stream=io.BytesIO();np.save(stream,original,allow_pickle=False)
            result=P.decode_array_bytes(stream.getvalue(),np=np)
            np.testing.assert_array_equal(result,original)

    def test_NPY_rejects_giant_object_fortran_wrong_schema_before_array_allocation(self):
        headers=[{'shape':(2**40,256,3),'fortran_order':False,'descr':'<f4'},
                 {'shape':(256,256,3),'fortran_order':False,'descr':'|O'},
                 {'shape':(256,256,3),'fortran_order':True,'descr':'<f4'},
                 {'shape':(256,256,3),'fortran_order':False,'descr':'<f8'},
                 {'shape':(128,128,3),'fortran_order':False,'descr':'<f4'}]
        for header in headers:
            stream=io.BytesIO();np.lib.format.write_array_header_1_0(stream,header)
            with self.subTest(header=header),mock.patch.object(np,'load',side_effect=AssertionError('Allocated before admission')):
                with self.assertRaises(ValueError):P.decode_array_bytes(stream.getvalue(),np=np)

    def test_NPY_refuses_trailing_truncated_oversized_header_and_unsupported_version(self):
        original=np.zeros((256,256,3),np.float32);stream=io.BytesIO();np.save(stream,original,allow_pickle=False);valid=stream.getvalue()
        bad_version=bytearray(valid);bad_version[6:8]=bytes([3,0])
        bad_header=bytearray(valid);bad_header[8:10]=(P.MAX_ARRAY_HEADER+1).to_bytes(2,'little')
        for raw in (valid+b'junk',valid[:-1],bytes(bad_version),bytes(bad_header)):
            with mock.patch.object(np,'load',side_effect=AssertionError('Array allocated before valid framing')):
                with self.assertRaises(ValueError):P.decode_array_bytes(raw,np=np)

    def test_actual_float32_floor_quantization_and_hashes(self):
        raw = np.full((256,256,3), np.float32(.5), dtype=np.float32)
        png = np.full(raw.shape, 127, dtype=np.uint8)
        result = P.pixel_summary(raw, png, np=np)
        self.assertEqual(result['RAW_pixel_bytes_sha256'], P.sha(raw.tobytes()))
        self.assertEqual(result['PNG_RGB8_sha256'], P.sha(png.tobytes()))
        self.assertEqual(raw[0,0,0], np.float32(.5)); self.assertEqual(png[0,0,0],127)

    def test_round_instead_of_floor_or_changed_png_is_refused(self):
        raw = np.full((256,256,3), np.float32(.5), dtype=np.float32)
        for png in (np.full(raw.shape, 128, dtype=np.uint8), np.full(raw.shape, 126, dtype=np.uint8)):
            with self.assertRaisesRegex(ValueError, 'floor-quantized'):
                P.pixel_summary(raw, png, np=np)

    def test_wrong_RAW_dtype_shape_nonfinite_or_range_is_refused(self):
        raw = np.zeros((256,256,3), np.float32); png = np.zeros(raw.shape, np.uint8)
        for value in (raw.astype(np.float64), raw[:128], np.full(raw.shape,np.nan,np.float32),
                      np.full(raw.shape,np.inf,np.float32), np.full(raw.shape,-.01,np.float32),
                      np.full(raw.shape,1.01,np.float32)):
            with self.assertRaisesRegex(ValueError, 'FP32 RAW'):
                P.pixel_summary(value,png,np=np)

    def test_wrong_actual_png_dtype_shape_is_refused(self):
        raw=np.zeros((256,256,3),np.float32); png=np.zeros(raw.shape,np.uint8)
        for value in (png.astype(np.float32),png[:128],png[:,:,0]):
            with self.assertRaisesRegex(ValueError,'RGB8'):
                P.pixel_summary(raw,value,np=np)

    def test_unclipped_array_may_be_outside_range_but_exact_shape_dtype_finite(self):
        P.unclip_shape(np.full((1,3,256,256),2.,np.float32),np=np)
        for value in (np.zeros((256,256,3),np.float32),np.zeros((1,3,256,256),np.float64),
                      np.full((1,3,256,256),np.nan,np.float32)):
            with self.assertRaises(ValueError): P.unclip_shape(value,np=np)

    def test_objective_arithmetic_policy_is_bounded_not_quality_epsilon(self):
        self.assertTrue(P.arithmetic_close(1.000001,1.0))
        self.assertFalse(P.arithmetic_close(1.0001,1.0))
        for a,b in ((True,1.),(math.nan,1.),(math.inf,1.),(1.,math.inf)):
            self.assertFalse(P.arithmetic_close(a,b))

    def test_recomputed_metric_equality_is_exact_no_added_arithmetic_tolerance(self):
        P.assert_subset({'metrics':{'mae':.1}}, {'metrics':{'mae':.1},'other':True},'test')
        with self.assertRaisesRegex(ValueError,'recomputed field differs'):
            P.assert_subset({'metrics':{'mae':.1}}, {'metrics':{'mae':math.nextafter(.1,1.)}},'test')
        with self.assertRaises(ValueError): P.assert_subset({'x':True},{'x':1},'test')

    def test_scope_sensitive_parity_joins_all_actual_metadata(self):
        row={'stock':{'native512':{'bytes_sha256':'a'}},'RAW_pixel_bytes_sha256':'b',
             'PNG_RGB8_sha256':'c','outputs':{'unclipped':{'sha256':'d'}},
             'input':{'path':'new.png','sha256':'e'},'target':{'path':'target.png','sha256':'f'}}
        self.assertTrue(P.parity_equal(row,copy.deepcopy(row)))
        changes=[lambda r:r['stock']['native512'].update(bytes_sha256='bad'),
                 lambda r:r.update(RAW_pixel_bytes_sha256='bad'),lambda r:r.update(PNG_RGB8_sha256='bad'),
                 lambda r:r['outputs']['unclipped'].update(sha256='bad'),
                 lambda r:r['input'].update(path='old.png'),lambda r:r['target'].update(path='oldtarget.png')]
        for change in changes:
            modified=copy.deepcopy(row);change(modified);self.assertFalse(P.parity_equal(row,modified))

    def test_failed_leaf_details_not_redundant_ancestors(self):
        value={'pass':False,'parent':{'pass':False,'rows':[{'pass':False,'view':'raw','role':'clear'},
                    {'pass':True,'view':'native_png'}]},'other':{'pass':True}}
        rows=P.failed_checks(value)
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['path'],'/parent/rows/0')
        self.assertEqual(rows[0]['role'],'clear')

    def test_strict_JSON_refuses_duplicates_nonfinite_float_overflow(self):
        for value in (b'{"x":1,"x":2}',b'[NaN]',b'[Infinity]',b'{"nested":{"value":1e999}}'):
            with self.assertRaises(ValueError): P.strict_json(value)
        self.assertEqual(P.strict_json(b'{"x":null}'),{'x':None})

    def test_canonical_path_refuses_escape_alias_and_windows_drive(self):
        for value in ('../x','/x','C:/x','x\\y','x//y','x/../y','.','x\ny','x/'):
            with self.assertRaises(ValueError): P.relative(value)
        self.assertEqual(P.relative('outputs/run/calls/0001/raw.npy'),'outputs/run/calls/0001/raw.npy')

    def test_audit_source_has_no_model_gradient_optimizer_or_VM_execution_calls(self):
        tree=ast.parse(SOURCE.read_bytes(),feature_version=(3,10))
        methods={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
        self.assertFalse({'backward','step','cuda','train','forward','load_state_dict','inference'} & methods)
        self.assertNotIn('extractall',methods)
        self.assertEqual(BLOCKER.attempted,[])
        self.assertFalse({'torch','tensorflow','jax'} & {n.split('.')[0] for n in sys.modules})


if __name__=='__main__': unittest.main()
