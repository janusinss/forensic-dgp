"""Pure schemas, safe arrays, source order and guards; never construct a model."""
import ast
import builtins
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

import numpy as np

from research.codeformer_u64_cleananchors_continuation_v1 import model as M


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'research/codeformer_u64_cleananchors_continuation_v1/model.py'
TERMINAL = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_inference_export/terminal_u0064_heads.npz'


def arrays():
    return {name: np.zeros(shape, dtype=np.float32) for name, shape in M.expected_head_shapes().items()}


def encoded(values=None):
    stream = io.BytesIO()
    np.savez(stream, **(arrays() if values is None else values))
    return stream.getvalue()


def rewrite(raw, transform, compression=zipfile.ZIP_STORED):
    stream = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw)) as source, zipfile.ZipFile(stream, 'w', compression=compression) as target:
        for item in source.infolist():
            name, payload = transform(item.filename, source.read(item.filename))
            target.writestr(name, payload)
    return stream.getvalue()


class HeadSchemaAndSafeArrays(unittest.TestCase):
    def test_exact_schema_and_independent_parameter_census(self):
        shapes = M.expected_head_shapes()
        self.assertEqual(len(shapes), 36)
        self.assertEqual(sum(math.prod(shape) for shape in shapes.values()), 168960)
        self.assertEqual(sum(math.prod(shape) for name, shape in shapes.items() if '.output.' in name), 25344)
        self.assertEqual(M.TOTAL_TENSORS, M.FROZEN_TENSORS + M.TRAINABLE_TENSORS)
        self.assertEqual((M.TOTAL_TENSORS, M.FROZEN_TENSORS), (575, 539))

    def test_complete_synthetic_roundtrip_is_not_terminal_authorization(self):
        original = arrays()
        raw = encoded(original)
        decoded = M.decode_head_npz(raw, np=np)
        self.assertEqual(set(original), set(decoded))
        for name in original:
            self.assertTrue(np.array_equal(original[name], decoded[name]))
            self.assertTrue(decoded[name].flags.c_contiguous)
            self.assertTrue(decoded[name].flags.writeable)
        self.assertNotEqual(M.head_arrays_canonical_sha256(decoded, np=np), M.TERMINAL_HEADS_CANONICAL_SHA256)

    def test_canonical_digest_independently_reconstructed(self):
        values = arrays()
        for index, value in enumerate(values.values()):
            value.fill(np.float32(index / 100))
        digest = hashlib.sha256()
        for name in sorted(values):
            value = values[name]
            header = json.dumps([name, 'torch.float32', list(value.shape)], separators=(',', ':')).encode()
            digest.update(len(header).to_bytes(8, 'big') + header + value.tobytes(order='C'))
        self.assertEqual(M.head_arrays_canonical_sha256(values, np=np), digest.hexdigest())

    def test_canonical_digest_distinguishes_signed_zero(self):
        positive = arrays()
        negative = {name: value.copy() for name, value in positive.items()}
        next(iter(negative.values())).fill(np.float32(-0.0))
        self.assertNotEqual(M.head_arrays_canonical_sha256(positive, np=np),
                            M.head_arrays_canonical_sha256(negative, np=np))

    def test_safe_parser_rejects_missing_extra_and_unsafe_names(self):
        raw = encoded()
        first = next(iter(M.expected_head_shapes())) + '.npy'
        for new in ('../escape.npy', 'heads/unexpected.npy', 'unexpected.npy'):
            with self.subTest(new=new), self.assertRaises(ValueError):
                M.decode_head_npz(rewrite(raw, lambda name, payload: (new if name == first else name, payload)), np=np)
        values = arrays()
        values.pop(next(iter(values)))
        with self.assertRaises(ValueError):
            M.decode_head_npz(encoded(values), np=np)
        values = arrays()
        values['unexpected'] = np.zeros((1,), dtype=np.float32)
        with self.assertRaises(ValueError):
            M.decode_head_npz(encoded(values), np=np)

    def test_duplicate_names_are_rejected_before_array_read(self):
        raw = encoded()
        with zipfile.ZipFile(io.BytesIO(raw)) as source:
            names = source.namelist()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            duplicate = rewrite(raw, lambda name, payload: (names[0] if name == names[1] else name, payload))
        with self.assertRaises(ValueError):
            M.decode_head_npz(duplicate, np=np)

    def test_wrong_shape_dtype_endianness_object_fortran_and_nonfinite_refused(self):
        for kind in ('shape', 'float64', 'big_endian', 'object', 'fortran', 'NaN', 'infinity'):
            values = arrays()
            key = 'heads.32.spatial.weight'
            if kind == 'shape':
                values[key] = np.zeros((1,), dtype=np.float32)
            elif kind == 'float64':
                values[key] = values[key].astype(np.float64)
            elif kind == 'big_endian':
                values[key] = values[key].astype('>f4')
            elif kind == 'object':
                values[key] = values[key].astype(object)
            elif kind == 'fortran':
                values[key] = np.asfortranarray(values[key])
            else:
                values[key].flat[0] = np.nan if kind == 'NaN' else np.inf
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                M.decode_head_npz(encoded(values), np=np)

    def test_payload_trailing_bytes_and_unsupported_header_refused(self):
        raw = encoded()
        first = next(iter(M.expected_head_shapes())) + '.npy'
        for label, transform in (
            ('trailing', lambda payload: payload + b'extra'),
            ('version3', lambda payload: payload[:6] + b'\x03\x00' + payload[8:]),
            ('truncated', lambda payload: payload[:-1]),
        ):
            changed = rewrite(raw, lambda name, payload: (name, transform(payload) if name == first else payload))
            with self.subTest(label=label), self.assertRaises(ValueError):
                M.decode_head_npz(changed, np=np)

    def test_compression_and_comments_are_refused(self):
        raw = encoded()
        compressed = rewrite(raw, lambda name, payload: (name, payload), compression=zipfile.ZIP_DEFLATED)
        with self.assertRaises(ValueError):
            M.decode_head_npz(compressed, np=np)
        stream = io.BytesIO(raw)
        with zipfile.ZipFile(stream, 'a') as archive:
            archive.comment = b'unreviewed'
        with self.assertRaises(ValueError):
            M.decode_head_npz(stream.getvalue(), np=np)

    def test_encrypted_or_symlink_member_metadata_is_refused(self):
        raw = encoded()
        original = zipfile.ZipFile.infolist
        for kind in ('encrypted', 'symlink'):
            def observed(archive):
                values = original(archive)
                if kind == 'encrypted':
                    values[0].flag_bits |= 1
                else:
                    values[0].external_attr = (stat.S_IFLNK | 0o777) << 16
                return values
            with self.subTest(kind=kind), mock.patch.object(zipfile.ZipFile, 'infolist', observed):
                with self.assertRaises(ValueError):
                    M.decode_head_npz(raw, np=np)

    def test_corrupt_member_crc_refused(self):
        raw = encoded()
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            first = archive.infolist()[0]
            offset = first.header_offset
        changed = bytearray(raw)
        filename_length = int.from_bytes(changed[offset + 26:offset + 28], 'little')
        extra_length = int.from_bytes(changed[offset + 28:offset + 30], 'little')
        position = offset + 30 + filename_length + extra_length + first.file_size - 1
        changed[position] ^= 1
        with self.assertRaises(zipfile.BadZipFile):
            M.decode_head_npz(bytes(changed), np=np)

    def test_input_and_canonical_schema_bounds(self):
        for raw in (b'', bytearray(b'x'), b'x' * M.MAX_HEAD_FILE_BYTES):
            with self.subTest(type=type(raw).__name__, size=len(raw)), self.assertRaises(ValueError):
                M.decode_head_npz(raw, np=np)
        with self.assertRaises(ValueError):
            M.head_arrays_canonical_sha256({}, np=np)
        values = arrays()
        values[next(iter(values))] = np.zeros((1,), dtype=np.float32)
        with self.assertRaises(ValueError):
            M.head_arrays_canonical_sha256(values, np=np)

    def test_actual_terminal_npz_only_read_as_arrays_without_model(self):
        # Explicitly permitted byte/array readback: no framework/model creation.
        values = M.read_terminal_head_state(TERMINAL, np=np)
        self.assertEqual(len(values), 36)
        self.assertEqual(M.head_arrays_canonical_sha256(values, np=np), M.TERMINAL_HEADS_CANONICAL_SHA256)
        self.assertEqual(M.file_pin(TERMINAL), {key: M.ARTIFACTS['heads'][key] for key in ('bytes', 'sha256')})
        self.assertNotIn('torch', sys.modules)

    def test_synthetic_reset_file_cannot_pass_terminal_reader(self):
        (ROOT / 'scratch').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / 'scratch') as tmp:
            path = Path(tmp) / 'heads.npz'
            path.write_bytes(encoded())
            with self.assertRaisesRegex(ValueError, 'Exact retained terminal'):
                M.read_terminal_head_state(path, np=np)


class ManualGuardsAndSource(unittest.TestCase):
    def context(self):
        home = '/home/janusdominic0'
        root = home + '/forensic-dgp/' + M.VM_NAME
        return {'system': 'Linux', 'root': root, 'home': home, 'cwd': root,
                'prefix': home + '/forensic-dgp/cctv_dgp_vm_bundle/.venv',
                'device': 'cuda:0', 'tmux': '/tmp/tmux-1000/default,1,0',
                'gpu_names': ('NVIDIA L4',), 'acknowledgement': M.MANUAL_ACKNOWLEDGEMENT}

    def test_pure_context_accepts_only_exact_declared_context(self):
        self.assertTrue(M.validate_manual_context(**self.context()))
        bad = {'system': 'Windows', 'root': '/home/janusdominic0/forensic-dgp/old',
               'home': '/home/a/../janusdominic0', 'cwd': '/tmp', 'prefix': '/usr',
               'device': 'cpu', 'tmux': '', 'gpu_names': ('NVIDIA L4', 'NVIDIA L4'),
               'acknowledgement': 'automatic'}
        for key, value in bad.items():
            context = self.context()
            context[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                M.validate_manual_context(**context)
        for names in (['NVIDIA L4'], ('Tesla T4',), ()):
            context = self.context()
            context['gpu_names'] = names
            with self.subTest(names=names), self.assertRaises(ValueError):
                M.validate_manual_context(**context)

    def test_wrong_platform_factory_stops_before_framework_import_or_files(self):
        original = builtins.__import__
        attempted = []
        def no_framework(name, *args, **kwargs):
            if name == 'torch' or name.startswith('torch.'):
                attempted.append(name)
                raise AssertionError('Torch import forbidden in local test')
            return original(name, *args, **kwargs)
        with mock.patch.object(M.platform, 'system', return_value='Windows'), \
             mock.patch.object(builtins, '__import__', no_framework), \
             mock.patch.object(M, 'file_pin', side_effect=AssertionError('No file work before context refusal')):
            with self.assertRaisesRegex(ValueError, 'Manual Linux'):
                M.build_restorer(object(), {}, r2_artifact='unused', u96_artifact='unused', head_artifact='unused',
                                 manual_vm_root=ROOT, acknowledgement=M.MANUAL_ACKNOWLEDGEMENT)
        self.assertEqual(attempted, [])
        self.assertNotIn('torch', sys.modules)

    def test_wrong_linux_root_stops_before_framework_or_GPU_probe(self):
        with mock.patch.object(M.platform, 'system', return_value='Linux'), \
             mock.patch.object(M.subprocess, 'run', side_effect=AssertionError('No GPU probe for wrong root')):
            with self.assertRaisesRegex(ValueError, 'exact future manual stage'):
                M.build_restorer(object(), {}, r2_artifact='unused', u96_artifact='unused', head_artifact='unused',
                                 manual_vm_root=ROOT, acknowledgement=M.MANUAL_ACKNOWLEDGEMENT)
        self.assertNotIn('torch', sys.modules)

    def test_guard_and_original_factory_order_from_AST_without_execution(self):
        source = SOURCE.read_text(encoding='utf-8')
        tree = ast.parse(source, feature_version=(3, 10))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'build_restorer')
        guard = next(node.lineno for node in ast.walk(function) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == '_guard_manual_vm')
        torch_import = next(node.lineno for node in ast.walk(function) if isinstance(node, ast.Import)
                            and any(alias.name == 'torch' for alias in node.names))
        self.assertLess(guard, torch_import)
        original_call = source.index('candidate = original.build_restorer')
        seeded_check = source.index('Original seeded zero-head constructor checks must precede terminal restoration')
        terminal_load = source.index('candidate.heads.load_state_dict')
        terminal_check = source.index('validate(candidate, terminal=True)')
        self.assertLess(original_call, seeded_check)
        self.assertLess(seeded_check, terminal_load)
        self.assertLess(terminal_load, terminal_check)
        self.assertNotIn('torch.optim', source)
        self.assertNotIn('torch.load(', source)
        self.assertNotIn('np.load(', source)
        self.assertNotIn('zeros_(', source)
        self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                             and node.func.id == 'candidate' for node in ast.walk(tree)))

    def test_original_factory_and_architecture_byte_pins_remain_exact(self):
        self.assertEqual(M.file_pin(ROOT / M.ORIGINAL_FACTORY['path']),
                         {key: M.ORIGINAL_FACTORY[key] for key in ('bytes', 'sha256')})
        for relative, digest in M.ARCHITECTURE_SOURCES.items():
            self.assertEqual(M.file_pin(ROOT / relative)['sha256'], digest)

    def test_complete_capture_and_inert_contract_source_boundaries(self):
        contract = M.architecture_contract()
        self.assertEqual((contract['total_tensors'], contract['frozen_tensors'], contract['trainable_tensors']), (575, 539, 36))
        self.assertFalse(contract['architecture_changed'])
        self.assertFalse(contract['runtime_parity_gradient_or_quality_verified'])
        self.assertFalse(contract['optimizer_state_loaded'])
        source = SOURCE.read_text(encoding='utf-8')
        self.assertIn('len(captured) == TOTAL_TENSORS', source)
        self.assertIn('value.detach().cpu().clone().contiguous()', source)
        self.assertIn('_tensor_state_sha256(captured, torch=torch)', source)
        self.assertIn('self._forward_lock.release()', source)
        self.assertNotIn('research.codeformer_u96_postfusion_v1.model', sys.modules)
        self.assertNotIn('torch', sys.modules)


class RNGSyntheticState(unittest.TestCase):
    class State:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.uint8).copy()
        def clone(self):
            return RNGSyntheticState.State(self.values)

    def fake_runtime(self):
        state = {'cpu': self.State([1, 2, 3]), 'cuda': [self.State([4, 5, 6])]}
        def set_cpu(value):
            state['cpu'] = value.clone()
        def set_cuda(values):
            state['cuda'] = [value.clone() for value in values]
        runtime = SimpleNamespace(
            get_rng_state=lambda: state['cpu'].clone(), set_rng_state=set_cpu,
            equal=lambda a, b: bool(np.array_equal(a.values, b.values)),
            cuda=SimpleNamespace(get_rng_state_all=lambda: [value.clone() for value in state['cuda']],
                                 set_rng_state_all=set_cuda))
        return runtime, state

    def test_mock_CPU_CUDA_RNG_restoration_success(self):
        runtime, state = self.fake_runtime()
        with M._preserve_runtime_rng(runtime) as receipt:
            state['cpu'] = self.State([0])
            state['cuda'] = [self.State([9])]
        self.assertEqual(receipt, {'CPU_restored_exact': True, 'CUDA_restored_exact': True})
        self.assertEqual(state['cpu'].values.tolist(), [1, 2, 3])
        self.assertEqual(state['cuda'][0].values.tolist(), [4, 5, 6])

    def test_mock_CPU_CUDA_RNG_restoration_on_failure(self):
        runtime, state = self.fake_runtime()
        with self.assertRaisesRegex(RuntimeError, 'fixture stop'):
            with M._preserve_runtime_rng(runtime) as receipt:
                state['cpu'] = self.State([0])
                state['cuda'] = [self.State([9])]
                raise RuntimeError('fixture stop')
        self.assertTrue(receipt['CPU_restored_exact'])
        self.assertTrue(receipt['CUDA_restored_exact'])
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main()
