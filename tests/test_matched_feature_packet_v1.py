"""Bounded transport primitives; synthetic bytes only, no VM/model execution."""
import ast
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


class Blocker(importlib.abc.MetaPathFinder):
    attempted = []
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'PIL', 'tensorflow', 'jax'} or fullname.endswith('.worker'):
            self.attempted.append(fullname)
            raise RuntimeError('Forbidden framework/model/worker import: ' + fullname)
        return None


BLOCKER = Blocker()
sys.meta_path.insert(0, BLOCKER)
from research.codeformer_matched_feature_learning_v1 import packet as P
from research.codeformer_matched_feature_learning_v1 import install_weights as I

ROOT = Path(__file__).resolve().parents[1]


class PacketPrimitives(unittest.TestCase):
    def setUp(self):
        (ROOT / 'scratch').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='matched_packet_unit_', dir=ROOT / 'scratch')
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_relative_paths_have_one_canonical_namespace(self):
        for value in ('data/inputs/x.png', 'research/a.py', 'weights/a.pth'):
            self.assertEqual(P.relative(value).as_posix(), value)
        for value in ('', '.', '..', '../x', 'x/../y', '/x', 'C:/x', 'x\\y', 'x//y',
                      './x', 'x/', 'x\x00y', 'x\ny', 3, True, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                P.relative(value)

    def test_strict_JSON_rejects_duplicate_nonfinite_overflow_and_deep_objects(self):
        for raw in (b'{"x":1,"x":2}', b'{"a":{"x":1,"x":2}}', b'[NaN]', b'[Infinity]',
                    b'[-Infinity]', b'{"x":1e999}', b'[' * 66 + b'0' + b']' * 66):
            with self.subTest(raw=raw[:30]), self.assertRaises(ValueError):
                P.strict_json(raw)
        for raw in ('{}', bytearray(b'{}')):
            with self.assertRaises(ValueError):
                P.strict_json(raw)
        self.assertEqual(P.strict_json(b'{"x":1,"y":[true,null,-0.0]}'), {'x': 1, 'y': [True, None, -0.0]})

    def test_pin_hashes_exact_regular_bytes_inside_root(self):
        path = self.root / 'a.bin'; path.write_bytes(b'known\x00payload\n')
        expected = {'path': 'a.bin', 'bytes': 14, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        self.assertEqual(P.pin(self.root, path), expected)
        self.assertEqual(P.sha256(path), expected['sha256'])
        outside = self.root.parent / (self.root.name + '_outside.bin')
        outside.write_bytes(b'x'); self.addCleanup(outside.unlink)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            P.pin(self.root, outside)

    def test_hardlinked_regular_files_are_refused(self):
        first = self.root / 'a'; second = self.root / 'b'; first.write_bytes(b'x')
        try:
            os.link(first, second)
        except OSError as exc:
            self.skipTest('Filesystem has no hard-link support: ' + str(exc))
        for path in (first, second):
            with self.assertRaisesRegex(ValueError, 'single-link'):
                P.regular(path)

    def test_linked_ancestry_and_symlink_file_are_refused_when_supported(self):
        original = self.root / 'original'; original.mkdir(); (original / 'a').write_bytes(b'x')
        alias = self.root / 'alias'
        try:
            os.symlink(original, alias, target_is_directory=True)
        except OSError as exc:
            self.skipTest('Symlink privilege unavailable: ' + str(exc))
        with self.assertRaises(ValueError):
            P.directory(alias)
        with self.assertRaises(ValueError):
            P.regular(alias / 'a')

    def test_mocked_hardlink_count_is_refused_without_filesystem_link_privilege(self):
        path = self.root / 'mocked_link'; path.write_bytes(b'x')
        original = Path.lstat
        info = path.lstat()
        def linked(item, *args, **kwargs):
            if item == path:
                return SimpleNamespace(st_mode=info.st_mode, st_size=info.st_size,
                                       st_nlink=2, st_file_attributes=0)
            return original(item, *args, **kwargs)
        with mock.patch.object(Path, 'lstat', linked):
            for guard in (P.regular, I.ordinary):
                with self.subTest(guard=guard.__module__), self.assertRaisesRegex(ValueError, 'single-link'):
                    guard(path)

    def test_mocked_symlink_file_and_ancestry_are_refused_without_privilege(self):
        path = self.root / 'mocked_symlink'; path.write_bytes(b'x')
        original = Path.lstat
        for target, directory in ((path, False), (self.root, True)):
            def linked(item, *args, **kwargs):
                if item == target:
                    return SimpleNamespace(st_mode=stat.S_IFLNK | 0o777, st_size=1,
                                           st_nlink=1, st_file_attributes=0)
                return original(item, *args, **kwargs)
            with self.subTest(directory=directory), mock.patch.object(Path, 'lstat', linked):
                with self.assertRaises(ValueError):
                    P.directory(target) if directory else P.regular(target)
                with self.assertRaises(ValueError):
                    I.ordinary(target, directory=directory)

    def test_mocked_Windows_reparse_file_and_ancestry_are_refused(self):
        path = self.root / 'mocked_reparse'; path.write_bytes(b'x')
        original = Path.lstat
        for target, directory in ((path, False), (self.root, True)):
            info = original(target)
            def reparse(item, *args, **kwargs):
                if item == target:
                    return SimpleNamespace(st_mode=info.st_mode, st_size=info.st_size,
                                           st_nlink=1, st_file_attributes=1024)
                return original(item, *args, **kwargs)
            with self.subTest(directory=directory), mock.patch.object(Path, 'lstat', reparse):
                with self.assertRaises(ValueError):
                    P.directory(target) if directory else P.regular(target)
                with self.assertRaises(ValueError):
                    I.ordinary(target, directory=directory)

    def test_file_changed_during_hash_is_refused(self):
        path = self.root / 'a'; path.write_bytes(b'a' * 20)
        actual_open = Path.open
        def changed_open(item, *args, **kwargs):
            if item == path and args and args[0] == 'rb':
                with actual_open(item, 'wb') as stream:
                    stream.write(b'b' * 21)
            return actual_open(item, *args, **kwargs)
        with mock.patch.object(Path, 'open', changed_open), self.assertRaisesRegex(ValueError, 'changed during hashing'):
            P.sha256(path)

    def test_binding_requires_manifest_identity_and_exact_actual_content(self):
        path = self.root / 'a'; path.write_bytes(b'test')
        binding = P.pin(self.root, path)
        table = {'a': {k: binding[k] for k in ('bytes', 'sha256')}}
        P._binding(self.root, table, binding)
        for item in (dict(binding, extra=True), dict(binding, sha256='0' * 64), dict(binding, path='unknown')):
            with self.subTest(item=item), self.assertRaises(ValueError):
                P._binding(self.root, table, item)
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Bound artifact differs'):
            P._binding(self.root, table, binding)

    def test_install_mode_omits_only_explicit_missing_binding(self):
        binding = {'path': 'a', 'bytes': 1, 'sha256': hashlib.sha256(b'x').hexdigest()}
        files = {'a': {'bytes': 1, 'sha256': binding['sha256']}}
        P._binding(self.root, files, binding, omitted={'a'})
        with self.assertRaises(FileNotFoundError):
            P._binding(self.root, files, binding)
        (self.root / 'a').write_bytes(b'bad')
        with self.assertRaises(ValueError):
            P._binding(self.root, files, binding, omitted={'a'})

    def test_protocol_hash_is_checked_before_decode_or_any_helper_import(self):
        (self.root / 'protocol.json').write_bytes(b'not JSON')
        with (mock.patch.object(P, 'strict_json', side_effect=AssertionError('Decoded before SHA')),
              mock.patch.object(P.importlib, 'import_module', side_effect=AssertionError('Imported before SHA'))):
            with self.assertRaisesRegex(ValueError, 'Exact supplied protocol SHA256'):
                P.verify_static(self.root, '0' * 64)

    def test_wrong_packet_type_and_manual_flags_fail_before_import(self):
        base = {'format': P.FORMAT, 'stage_name': P.STAGE_NAME, 'output': P.OUTPUT_RELATIVE,
                'manual_launch_only': True, 'automatic_retry_resume': False,
                'model_qualified': False, 'training_success_not_implied': True}
        for field, value in (('format', 'historical'), ('manual_launch_only', False),
                             ('automatic_retry_resume', True), ('model_qualified', True)):
            changed = dict(base); changed[field] = value
            raw = (json.dumps(changed) + '\n').encode(); (self.root / 'protocol.json').write_bytes(raw)
            with self.subTest(field=field), mock.patch.object(P.importlib, 'import_module', side_effect=AssertionError('Unexpected import')):
                with self.assertRaises(ValueError):
                    P.verify_static(self.root, hashlib.sha256(raw).hexdigest())

    def test_CRLF_or_missing_final_newline_cannot_be_new_sealed_protocol(self):
        for raw in (b'{}\r\n', b'{}'):
            (self.root / 'protocol.json').write_bytes(raw)
            with self.assertRaisesRegex(ValueError, 'LF/final newline'):
                P.verify_static(self.root, hashlib.sha256(raw).hexdigest())

    def test_bounded_verified_copy_is_exclusive_and_preserves_source(self):
        source = self.root / 'source'; target = self.root / 'target'; source.write_bytes(b'payload\n')
        binding = {'bytes': source.stat().st_size, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
        guard = mock.Mock()
        I.copy_verified(source, target, binding, guard)
        self.assertEqual(target.read_bytes(), source.read_bytes())
        self.assertEqual(guard.call_count, 2)
        with self.assertRaises(FileExistsError):
            I.copy_verified(source, target, binding, guard)

    def test_wrong_copy_hash_stops_before_creating_destination(self):
        source = self.root / 'source'; source.write_bytes(b'x'); target = self.root / 'target'
        with self.assertRaisesRegex(ValueError, 'differs before copy'):
            I.copy_verified(source, target, {'bytes': 1, 'sha256': '0' * 64}, lambda: None)
        self.assertFalse(target.exists())

    def test_midcopy_failure_retains_partial_file_and_refuses_retry(self):
        source = self.root / 'source'; target = self.root / 'target'; source.write_bytes(b'x' * 32)
        binding = {'bytes': 32, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
        def stop():
            raise TimeoutError('Synthetic finite stop')
        with self.assertRaises(TimeoutError):
            I.copy_verified(source, target, binding, stop)
        self.assertTrue(target.exists())
        self.assertEqual(target.stat().st_size, 0)
        self.assertEqual(source.read_bytes(), b'x' * 32)
        with self.assertRaises(FileExistsError):
            I.copy_verified(source, target, binding, lambda: None)

    def bootstrap_fixture(self):
        names = ('research/__init__.py', I.PACKAGE + '/__init__.py',
                 I.PACKAGE + '/packet.py', I.PACKAGE + '/install_weights.py')
        files = {}
        for index, name in enumerate(names):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = ('# synthetic bootstrap source ' + str(index) + '\n').encode('ascii')
            path.write_bytes(raw)
            files[name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        return names, {'files': files}

    def test_bootstrap_authenticates_all_four_small_sources_without_importing_them(self):
        names, protocol = self.bootstrap_fixture()
        with mock.patch.object(I, 'digest', wraps=I.digest) as digests, mock.patch.object(
                I.importlib, 'import_module', side_effect=AssertionError('Fixture source import forbidden')) as imported:
            I.authenticate_bootstrap(self.root, protocol)
        self.assertEqual([call.args[0] for call in digests.call_args_list],
                         [self.root / name for name in names])
        imported.assert_not_called()

    def test_modified_parent_initializer_is_refused_before_staged_import(self):
        _, protocol = self.bootstrap_fixture()
        (self.root / 'research/__init__.py').write_bytes(b'# altered parent\n')
        with mock.patch.object(I.importlib, 'import_module') as imported:
            with self.assertRaisesRegex(ValueError, 'Sealed bootstrap source differs: research/__init__'):
                I.authenticate_bootstrap(self.root, protocol)
            imported.assert_not_called()

    def test_each_bootstrap_source_missing_pin_or_wrong_digest_is_refused(self):
        names, protocol = self.bootstrap_fixture()
        for name in names:
            for missing in (False, True):
                changed = json.loads(json.dumps(protocol))
                if missing:
                    del changed['files'][name]
                else:
                    changed['files'][name]['sha256'] = '0' * 64
                with self.subTest(name=name, missing=missing), self.assertRaisesRegex(
                        ValueError, 'Sealed bootstrap source differs'):
                    I.authenticate_bootstrap(self.root, changed)

    def test_bootstrap_authentication_call_precedes_first_package_import_in_install_AST(self):
        tree = ast.parse(Path(I.__file__).read_text(), feature_version=(3, 10))
        install = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_install')
        authentication = next(node.lineno for node in ast.walk(install) if isinstance(node, ast.Call)
                              and isinstance(node.func, ast.Name) and node.func.id == 'authenticate_bootstrap')
        staged_import = min(node.lineno for node in ast.walk(install) if isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Attribute) and node.func.attr == 'import_module')
        self.assertLess(authentication, staged_import)

    def test_installer_and_packet_are_inert_sources(self):
        for module in (P, I):
            tree = ast.parse(Path(module.__file__).read_text(), feature_version=(3, 10))
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(a.name for a in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.append(node.module or '')
            self.assertFalse(any(n.split('.')[0] in {'torch', 'numpy', 'PIL', 'tensorflow', 'jax'} or n.endswith('.worker') for n in imports))
        self.assertEqual(BLOCKER.attempted, [])
        self.assertFalse({'torch', 'numpy', 'PIL', 'tensorflow', 'jax'} & {n.split('.')[0] for n in sys.modules})


if __name__ == '__main__':
    unittest.main()
