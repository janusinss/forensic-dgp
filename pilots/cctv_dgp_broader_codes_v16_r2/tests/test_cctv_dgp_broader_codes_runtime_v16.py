"""VM entry, bounded cache, numeric and archive regressions; zero backward/updates."""
import ast
from contextlib import contextmanager
import importlib.util
import io
from pathlib import Path
import tarfile
import shutil
import unittest
from unittest.mock import patch
import uuid

import numpy as np
import cctv_dgp_broader_codes_v16 as v

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def scratch_directory(prefix):
    # Windows sandbox forbids tempfile's owner-only mode700; use inherited workspace ACLs.
    path = ROOT / 'scratch' / (prefix + uuid.uuid4().hex)
    path.mkdir()
    try: yield str(path)
    finally:
        if path.resolve().is_relative_to((ROOT / 'scratch').resolve()) and path.name.startswith(prefix):
            shutil.rmtree(path)


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class RuntimeBoundaries(unittest.TestCase):
    def test_all_neural_job_entries_reject_local_before_writing_or_launching(self):
        with scratch_directory('v16-entry-') as name:
            target = Path(name) / 'not-created'
            with self.assertRaisesRegex(RuntimeError, 'VM'):
                load('train_cctv_dgp_broader_codes_v16').train(target, ROOT, ROOT, ROOT, 'x')
            with self.assertRaisesRegex(RuntimeError, 'VM'):
                load('supervise_cctv_dgp_broader_codes_v16').supervise(target, ROOT, ROOT, ROOT, 'x')
            with self.assertRaisesRegex(RuntimeError, 'VM'):
                load('launch_cctv_dgp_broader_codes_v16').launch('x', 'x')
            self.assertFalse(target.exists())

    def test_streamed_cache_rejects_corruption_and_wrong_field_schema(self):
        with scratch_directory('v16-cache-') as name:
            root = Path(name); path = v.cache_path(root, 'case'); path.parent.mkdir()
            values = dict(dgp=np.zeros((3, 256, 256), np.float32),
                          features=np.zeros((256, 16, 16), np.float32), logits=np.zeros((256, 1024), np.float32))
            np.savez(path, **values); manifest = {'case': {'sha256': v.sha(path)}}
            self.assertEqual([x.shape for x in v.load_cache(root, 'case', manifest)], [x.shape for x in values.values()])
            with path.open('ab') as f: f.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'fingerprint'): v.load_cache(root, 'case', manifest)
            np.savez(path, **values, forbidden_labels=np.zeros(256, np.int64))
            manifest['case']['sha256'] = v.sha(path)
            with self.assertRaisesRegex(ValueError, 'fields'): v.load_cache(root, 'case', manifest)

    def test_observed_token_ce_stays_finite_and_excludes_unobserved_tokens(self):
        x = np.zeros((256, 1024), np.float32); labels = np.zeros(256, np.int64)
        mask = np.zeros(256, bool); mask[:2] = True
        stats = v.code_metrics(x, labels, mask)
        self.assertAlmostEqual(stats['code_ce'], np.log(1024), places=10)
        x[2:, 12] = 1e30
        self.assertEqual(stats, v.code_metrics(x, labels, mask))
        x[:2, 0] = 10000
        self.assertEqual(v.code_metrics(x, labels, mask), {'code_ce': 0., 'code_accuracy': 1.})
        with self.assertRaisesRegex(ValueError, 'code metric'): v.code_metrics(x, labels, np.zeros(256, bool))

    def test_safe_import_rejects_traversal_links_duplicate_and_aliased_names(self):
        for names, link in [(['../outside'], False), (['/outside'], False), (['C:/outside'], False),
                            (['a\\b'], False), (['good', 'good'], False), (['a//b'], False), (['link'], True)]:
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode='w') as stream:
                for name in names:
                    info = tarfile.TarInfo(name)
                    if link: info.type = tarfile.SYMTYPE; info.linkname = '/outside'
                    stream.addfile(info)
            buffer.seek(0)
            with self.subTest(names=names), tarfile.open(fileobj=buffer) as stream, self.assertRaises(ValueError):
                v.archive_members(stream, ROOT / 'scratch/never-extract-v16', 1024)

    def test_failure_import_refuses_overwrite_and_hash_mismatch(self):
        importer = load('import_cctv_dgp_broader_codes_v16')
        with self.assertRaisesRegex(ValueError, 'Preserve'):
            importer.collect(ROOT / 'outputs/nonexistent.tar.gz', ROOT / 'outputs/nonexistent.json', ROOT / 'outputs')
        with patch.object(v, 'sha', return_value='pin'), patch.object(v, 'read', return_value={
                'complete': True, 'protocol_sha256': 'pin', 'archive_sha256': 'other', 'bytes': 1}):
            with self.assertRaisesRegex(ValueError, 'hash'):
                importer.collect(ROOT / 'outputs/nonexistent.tar.gz', ROOT / 'outputs/nonexistent.json',
                                 ROOT / 'outputs/nonexistent-v16-test-return')

    def test_python310_syntax_and_audit_has_no_neural_imports(self):
        for name in v.SOURCES_FILES:
            with self.subTest(file=name): ast.parse((ROOT / name).read_text(), feature_version=(3, 10))
        tree = ast.parse((ROOT / 'scripts/audit_cctv_dgp_broader_codes_v16.py').read_text())
        imports = [n.name for node in ast.walk(tree) if isinstance(node, ast.Import) for n in node.names]
        imports += [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        self.assertFalse(any(x and ('torch' in x or x.startswith('dgp_') or x.startswith('models')) for x in imports))

    def test_failed_return_audits_partial_trace_and_rejects_heldout_exposure(self):
        auditor = load('audit_cctv_dgp_broader_codes_v16')
        step = {'update': 1, 'epoch': 1, 'case_ids': ['train'] * 10}
        row = {**step, 'loss': 2., 'code_ce': 2., 'code_accuracy': .1, 'gradient_norm_before_clip': 1., 'seconds': 1.}
        with scratch_directory('v16-failure-') as name:
            root = Path(name); out = root / 'output'; out.mkdir()
            v.write(root / 'schedule_v16.json', {'steps': [step]})
            v.write(root / 'supervisor_failure.json', {'complete': False, 'protocol_sha256': 'pin', 'resume_permitted': False})
            v.write(out / 'failure.json', {'complete': False, 'resume_permitted': False, 'optimizer_updates': 1, 'backward_calls': 2})
            import json
            trace = out / 'update_trace.jsonl'; trace.write_text(json.dumps(row) + '\n')
            with patch.object(v, 'verify', return_value={}):
                auditor.audit_failure(root, root, root, root, 'pin', out, root / 'supervisor_failure.json', root / 'audit.json')
                self.assertEqual(v.read(root / 'audit.json')['completed_trace_records_checked'], 1)
                row['case_ids'] = ['validation'] * 10; trace.write_text(json.dumps(row) + '\n')
                with self.assertRaisesRegex(ValueError, 'trace differs'):
                    auditor.audit_failure(root, root, root, root, 'pin', out, root / 'supervisor_failure.json', root / 'bad-audit.json')


if __name__ == '__main__': unittest.main()
