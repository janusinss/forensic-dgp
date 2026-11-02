"""Tiny exporter evidence tests; fake bytes only, no models/training/VM."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('completion_finetune_export',ROOT/'scripts/export_completion_codeformer_finetune_v1.py')
exporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(exporter)

class ExporterTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='completion-export-fixture-');self.addCleanup(self.temp.cleanup)
        self.home=Path(self.temp.name);self.root=self.home/'forensic-dgp'/exporter.VM_NAME;self.root.mkdir(parents=True)
        (self.root/'protocol.json').write_text('{"format":"tiny-fixture"}\n',encoding='utf-8')
        (self.root/'supervisor.json').write_text(json.dumps({'complete':True,'worker_exit':2,'timeout':False}),encoding='utf-8')
        (self.root/'trainer.log').write_bytes(b'failed-quality-screen retained\n')
        out=self.root/'outputs';out.mkdir();(out/'terminal_training_state.pth.partial').write_bytes(b'not-a-resumable-model-fixture')
        (out/'failure.json').write_text('{"complete":false}\n',encoding='utf-8')
        (self.root/'weights').mkdir();(self.root/'weights/original.pth').write_bytes(b'excluded-original-fixture')
        (self.root/'excluded_source.py').write_bytes(b'# excluded source fixture\n')
        fake_os=types.SimpleNamespace(name='posix',fsync=os.fsync,replace=os.replace)
        context=patch.object(exporter,'os',fake_os);context.start();self.addCleanup(context.stop)
    def export(self):
        with contextlib.redirect_stdout(io.StringIO()):return exporter.export(self.root,exporter.sha(self.root/'protocol.json'),self.home)
    def test_failed_worker_all_evidence_included_base_assets_excluded_LF(self):
        result=self.export();self.assertEqual(result['worker_exit'],2);self.assertFalse(result['terminal_state_committed'])
        archive=self.home/(exporter.STEM+'-results.tar.gz')
        self.assertEqual(result['archive_sha256'],hashlib.sha256(archive.read_bytes()).hexdigest())
        checksum=Path(str(archive)+'.sha256').read_bytes();self.assertNotIn(b'\r',checksum);self.assertTrue(checksum.endswith(b'\n'))
        with tarfile.open(archive,'r:gz') as tar:
            names=tar.getnames();self.assertTrue(all(m.isfile() for m in tar.getmembers()))
            self.assertFalse(any('/weights/' in n or n.endswith('.py') for n in names))
            manifest=json.load(tar.extractfile(exporter.MEMBER_ROOT+'/export_manifest.json'))
            self.assertEqual(manifest['partial_files_retained'],['outputs/terminal_training_state.pth.partial'])
            self.assertFalse(manifest['model_qualified']);self.assertTrue(manifest['training_success_not_implied'])
            for row in manifest['files']:
                data=tar.extractfile(exporter.MEMBER_ROOT+'/'+row['path']).read()
                self.assertEqual(len(data),row['bytes']);self.assertEqual(hashlib.sha256(data).hexdigest(),row['sha256'])
    def test_existing_export_refused_unchanged(self):
        self.export();archive=self.home/(exporter.STEM+'-results.tar.gz');before=exporter.sha(archive)
        with self.assertRaisesRegex(ValueError,'refusing overwrite'):self.export()
        self.assertEqual(before,exporter.sha(archive))
    def test_byte_cap_refuses_before_archive(self):
        with patch.object(exporter,'MAX_BYTES',1):
            with self.assertRaisesRegex(ValueError,'Uncompressed evidence exceeds'):self.export()
        self.assertFalse((self.home/(exporter.STEM+'-results.tar.gz')).exists())

if __name__=='__main__':unittest.main(verbosity=2)
