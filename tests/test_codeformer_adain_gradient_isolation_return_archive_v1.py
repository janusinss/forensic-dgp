"""Source-bound CPU fixtures for the two AdaIN-specific transport boundaries."""
import ast
from pathlib import Path
import subprocess
import sys
import unittest

from research.codeformer_adain_gradient_isolation_v1 import return_archive as audit


class AdaINTransportRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(Path(audit.__file__).read_text(encoding='utf-8'))

    def png_clause(self, name, size):
        clause = next(node for node in ast.walk(self.tree) if isinstance(node, ast.If) and
                      ast.unparse(node.test) == "suffix == '.png'")
        # Execute exactly the existing auditor clause with inert metadata.
        # No TAR payload, returned code, neural framework or checkpoint loads.
        code = compile(ast.fix_missing_locations(ast.Module(body=clause.body, type_ignores=[])), '<png-cap-fixture>', 'exec')
        scope = {'name': name, 'member': type('Member', (), {'size': size})(),
                 'p': {'limits': {'PNG_file_bytes': 198656}},
                 'OUTPUT': audit.OUTPUT, 'require': audit.require}
        exec(code, scope)

    def test_declared_clean512_example_has_separate_exact_bound(self):
        path = audit.OUTPUT + '/first_training_example/clean512.png'
        self.png_clause(path, 3 * 512 * 512 + 4096)
        with self.assertRaises(ValueError): self.png_clause(path, 3 * 512 * 512 + 4097)

    def test_delivered_and_unlisted_clean512_names_retain_256_cap(self):
        for path in (audit.OUTPUT + '/snapshot_0050/face/native.png',
                     audit.OUTPUT + '/gradient_route_witness/original/face/native.png',
                     audit.OUTPUT + '/other/clean512.png'):
            self.png_clause(path, 198656)
            with self.assertRaises(ValueError): self.png_clause(path, 198657)

    def test_result_accounting_includes_manifest_and_excludes_companions(self):
        statement = next(node for node in ast.walk(self.tree) if isinstance(node, ast.Assign) and
                         any(isinstance(target, ast.Name) and target.id == 'out_records' for target in node.targets))
        code = compile(ast.fix_missing_locations(ast.Module(body=[statement], type_ignores=[])), '<accounting-fixture>', 'exec')
        records = {audit.MANIFEST: {'bytes': 100}, audit.OUTPUT + '/failure.json': {'bytes': 40},
                   audit.OUTPUT + '/snapshot_0000/checkpoint.pt': {'bytes': 200},
                   'protocol.json': {'bytes': 20}, 'outputs/supervision/trainer.log': {'bytes': 30}}
        scope = {'records': records, 'OUTPUT': audit.OUTPUT}
        exec(code, scope)
        actual = scope['out_records']
        self.assertEqual(set(actual), {audit.MANIFEST, audit.OUTPUT + '/failure.json',
                                      audit.OUTPUT + '/snapshot_0000/checkpoint.pt'})
        self.assertEqual(sum(row['bytes'] for row in actual.values()), 340)
        self.assertEqual(sum(row['bytes'] for name, row in actual.items() if audit.metadata(name)), 140)

    def test_auditor_import_has_no_scientific_framework_or_io(self):
        code = ('import sys; from research.codeformer_adain_gradient_isolation_v1 import return_archive; '
                'assert not any(n in sys.modules for n in ("torch", "numpy", "PIL", "cv2"))')
        result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__': unittest.main()
