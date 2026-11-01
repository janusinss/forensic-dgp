"""Managed repair publishes only a completely verified computed generation."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from implementation import environment


class EnvironmentRepairTests(unittest.TestCase):
    def test_generation_record_never_accepts_arbitrary_executable_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'code'; root.mkdir(); (root/'implementation').mkdir()
            (root/'requirements.txt').write_text('demo==1\n')
            (root/'implementation/capture_install.py').write_text('capture')
            managed=Path(temporary)/'runtime'; managed.mkdir()
            with patch.object(environment,'managed_root',return_value=managed):
                record=environment.expected_record(root,temporary,generation='a'*32)
                self.assertEqual(record['schema_version'],2)
                path=Path(record['python']); path.parent.mkdir(parents=True); path.write_bytes(b'fixture')
                config=Path(temporary)/'config.json'; config.write_text(json.dumps(record))
                with patch.object(environment,'_verify_generation_marker'):
                    self.assertEqual(environment.configured_python(config,root,temporary),path)
                    record['python']=str(Path(temporary)/'outside.exe'); config.write_text(json.dumps(record))
                    with self.assertRaises(environment.EnvironmentError): environment.configured_python(config,root,temporary)

    def test_failed_candidate_never_overwrites_prior_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); config=root/'installation.json'; config.write_bytes(b'prior record')
            managed=root/'runtime'; managed.mkdir(); (root/'implementation').mkdir(); (root/'requirements.txt').write_text('demo==1\n'); (root/'implementation/capture_install.py').write_text('capture')
            fake=object()
            with patch.object(environment,'ROOT',root), patch.object(environment,'CONFIG_PATH',config), \
                 patch.object(environment,'managed_root',return_value=managed), \
                 patch.object(environment,'expected_record',side_effect=lambda *a,**kw: {'schema_version':2,'generation':kw.get('generation','b'*32),'environment_root':str(root/'new-generation'),'python':str(root/'new-generation/Scripts/python.exe'),'checkout_root':str(root)}), \
                 patch.object(environment,'check_python'), patch.object(environment.venv,'EnvBuilder') as builder, \
                 patch.object(environment,'run_install_tasks',return_value=37), \
                 patch.object(environment,'write_record') as publish, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(environment._install_offline(fake),37)
                publish.assert_not_called()
                self.assertEqual(config.read_bytes(),b'prior record')

    def test_transitive_lock_change_changes_environment_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); (root/'implementation/locks').mkdir(parents=True)
            (root/'requirements.txt').write_text('demo==1\n'); (root/'implementation/capture_install.py').write_text('capture')
            lock=root/'implementation/locks/windows-cp313-application.txt'; lock.write_text('transitive==1\n')
            first=environment.dependency_identity(root); lock.write_text('transitive==2\n')
            self.assertNotEqual(environment.dependency_identity(root),first)

    def test_invalid_generation_token_is_rejected(self):
        with self.assertRaises(environment.EnvironmentError):
            environment.expected_record(generation='../outside')

if __name__=='__main__': unittest.main()
