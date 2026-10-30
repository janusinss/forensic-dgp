"""Installer integration regressions: capture setup and dependency cache identity."""
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from implementation import environment

class ManagedEnvironmentTests(unittest.TestCase):
    def test_dependency_identity_stays_equal_for_ui_only_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); (root/'implementation').mkdir()
            (root/'requirements.txt').write_text('torch==2.13.0\n')
            (root/'implementation/capture_install.py').write_text('capture version 1')
            with patch.object(environment,'ROOT',root):
                first=environment.dependency_identity()
                (root/'templates').mkdir(); (root/'templates/implementation.html').write_text('new UI')
                self.assertEqual(environment.dependency_identity(),first)
                (root/'requirements.txt').write_text('torch==2.13.1\n')
                self.assertNotEqual(environment.dependency_identity(),first)

    def test_capture_failure_prevents_environment_record_publication(self):
        with patch.object(environment.sys,'version_info',(3,13,0)), \
             patch.object(environment,'expected_record',return_value={'environment_root':str(Path(os.__file__).parent),'python':__import__('sys').executable}), \
             patch.object(environment,'check_python'), patch.object(environment,'install_dependencies',return_value=0), \
             patch.object(environment,'install_models',return_value=0), patch.object(environment,'install_capture',return_value=37), \
             patch.object(environment,'write_record') as record, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertNotEqual(environment.install(),0)
            record.assert_not_called()

if __name__=='__main__': unittest.main()
