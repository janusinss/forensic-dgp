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

class CrossCommitReuseTests(unittest.TestCase):
    def fixture(self):
        temporary=tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        base=Path(temporary.name); runtime=base/'runtime'
        old=runtime/'versions'/('a'*40); new=runtime/'versions'/('b'*40)
        for source in (old,new):
            (source/'implementation').mkdir(parents=True)
            (source/'requirements.txt').write_text('demo==1\n')
            (source/'implementation/capture_install.py').write_text('same capture implementation')
        identity=environment.dependency_identity(new); generation='c'*32
        folder=runtime/'environments'/(identity+'-g'+generation)
        (folder/'Scripts').mkdir(parents=True); (folder/'Scripts/python.exe').write_bytes(b'fixture')
        (folder/'.dgp-environment.json').write_text(json.dumps({'schema':1,'dependency_identity':identity,'generation':generation,'state':'ready'}))
        (runtime/'data/installations').mkdir(parents=True)
        (runtime/'current.json').write_text(json.dumps({'schema':1,'commit':'a'*40}))
        prior=environment._record_for_path(old,folder,generation)
        (runtime/'data/installations'/('a'*40+'.json')).write_text(json.dumps(prior))
        return base,runtime,old,new,folder,generation

    def test_ui_only_commit_reuses_ready_generation_after_full_verification(self):
        base,runtime,old,new,folder,generation=self.fixture(); calls=[]
        def verify(python,**kwargs):
            self.assertEqual(python,folder/'Scripts/python.exe')
            self.assertEqual(os.environ.get('DGP_SETUP_GENERATION'),generation)
            calls.append(kwargs.get('include_models',True)); return 0
        with patch.dict(os.environ,{'LOCALAPPDATA':str(base)},clear=False),patch.object(environment,'ROOT',new),patch.object(environment,'managed_root',return_value=runtime),patch.object(environment,'verify_runtime',side_effect=verify),patch.object(environment,'install_models',return_value=0),patch.object(environment.venv,'EnvBuilder') as create,patch.object(environment,'write_record') as publish,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(environment._install_offline(object()),0)
        create.assert_not_called(); self.assertEqual(calls,[False,True])
        record=publish.call_args.args[0]
        self.assertEqual(record['checkout_root'],str(new.resolve()))
        self.assertEqual(record['generation'],generation)
        self.assertEqual(record['environment_root'],str(folder))

    def test_active_previous_generation_is_preferred_over_other_ready_candidates(self):
        base,runtime,old,new,folder,generation=self.fixture()
        other='0'*32; another=folder.with_name(folder.name[:-32]+other)
        (another/'Scripts').mkdir(parents=True); (another/'Scripts/python.exe').write_bytes(b'fixture')
        marker=json.loads((folder/'.dgp-environment.json').read_text()); marker['generation']=other
        (another/'.dgp-environment.json').write_text(json.dumps(marker))
        with patch.dict(os.environ,{'LOCALAPPDATA':str(base)}),patch.object(environment,'ROOT',new),patch.object(environment,'managed_root',return_value=runtime):
            candidates=environment._reuse_candidates(None)
        self.assertEqual(candidates[0]['generation'],generation)
        self.assertEqual(candidates[1]['generation'],other)

    def test_reuse_scan_is_bounded(self):
        base,runtime,old,new,folder,generation=self.fixture()
        for index in range(1024):
            (runtime/'environments'/('unrecognized-'+str(index))).write_bytes(b'preserved')
        with patch.dict(os.environ,{'LOCALAPPDATA':str(base)}),patch.object(environment,'ROOT',new),patch.object(environment,'managed_root',return_value=runtime):
            with self.assertRaisesRegex(environment.EnvironmentError,'bounded'):
                environment._reuse_candidates(None)
        self.assertTrue((runtime/'environments'/'unrecognized-0').is_file())

    def test_invalid_ready_marker_is_preserved_and_not_selected(self):
        base,runtime,old,new,folder,generation=self.fixture()
        marker=folder/'.dgp-environment.json'; original=marker.read_bytes(); marker.write_text(original.decode().replace('ready','preparing'))
        with patch.dict(os.environ,{'LOCALAPPDATA':str(base)}),patch.object(environment,'ROOT',new),patch.object(environment,'managed_root',return_value=runtime):
            candidates=environment._reuse_candidates(None)
        self.assertEqual(candidates,[])
        self.assertIn('preparing',marker.read_text())

    def test_previous_config_cannot_redirect_reuse_to_arbitrary_python(self):
        base,runtime,old,new,folder,generation=self.fixture()
        config=runtime/'data/installations'/('a'*40+'.json')
        value=json.loads(config.read_text()); value['python']=str(base/'outside.exe'); config.write_text(json.dumps(value))
        with patch.dict(os.environ,{'LOCALAPPDATA':str(base)}),patch.object(environment,'ROOT',new),patch.object(environment,'managed_root',return_value=runtime):
            candidates=environment._reuse_candidates(None)
        self.assertEqual(len(candidates),1)
        self.assertEqual(candidates[0]['python'],str(folder/'Scripts/python.exe'))

if __name__=='__main__': unittest.main()
