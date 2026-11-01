"""Chosen installation destinations remain owned and isolated from legacy data."""
import contextlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from implementation import managed_paths
from tests import test_inno_backend as backend_tests
COMMIT = backend_tests.COMMIT


class CustomDestinationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.local=Path(self.temp.name)
        self.shell=self.local/'Forensic DGP ä'
        self.root=self.shell/'runtime'
        self.code=self.root/'versions'/('a'*40)
        self.code.mkdir(parents=True)
        (self.shell/'shell-owner.json').write_text('{"schema":1,"application":"Forensic DGP"}')
        self.marker={'schema_version':2,'repository':'janusinss/forensic-dgp','layout':'desktop-runtime-v1','install_root':str(self.shell)}
        (self.root/'owner.json').write_text(json.dumps(self.marker))
        self.variables={'DGP_INSTALL_ROOT':str(self.shell),'DGP_MANAGED_ROOT':str(self.root),'LOCALAPPDATA':str(self.local)}
        self.patch=patch.dict(os.environ,self.variables)
        self.patch.start(); self.addCleanup(self.patch.stop)

    def test_custom_owned_root_binds_code_models_and_records(self):
        self.assertEqual(managed_paths.data_root(self.code),self.root/'data')
        self.assertEqual(managed_paths.data_path(self.code,'models','implementation'),self.root/'data/models/implementation')
        with patch.object(managed_paths,'__file__',str(self.code/'implementation/managed_paths.py')):
            self.assertEqual(managed_paths.managed_root(),self.root)

    def test_schema2_requires_exact_shell_binding_and_declared_contract(self):
        for mutation in ({'install_root':str(self.local/'other')},{'schema_version':True},{'layout':'other'},{'extra':1}):
            marker=dict(self.marker,**mutation)
            (self.root/'owner.json').write_text(json.dumps(marker))
            with self.subTest(mutation=mutation), self.assertRaises(managed_paths.ManagedPathError):
                managed_paths.data_root(self.code)
        (self.root/'owner.json').write_text(json.dumps(self.marker))
        with patch.dict(os.environ,{'DGP_INSTALL_ROOT':str(self.local/'wrong')}):
            with self.assertRaises(managed_paths.ManagedPathError): managed_paths.data_root(self.code)
        with patch.dict(os.environ,{},clear=True):
            os.environ.update(LOCALAPPDATA=str(self.local),DGP_MANAGED_ROOT=str(self.root))
            with self.assertRaises(managed_paths.ManagedPathError): managed_paths.data_root(self.code)

    def test_custom_environment_cannot_redirect_an_unrelated_checkout(self):
        for code in (self.local/'source',self.root/'versions'/'not-a-commit',self.root/'versions'/('a'*40)/'nested'):
            with self.subTest(code=code), self.assertRaises(managed_paths.ManagedPathError): managed_paths.data_root(code)

    def test_shell_marker_is_required_and_default_assets_are_untouched(self):
        legacy=self.local/'DGP/app/data/models'
        legacy.mkdir(parents=True); asset=legacy/'preserve.pth'; asset.write_bytes(b'original model')
        (self.shell/'shell-owner.json').unlink()
        with self.assertRaises(managed_paths.ManagedPathError): managed_paths.data_root(self.code)
        self.assertEqual(asset.read_bytes(),b'original model')

    def test_custom_marker_is_rejected_at_wrong_runtime_sibling(self):
        other=self.shell/'other'; other.mkdir(); (other/'owner.json').write_text(json.dumps(self.marker))
        with patch.dict(os.environ,{'DGP_MANAGED_ROOT':str(other)}):
            with self.assertRaises(managed_paths.ManagedPathError): managed_paths.data_root(self.code)


class CustomPowerShellDestinationTests(backend_tests.InnoBundleTests):
    def setUp(self):
        super().setUp()
        self.shell=self.local/'Chosen DGP ä'
        self.shell.mkdir()
        (self.shell/'shell-owner.json').write_text('{"schema":1,"application":"Forensic DGP"}',encoding='utf8')
        self.root=self.shell/'runtime'
        self.env['DGP_INSTALL_ROOT']=str(self.shell)

    def prior(self,commit='b'*40):
        pointer,content=super().prior(commit)
        (self.root/'owner.json').write_text(json.dumps({'schema_version':2,'repository':'janusinss/forensic-dgp','layout':'desktop-runtime-v1','install_root':str(self.shell)}))
        return pointer,content

    def test_missing_shell_owner_is_refused(self):
        (self.shell/'shell-owner.json').unlink()
        result=self.run_install()
        self.assertNotEqual(result.returncode,0)
        self.assertFalse((self.root/'current.json').exists())

    def test_shell_runtime_mismatch_is_refused(self):
        self.env['DGP_INSTALL_ROOT']=str(self.local)
        result=self.run_install()
        self.assertNotEqual(result.returncode,0)
        self.assertFalse((self.root/'current.json').exists())

    def test_custom_install_keeps_legacy_models_untouched(self):
        legacy=self.local/'DGP/app/data/models'; legacy.mkdir(parents=True)
        model=legacy/'preserved.pth'; model.write_bytes(b'original model')
        result=self.run_install()
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        marker=json.loads((self.root/'owner.json').read_text(encoding='utf-8-sig'))
        self.assertEqual(marker['schema_version'],2)
        self.assertEqual(marker['install_root'],str(self.shell))
        self.assertEqual(model.read_bytes(),b'original model')


class CustomPrivateBootstrapTests(CustomDestinationTests):
    def run_bootstrap(self, *, valid=True):
        import shutil
        import subprocess
        import sys
        import venv
        implementation=self.code/'implementation'; implementation.mkdir()
        original=Path(__file__).resolve().parents[1]/'implementation'
        for name in ('bootstrap_python.ps1','install_paths.ps1'):
            shutil.copyfile(original/name,implementation/name)
        target=self.root/'python/python-3.13.9-amd64'
        if valid:
            venv.EnvBuilder(with_pip=False).create(target)
            shutil.copyfile(target/'Scripts/python.exe',target/'python.exe')
        else:
            target.mkdir(parents=True)
            (target/'python.exe').write_bytes(b'preserve invalid private Python')
        environment=dict(os.environ,**self.variables)
        result=subprocess.run([str(backend_tests.PS),'-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass',
            '-File',str(implementation/'bootstrap_python.ps1'),'-ManagedRoot',str(self.root),'-VerifyOnly'],
            capture_output=True,text=True,env=environment,timeout=20)
        return result,target

    @unittest.skipUnless(os.name=='nt','Windows private runtime fixture')
    def test_custom_private_python_is_verified_under_selected_root(self):
        result,target=self.run_bootstrap()
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('without downloads',result.stdout)
        self.assertFalse((self.local/'DGP/python-3.13.9-amd64').exists())

    @unittest.skipUnless(os.name=='nt','Windows private runtime fixture')
    def test_invalid_custom_private_python_is_preserved_without_global_fallback(self):
        result,target=self.run_bootstrap(valid=False)
        self.assertNotEqual(result.returncode,0)
        self.assertEqual((target/'python.exe').read_bytes(),b'preserve invalid private Python')


class CustomPackageCacheTests(unittest.TestCase):
    def test_selected_runtime_overrides_inherited_global_package_caches(self):
        import subprocess
        from implementation import environment
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); calls=[]
            def cache(name): return root/'cache'/name
            with patch.object(environment,'custom_install_root',return_value=root.parent), \
                 patch.object(environment,'managed_cache',side_effect=cache), \
                 patch.object(environment.subprocess,'run',return_value=subprocess.CompletedProcess([],1)), \
                 patch.object(environment,'run_foreground',side_effect=lambda command,**kwargs: calls.append((command,kwargs)) or 0), \
                 patch.dict(os.environ,{'UV_CACHE_DIR':'C:/global/uv','PIP_CACHE_DIR':'C:/global/pip'}):
                self.assertEqual(environment.install_dependencies(root/'python.exe'),0)
            for command,options in calls:
                self.assertEqual(options['env']['UV_CACHE_DIR'],str(cache('uv')))
                self.assertEqual(options['env']['PIP_CACHE_DIR'],str(cache('pip')))
            bootstrap=calls[0][0]
            self.assertEqual(bootstrap[bootstrap.index('--cache-dir')+1],str(cache('pip')))


class InstalledModuleBindingTests(unittest.TestCase):
    @unittest.skipUnless(os.name=='nt','Windows installed-layout import')
    def test_actual_environment_import_under_chosen_localappdata_layout(self):
        import shutil
        import subprocess
        import sys
        # A real LocalAppData layout exposes Windows packaged-process filesystem
        # virtualization that does not occur in a mocked code_root argument.
        with tempfile.TemporaryDirectory(prefix='DGP import ',dir=os.environ['LOCALAPPDATA']) as temporary:
            shell=Path(temporary); runtime=shell/'runtime'; code=runtime/'versions'/('d'*40)
            implementation=code/'implementation'; implementation.mkdir(parents=True)
            source=Path(__file__).resolve().parents[1]
            for name in ('__init__.py','managed_paths.py','environment.py','setup_events.py','capture_install.py','offline_payload.py'):
                shutil.copyfile(source/'implementation'/name,implementation/name)
            shutil.copyfile(source/'requirements.txt',code/'requirements.txt')
            (shell/'shell-owner.json').write_text('{"schema":1,"application":"Forensic DGP"}')
            (runtime/'owner.json').write_text(json.dumps({'schema_version':2,'repository':'janusinss/forensic-dgp',
                'layout':'desktop-runtime-v1','install_root':str(shell)}))
            env=dict(os.environ,DGP_INSTALL_ROOT=str(shell),DGP_MANAGED_ROOT=str(runtime))
            result=subprocess.run([sys.executable,'-B','-c',
                'import implementation.environment as e; import implementation.managed_paths as p; '
                'assert p.managed_root() is not None; print(e.CONFIG_PATH)'],
                cwd=code,env=env,capture_output=True,text=True,timeout=15)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('installations',result.stdout)

    def test_version_directory_identity_alias_remains_bound_to_exact_commit(self):
        from implementation import managed_paths
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); shell=root/'shell'; runtime=shell/'runtime'
            code=runtime/'versions'/('a'*40); code.mkdir(parents=True)
            physical=root/'virtualized'/'runtime'/'versions'/('a'*40)
            physical.mkdir(parents=True)
            with patch.dict(os.environ,{'DGP_INSTALL_ROOT':str(shell),'DGP_MANAGED_ROOT':str(runtime)}), \
                 patch.object(managed_paths,'_validate_custom_shell'), \
                 patch.object(managed_paths.os.path,'samefile',return_value=True) as identity:
                self.assertEqual(managed_paths.custom_install_root(physical),shell)
                identity.assert_called_once_with(physical.parent,runtime/'versions')
            with patch.dict(os.environ,{'DGP_INSTALL_ROOT':str(shell),'DGP_MANAGED_ROOT':str(runtime)}), \
                 patch.object(managed_paths,'_validate_custom_shell'), \
                 patch.object(managed_paths.os.path,'samefile',return_value=False):
                with self.assertRaises(managed_paths.ManagedPathError): managed_paths.custom_install_root(physical)
