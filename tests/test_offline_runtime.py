"""Offline release payload invariants, tested without executing packages."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class OfflinePayloadTests(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module('implementation.offline_payload')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.content = {
            'python/python-3.13.9-amd64.zip': b'python fixture',
            'locks/application.txt': ('demo==1.0 --hash=sha256:' + hashlib.sha256(b'wheel').hexdigest() + '\n').encode(),
            'locks/capture.txt': ('capture==2.0 --hash=sha256:' + hashlib.sha256(b'capture wheel').hexdigest() + '\n').encode(),
            'wheels/application/demo-1.0-py3-none-any.whl': b'wheel',
            'wheels/capture/capture-2.0-py3-none-any.whl': b'capture wheel',
            'models/face.pth': b'model fixture',
        }
        self.manifest = dict(schema=1, platform='win_amd64', python=dict(version='3.13.9', path='python/python-3.13.9-amd64.zip', bytes=len(self.content['python/python-3.13.9-amd64.zip']), sha256=hashlib.sha256(self.content['python/python-3.13.9-amd64.zip']).hexdigest()), locks=dict(application='locks/application.txt', capture='locks/capture.txt'), files=[dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()) for name,data in self.content.items()], models=[dict(group='restoration',id='face',filename='face.pth',path='models/face.pth')])
        for name,data in self.content.items():
            target=self.root/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(data)
        self._pin()
        self.environment = patch.dict(os.environ, {'DGP_OFFLINE_PAYLOAD':str(self.root),'DGP_OFFLINE_PAYLOAD_SHA256':self.pin})
        self.environment.start(); self.addCleanup(self.environment.stop)

    def _pin(self):
        data=(json.dumps(self.manifest)+'\n').encode()
        (self.root/'manifest.json').write_bytes(data)
        self.pin=hashlib.sha256(data).hexdigest()
        if hasattr(self,'environment'):
            os.environ['DGP_OFFLINE_PAYLOAD_SHA256']=self.pin

    def test_valid_bundle_builds_offline_hash_locked_command(self):
        payload=self.module.load_payload()
        command=payload.package_command(self.root/'environment'/'Scripts'/'python.exe','application')
        for flag in ('--isolated','--no-index','--require-hashes','--only-binary=:all:','--no-cache-dir'):
            self.assertIn(flag,command)
        self.assertNotIn('--index-url',command)
        if os.name=='nt':
            self.assertEqual(command[command.index('--prefix')+1], '\\\\?\\'+str(self.root/'environment'))
        self.assertEqual(payload.locked_versions('application'), [('demo','1.0')])

    def test_cache_rechecks_consumed_models_after_same_size_change(self):
        payload=self.module.load_payload()
        path=self.root/'models/face.pth'; before=path.stat()
        path.write_bytes(b'wrong fixture'); os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
        asset=dict(filename='face.pth',size=len(b'model fixture'),sha256=hashlib.sha256(b'model fixture').hexdigest())
        with self.assertRaises(self.module.OfflinePayloadError): payload.model_source('restoration','face',asset)

    def test_manifest_hash_mismatch_is_rejected(self):
        os.environ['DGP_OFFLINE_PAYLOAD_SHA256']='0'*64
        with self.assertRaisesRegex(self.module.OfflinePayloadError,'manifest'):
            self.module.load_payload()

    def test_corrupted_wheel_is_rejected_before_install(self):
        (self.root/'wheels/application/demo-1.0-py3-none-any.whl').write_bytes(b'other')
        with self.assertRaises(self.module.OfflinePayloadError): self.module.load_payload()

    def test_unlisted_file_is_rejected(self):
        (self.root/'unexpected.exe').write_bytes(b'unlisted')
        with self.assertRaisesRegex(self.module.OfflinePayloadError,'inventory'):
            self.module.load_payload()

    def test_traversal_manifest_entry_is_rejected(self):
        self.manifest['files'][0]['path']='../outside'
        self._pin()
        with self.assertRaises(self.module.OfflinePayloadError): self.module.load_payload()

    def test_case_duplicate_inventory_is_rejected(self):
        self.manifest['files'].append(dict(self.manifest['files'][0],path='PYTHON/python-3.13.9-amd64.zip'))
        self._pin()
        with self.assertRaises(self.module.OfflinePayloadError): self.module.load_payload()

    def test_lock_url_does_not_enable_network_fallback(self):
        name='locks/application.txt'; data=b'demo @ https://example.invalid/demo.whl\n'; (self.root/name).write_bytes(data)
        for item in self.manifest['files']:
            if item['path']==name: item.update(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        self._pin()
        with self.assertRaisesRegex(self.module.OfflinePayloadError,'lock'):
            self.module.load_payload()

    def test_model_source_must_match_admitted_asset(self):
        payload=self.module.load_payload()
        asset=dict(filename='face.pth',size=len(b'model fixture'),sha256=hashlib.sha256(b'model fixture').hexdigest())
        self.assertEqual(payload.model_source('restoration','face',asset),self.root/'models/face.pth')
        asset['sha256']='f'*64
        with self.assertRaises(self.module.OfflinePayloadError): payload.model_source('restoration','face',asset)

    def test_incomplete_offline_environment_is_rejected(self):
        del os.environ['DGP_OFFLINE_PAYLOAD_SHA256']
        with self.assertRaises(self.module.OfflinePayloadError): self.module.load_payload()

    def test_no_offline_request_returns_none(self):
        del os.environ['DGP_OFFLINE_PAYLOAD']; del os.environ['DGP_OFFLINE_PAYLOAD_SHA256']
        self.assertIsNone(self.module.load_payload())

class PrivatePythonPipTests(unittest.TestCase):
    def test_staged_ensurepip_upgrade_retains_derived_proof(self):
        module=importlib.import_module('implementation.offline_payload')
        with tempfile.TemporaryDirectory() as temporary:
            runtime=Path(temporary)/'runtime'; runtime.mkdir()
            staged=runtime/'python'/('python-bootstrap-'+'a'*32)/'python'
            bundled=staged/'Lib/ensurepip/_bundled'; bundled.mkdir(parents=True)
            init=bundled.parent/'__init__.py'; init.write_text('_PIP_VERSION = "25.2"\n')
            (bundled/'pip-25.2-py3-none-any.whl').write_bytes(b'old archive input')
            source=Path(temporary)/'pip-26.2.2-py3-none-any.whl'; source.write_bytes(b'verified pip fixture')
            fake=unittest.mock.Mock()
            fake.locked_versions.return_value=[('pip','26.2.2')]
            fake.file.return_value=source
            fake.manifest={'python':{'sha256':'a'*64}}
            fake.files={'wheels/application/pip-26.2.2-py3-none-any.whl':{'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}}
            with patch.object(module,'load_payload',return_value=fake), patch('implementation.managed_paths.managed_root',return_value=runtime), patch.object(module.subprocess,'run') as run:
                module.prepare_private_python(staged)
            self.assertIn('26.2.2',init.read_text())
            self.assertFalse((bundled/'pip-25.2-py3-none-any.whl').exists())
            self.assertEqual((bundled/source.name).read_bytes(),source.read_bytes())
            self.assertEqual(run.call_count,2)
            proof=json.loads((staged/'.dgp-derived-runtime.json').read_text())
            self.assertEqual(proof['pip_version'],'26.2.2')
            self.assertEqual(proof['python_archive_sha256'],'a'*64)

    def test_private_python_mutation_rejects_nonstaging_destination(self):
        module=importlib.import_module('implementation.offline_payload')
        with tempfile.TemporaryDirectory() as temporary:
            runtime=Path(temporary)/'runtime'; runtime.mkdir()
            outside=Path(temporary)/'unrelated'; outside.mkdir()
            with patch('implementation.managed_paths.managed_root',return_value=runtime):
                with self.assertRaises(module.OfflinePayloadError): module.prepare_private_python(outside)

class ActiveModelProtectionTests(unittest.TestCase):
    def fixture(self, *, corrupt=False):
        temp=tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        root=Path(temp.name); runtime=root/'runtime'; version=runtime/'versions'/('a'*40)
        (version/'implementation').mkdir(parents=True); (runtime/'receipts').mkdir()
        data=b'previous valid model'; asset=dict(filename='face.pth',size=len(data),sha256=hashlib.sha256(data).hexdigest())
        manifest=version/'implementation/models.json'; manifest.write_text(json.dumps({'schema_version':1,'assets':{'face':asset}}))
        (runtime/'current.json').write_text(json.dumps({'schema':1,'commit':'a'*40}))
        receipt={'schema':1,'commit':'a'*40,'files':{'implementation/models.json':hashlib.sha256(manifest.read_bytes()).hexdigest()}}
        (runtime/'receipts'/('a'*40+'.json')).write_text(json.dumps(receipt))
        destination=runtime/'data/models/implementation/face.pth'; destination.parent.mkdir(parents=True); destination.write_bytes(b'corrupt' if corrupt else data)
        return runtime,destination,dict(filename='face.pth',size=3,sha256=hashlib.sha256(b'new').hexdigest())

    def test_valid_active_previous_model_requires_move(self):
        module=importlib.import_module('implementation.offline_payload'); runtime,path,new=self.fixture()
        before=path.read_bytes()
        with patch('implementation.managed_paths.managed_root',return_value=runtime):
            with self.assertRaisesRegex(module.OfflinePayloadError,'Move'):
                module.protect_active_model(path,'restoration',new)
        self.assertEqual(path.read_bytes(),before)

    def test_corrupt_active_model_can_be_repaired_from_bundle(self):
        module=importlib.import_module('implementation.offline_payload'); runtime,path,new=self.fixture(corrupt=True)
        with patch('implementation.managed_paths.managed_root',return_value=runtime):
            module.protect_active_model(path,'restoration',new)
        self.assertEqual(path.read_bytes(),b'corrupt')


class CommittedLockHealthTests(unittest.TestCase):
    def test_committed_lock_rejects_urls_without_payload(self):
        module=importlib.import_module('implementation.offline_payload')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); folder=root/'implementation/locks'; folder.mkdir(parents=True)
            (folder/'windows-cp313-application.txt').write_text('demo @ https://example.invalid/demo.whl\n')
            with self.assertRaises(module.OfflinePayloadError): module.committed_versions('application',root)

    def test_app_verification_checks_transitive_lock_without_payload(self):
        import subprocess
        from implementation import environment
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); (root/'requirements.txt').write_text('demo==1\n')
            def run(command,**kwargs):
                if command[-1].startswith('import importlib.metadata'):
                    self.assertIn("('pip', '26.2.2')",command[-1])
                    self.assertIn("('transitive', '2')",command[-1])
                    return 19
                return 0
            with patch.object(environment,'ROOT',root),patch.object(environment,'check_python'),patch.object(environment,'load_payload',return_value=None),patch.object(environment,'committed_versions',return_value=[('pip','26.2.2'),('transitive','2')]),patch.object(environment,'run_foreground',side_effect=run):
                self.assertEqual(environment.verify_runtime(Path('fixture.exe'),include_models=False),19)

class CaptureOfflineLockTests(unittest.TestCase):
    def test_correct_detector_with_obsolete_pip_is_not_reused(self):
        import importlib.metadata
        import subprocess
        from types import SimpleNamespace
        from implementation import capture_install
        with tempfile.TemporaryDirectory() as temporary:
            python=Path(temporary)/'python.exe'; python.write_bytes(b'fixture')
            payload=unittest.mock.Mock(); payload.locked_versions.return_value=[('opencv-python-headless','4.13.0.92'),('numpy','2.5.2'),('pip','26.2.2')]
            cv2=SimpleNamespace(CascadeClassifier=lambda value:SimpleNamespace(empty=lambda:False),data=SimpleNamespace(haarcascades='fixture/'))
            versions={'opencv-python-headless':'4.13.0.92','numpy':'2.5.2','pip':'25.2'}
            def run(command,**kwargs):
                try:
                    with patch.dict(__import__('sys').modules,{'cv2':cv2,'numpy':SimpleNamespace()}), patch.object(importlib.metadata,'version',side_effect=versions.__getitem__):
                        exec(command[-1],{})
                    return subprocess.CompletedProcess(command,0)
                except AssertionError:
                    return subprocess.CompletedProcess(command,1)
            with patch.object(capture_install,'load_payload',return_value=payload),patch.object(capture_install.subprocess,'run',side_effect=run):
                self.assertFalse(capture_install.verified(python))
            with patch.object(capture_install,'load_payload',return_value=None),patch.object(capture_install,'committed_versions',return_value=payload.locked_versions('capture')),patch.object(capture_install.subprocess,'run',side_effect=run):
                self.assertFalse(capture_install.verified(python))

class CaptureProbeDeadlineTests(unittest.TestCase):
    def test_setup_probe_allows_cold_load_but_reports_timeout_as_failure(self):
        import contextlib,io,subprocess
        from implementation import capture_install
        with tempfile.TemporaryDirectory() as temporary:
            python=Path(temporary)/'python.exe'; python.write_bytes(b'fixture')
            with patch.object(capture_install,'capture_python',return_value=python), patch.object(capture_install,'load_payload',return_value=None),patch.object(capture_install,'committed_versions',return_value=None), patch.object(capture_install.subprocess,'run',side_effect=subprocess.TimeoutExpired('capture-probe',120)) as run, patch.object(capture_install,'event') as events,contextlib.redirect_stderr(io.StringIO()) as error:
                self.assertEqual(capture_install.main([]),1)
            self.assertEqual(run.call_args.kwargs['timeout'],120)
            self.assertIn('120 seconds',error.getvalue())
            self.assertIn('retry',error.getvalue().lower())
            self.assertTrue(any(call.args[1]=='error' for call in events.call_args_list))

    def test_regular_capture_health_keeps_short_deadline(self):
        import subprocess
        from implementation import capture_install
        with tempfile.TemporaryDirectory() as temporary:
            python=Path(temporary)/'python.exe'; python.write_bytes(b'fixture')
            with patch.object(capture_install,'load_payload',return_value=None),patch.object(capture_install,'committed_versions',return_value=None),patch.object(capture_install.subprocess,'run',return_value=subprocess.CompletedProcess([],0)) as run:
                self.assertTrue(capture_install.verified(python))
            self.assertEqual(run.call_args.kwargs['timeout'],30)


class DeferredCaptureSetupTests(unittest.TestCase):
    def test_bulk_capture_task_defers_import_until_outer_verification(self):
        import contextlib,io,subprocess
        from implementation import capture_install
        with tempfile.TemporaryDirectory() as temporary:
            python=Path(temporary)/'capture/Scripts/python.exe'; python.parent.mkdir(parents=True); python.write_bytes(b'fixture')
            payload=unittest.mock.Mock(); payload.package_command.return_value=['fixture.exe','install']
            with patch.object(capture_install,'capture_python',return_value=python),patch.object(capture_install,'load_payload',return_value=payload),patch.object(capture_install,'_deferred_setup_allowed',return_value=True),patch.object(capture_install,'verified') as verify,patch.object(capture_install,'custom_install_root',return_value=None),patch.object(capture_install.subprocess,'run',return_value=subprocess.CompletedProcess([],0)),patch.object(capture_install,'event') as emitted,contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(capture_install.main(['--defer-verification']),0)
            verify.assert_not_called()
            self.assertFalse(any(call.args[1]=='complete' for call in emitted.call_args_list))
            self.assertTrue(any('verification follows' in call.args[2] for call in emitted.call_args_list))

    def test_deferred_mode_requires_exact_preparing_generation_marker(self):
        from implementation import capture_install
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)/'env'; folder.mkdir(); python=folder/'capture/Scripts/python.exe'
            marker=folder/'.dgp-environment.json'; body={'schema':1,'dependency_identity':'a'*16,'generation':'b'*32,'state':'preparing'}
            marker.write_text(json.dumps(body))
            with patch.dict(os.environ,{'DGP_SETUP_GENERATION':'b'*32}),patch.object(capture_install,'expected_environment',return_value=folder),patch.object(capture_install,'dependency_identity',return_value='a'*16):
                self.assertTrue(capture_install._deferred_setup_allowed(python,object()))
                body['state']='ready'; marker.write_text(json.dumps(body))
                self.assertFalse(capture_install._deferred_setup_allowed(python,object()))

    def test_timeout_still_honors_pending_cancellation(self):
        import subprocess
        from implementation import capture_install
        from implementation.setup_events import SetupCancelled
        with tempfile.TemporaryDirectory() as temporary:
            python=Path(temporary)/'python.exe'; python.write_bytes(b'fixture')
            with patch.object(capture_install,'load_payload',return_value=None),patch.object(capture_install,'committed_versions',return_value=None),patch.object(capture_install,'check_cancel',side_effect=[None,SetupCancelled('cancelled')]),patch.object(capture_install.subprocess,'run',side_effect=subprocess.TimeoutExpired('probe',120)):
                with self.assertRaises(SetupCancelled): capture_install.verified(python,timeout=120)

class CopyProgressThrottleTests(unittest.TestCase):
    def test_progress_is_throttled_but_final_bytes_are_always_reported(self):
        module=importlib.import_module('implementation.offline_payload')
        emitted=[]; clock=[0.0]; total=20*1024*1024+1
        progress=module.CopyProgress('face.pth',total,emit=lambda *a,**k:emitted.append((a,k)),clock=lambda:clock[0])
        for current in range(1024*1024,total,1024*1024):
            clock[0]+=0.02; progress.update(current)
        progress.update(total)
        self.assertEqual(emitted[0][1]['current'],0)
        self.assertEqual(emitted[-1][1]['current'],total)
        self.assertEqual(emitted[-1][1]['total'],total)
        self.assertLessEqual(len(emitted),5)
        self.assertTrue(all(item[0][1]=='copying' for item in emitted))

if __name__=='__main__': unittest.main()
