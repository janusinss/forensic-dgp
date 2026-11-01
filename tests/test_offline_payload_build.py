"""Behavioral tests for fully pinned offline payloads and release staging."""
import hashlib
import io
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('offline_preparer', ROOT / 'packaging' / 'prepare_offline.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)
spec_builder = importlib.util.spec_from_file_location('offline_release_builder', ROOT / 'packaging' / 'build_release.py')
builder = importlib.util.module_from_spec(spec_builder)
spec_builder.loader.exec_module(builder)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class OfflineBuildTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'payload'
        self.root.mkdir()
        def archive_bytes(name, data):
            stream = io.BytesIO()
            with zipfile.ZipFile(stream,'w') as archive:
                archive.writestr(name,data)
            return stream.getvalue()
        python_zip = archive_bytes('python.exe', b'python')
        app_wheel = archive_bytes('demo-1.0.dist-info/METADATA', 'Metadata-Version: 2.1\nName: demo\nVersion: 1.0\n')
        capture_wheel = archive_bytes('capture-1.0.dist-info/METADATA', 'Metadata-Version: 2.1\nName: capture\nVersion: 1.0\n')
        self.python = {'version': '3.13.9', 'path': 'python/python.zip', 'bytes': len(python_zip), 'sha256': sha(python_zip)}
        self.models = [dict(group='restoration', id='sample', filename='model.pth', size=5, sha256=sha(b'model'))]
        self.content = {
            'python/python.zip': python_zip,
            'locks/application.txt': ('demo==1.0 --hash=sha256:' + sha(app_wheel) + '\n').encode(),
            'locks/capture.txt': ('capture==1.0 --hash=sha256:' + sha(capture_wheel) + '\n').encode(),
            'wheels/application/demo-1.0-py3-none-any.whl': app_wheel,
            'wheels/capture/capture-1.0-py3-none-any.whl': capture_wheel,
            'models/model.pth': b'model',
            'licenses/NOTICE.txt': b'license',
        }
        for name, data in self.content.items():
            output = self.root / name
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(data)
        self.manifest = dict(schema=1, platform='win_amd64', python=self.python,
            locks={'application':'locks/application.txt', 'capture':'locks/capture.txt'},
            files=[{'path':name,'bytes':len(data),'sha256':sha(data)} for name,data in sorted(self.content.items())],
            models=[{'group':'restoration','id':'sample','filename':'model.pth','path':'models/model.pth'}])
        self.save_manifest()

    def save_manifest(self):
        (self.root / 'manifest.json').write_text(json.dumps(self.manifest), encoding='utf8')

    def verify(self):
        return prepare.verify_payload(self.root, python_spec=self.python, model_specs=self.models)

    def test_inventory_accepts_only_exact_unchanged_files(self):
        result = self.verify()
        self.assertEqual(result['payload_bytes'], sum(map(len,self.content.values())) + (self.root/'manifest.json').stat().st_size)
        (self.root / 'models/model.pth').write_bytes(b'other')
        with self.assertRaisesRegex(ValueError, 'hash|checksum'):
            self.verify()

    def test_unlisted_file_is_rejected(self):
        (self.root / 'unexpected.exe').write_bytes(b'x')
        with self.assertRaisesRegex(ValueError, 'inventory|unexpected'):
            self.verify()

    def test_case_collisions_and_traversal_are_rejected(self):
        for path in ('../outside', 'C:/outside', 'models/../outside', 'models/CON.txt', 'models/file.', 'models\\evil'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                prepare.safe_relative(path)
        self.manifest['files'].append(dict(self.manifest['files'][0],path=self.manifest['files'][0]['path'].upper()))
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'duplicate|case'):
            self.verify()

    def test_model_descriptor_cannot_redirect_an_admitted_model(self):
        self.manifest['models'][0]['path'] = 'wheels/application/demo-1.0-py3-none-any.whl'
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'model'):
            self.verify()

    def test_incomplete_or_remote_locks_are_rejected(self):
        for lock in ('demo>=1.0\n', 'demo @ https://example.invalid/demo.whl\n', '--index-url https://example.invalid\n',
                     'demo==1.0 --hash=sha256:'+'a'*64+'\nDemo==1.0 --hash=sha256:'+'a'*64+'\n'):
            with self.subTest(lock=lock), self.assertRaises(ValueError):
                prepare.validate_lock(lock)

    def test_expected_source_locks_are_required_before_staging(self):
        destination = Path(self.temporary.name) / 'stage' / 'payload'
        with self.assertRaisesRegex(ValueError, 'lock'):
            prepare.stage_payload(self.root, destination, {'application':b'wrong','capture':self.content['locks/capture.txt']},
                                  python_spec=self.python, model_specs=self.models)
        self.assertFalse(destination.exists())

    def test_stage_copies_verified_bytes_without_changing_source(self):
        destination = Path(self.temporary.name) / 'stage' / 'payload'
        result = prepare.stage_payload(self.root, destination,
            {group:self.content['locks/'+group+'.txt'] for group in ('application','capture')},
            python_spec=self.python, model_specs=self.models)
        self.assertEqual((destination/'models/model.pth').read_bytes(),b'model')
        self.assertEqual(result['manifest_sha256'],sha((self.root/'manifest.json').read_bytes()))
        self.assertEqual((self.root/'models/model.pth').read_bytes(),b'model')

    def test_verified_staging_can_be_reused_after_an_interrupted_build(self):
        destination = Path(self.temporary.name) / 'stage' / 'payload'
        locks = {group:self.content['locks/'+group+'.txt'] for group in ('application','capture')}
        first = prepare.stage_payload(self.root,destination,locks,python_spec=self.python,model_specs=self.models)
        second = prepare.stage_payload(self.root,destination,locks,python_spec=self.python,model_specs=self.models)
        self.assertEqual(first['manifest_sha256'],second['manifest_sha256'])
        (destination/'models/model.pth').write_bytes(b'other')
        with self.assertRaises(ValueError):
            prepare.stage_payload(self.root,destination,locks,python_spec=self.python,model_specs=self.models)

    def test_offline_defines_include_payload_size_contract(self):
        release = dict(files={'uninstall.ps1':'a'*64},payload='offline-bundled',payload_bytes=100,payload_expanded_bytes=300)
        result = builder.release_defines(release,'b'*64)
        self.assertIn('#define OfflinePayload 1',result)
        self.assertIn('#define OfflinePayloadBytes 100',result)
        self.assertIn('#define OfflineExpandedBytes 300',result)
        release['payload']='online-source-bundled'
        self.assertNotIn('#define OfflinePayload 1',builder.release_defines(release,'b'*64))

    def test_artifact_hash_is_measured_and_oversize_is_rejected(self):
        artifact = Path(self.temporary.name)/'installer.exe'
        artifact.write_bytes(b'installer')
        self.assertEqual(builder.artifact_record(artifact),dict(artifact_bytes=9,artifact_sha256=sha(b'installer')))
        with mock.patch.object(Path,'stat',return_value=mock.Mock(st_size=2*1024**3)):
            with self.assertRaisesRegex(RuntimeError,'2 GiB'):
                builder.artifact_record(artifact)

    def test_cached_old_pins_are_not_copied_into_the_new_wheelhouse(self):
        cache=Path(self.temporary.name)/'cache'
        source=cache/'capture'
        source.mkdir(parents=True)
        def wheel(directory,name,version):
            filename=name+'-'+version+'-py3-none-any.whl'
            with zipfile.ZipFile(directory/filename,'w') as archive:
                archive.writestr(name+'-'+version+'.dist-info/METADATA',
                                 'Metadata-Version: 2.1\nName: '+name+'\nVersion: '+version+'\n')
        wheel(source,'demo','1.0')
        wheel(source,'unused','1.0')
        output=Path(self.temporary.name)/'new-wheels'
        calls=[]
        def resolve(arguments):
            calls.append(arguments)
            wheel(output,'demo','2.0')
            wheel(output,'pip','26.2.2')
        with mock.patch.object(prepare,'_run_pip',side_effect=resolve):
            prepare.download_wheels(output,'capture',{'demo':'2.0','pip':'26.2.2'},cache)
        self.assertEqual({prepare.wheel_metadata(p)[:2] for p in output.iterdir()},
                         {('demo','2.0'),('pip','26.2.2')})
        self.assertIn('demo==2.0',calls[0])
        self.assertIn('pip==26.2.2',calls[0])
        self.assertEqual(len(list(source.iterdir())),2)

    def test_security_bootstrap_tools_are_pinned_for_both_environments(self):
        application=prepare._required_application()
        self.assertEqual(application['uv'],'0.13.4')
        self.assertEqual(application['pip'],'26.2.2')
        self.assertEqual(prepare.capture_requirements()['pip'],'26.2.2')

    def test_deep_license_paths_are_flattened_without_changing_bytes(self):
        wheelhouse=Path(self.temporary.name)/'deep-wheels'
        wheelhouse.mkdir()
        wheel=wheelhouse/'demo-1.0-py3-none-any.whl'
        original='demo-1.0.dist-info/licenses/'+'nested/'*25+'LICENSE'
        with zipfile.ZipFile(wheel,'w') as archive:
            archive.writestr('demo-1.0.dist-info/METADATA','Metadata-Version: 2.1\nName: demo\nVersion: 1.0\n')
            archive.writestr(original,b'original license bytes\r\n')
        destination=Path(self.temporary.name)/'notices'
        prepare.write_notices(wheelhouse,destination,'application')
        index=json.loads((destination/'application-INDEX.json').read_text())
        self.assertEqual(index['schema'],1)
        self.assertEqual(index['files'][0]['original_path'],original)
        relative=index['files'][0]['saved_path']
        self.assertLess(len(relative),100)
        self.assertEqual((destination/relative).read_bytes(),b'original license bytes\r\n')
        self.assertEqual(index['files'][0]['sha256'],sha(b'original license bytes\r\n'))

    def test_vendored_dist_info_does_not_count_as_the_wheels_own_metadata(self):
        wheel = Path(self.temporary.name)/'setuptools-1.0-py3-none-any.whl'
        with zipfile.ZipFile(wheel,'w') as archive:
            archive.writestr('setuptools-1.0.dist-info/METADATA','Metadata-Version: 2.1\nName: setuptools\nVersion: 1.0\n')
            archive.writestr('setuptools/_vendor/wheel-1.0.dist-info/METADATA','Metadata-Version: 2.1\nName: wheel\nVersion: 1.0\n')
        self.assertEqual(prepare.wheel_metadata(wheel)[:2],('setuptools','1.0'))
        with zipfile.ZipFile(wheel,'a') as archive:
            archive.writestr('other-1.0.dist-info/METADATA','Metadata-Version: 2.1\nName: other\nVersion: 1.0\n')
        with self.assertRaisesRegex(ValueError,'exactly one'):
            prepare.wheel_metadata(wheel)

    def test_wheel_lock_uses_metadata_and_hash_for_every_wheel(self):
        wheelhouse = Path(self.temporary.name) / 'wheels'
        wheelhouse.mkdir()
        wheel = wheelhouse / 'demo-1.0-py3-none-any.whl'
        with zipfile.ZipFile(wheel,'w') as archive:
            archive.writestr('demo-1.0.dist-info/METADATA','Metadata-Version: 2.1\nName: Demo\nVersion: 1.0\nLicense: MIT\n')
        lock = prepare.wheel_lock(wheelhouse, {'demo':'1.0'})
        self.assertIn('demo==1.0 --hash=sha256:'+sha(wheel.read_bytes()),lock)
        with self.assertRaisesRegex(ValueError,'version|missing'):
            prepare.wheel_lock(wheelhouse, {'demo':'2.0'})


if __name__ == '__main__':
    unittest.main()
