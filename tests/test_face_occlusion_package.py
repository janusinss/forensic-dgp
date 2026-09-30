"""Packaging must preserve evidence and cover exact archive bytes."""
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from scripts.package_face_occlusion_vm import build_bundle


class FaceOcclusionPackageTests(unittest.TestCase):
    def test_members_inventory_and_linux_checksum_match_exact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'scripts').mkdir()
            (root/'scripts/pilot.py').write_bytes(b'print("fixture")\n')
            (root/'initial.pth').write_bytes(b'initial-state-fixture')
            archive = root/'outputs/pilot.tar.gz'
            inventory = root/'outputs/package/inventory.json'
            report = build_bundle(root, ['scripts/pilot.py','initial.pth'], inventory, archive)
            expected = json.loads(inventory.read_text())
            expected['outputs/package/inventory.json'] = hashlib.sha256(inventory.read_bytes()).hexdigest()
            with tarfile.open(archive,'r:gz') as tar:
                self.assertEqual(set(tar.getnames()), set(expected))
                for member in tar.getmembers():
                    self.assertTrue(member.isfile())
                    self.assertEqual(hashlib.sha256(tar.extractfile(member).read()).hexdigest(),expected[member.name])
            checksum = archive.with_name(archive.name+'.sha256').read_bytes()
            self.assertNotIn(b'\r', checksum)
            self.assertEqual(checksum, (report['sha256']+'  pilot.tar.gz\n').encode())

    def test_existing_evidence_and_escape_paths_fail_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'pilot.py').write_bytes(b'fixture')
            archive=root/'outputs/pilot.tar.gz';inventory=root/'outputs/package/inventory.json'
            for files in (['../outside.py'], ['C:/outside.py'], ['pilot.py','pilot.py']):
                with self.assertRaises(ValueError):
                    build_bundle(root,files,inventory,archive)
                self.assertFalse((root/'outputs').exists())
            archive.parent.mkdir();archive.write_bytes(b'old-evidence')
            with self.assertRaises(ValueError):
                build_bundle(root,['pilot.py'],inventory,archive)
            self.assertEqual(archive.read_bytes(),b'old-evidence')
            self.assertFalse(inventory.exists())


if __name__=='__main__':unittest.main()
