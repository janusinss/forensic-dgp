"""Model-free finite overlay/installer/sealing fixtures; no actual packet seal."""
from contextlib import contextmanager
from copy import deepcopy
import io
from pathlib import Path
import subprocess
import sys
import tarfile
import unittest
from unittest.mock import patch
import uuid

from scripts import prepare_codeformer_adain_gradient_isolation_v1 as p
from scripts import seal_codeformer_adain_gradient_isolation_v1 as s

ROOT=Path(__file__).resolve().parents[1]


@contextmanager
def directory():
    root=ROOT/'scratch'/('adain-prepare-fixture-'+uuid.uuid4().hex);root.mkdir(parents=True);yield root


@contextmanager
def install_fixture():
    with directory() as home:
        base=home/'forensic-dgp';root=base/p.PACKAGE;parent=base/p.PARENT
        root.mkdir(parents=True);parent.mkdir()
        (parent/'data').mkdir();(parent/'data/input.bin').write_bytes(b'immutable data fixture')
        (parent/'weights').mkdir();(parent/'weights/initializer.pth').write_bytes(b'inert never loaded weights')
        files={name:p.sha(parent/name) for name in ('data/input.bin','weights/initializer.pth')}
        pp={'files_sha256':files};p.write_json(parent/'protocol.json',pp);parent_sha=p.sha(parent/'protocol.json')
        (root/'overlay.py').write_bytes(b'raise RuntimeError("must never execute")')
        plan={'format':p.FORMAT,'parent_dir':p.PARENT,'parent_protocol_sha256':parent_sha,
            'dependency_files_sha256':files,'dependency_files_bytes':{n:(parent/n).stat().st_size for n in files},
            'files_sha256':{**files,'overlay.py':p.sha(root/'overlay.py')},'limits':{'initial_free_bytes':0}}
        p.write_json(root/'protocol.json',plan);protocol_sha=p.sha(root/'protocol.json')
        with patch.object(p,'PARENT_PROTOCOL_SHA',parent_sha):yield home,root,parent,protocol_sha,plan


class PreparationTests(unittest.TestCase):
    def test_dependency_allowlist_excludes_old_executors_outputs_mutable_and_replaced_approval_files(self):
        keep=('data/x.png','weights/codeformer.pth','parity/training_inputs/update_0001/slot_1/pair_record.json',
              'research/codeformer_adaptation_v2/model.py','instructions/OLD_RESEARCH_DESIGN.md','training_parity_v1.json')
        reject=('outputs/old/checkpoint.pt','scripts/old.py','tests/old.py','research/codeformer_fp32_comparison_v1/worker.py',
            'research/codeformer_fp32_comparison_v1/preparation_audit.py','source_readiness.json',
            'research/codeformer_adaptation_v1/worker.py','research/codeformer_adaptation_v2/worker.py',
            'provenance/resource_ledger.json','provenance/independent_input_readback.json','instructions/SYSTEM_WORKFLOW_AND_GOAL.md')
        self.assertEqual(p.dependency_names({'files_sha256':{n:'a'*64 for n in keep+reject}}),sorted(keep))

    def test_portable_paths_reject_traversal_and_windows_devices(self):
        for name in ('../x','/x','x//y','x\\y','C:a','x/CON.txt','x/COM¹','x/trailing.','x/a\x00'):
            with self.subTest(name=name),self.assertRaises(ValueError):p.canonical(name)

    def test_manual_installer_requires_linux_and_exact_new_root(self):
        with install_fixture() as (home,root,parent,digest,plan):
            with self.assertRaises(ValueError):p.manual_dependency_roots(root,system='Windows',home=home)
            self.assertEqual(p.manual_dependency_roots(root,system='Linux',home=home),(root,parent))
            with self.assertRaises(ValueError):p.manual_dependency_roots(parent,system='Linux',home=home)

    def test_installer_validates_every_source_before_regular_copies_and_refuses_repeat(self):
        with install_fixture() as (home,root,parent,digest,plan):
            result=p.stage_dependencies(root,digest,system='Linux',home=home)
            self.assertTrue(result['complete']);self.assertEqual(result['neural_calls'],0)
            for name in plan['dependency_files_sha256']:
                self.assertEqual(p.sha(root/name),p.sha(parent/name));self.assertNotEqual((root/name).stat().st_ino,(parent/name).stat().st_ino)
            with self.assertRaises(ValueError):p.stage_dependencies(root,digest,system='Linux',home=home)

    def test_corrupt_parent_unbound_overlay_or_partial_copy_blocks_before_journal(self):
        for variant in ('parent','overlay','partial'):
            with self.subTest(variant=variant),install_fixture() as (home,root,parent,digest,plan):
                if variant=='parent':(parent/'data/input.bin').write_bytes(b'changed')
                if variant=='overlay':(root/'unbound.py').write_bytes(b'never imported')
                if variant=='partial':(root/'data').mkdir();(root/'data/input.bin').write_bytes(b'partial retained')
                with self.assertRaises(ValueError):p.stage_dependencies(root,digest,system='Linux',home=home)
                self.assertFalse((root/'dependency_install_journal.jsonl').exists())

    def test_disk_and_finite_install_stop_preserve_partial_evidence(self):
        with install_fixture() as (home,root,parent,digest,plan),patch.object(p.shutil,'disk_usage',return_value=type('D',(),{'free':0})()):
            with self.assertRaises(ValueError):p.stage_dependencies(root,digest,system='Linux',home=home)
            self.assertFalse((root/'dependency_install_journal.jsonl').exists())
        with install_fixture() as (home,root,parent,digest,plan),patch.object(p.time,'monotonic',side_effect=[0,901]):
            with self.assertRaises(ValueError):p.stage_dependencies(root,digest,system='Linux',home=home)
            self.assertTrue((root/'dependency_install_journal.jsonl').is_file())
            self.assertFalse((root/'dependency_install_receipt.json').exists())

    def test_archive_readback_exact_single_package_regular_hashes_without_extraction(self):
        with directory() as root:
            (root/'a.txt').write_bytes(b'one');expected={'a.txt':p.sha(root/'a.txt')}
            archive=root/'tiny.tar.gz'
            with tarfile.open(archive,'w:gz') as tar:
                info=tarfile.TarInfo(p.PACKAGE+'/a.txt');info.size=3;tar.addfile(info,io.BytesIO(b'one'))
            proof=s.archive_readback(archive,root,expected);self.assertEqual(proof['members'],1)
            self.assertFalse(proof['returned_or_packet_code_executed'])
            with self.assertRaises(ValueError):s.archive_readback(archive,root,{'a.txt':'a'*64})
            with patch.object(s,'MAX_UPLOAD_BYTES',1),self.assertRaises(ValueError):s.archive_readback(archive,root,expected)

    def test_archive_rejects_links_duplicate_traversal_and_wrong_prefix(self):
        variants=('link','duplicate','traversal','wrong_prefix')
        for variant in variants:
            with self.subTest(variant=variant),directory() as root:
                (root/'a.txt').write_bytes(b'one');archive=root/'bad.tar.gz'
                name=p.PACKAGE+'/a.txt' if variant not in ('traversal','wrong_prefix') else '../a.txt' if variant=='traversal' else 'other/a.txt'
                with tarfile.open(archive,'w:gz') as tar:
                    info=tarfile.TarInfo(name);info.size=3
                    if variant=='link':info.type=tarfile.SYMTYPE;info.linkname='other';info.size=0
                    tar.addfile(info,io.BytesIO(b'one') if info.isreg() else None)
                    if variant=='duplicate':tar.addfile(info,io.BytesIO(b'one'))
                with self.assertRaises(ValueError):s.archive_readback(archive,root,{'a.txt':p.sha(root/'a.txt')})

    def test_source_readiness_required_before_any_freeze_append(self):
        with directory() as root:
            evidence=root/p.EVIDENCE;evidence.mkdir(parents=True)
            p.write_json(evidence/'source_readiness.json',{'complete':True,'approved_for_protocol_freeze':False,'training_authorized':False})
            with self.assertRaises(ValueError):s.freeze_and_seal(root,p.sha(evidence/'source_readiness.json'))
            self.assertFalse((root/'outputs'/p.PACKAGE).exists())

    def test_resource_proof_rejects_wrong_caps_fourth_state_and_bool_file_count(self):
        protocol={'limits':{'result_payload_bytes':4000,'archive_bytes':5000,'checkpoint_temporary_bytes':100,
                            'initial_free_bytes':16000,'reserve_free_bytes':4000}}
        ledger={'conservative_payload_bytes':3000,'payload_cap_bytes':4000,'conservative_tar_gzip_upper_bytes':4000,
            'archive_cap_bytes':5000,'planned_file_count_upper':6900,'regular_file_count_cap':7000,
            'atomic_checkpoint_temporary_bytes':100,'startup_free_bytes_after_install':16000,'reserve_bytes':4000,
            'extra_safe_checkpoint_copy_allowed':False,'last_safe_checkpoint_must_be_a_reference_not_fourth_serialized_file':True}
        s.resource_proof({'complete':True,'approved':True,'ledger':ledger},protocol)
        for key,value in [('conservative_payload_bytes',4001),('planned_file_count_upper',True),
                          ('extra_safe_checkpoint_copy_allowed',True),('archive_cap_bytes',9000)]:
            altered=deepcopy(ledger);altered[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):s.resource_proof({'complete':True,'approved':True,'ledger':altered},protocol)

    def test_pure_import_does_not_load_frameworks_or_run_external_processes(self):
        code='import sys;from scripts import prepare_codeformer_adain_gradient_isolation_v1,seal_codeformer_adain_gradient_isolation_v1;assert not any(n in sys.modules for n in ("torch","numpy","PIL","cv2"))'
        result=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
