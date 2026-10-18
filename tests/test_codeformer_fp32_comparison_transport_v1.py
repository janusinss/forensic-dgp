"""CPU synthetic preparation, dependency-copy and finite-supervision guards."""
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest import mock
import uuid

from scripts import prepare_codeformer_fp32_comparison_v1 as p
from scripts import supervise_codeformer_fp32_comparison_v1 as s


class TransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = Path(__file__).resolve().parents[1] / 'scratch'
        cls.scratch.mkdir(exist_ok=True)

    @contextmanager
    def directory(self):
        path = self.scratch / ('fp32_transport_fixture_' + uuid.uuid4().hex)
        path.mkdir(); yield path

    def fixture(self, home):
        root = home / 'forensic-dgp' / p.PACKAGE; root.mkdir(parents=True)
        parent = home / 'forensic-dgp' / p.PARENT; parent.mkdir()
        source = parent / 'data/source.png'; source.parent.mkdir(); source.write_bytes(b'synthetic-source')
        parent_p = {'files_sha256': {'data/source.png': p.sha(source)}}
        p.write_json(parent / 'protocol.json', parent_p)
        overlay = root / 'overlay.json'; overlay.write_text('{}')
        protocol = {'format': p.FORMAT, 'parent_dir': p.PARENT,
            'parent_protocol_sha256': p.sha(parent / 'protocol.json'),
            'dependency_files_sha256': parent_p['files_sha256'],
            'dependency_files_bytes': {'data/source.png': source.stat().st_size},
            'files_sha256': {**parent_p['files_sha256'], 'overlay.json': p.sha(overlay)},
            'limits': {'initial_free_bytes': 16 * 1024**3}}
        p.write_json(root / 'protocol.json', protocol)
        return root, parent, protocol, p.sha(root / 'protocol.json')

    def test_canonical_paths(self):
        for path in ('../x','/abs','./x','x/../y','x//y','C:/x','x\\y','x:a','x/NUL.txt','x/COM¹','x/end.','x/end ','x/a\x01'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                p.canonical(path)
        self.assertEqual(p.canonical('data/sources/train.png'), 'data/sources/train.png')

    def test_dependency_allowlist_excludes_historical_execution(self):
        parent = {'files_sha256': dict.fromkeys(('data/source.png','weights/initializer.pth',
            'provenance/original.json','research/codeformer_adaptation_v2/data.py',
            'research/codeformer_adaptation_v2/worker.py','scripts/run.sh','tests/old.py',
            'outputs/old_checkpoint.pt','READ_ME_FIRST.md'), 'a'*64)}
        self.assertEqual(set(p.dependency_names(parent)), {'data/source.png','weights/initializer.pth',
            'provenance/original.json','research/codeformer_adaptation_v2/data.py'})

    def test_manual_root_linux_and_tmux_before_worker(self):
        with self.directory() as home:
            root, _, _, _ = self.fixture(home)
            self.assertEqual(s.manual_root(root, system='Linux',home=home,environment={'TMUX':'session'}),root)
            for system, env in (('Windows',{'TMUX':'session'}),('Linux',{})):
                with self.subTest(system=system,env=env), self.assertRaises(ValueError):
                    s.manual_root(root,system=system,home=home,environment=env)

    def test_crossroot_and_parent_rejected(self):
        with self.directory() as home:
            root, parent, _, _ = self.fixture(home)
            with self.assertRaisesRegex(ValueError, 'Exact new'):
                s.manual_root(parent,system='Linux',home=home,environment={'TMUX':'session'})
            with self.assertRaisesRegex(ValueError, 'Linux'):
                p.manual_dependency_roots(root,system='Windows',home=home)
            self.assertEqual(p.manual_dependency_roots(root,system='Linux',home=home),(root,parent))

    def test_valid_dependency_copy_regular_no_parent_mutation(self):
        with self.directory() as home:
            root,parent,protocol,digest = self.fixture(home)
            old = (parent/'data/source.png').read_bytes()
            with mock.patch.object(p,'PARENT_PROTOCOL_SHA',protocol['parent_protocol_sha256']), \
                 mock.patch.object(p.shutil,'disk_usage',return_value=type('D',(),{'free':100*1024**3})()):
                result=p.stage_dependencies(root,digest,system='Linux',home=home)
            self.assertTrue(result['complete']); self.assertEqual(result['files'],1)
            self.assertEqual((root/'data/source.png').read_bytes(),old)
            self.assertEqual((parent/'data/source.png').read_bytes(),old)
            self.assertNotEqual((root/'data/source.png').stat().st_ino,(parent/'data/source.png').stat().st_ino)
            self.assertTrue((root/'dependency_install_journal.jsonl').is_file())
            self.assertFalse((root/'outputs').exists())

    def test_dependency_plan_corruption_and_overwrite(self):
        for mode in ('corrupt_parent','bad_protocol','existing_destination','bad_size','bad_overlay'):
            with self.subTest(mode=mode),self.directory() as home:
                root,parent,protocol,digest=self.fixture(home)
                parent_digest=protocol['parent_protocol_sha256']
                if mode=='corrupt_parent': (parent/'data/source.png').write_bytes(b'corruption')
                elif mode=='bad_protocol': digest='0'*64
                elif mode=='existing_destination':
                    (root/'data').mkdir(); (root/'data/source.png').write_bytes(b'preexisting')
                elif mode=='bad_size':
                    protocol['dependency_files_bytes']['data/source.png']=1
                    (root/'protocol.json').write_text(json.dumps(protocol));digest=p.sha(root/'protocol.json')
                elif mode=='bad_overlay': (root/'overlay.json').write_text('{"changed":true}')
                with mock.patch.object(p,'PARENT_PROTOCOL_SHA',parent_digest),self.assertRaises(ValueError):
                    p.validate_dependency_plan(root,parent,digest)
                self.assertFalse((root/'dependency_install_receipt.json').exists())

    def test_dependency_no_partial_install_on_space_failure(self):
        with self.directory() as home:
            root,parent,protocol,digest=self.fixture(home)
            with mock.patch.object(p,'PARENT_PROTOCOL_SHA',protocol['parent_protocol_sha256']), \
                 mock.patch.object(p.shutil,'disk_usage',return_value=type('D',(),{'free':4*1024**3})()), \
                 self.assertRaisesRegex(ValueError,'16GiB'):
                p.stage_dependencies(root,digest,system='Linux',home=home)
            self.assertFalse((root/'data').exists())
            self.assertFalse((root/'dependency_install_journal.jsonl').exists())

    def test_dependency_existing_journal_prevents_retry(self):
        with self.directory() as home:
            root,_,_,digest=self.fixture(home)
            (root/'dependency_install_journal.jsonl').write_text('partial install retained\n')
            with self.assertRaisesRegex(ValueError,'no retry'):
                p.stage_dependencies(root,digest,system='Linux',home=home)

    def test_busy_gpu_stops_without_process_interference(self):
        response=type('R',(),{'stdout':'123, python, 1000\n'})()
        with mock.patch.object(s.subprocess,'run',return_value=response) as run,self.assertRaisesRegex(ValueError,'GPU occupied'):
            s.idle()
        self.assertEqual(run.call_count,1)

    def test_competing_commands_idle_tmux_not_workload(self):
        rows=[(1,'tmux new-session -A -s training'),(2,'python -m research.codeformer_fp32_comparison_v1.worker --run'),
              (3,'python scripts/supervise_codeformer_fp32_comparison_v1.py --root /x'),
              (4,'python unrelated_app.py'),(5,'python train_cctv_dgp_original.py')]
        self.assertEqual([r['pid'] for r in s.competing_commands(rows,3)],[2,5])

    def test_finite_stage_has_process_group_and_no_retry(self):
        with self.directory() as root:
            log=root/'trainer.log'
            process=mock.Mock(pid=321); process.wait.return_value=2
            with mock.patch.object(s.subprocess,'Popen',return_value=process) as popen, \
                 mock.patch.object(s.sys,'stdout',new=io.StringIO()):
                result=s.stage(root,'a'*64,'--run',3000,180,log)
            command=popen.call_args.args[0]
            self.assertEqual(command[:4],['timeout','--signal=TERM','--kill-after=180','3000'])
            self.assertTrue(popen.call_args.kwargs['start_new_session'])
            self.assertEqual(result['exit_code'],2); self.assertEqual(popen.call_count,1)
            self.assertEqual(result['external_cap_seconds'],3000)

    def test_unknown_or_unbounded_stage_rejected(self):
        with self.directory() as root:
            for flag,seconds in (('--resume',3000),('--run',0)):
                with self.assertRaises(ValueError): s.stage(root,'a'*64,flag,seconds,180,root/'log')

    def test_protocol_template_preserves_schedule_and_filters_arrays(self):
        parent={'training':{'call_caps':{}},'limits':{},'export_files':[], 'provenance':{}}
        staged={'dependency_files_sha256':{},'dependency_files_bytes':{},
            'files_sha256':{'parity/input.png':'a'*64,'parity/anchor.npy':'b'*64,
                            'training_parity_v1.json':'c'*64,'provenance/history.jsonl':'d'*64}}
        protocol=p.protocol_template(parent,staged,parity_sha='c'*64)
        self.assertEqual(protocol['training']['max_updates'],50)
        self.assertEqual(protocol['training']['schedule_horizon_updates'],600)
        self.assertEqual(protocol['training']['data_horizon_updates'],600)
        self.assertEqual(protocol['training']['call_caps']['perceptual_training_forwards'],148)
        self.assertEqual(protocol['export_files'],['training_parity_v1.json'])
        self.assertEqual(protocol['limits']['result_payload_bytes'],int(3.5*1024**3))


if __name__=='__main__': unittest.main()
