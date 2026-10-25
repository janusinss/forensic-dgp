"""Pure policy/math/transport fixtures. No VM, network, sudo, unlink or neural calls."""
import ast
import base64
import copy
import importlib.util
import json
from pathlib import Path
import shlex
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
def load(alias, name):
    spec = importlib.util.spec_from_file_location(alias, ROOT / 'scripts' / name)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module
g = load('vm_archive_guest_tested', 'vm_archive_offload_20261018_v1_guest.py')
c = load('vm_archive_client_tested', 'vm_archive_offload_20261018_v1.py')
SCOPE = json.loads((ROOT/'outputs/vm_archive_offload_20261018_v1/windows_backup_proof_v1/scope.json').read_bytes())


def request():
    with patch.object(c, 'require_backup', return_value=(None, None)):
        return c.build_request(SCOPE)


def identity():
    return {'project/project-id': g.INSTANCE, 'instance/id': g.INSTANCE_ID,
            'instance/name': g.INSTANCE, 'instance/zone': 'projects/x/zones/us-central1-a',
            'instance/machine-type': 'projects/x/machineTypes/g2-standard-4'}


class Policy(unittest.TestCase):
    def test_exact_six_paths_and_current_verified_backup_scope(self):
        r = request(); rows = g.validate_request(r)
        self.assertEqual(len(rows), 6)
        self.assertEqual(sum(x['expected_metadata']['bytes'] for x in rows), 16692148382)
        self.assertFalse(any('filter-fusion' in x['vm_path'] or '/forensic-dgp/' in x['vm_path'] for x in rows))

    def test_no_checkpoint_cache_MD_nested_or_other_archive_path_admitted(self):
        for name in ('/home/janusdominic0/forensic-dgp/checkpoint.pth',
                     '/home/janusdominic0/notes.md', '/tmp/cctv-codeformer-adaptation-v1-results.tar.gz',
                     '/home/janusdominic0/../janusdominic0/cctv-codeformer-adaptation-v1-results.tar.gz',
                     '/home/janusdominic0/cctv-codeformer-filter-fusion-training-v1-results.tar.gz'):
            with self.assertRaises(ValueError): g.path_policy(name)

    def test_metadata_real_identity_drift_rejected(self):
        row = request()['candidates'][0]; expected = row['expected_metadata']; actual = dict(expected, gid=1001)
        self.assertTrue(g.same_expected(actual, expected))
        for k in ('bytes','allocated_bytes','uid','nlink','inode','device','mtime_ns','ctime_ns','mode'):
            bad = dict(actual); bad[k] += 1
            self.assertFalse(g.same_expected(bad, expected), k)
        missing = dict(expected); del missing['inode']
        with self.assertRaises(ValueError): g.same_expected(actual, missing)

    def test_request_needs_complete_backup_hash_and_exact_count(self):
        for kind in ('incomplete','missing','uid','hardlink','hash','backup_unverified','size'):
            r = request()
            if kind == 'incomplete': r['complete'] = False
            if kind == 'missing': r['candidates'].pop()
            if kind == 'uid': r['candidates'][0]['expected_metadata']['uid'] = 0
            if kind == 'hardlink': r['candidates'][0]['expected_metadata']['nlink'] = 2
            if kind == 'hash': r['candidates'][0]['backup']['sha256'] = '0'*64
            if kind == 'backup_unverified': r['candidates'][0]['backup']['fresh_full_windows_hash_verified'] = False
            if kind == 'size': r['candidates'][0]['backup']['bytes'] += 1
            with self.assertRaises(ValueError): g.validate_request(r)

    def test_fresh_identity_exact_host_uid_project_instance_zone_machine(self):
        g.validate_identity_values(identity(),1001,g.INSTANCE,'/home/janusdominic0')
        for key in identity():
            wrong = identity(); wrong[key] = 'wrong'
            with self.assertRaises(ValueError): g.validate_identity_values(wrong,1001,g.INSTANCE,'/home/janusdominic0')
        for uid,host,home in ((0,g.INSTANCE,'/home/janusdominic0'),(1001,'other','/home/janusdominic0'),
                              (1001,g.INSTANCE,'/home/root')):
            with self.assertRaises(ValueError): g.validate_identity_values(identity(),uid,host,home)

    def test_real_reader_census_blocks_and_no_root_access_cannot_be_waived(self):
        rows=request()['candidates']
        for value in ({'complete':False,'root_read_only_proc_fd_census':True,'readers':[]},
                      {'complete':True,'root_read_only_proc_fd_census':False,'readers':[]},
                      {'complete':True,'root_read_only_proc_fd_census':True,'readers':[{'pid':99,'fd':4}]}):
            with patch.object(g,'call',return_value={'stdout':json.dumps(value)}):
                with self.assertRaises(ValueError):g.open_reader_guard(rows)

    def test_reader_scan_skips_only_exact_current_one_hash_fd(self):
        okay={'complete':True,'root_read_only_proc_fd_census':True,'readers':[]}
        with patch.object(g,'call',return_value={'stdout':json.dumps(okay)}) as call:
            self.assertTrue(g.open_reader_guard(request()['candidates'],allowed_own_fd=123)['complete'])
        argv=call.call_args.args[0]
        self.assertEqual(argv[:4],['sudo','-n','python3','-B'])
        self.assertEqual(json.loads(argv[-1]),[[g.os.getpid(),123]])
        self.assertEqual(len(json.loads(argv[-2])),6)
        ast.parse(g.PROC_FD_SOURCE,feature_version=(3,10))

    def test_active_candidate_reader_or_training_blocks_but_disjoint_app_allowed(self):
        names=[request()['candidates'][0]['vm_path']]
        self.assertIsNone(g.classify_process((4,1,'python','python app.py'),9,8,names))
        self.assertIsNone(g.classify_process((9,1,'python','python train.py'),9,8,names))
        self.assertTrue(g.classify_process((4,1,'python','python audit.py '+names[0]),9,8,names))
        self.assertTrue(g.classify_process((4,1,'python','python '+str(g.RESEARCH)+'/scripts/training.py --run'),9,8,names))
        self.assertTrue(g.classify_process((4,1,'torchrun','torchrun workload'),9,8,names))

    def test_idle_tmux_shell_allowed_busy_research_or_gpu_blocks(self):
        def responses(busy=False,gpu=False):
            return [{'stdout':'NVIDIA L4, 0 MiB, 23034 MiB\n'},
                    {'stdout':'99, python, 100 MiB\n' if gpu else ''},
                    {'stdout':'session|0|'+('python' if busy else 'bash')+'|44|'+str(g.RESEARCH)+'/current|0\n'},
                    {'stdout':'77 1 sshd sshd: user\n'}]
        with patch.object(g,'call',side_effect=responses()):
            self.assertTrue(g.workload_guard(request()['candidates'])['complete'])
        for b, gpu in ((True,False),(False,True)):
            with patch.object(g,'call',side_effect=responses(b,gpu)):
                with self.assertRaises(ValueError):g.workload_guard(request()['candidates'])

    def test_model_provenance_small_hash_policy_and_entire_latest_roots(self):
        space={str(p):{'kind':'directory','bytes':0} for p in g.FULL_ROOTS}
        examples={'checkpoint.pth':10,'recognizer.onnx':10,'failure.json':10,'terms.md':10,
                  'large_old.log':g.SMALL_RECORD_BYTES+1,'cache/features.bin':20}
        for n,size in examples.items():space[str(g.RESEARCH/n)]={'kind':'regular','bytes':size}
        fusion=str(g.FULL_ROOTS[0]/'data/opaque.bin');space[fusion]={'kind':'regular','bytes':99999999}
        r2=str(g.FULL_ROOTS[1]/'outputs/pixels.npy');space[r2]={'kind':'regular','bytes':99999999}
        home=types.SimpleNamespace(name='cctv-codeformer-filter-fusion-training-v1-export.json',__str__=lambda self:'')
        # Real Path objects are supplied through a patched HOME iterator, with no filesystem reads.
        p=g.HOME/'cctv-codeformer-filter-fusion-training-v1-export.json'
        with patch.object(type(g.HOME),'iterdir',return_value=iter([p])),patch.object(g,'safe',return_value=p):
            names=set(g.protected_names(space))
        self.assertIn(str(g.RESEARCH/'checkpoint.pth'),names);self.assertIn(str(g.RESEARCH/'recognizer.onnx'),names)
        self.assertIn(str(g.RESEARCH/'failure.json'),names);self.assertIn(str(g.RESEARCH/'terms.md'),names)
        self.assertIn(fusion,names);self.assertIn(r2,names)
        self.assertNotIn(str(g.RESEARCH/'large_old.log'),names);self.assertNotIn(str(g.RESEARCH/'cache/features.bin'),names)

    def test_preapply_review_must_bind_windows_backup_current_VM_and_protection(self):
        plan={'complete':True,'format':'vm-six-home-archive-exact-apply-plan-v1',
              'guest_source':{'sha256':'a'*64},'windows_backup_scope':{'sha256':'b'*64},
              'verified':{'protected_sha256':'c'*64,'candidate_current_sha256':'d'*64}}
        review={'complete':True,'approved_for_exact_six_HOME_archive_apply':True,'plan_sha256':'e'*64,
                'guest_source_sha256':'a'*64,'windows_backup_scope_sha256':'b'*64,
                'protected_sha256':'c'*64,'candidate_current_sha256':'d'*64}
        g.require_review(plan,'e'*64,review)
        for key in review:
            bad=copy.deepcopy(review);bad[key]=False if isinstance(bad[key],bool) else 'wrong'
            with self.assertRaises(ValueError):g.require_review(plan,'e'*64,bad)

    def test_transport_base64_roundtrip_and_short_command_no_raw_multiline(self):
        code='from pathlib import Path\nprint("quoted \' text", Path("/home/janusdominic0"))\n'
        argv=c.ssh_args(code);command=argv[-1].removeprefix('--command=')
        tokens=shlex.split(command);self.assertEqual(tokens[:3],['python3','-B','-c'])
        tree=ast.parse(tokens[3]);encoded=tree.body[1].value.args[0].args[0].value
        self.assertEqual(base64.b64decode(encoded).decode(),code)
        self.assertNotIn('\n',command);self.assertLess(len(c.subprocess.list2cmdline(argv)),7900)

    def test_single_remote_SCP_source_upload_and_download_pinned_hostkey(self):
        for download in (False,True):
            argv=c.scp_args(ROOT/'scratch/opaque.bin','janusdominic0@forensic-dgp-thesis:/home/janusdominic0/test',download)
            self.assertEqual(argv[:2],['compute','scp']);self.assertEqual(sum('@forensic-dgp-thesis:' in a for a in argv),1)
            self.assertIn('--scp-flag=-batch',argv);self.assertIn('--scp-flag='+c.trust.HOSTKEY,argv)

    def test_launcher_executes_only_uploaded_hash_bound_guest_and_documents(self):
        code=c.launcher('apply','a'*64,'exact_plan.json','b'*64,'c'*64)
        ast.parse(code,feature_version=(3,10));self.assertIn('hashlib.sha256',code)
        self.assertIn('b'*64,code);self.assertIn('c'*64,code);self.assertIn('runpy.run_path',code)
        self.assertNotIn('forensic-dgp/',code)

    def test_no_NN_import_or_any_automatic_entry_execution(self):
        self.assertNotIn('torch',sys.modules)
        source=(ROOT/'scripts/vm_archive_offload_20261018_v1_guest.py').read_text()
        tree=ast.parse(source);unlinks=[n for n in ast.walk(tree) if isinstance(n,ast.Call)
                and isinstance(n.func,ast.Attribute) and n.func.attr in ('unlink','rmdir','rename','remove','rmtree')]
        self.assertEqual(len(unlinks),1);self.assertEqual(unlinks[0].func.attr,'unlink')
        self.assertIn('dir_fd', [k.arg for k in unlinks[0].keywords])


if __name__=='__main__':unittest.main()
