"""CPU synthetic contract/resource/supervisor fixtures; no neural calls."""
import ast
from contextlib import contextmanager, ExitStack, redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import numpy as np

from research.codeformer_bounded_fusion_feasibility_v1 import contract as c
from research.codeformer_bounded_fusion_feasibility_v1 import worker as w
from scripts import supervise_codeformer_bounded_fusion_feasibility_v1 as s

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def directory():
    scratch = ROOT / 'scratch'; scratch.mkdir(exist_ok=True)
    root = scratch / ('bounded-fusion-worker-fixture-' + uuid.uuid4().hex)
    root.mkdir(); yield root


def protocol():
    return deepcopy(c.SPEC)


def complete_protocol():
    p = protocol()
    p['runtime_expected'] = {'torch': '2.9.1+cu129', 'numpy': '2.2.6', 'CUDA': '12.9',
                             'GPU': 'NVIDIA L4', 'capability': [8, 9]}
    files = {name: 'b' * 64 for name in c.REQUIRED_SOURCES}
    files[c.DESIGN_PATH] = c.DESIGN_SHA256
    files.update(c.IMMUTABLE_SOURCES)
    calibration = []
    for update in range(1, 51):
        for slot in range(1, 5):
            case_id = 'train_u%04d_slot%d' % (update, slot)
            row = {'id': case_id, 'update': update, 'slot': slot,
                   'input_path': 'data/inputs/' + case_id + '.png', 'input_sha256': 'a' * 64}
            files[row['input_path']] = row['input_sha256']; calibration.append(row)
    by_position = {(r['update'], r['slot']): r for r in calibration}
    cases = []
    for index, position in enumerate(c.SELECTED_ORDER):
        row = deepcopy(by_position[position]); row['reference_id'] = 'tr_ffhq_%05d' % (index // 2)
        for path_key, sha_key, directory_name in [('source_path', 'source_sha256', 'sources'),
                ('target512_path', 'target512_sha256', 'clean512'),
                ('target256_path', 'target256_sha256', 'clean256')]:
            row[path_key] = 'data/' + directory_name + '/' + row['reference_id'] + '.png'
            row[sha_key] = 'c' * 64; files[row[path_key]] = row[sha_key]
        cases.append(row)
    p.update(files_sha256=files, export_files=list(c.REQUIRED_SOURCES), cases=cases,
             calibration_inputs=calibration, canary_ids=list(c.CANARY_IDS))
    return p


class FiniteMeasurementTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack(); self.addCleanup(stack.close)
        stack.enter_context(redirect_stdout(io.StringIO()))
        stack.enter_context(redirect_stderr(io.StringIO()))

    def test_fixed_caps_and_gate_are_zero_training(self):
        self.assertEqual(c.CALL_CAPS['codeformer_forwards'], 170)
        self.assertEqual(c.CALL_CAPS['fusion_block_evaluations'], 960)
        self.assertTrue(all(value == 0 for name, value in c.CALL_CAPS.items() if name not in
                            ('codeformer_forwards', 'fusion_block_evaluations')))
        self.assertEqual(c.LIMITS['worker_seconds'], 900)
        self.assertEqual(c.LIMITS['measurement_seconds'], 600)
        self.assertEqual(c.GATE['central_support'], [32, 224])

    def test_paths_reject_escape_devices_collisions(self):
        for name in ('', '.', '../x', '/x', './x', 'x//y', 'x/../y', 'x\\y', 'x:a',
                     'x/CON.txt', 'x/CONIN$', 'x/COM¹', 'x/trailing.', 'x/a\x00'):
            with self.subTest(name=name), self.assertRaises(ValueError): c.canonical(name)
        with directory() as root:
            with self.assertRaises(ValueError): c.bounded(root, '../escape', must_exist=False)

    def test_complete_protocol_exact_cohort_and_frozen_runtime(self):
        p = complete_protocol()
        self.assertIs(c.validate_protocol(p), p)
        for key, value in [('seed', 42), ('canary_ids', list(reversed(c.CANARY_IDS))),
                           ('call_caps', {**c.CALL_CAPS, 'optimizer_updates': 1}),
                           ('gate', {**c.GATE, 'beta': .2}),
                           ('gate', {**c.GATE, 'gate_ceiling': True}),
                           ('call_caps', {**c.CALL_CAPS, 'optimizer_updates': False})]:
            changed = deepcopy(p); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): c.validate_protocol(changed)
        for key, value in [('GPU', 'NVIDIA T4'), ('capability', [7, 5]), ('extra', 'unbound')]:
            changed = deepcopy(p); changed['runtime_expected'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): c.validate_protocol(changed)

    def test_protocol_rejects_unmatched_targets_immutable_sources_and_final_or_reordered_cohort(self):
        p = complete_protocol()
        mutations = []
        changed = deepcopy(p); changed['cases'][0]['target512_sha256'] = 'd' * 64; mutations.append(changed)
        changed = deepcopy(p); changed['files_sha256']['third_party/codeformer/codeformer_arch.py'] = 'd' * 64; mutations.append(changed)
        changed = deepcopy(p); changed['cases'][0]['reference_id'] = 'final_identity'; mutations.append(changed)
        changed = deepcopy(p); changed['cases'] = list(reversed(changed['cases'])); mutations.append(changed)
        changed = deepcopy(p); changed['calibration_inputs'][0]['update'] = True; mutations.append(changed)
        changed = deepcopy(p); changed['export_files'].append(p['cases'][0]['input_path']); mutations.append(changed)
        changed = deepcopy(p); changed['files_sha256']['DATA/inputs/TRAIN_u0001_slot1.png'] = 'a' * 64; mutations.append(changed)
        for index, changed in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(ValueError): c.validate_protocol(changed)

    def test_manual_guard_precedes_output_or_neural_import(self):
        with directory() as root, patch.object(w.platform, 'system', return_value='Windows'):
            with self.assertRaises(ValueError): w.run(root, 'a' * 64)
            self.assertEqual(list(root.iterdir()), [])

    def test_manual_exact_root_and_tmux(self):
        with directory() as home:
            root = home / 'forensic-dgp' / c.PACKAGE_DIR; root.mkdir(parents=True)
            wrong = root.parent / 'old'; wrong.mkdir()
            self.assertEqual(w.manual_root(root, system='Linux', home=home, environment={'TMUX':'human'}), root)
            with self.assertRaises(ValueError): w.manual_root(root, system='Linux', home=home, environment={})
            with self.assertRaises(ValueError): w.manual_root(wrong, system='Linux', home=home, environment={'TMUX':'human'})

    def test_writer_exact_arrays_png_and_no_overwrite(self):
        with directory() as root:
            p = protocol(); out = root / p['outputs']; out.mkdir(parents=True)
            writer = w.EvidenceWriter(root, p, lambda: None)
            value = np.zeros((256,256,3), np.float32)
            row = writer.array('case/raw.npy', value)
            self.assertEqual(row['sha256'], c.sha256(root / row['path']))
            self.assertTrue(np.array_equal(np.load(root / row['path'], allow_pickle=False),value))
            writer.png('case/native.png', np.zeros((256,256,3),np.uint8))
            with self.assertRaises(ValueError): writer.array('case/raw.npy', value)
            with self.assertRaises(ValueError): writer.array('weights.pt', value)
            with self.assertRaises(ValueError): writer.png('case/wrong.png', np.zeros((512,512,3),np.uint8))
            with self.assertRaises(ValueError): writer.array('case/nonfinite.npy', np.asarray([float('nan')],np.float32))
            with self.assertRaises(ValueError): writer.array('case/float64.npy', np.zeros(2,np.float64))

    def test_prospective_writer_caps_preserve_failure_slack(self):
        with directory() as root:
            p=protocol();out=root/p['outputs'];out.mkdir(parents=True)
            p['limits'].update(regular_files=3,metadata_bytes=1024**2+100,result_bytes=1024**2+100)
            writer=w.EvidenceWriter(root,p,lambda:None)
            writer.json('one.json', {'small':True})
            with self.assertRaises(ValueError): writer.json('two.json', {'small':True})
            terminal=w.EvidenceWriter(root,p,lambda:None,terminal=True)
            terminal.json('worker_failure.json', {'complete':False})
            self.assertTrue((out/'worker_failure.json').is_file())

    def test_per_array_cap_before_write(self):
        with directory() as root:
            p=protocol();out=root/p['outputs'];out.mkdir(parents=True)
            p['limits']['array_file_bytes']=4097
            with self.assertRaises(ValueError): w.EvidenceWriter(root,p,lambda:None).array('large.npy',np.zeros(2,np.float32))
            self.assertFalse((out/'large.npy').exists())

    def test_budgeted_call_reservations_and_terminal_counts(self):
        with directory() as root:
            p=protocol();p['call_caps']={'codeformer_forwards':2,'fusion_block_evaluations':0,'optimizer_updates':0}
            def execute(where,current,original,stopped,cb):
                cb['guard'](where,current);cb['clock']('measurement')
                for i in range(2): cb['reserve']('codeformer_forwards', {'case_id': 'fixture'+str(i), 'mode':'original','section':'paths'})
                return {'complete':True,'state_before_after_equal':True,'parameter_grad_fields_all_none':True,
                        'content_preservation_verified':True,'repeat_zero_identity_canaries_pass':True,
                        'counts':dict(current['call_caps'])}
            fake=SimpleNamespace(execute=execute)
            with patch.object(w,'manual_root',return_value=root), patch.object(c,'verify_transfer',return_value=p), \
                 patch.object(c,'resolve_assets',return_value={'original':root/'o','stopped':root/'s'}), \
                 patch.dict(sys.modules,{'research.codeformer_bounded_fusion_feasibility_v1.neural':fake}):
                self.assertEqual(w.run(root,'a'*64),0)
            result=c.read_json(root/p['outputs']/'worker_complete.json')
            self.assertEqual(result['counts']['codeformer_forwards'],2)
            self.assertEqual(len(list((root/p['outputs']/'call_trace').glob('*.json'))),2)
            self.assertEqual(result['counts']['optimizer_updates'],0)

    def test_extra_call_refused_and_failure_retained(self):
        with directory() as root:
            p=protocol();p['call_caps']={'codeformer_forwards':1,'optimizer_updates':0}
            def execute(where,current,original,stopped,cb):
                cb['clock']('measurement');cb['reserve']('codeformer_forwards',{'case_id':'one','mode':'original','section':'paths'})
                cb['reserve']('codeformer_forwards',{'case_id':'extra','mode':'original','section':'paths'})
            with patch.object(w,'manual_root',return_value=root), patch.object(c,'verify_transfer',return_value=p), \
                 patch.object(c,'resolve_assets',return_value={'original':root/'o','stopped':root/'s'}), \
                 patch.dict(sys.modules,{'research.codeformer_bounded_fusion_feasibility_v1.neural':SimpleNamespace(execute=execute)}):
                self.assertEqual(w.run(root,'a'*64),1)
            failure=c.read_json(root/p['outputs']/'worker_failure.json')
            self.assertEqual(failure['counts']['codeformer_forwards'],1)
            self.assertFalse((root/p['outputs']/'worker_complete.json').exists())

    def test_missing_content_proof_and_malformed_call_provenance_fail_closed(self):
        for variant in ('missing_canary', 'oversized_context', 'nonstr_context', 'unexpected_context_key'):
            with self.subTest(variant=variant), directory() as root:
                p=protocol();p['call_caps']={'codeformer_forwards':1,'optimizer_updates':0}
                def execute(where,current,original,stopped,cb):
                    cb['clock']('measurement')
                    context={'case_id':'one','mode':'original','section':'paths'}
                    if variant=='oversized_context': context['case_id']='x'*1025
                    if variant=='nonstr_context': context['case_id']=float('nan')
                    if variant=='unexpected_context_key': context['clean_target']='forbidden'
                    cb['reserve']('codeformer_forwards',context)
                    return {'complete':True,'counts':dict(current['call_caps']),
                            'state_before_after_equal':True,'parameter_grad_fields_all_none':True,
                            'content_preservation_verified':True}
                with patch.object(w,'manual_root',return_value=root),patch.object(c,'verify_transfer',return_value=p), \
                     patch.object(c,'resolve_assets',return_value={'original':root/'o','stopped':root/'s'}), \
                     patch.dict(sys.modules,{'research.codeformer_bounded_fusion_feasibility_v1.neural':SimpleNamespace(execute=execute)}):
                    self.assertEqual(w.run(root,'a'*64),1)
                self.assertTrue((root/p['outputs']/'worker_failure.json').is_file())
                self.assertFalse((root/p['outputs']/'worker_complete.json').exists())

    def test_initialization_measurement_whole_time_and_gpu_peak_stops_preserve_failure(self):
        for variant in ('initialization', 'measurement', 'whole_worker', 'peak_gpu'):
            with self.subTest(variant=variant), directory() as root:
                p=protocol();now=[0.]
                p['call_caps']={'codeformer_forwards':0,'optimizer_updates':0}
                if variant=='initialization':p['limits']['initialization_seconds']=10
                if variant=='measurement':p['limits']['measurement_seconds']=5
                if variant=='whole_worker':p['limits']['worker_seconds']=4
                def execute(where,current,original,stopped,cb):
                    if variant=='initialization':now[0]=11.;cb['clock']()
                    elif variant=='measurement':cb['clock']('measurement');now[0]=6.;cb['clock']()
                    elif variant=='whole_worker':now[0]=5.;cb['clock']()
                    else:
                        cuda=SimpleNamespace(is_initialized=lambda:True,memory_reserved=lambda:0,
                            max_memory_reserved=lambda:current['limits']['max_reserved_GPU_bytes']+1)
                        with patch.dict(sys.modules,{'torch':SimpleNamespace(cuda=cuda)}):cb['clock']()
                    raise AssertionError('finite stop was bypassed')
                with patch.object(w,'manual_root',return_value=root),patch.object(c,'verify_transfer',return_value=p), \
                     patch.object(c,'resolve_assets',return_value={'original':root/'o','stopped':root/'s'}), \
                     patch.object(w.time,'monotonic',side_effect=lambda:now[0]), \
                     patch.dict(sys.modules,{'research.codeformer_bounded_fusion_feasibility_v1.neural':SimpleNamespace(execute=execute)}):
                    self.assertEqual(w.run(root,'a'*64),1)
                failure=c.read_json(root/p['outputs']/'worker_failure.json')
                self.assertEqual(failure['counts']['codeformer_forwards'],0)
                self.assertNotIn('finite stop was bypassed',failure['message'])
                self.assertTrue(failure['partial_evidence_retained'])

    def test_static_verify_has_no_parent_fallback_and_rejects_extra_source_or_modified_input(self):
        with directory() as root:
            p=protocol();p['files_sha256']={};p['export_files']=[]
            (root/'input.png').write_bytes(b'exact retained PNG fixture')
            p['files_sha256']['input.png']=c.sha256(root/'input.png')
            c.write_new_json(root/'protocol.json',p);digest=c.sha256(root/'protocol.json')
            with patch.object(c,'validate_protocol',side_effect=lambda p:p),patch.object(c,'resolve_assets') as parent:
                c.verify_transfer(root,digest,verify_parent=False);parent.assert_not_called()
                c.verify_transfer(root,digest,verify_parent=True);parent.assert_called_once()
                (root/'unbound.py').write_text('raise RuntimeError("never import")')
                with self.assertRaises(ValueError):c.verify_transfer(root,digest,verify_parent=False)
                (root/'unbound.py').rename(root/'retained-unbound.py')
                (root/'input.png').write_bytes(b'modified')
                with self.assertRaises(ValueError):c.verify_transfer(root,digest,verify_parent=False)

    def test_terminal_pid_live_or_unrelated_reuse(self):
        with directory() as root:
            path=root/'outputs/bounded_fusion_feasibility_v1';path.mkdir(parents=True)
            c.write_new_json(path/'manual_launch.json',{'pid':123,'protocol_sha256':'a'*64})
            proc=root/'proc';(proc/'123').mkdir(parents=True)
            (proc/'123/cmdline').write_bytes(b'python\0-m\0research.codeformer_bounded_fusion_feasibility_v1.worker')
            with self.assertRaises(ValueError): w.require_worker_terminal(root,'a'*64,proc_base=proc)
            (proc/'123/cmdline').write_bytes(b'unrelated-process')
            w.require_worker_terminal(root,'a'*64,proc_base=proc)

    def test_no_model_or_pickle_result(self):
        with directory() as root:
            (root/'saved.pth').write_bytes(b'not a permitted result')
            with self.assertRaises(ValueError): w.reject_unexpected_files(root)

    def test_export_closed_failure_no_live_log(self):
        with directory() as root:
            p=protocol();p['limits']['reserve_free_bytes']=0
            out=root/p['outputs'];out.mkdir(parents=True)
            c.write_new_json(out/'worker_failure.json',{'complete':False})
            c.write_new_json(out/'manual_launch.json',{'pid':123,'protocol_sha256':'a'*64})
            p['export_files']=[];p['files_sha256']={}
            (root/'protocol.json').write_text('{}')
            supervision=root/'outputs/supervision';supervision.mkdir()
            for name in ('preflight.log','diagnostic.log'): (supervision/name).write_text('closed')
            c.write_new_json(supervision/'run_receipt.json',{'protocol_sha256':'a'*64,'stages':[]})
            (supervision/'export.log').write_text('live-export-log')
            with patch.object(w,'manual_root',return_value=root),patch.object(c,'verify_transfer',return_value=p), \
                 patch.object(w,'require_worker_terminal',return_value=None),patch.object(w.Path,'home',return_value=root):
                self.assertEqual(w.export(root,'a'*64),0)
            with tarfile.open(root/(p['export_prefix']+'-results.tar.gz')) as tar:
                self.assertNotIn('outputs/supervision/export.log',tar.getnames())
                self.assertIn('outputs/supervision/diagnostic.log',tar.getnames())
            self.assertNotIn(b'\r',(root/(p['export_prefix']+'-results.tar.gz.sha256')).read_bytes())

    def test_supervisor_exact_finite_group_and_streamed_log_cap(self):
        with directory() as root:
            log=root/'diagnostic.log'
            fake=SimpleNamespace(pid=321,stdout=io.BytesIO(b'closed output'),wait=lambda **kwargs:0)
            with patch.object(s.subprocess,'Popen',return_value=fake) as popen:
                receipt=s.stage(root,'a'*64,'--run',900,60,log)
            self.assertEqual(popen.call_args.args[0][:4],['timeout','--signal=TERM','--kill-after=60','900'])
            self.assertTrue(popen.call_args.kwargs['start_new_session'])
            self.assertFalse(receipt['log_capped'])
            fake.stdout=io.BytesIO(b'X'*(2*1024**2+1))
            with patch.object(s.subprocess,'Popen',return_value=fake),patch.object(s.os,'killpg',create=True) as kill:
                receipt=s.stage(root,'a'*64,'--run',900,60,root/'capped.log')
            self.assertTrue(receipt['log_capped']);self.assertEqual(receipt['exit_code'],125)
            self.assertEqual((root/'capped.log').stat().st_size,2*1024**2)
            kill.assert_called_once()

    def test_import_static_no_framework_and_python310_parse(self):
        for module in (c,w,s): ast.parse(Path(module.__file__).read_text(encoding='utf-8'),feature_version=(3,10))
        code='import sys;from research.codeformer_bounded_fusion_feasibility_v1 import worker;assert not any(n in sys.modules for n in ("torch","numpy","PIL","cv2"))'
        result=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__': unittest.main()
