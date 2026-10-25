"""Independent pure ledger/AST fixtures; no actual worker or model execution.

Synthetic completed operations prove accounting behavior only, not a usable
GPU worker, finite candidate eligibility, quality gains or manual VM readiness.
"""
import ast
import copy
from pathlib import Path
import subprocess
import sys
import unittest

from research.codeformer_u64_png_checked_step_v1 import contract as C, worker as W

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(W.__file__)


def plan():
    base = ROOT / 'outputs/cctv_codeformer_u64_role_gradients_v1_vm'
    selection = C.select_cases((base / 'evidence/parent_protocol.json').read_bytes(),
                               (base / 'evidence/standard_protocol.json').read_bytes())
    return C.operation_plan(selection)


def simulate(calls):
    journal = []
    ledger = W.OperationLedger(lambda row: journal.append(copy.deepcopy(row)))
    for call in calls:
        metadata = {'id': call['id'], 'plan_slot': call['slot'], 'trial_index': call['trial_index']}
        if call['kind'] == 'forward':
            token = ledger.reserve('forward', category=call['category'], **metadata)
            ledger.returned(token, synchronized_return=True, synthetic_only=True)
        else:
            token = ledger.reserve('parameter_application', action=call['action'], **metadata)
            ledger.returned(token, synchronized_return=True, synthetic_only=True,
                actual_before_head_canonical_sha256='a' * 64,
                actual_after_head_canonical_sha256=C.HEAD_CANONICAL_SHA256 if call['action'] == 'trial_restore' else 'b' * 64)
    return ledger, journal


class LedgerFixtures(unittest.TestCase):
    def test_full445pairs890events436forwards9applications_action4_4_1(self):
        ledger, journal = simulate(plan())
        expected = {k: C.LIMITS[k] for k in W.COUNTER_KEYS}
        self.assertEqual(ledger.counters, expected)
        self.assertEqual(ledger.sequence, 445)
        self.assertEqual(len(journal), 890)
        self.assertEqual(ledger.counters['trial_parameter_applications'], 4)
        self.assertEqual(ledger.counters['original_restoration_applications'], 4)
        self.assertEqual(ledger.counters['selected_parameter_commits'], 1)
        self.assertEqual(ledger.counters['gradient_query_returns'], 0)
        self.assertFalse(ledger.snapshot()['operation_outcome_unknown'])
        for index in range(0, len(journal), 2):
            self.assertEqual(journal[index]['token'], journal[index + 1]['token'])
            self.assertTrue(journal[index]['kind'].endswith('_reserved'))
            self.assertTrue(journal[index + 1]['kind'].endswith('_returned'))

    def test_noeligible_known388pairs776events380forwards8apps_action4_4_0(self):
        calls = plan()[:C.NO_ELIGIBLE_PREFIX['plan_rows']]
        ledger, journal = simulate(calls)
        self.assertEqual(ledger.sequence, 388)
        self.assertEqual(len(journal), 776)
        self.assertEqual(ledger.counters['full_forward_returns'], 380)
        self.assertEqual(ledger.counters['parameter_applications'], 8)
        self.assertEqual(ledger.counters['trial_parameter_applications'], 4)
        self.assertEqual(ledger.counters['original_restoration_applications'], 4)
        self.assertEqual(ledger.counters['selected_parameter_commits'], 0)
        self.assertEqual(ledger.counters['selected_quality_forwards'], 0)
        self.assertEqual(journal[-1]['action'], 'trial_restore')
        self.assertEqual(journal[-1]['trial_index'], 3)
        self.assertEqual(journal[-1]['actual_after_head_canonical_sha256'], C.HEAD_CANONICAL_SHA256)
        self.assertFalse(ledger.pending)

    def test_interrupted_copy_stays_unknown_without_automatic_restore(self):
        rows = []
        ledger = W.OperationLedger(rows.append)
        token = ledger.reserve('parameter_application', action='trial_apply', trial_index=0)
        self.assertEqual(ledger.counters['parameter_application_attempts'], 1)
        self.assertEqual(ledger.counters['parameter_applications'], 0)
        self.assertEqual(ledger.counters['trial_parameter_applications'], 0)
        self.assertTrue(ledger.snapshot()['operation_outcome_unknown'])
        self.assertEqual(ledger.snapshot()['unknown_reservations'][0]['token'], token)
        with self.assertRaises(ValueError):
            ledger.reserve('parameter_application', action='trial_restore', trial_index=0)
        self.assertEqual(len(rows), 1)

    def test_failed_durable_return_does_not_count_action_or_restore(self):
        def persist(row):
            if row['kind'].endswith('_returned'):
                raise OSError('Synthetic durable-return failure')
        ledger = W.OperationLedger(persist)
        token = ledger.reserve('parameter_application', action='trial_restore', trial_index=0)
        with self.assertRaises(OSError):
            ledger.returned(token, actual_before_head_canonical_sha256='b' * 64,
                actual_after_head_canonical_sha256=C.HEAD_CANONICAL_SHA256)
        self.assertEqual(ledger.counters['parameter_applications'], 0)
        self.assertEqual(ledger.counters['original_restoration_applications'], 0)
        self.assertIn(token, ledger.pending)
        self.assertTrue(ledger.snapshot()['operation_outcome_unknown'])

    def test_failed_reservation_remains_conservatively_unknown(self):
        ledger = W.OperationLedger(lambda row: (_ for _ in ()).throw(OSError('Synthetic fsync failure')))
        with self.assertRaises(OSError):
            ledger.reserve('parameter_application', action='trial_apply', trial_index=0)
        self.assertEqual(ledger.counters['parameter_application_attempts'], 1)
        self.assertEqual(ledger.counters['parameter_applications'], 0)
        self.assertTrue(ledger.pending)

    def test_caps_unknown_actions_kinds_and_reserved_metadata_refused(self):
        ledger = W.OperationLedger(lambda row: None)
        for kind in ('gradient_query', 'backward', 'optimizer_update', 'unknown'):
            with self.assertRaises(ValueError):
                ledger.reserve(kind)
        for metadata in ({}, {'action': 'unknown'}, {'action': 'trial_apply', 'token': 1},
                         {'action': 'trial_apply', 'counters': {}}):
            with self.assertRaises(ValueError):
                ledger.reserve('parameter_application', **metadata)
        with self.assertRaises(ValueError):
            ledger.reserve('forward', category='unknown')
        for action, cap in (('trial_apply', 4), ('trial_restore', 4), ('selected_commit', 1)):
            for index in range(cap):
                ledger.returned(ledger.reserve('parameter_application', action=action, trial_index=index))
            with self.assertRaises(ValueError):
                ledger.reserve('parameter_application', action=action)
        self.assertEqual(ledger.counters['parameter_applications'], 9)

    def test_known_return_canonicals_preserved_bad_token_duplicate_refused(self):
        rows = []
        ledger = W.OperationLedger(lambda row: rows.append(copy.deepcopy(row)))
        token = ledger.reserve('parameter_application', action='trial_apply', trial_index=0)
        for wrong in (True, '1', 0, token + 1):
            with self.assertRaises(ValueError):
                ledger.returned(wrong)
        with self.assertRaises(ValueError):
            ledger.returned(token, counters={})
        ledger.returned(token, actual_before_head_canonical_sha256=C.HEAD_CANONICAL_SHA256,
                        actual_after_head_canonical_sha256='b' * 64)
        self.assertEqual(rows[-1]['actual_before_head_canonical_sha256'], C.HEAD_CANONICAL_SHA256)
        self.assertEqual(rows[-1]['actual_after_head_canonical_sha256'], 'b' * 64)
        self.assertEqual(rows[-1]['action'], 'trial_apply')
        with self.assertRaises(ValueError):
            ledger.returned(token)

    def test_return_cannot_change_reserved_action_id_or_plan_slot(self):
        ledger = W.OperationLedger(lambda row: None)
        token = ledger.reserve('parameter_application', action='trial_apply',
                               id='trial_apply_trial_0', plan_slot=125, trial_index=0)
        for metadata in ({'action': 'selected_commit'}, {'id': 'forged'},
                         {'plan_slot': 126}, {'trial_index': 1}):
            with self.subTest(metadata=metadata):
                with self.assertRaises(ValueError):
                    ledger.returned(token, **metadata)
                self.assertIn(token, ledger.pending)
                self.assertEqual(ledger.counters['parameter_applications'], 0)
                self.assertEqual(ledger.counters['trial_parameter_applications'], 0)
        ledger.returned(token, action='trial_apply', id='trial_apply_trial_0', plan_slot=125, trial_index=0)
        self.assertEqual(ledger.counters['trial_parameter_applications'], 1)


class SourceFixtures(unittest.TestCase):
    def test_stdlib_import_blocks_numpy_PIL_neural_scipy(self):
        code = '''import importlib.abc,importlib,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path,target=None):
  if fullname.split('.')[0] in {'numpy','PIL','torch','tensorflow','jax','scipy'}:raise RuntimeError('Scientific import forbidden:'+fullname)
  return None
sys.meta_path.insert(0,Block())
w=importlib.import_module('research.codeformer_u64_png_checked_step_v1.worker')
assert not any(n.split('.')[0] in {'numpy','PIL','torch','tensorflow','jax','scipy'} for n in sys.modules)
print('INERT_WORKER_STDLIB')
'''
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.strip(), 'INERT_WORKER_STDLIB')

    def test_manual_supervisor_exact_protocol_and_actual_parent_pid(self):
        sha = 'a' * 64
        env = {C.ENV + 'SUPERVISED': '1', C.ENV + 'PROTOCOL': sha, C.ENV + 'SUPERVISOR_PID': '42'}
        W.supervisor_context(sha, environ=env, parent_pid=42)
        for key in env:
            bad = dict(env); bad[key] = 'wrong'
            with self.assertRaises(ValueError):
                W.supervisor_context(sha, environ=bad, parent_pid=42)
        with self.assertRaises(ValueError):
            W.supervisor_context(sha, environ=env, parent_pid=43)

    def test_no_autograd_backward_optimizer_and_neural_work_only_guarded_run(self):
        tree = ast.parse(SOURCE.read_bytes(), feature_version=(3, 10))
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        def owned(node):
            while node in parents:
                node = parents[node]
                if isinstance(node, ast.FunctionDef) and node.name == 'run':
                    return True
            return False
        attributes = [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)]
        self.assertNotIn('optim', attributes)
        self.assertNotIn('backward', attributes)
        actual_queries, copies = [], []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = ast.unparse(node.func)
            if name == 'torch.autograd.grad':
                actual_queries.append(node)
            if name.endswith('.copy_'):
                copies.append(node)
            if name.startswith(('torch.', 'candidate.', 'model.')) or name == 'CodeFormer':
                self.assertTrue(owned(node), name)
        self.assertEqual(actual_queries, [])
        self.assertEqual(len(copies), 1)
        top_calls = [n for top in tree.body if not isinstance(top, (ast.FunctionDef, ast.ClassDef))
                     for n in ast.walk(top) if isinstance(n, ast.Call)]
        self.assertEqual(top_calls, [])
        self.assertTrue(all(not c['gradient_queries'] for c in plan()))

    def test_actual_before_after_bytes_frozen_RNG_check_precedes_durable_return(self):
        text = SOURCE.read_text()
        reserve = text.index("token = ledger.reserve('parameter_application'")
        copy_call = text.index('parameter.copy_(', reserve)
        after = text.index('after_heads = arrays(dict(head_items))', copy_call)
        bytes_check = text.index('after_heads[key].tobytes() == destination[key].tobytes()', after)
        frozen = text.index('assert_frozen()', bytes_check)
        rng = text.index("need(rng_record() == run_rng, 'Application changed fixed run RNG", frozen)
        returned = text.index('ledger.returned(token, synchronized_return=True,', rng)
        self.assertEqual([reserve, copy_call, after, bytes_check, frozen, rng, returned],
                         sorted([reserve, copy_call, after, bytes_check, frozen, rng, returned]))
        self.assertIn('actual_before_head_canonical_sha256=before_sha, actual_after_head_canonical_sha256=after_sha', text[returned:returned + 230])
        self.assertIn('Actual copied heads differ; keep pending application unknown', text)

    def test_fourth_original_restore_before_optimization_only_selection(self):
        text = SOURCE.read_text()
        restore = text.index("if action == 'trial_restore':", text.index("token = ledger.reserve('parameter_application'"))
        original = text.index("need(after_sha == contract.HEAD_CANONICAL_SHA256", restore)
        all4 = text.index('all(len(bank) == 64 for bank in trial_banks)', original)
        choose = text.index('choice = step_selection.select_candidate(selector_bank(initial_opt)', all4)
        self.assertLess(original, all4)
        self.assertLess(all4, choose)
        self.assertIn("'all_four_original_restores_completed': True, 'monitor_records_supplied': 0", text[choose:choose + 650])
        self.assertIn("raise QualityStop('No eligible actual-PNG/RAW candidate in fixed four trials; retain original stop')", text)
        self.assertIn('Selected candidate failed unchanged terminal monitor screen; no fallback', text)

    def test_independent_original_candidate_restore_and_final_precommit_caps(self):
        text = SOURCE.read_text()
        for fragment in ('proposal = step_selection.propose_from_original_fp32(original_flat, direction, ordered,',
            'step_selection.CANDIDATE_MAXABS[index])',
            'destination = {key: value.copy() for key, value in original_heads.items()}',
            "certificate = {'restore_reversal_not_descent': True, 'expected_after': contract.HEAD_CANONICAL_SHA256}",
            'float(np.max(np.abs(actual_delta))) <= step_selection.OLD_MAXABS_STEP',
            'float(np.max(np.abs(cumulative))) <= step_selection.MAX_CUMULATIVE',
            "if action != 'trial_restore':", 'Final rounded nonrestore application lacks all8 RAW descent dots'):
            self.assertIn(fragment, text)

    def test_guards_precede_Torch_and_unrestored_failure_never_auto_applies(self):
        text = SOURCE.read_text()
        framework = text.index('import torch as torch_module')
        for fragment in ('runtime.verify_static(root, parent, expected_sha)',
            'runtime.require_vm(root, parent)',
            'supervisor_context(expected_sha, environ=os.environ, parent_pid=os.getppid())',
            "contract.LIMITS['initial_free_disk_bytes']"):
            self.assertLess(text.index(fragment), framework)
        failure = text[text.index('    except BaseException as error:', text.index("store.json('result.json'")):]
        self.assertNotIn('parameter.copy_(', failure)
        self.assertNotIn('propose_from_original_fp32(', failure)
        self.assertIn('ledger.snapshot()', failure)
        self.assertIn('return 2 if isinstance(error, QualityStop) else 1', failure)

    def test_checkpoint_nulloptimizer_allapplications_and_terminal_original_gate(self):
        text = SOURCE.read_text()
        for fragment in ("'optimizer': None", "'optimizer_constructions': 0",
            "'all_trial_restore_commit_parameter_applications': ledger.counters['parameter_applications']",
            "'automatic_resume_permitted': False", 'frozen539_canonical_sha256',
            'quality.evaluate_quality(means[0], means[1], means[1], True, inherited)',
            "'selected_training_updates': 1", "'new_autograd_queries': 0"):
            self.assertIn(fragment, text)


if __name__ == '__main__':
    unittest.main()
