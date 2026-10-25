"""Pure worker ledger/import-path/static-contract fixtures; no model runtime."""
import ast
import importlib.machinery
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'research/codeformer_u64_role_gradients_v1/worker.py'
FRAMEWORKS_BEFORE={name for name in sys.modules if name.split('.')[0] in ('torch','numpy')}
SPEC=importlib.util.spec_from_file_location('role_gradients_worker_pure_fixture',SOURCE)
W=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(W)
FRAMEWORKS_AFTER={name for name in sys.modules if name.split('.')[0] in ('torch','numpy')}


class Ledger(unittest.TestCase):
    def test_import_is_inert_no_framework(self):
        self.assertEqual(FRAMEWORKS_AFTER,FRAMEWORKS_BEFORE)

    def test_72_known_forwards_and192_queries(self):
        rows=[];ledger=W.OperationLedger(rows.append)
        for category,maximum in W.CATEGORIES.items():
            for _ in range(maximum):
                token=ledger.reserve('forward',category=category);ledger.returned(token,synchronized_return=True)
                if category=='diagnostic_forwards':
                    for name in W.QUERIES:
                        token=ledger.reserve('gradient_query',query=name);ledger.returned(token,synchronized_return=True)
        self.assertEqual(ledger.counters['full_forward_returns'],72)
        self.assertEqual(ledger.counters['gradient_query_returns'],192)
        self.assertFalse(ledger.snapshot()['operation_outcome_unknown'])
        self.assertEqual(len(rows),528)

    def test_no_optimizer_application_or_backward_operation(self):
        ledger=W.OperationLedger(lambda row:None)
        for kind in ('optimizer_construction','optimizer_update','application','backward'):
            with self.assertRaises(ValueError):ledger.reserve(kind)

    def test_query_reservation_stays_unknown_until_durable_return(self):
        rows=[];ledger=W.OperationLedger(rows.append);token=ledger.reserve('gradient_query',query='gate_mae')
        self.assertTrue(ledger.snapshot()['operation_outcome_unknown'])
        self.assertEqual(ledger.counters['gradient_query_returns'],0)
        ledger.returned(token,synchronized_return=True)
        self.assertEqual(ledger.counters['gradient_query_returns'],1)

    def test_return_write_failure_does_not_invent_return(self):
        def persist(row):
            if row['kind'].endswith('_returned'):raise OSError('synthetic fsync failure')
        ledger=W.OperationLedger(persist);token=ledger.reserve('gradient_query')
        with self.assertRaises(OSError):ledger.returned(token)
        self.assertEqual(ledger.counters['gradient_query_returns'],0)
        self.assertTrue(ledger.snapshot()['operation_outcome_unknown'])

    def test_no_overlapping_reservation_or_duplicate_return(self):
        ledger=W.OperationLedger(lambda row:None);token=ledger.reserve('forward',category='parity_forwards')
        with self.assertRaises(ValueError):ledger.reserve('gradient_query')
        ledger.returned(token)
        with self.assertRaises(ValueError):ledger.returned(token)

    def test_category_and_total_query_caps(self):
        ledger=W.OperationLedger(lambda row:None)
        for _ in range(4):ledger.returned(ledger.reserve('forward',category='parity_forwards'))
        with self.assertRaises(ValueError):ledger.reserve('forward',category='parity_forwards')
        for _ in range(192):ledger.returned(ledger.reserve('gradient_query'))
        with self.assertRaises(ValueError):ledger.reserve('gradient_query')

    def test_boolean_return_token_and_reserved_metadata_refused(self):
        ledger=W.OperationLedger(lambda row:None);token=ledger.reserve('gradient_query')
        with self.assertRaises(ValueError):ledger.returned(True)
        with self.assertRaises(ValueError):ledger.returned(token,counters={})


class Boundaries(unittest.TestCase):
    def test_supervisor_requires_real_parent_pid_and_own_seal(self):
        sha='a'*64;env={'FORENSIC_U64_ROLE_GRADIENTS_SUPERVISED':'1',
            'FORENSIC_U64_ROLE_GRADIENTS_PROTOCOL':sha,'FORENSIC_U64_ROLE_GRADIENTS_SUPERVISOR_PID':'42'}
        W.supervisor_context(sha,environ=env,parent_pid=42)
        with self.assertRaises(ValueError):W.supervisor_context(sha,environ=env,parent_pid=43)
        with self.assertRaises(ValueError):W.supervisor_context('b'*64,environ=env,parent_pid=42)

    def test_bare_staged_regular_and_namespace_research_paths_resolve_parent_history(self):
        (ROOT/'scratch').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'scratch') as directory:
            base=Path(directory);parent=base/'parent';stage=base/'stage'
            (parent/'research'/'historical').mkdir(parents=True);(stage/'research'/'newpackage').mkdir(parents=True)
            (parent/'research'/'historical'/'__init__.py').write_text('MARKER="parent"\n')
            (stage/'research'/'newpackage'/'__init__.py').write_text('MARKER="stage"\n')
            (stage/'research'/'__init__.py').write_text('"""Inert stage namespace."""\n')
            class IterableNamespace:
                def __iter__(self):return iter([str(stage/'research')])
            for package_path in ([str(stage/'research')],IterableNamespace()):
                package=types.SimpleNamespace(__path__=package_path);before=list(sys.path)
                try:
                    W.install_parent_import_paths(parent,package)
                    historical=importlib.machinery.PathFinder.find_spec('research.historical',package.__path__)
                    new=importlib.machinery.PathFinder.find_spec('research.newpackage',package.__path__)
                    self.assertEqual(Path(historical.origin),parent/'research'/'historical'/'__init__.py')
                    self.assertEqual(Path(new.origin),stage/'research'/'newpackage'/'__init__.py')
                    self.assertEqual(sys.path[0],str(parent))
                finally:sys.path[:]=before

    def test_no_actual_run_at_module_top_level(self):
        tree=ast.parse(SOURCE.read_bytes(),feature_version=(3,10))
        imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
        self.assertFalse(any((getattr(n,'module','') or '').startswith(('torch','numpy')) for n in imports))
        self.assertFalse(any(isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) for n in tree.body))

    def test_no_optimizer_constructor_step_or_backward_call_site(self):
        tree=ast.parse(SOURCE.read_bytes());attributes=[n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)]
        self.assertNotIn('optim',attributes);self.assertNotIn('backward',attributes);self.assertNotIn('step',attributes)
        grad=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='grad']
        self.assertEqual(len(grad),1)
        keywords={k.arg:k.value for k in grad[0].keywords}
        self.assertIs(keywords['allow_unused'].value,True)
        self.assertIs(keywords['create_graph'].value,False)
        self.assertIs(keywords['materialize_grads'].value,False)

    def test_source_contains_exact_parent_clamp_and_readback_boundaries(self):
        text=SOURCE.read_text()
        for item in ['runtime.verify_static(root,parent,expected_sha)','install_parent_import_paths(parent,research)',
                     'native.clamp(0,1)','initial[n].tobytes()','decoded[n].tobytes()','gradient_fields_all_None',
                     'unused_names_explicit_zero','with query_warning_scope():','synchronized_return=True']:
            self.assertIn(item,text)

    def test_input_and_target_use_original_contiguous_CHW_before_GPU(self):
        import numpy as np
        observed=np.zeros((256,256,3),np.uint8)
        contiguous=observed.transpose(2,0,1).copy()
        self.assertTrue(contiguous.flags.c_contiguous)
        self.assertEqual(contiguous.strides,(65536,256,1))
        self.assertFalse(observed.transpose(2,0,1).flags.c_contiguous)
        tree=ast.parse(SOURCE.read_bytes())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
               and n.func.attr=='from_numpy']
        self.assertEqual(len(calls),2)
        for node in calls:
            expression=ast.unparse(node.args[0])
            self.assertIn('.transpose(2, 0, 1).copy()',expression)

    def test_initial8GiB_guard_precedes_Torch_import(self):
        text=SOURCE.read_text()
        self.assertLess(text.index("contract.LIMITS['initial_free_disk_bytes']"),text.index('import torch as torch_module'))
        self.assertIn('Initial8GiB free disk admission required before Torch',text)


if __name__=='__main__':unittest.main()
