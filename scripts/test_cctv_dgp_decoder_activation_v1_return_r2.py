"""Metadata-only R2 schema/delta rejection fixtures; no heavy arrays or inference."""
import ast
import copy
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import time
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT/'scripts'
sys.path.insert(0,str(SCRIPTS))
import audit_cctv_dgp_decoder_activation_v1_return as original

ANALYSIS = ROOT/'outputs/cctv_dgp_decoder_activation_v1_analysis'
RETURN = ROOT/'outputs/cctv_dgp_decoder_activation_v1_return'
REPAIR = ROOT/'outputs/cctv_dgp_decoder_hinge_repair_v1_return'
SOURCE = ROOT/'outputs/cctv_dgp_spatial_readout_v1_return'
ADAPTER = SCRIPTS/'audit_cctv_dgp_decoder_activation_v1_return_r2.py'
adapter = None


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


class TypedRecords(unittest.TestCase):
    def test_valid_nested_record_lists_keep_numeric_bounds(self):
        left={'rows':[{'flag':False,'text':'held_out','maybe':None,'values':[1.,2.]},{'flag':True,'text':'TRAIN','maybe':None,'values':[3.,4.]}]}
        right=copy.deepcopy(left);right['rows'][0]['values'][1]+=5e-11
        self.assertGreaterEqual(adapter.typed_record_compare(left,right),0.)
        with self.assertRaises((AssertionError,ValueError,TypeError)):
            adapter.typed_record_compare(left,right,atol=0.,rtol=0.)

    def test_record_list_lengths_and_keys_are_exact(self):
        left=[{'a':1.,'b':'text'},{'a':2.,'b':'text'}]
        for changed in (left[:1],[{'a':1.,'other':'text'},left[1]],{'0':left[0],'1':left[1]}):
            with self.subTest(changed=changed),self.assertRaises((AssertionError,ValueError,TypeError)):
                adapter.typed_record_compare(left,changed)

    def test_boolean_string_None_and_container_types_do_not_coerce(self):
        for left,right in [(True,1),(False,0),(1,True),(0,False),('1',1),(1,'1'),(None,0),(0,None),([],{}),([1.],(1.,)),({'flag':True},{'flag':1})]:
            with self.subTest(left=left,right=right),self.assertRaises((AssertionError,ValueError,TypeError)):
                adapter.typed_record_compare(left,right)

    def test_nonnumeric_unchanged_values_and_finite_numbers(self):
        for value in (True,False,None,'TRAIN',{'flag':False,'value':1.},[{'flag':False,'value':1.}]):
            self.assertEqual(adapter.typed_record_compare(value,copy.deepcopy(value)),0.)
        for value in (float('nan'),float('inf'),float('-inf')):
            with self.subTest(value=value),self.assertRaises((AssertionError,ValueError,TypeError)):
                adapter.typed_record_compare(value,value)


class SavedDelta(unittest.TestCase):
    def rows(self):
        for label in ('initial','probe_1_16','probe_1_4','fit_u001'):
            yield from read(RETURN/'outputs/decoder_activation_v1'/label/'objective_components.json')['rows']

    def test_all80_saved_VM_delta_relations_are_exact(self):
        rows=list(self.rows());self.assertEqual(len(rows),80)
        for row in rows:adapter.verify_saved_delta(row)

    def test_one_ULP_delta_tamper_rejected(self):
        rows=list(self.rows())
        for row in (rows[0],next(r for r in rows if r['finite_minus_torch_objective']!=0.)):
            changed=copy.deepcopy(row)
            changed['finite_minus_torch_objective']=math.nextafter(row['finite_minus_torch_objective'],math.inf)
            with self.assertRaises((AssertionError,ValueError,TypeError)):adapter.verify_saved_delta(changed)

    def test_stored_delta_must_not_be_replaced_by_local_recomputation(self):
        # This retains the saved operands as the authoritative subtraction even
        # when the local scalar has an independently bounded reduction drift.
        row=copy.deepcopy(next(self.rows()))
        row['objective']=1.
        row['finite_arithmetic']['objective']=math.nextafter(1.,math.inf)
        row['finite_minus_torch_objective']=row['finite_arithmetic']['objective']-row['objective']
        adapter.verify_saved_delta(row)
        changed=copy.deepcopy(row);changed['finite_minus_torch_objective']=0.
        with self.assertRaises((AssertionError,ValueError,TypeError)):adapter.verify_saved_delta(changed)


class ProspectiveSchema(unittest.TestCase):
    def test_actual_metadata_preflight_uses_no_heavy_arrays(self):
        import numpy as np
        p=read(RETURN/'protocol.json');old=read(SOURCE/'protocol.json')
        with patch.object(np,'load',side_effect=AssertionError('No heavy array reads in schema preflight')):
            evidence=adapter.preflight_saved_schema(RETURN,REPAIR,SOURCE,p,old)
        self.assertIsInstance(evidence,dict)
        self.assertTrue(evidence['complete'])
        self.assertEqual(len(evidence['objective_rows']),80)
        self.assertEqual(evidence['exact_stored_VM_delta_rows'],80)
        self.assertEqual(evidence['gate_records'],8)
        self.assertEqual(len(evidence['gradient_records']),2)

    def test_objective_and_gradient_metadata_corruption_rejected(self):
        p=read(RETURN/'protocol.json');old=read(SOURCE/'protocol.json')
        objective=RETURN/'outputs/decoder_activation_v1/initial/objective_components.json'
        gradient=RETURN/'outputs/decoder_activation_v1/gradients/r12/measurement.json'
        changes=[(objective,lambda x:x['rows'].pop()),
                 (objective,lambda x:x['rows'][0].update(raw_control_exact=1)),
                 (objective,lambda x:x['rows'][0]['current']['MSE'].__setitem__(0,'0.0')),
                 (objective,lambda x:x['rows'][0].update(extra_field=False)),
                 (gradient,lambda x:x['scalar_values'].pop()),
                 (gradient,lambda x:x.update(optimizer_updates=False)),
                 (gradient,lambda x:x.update(extra_field=False))]
        actual_read=adapter.read
        for target,change in changes:
            with self.subTest(path=str(target),change=str(change)):
                modified=copy.deepcopy(read(target));change(modified)
                def substitute(path):
                    return copy.deepcopy(modified) if Path(path)==target else actual_read(path)
                with patch.object(adapter,'read',side_effect=substitute),self.assertRaises((AssertionError,ValueError,TypeError,KeyError)):
                    adapter.preflight_saved_schema(RETURN,REPAIR,SOURCE,p,old)

    def test_retained_gate_boolean_to_int_rejected_before_heavy_arrays(self):
        import numpy as np
        p=read(RETURN/'protocol.json');old=read(SOURCE/'protocol.json')
        target=REPAIR/'outputs/decoder_hinge_repair_v1/probe_1_16_outcome.json'
        retained=read(target);modified=copy.deepcopy(retained)
        self.assertIs(modified['gates']['raw']['numerical_sample_gate_only'],True)
        modified['gates']['raw']['numerical_sample_gate_only']=1
        # Python dictionary equality silently accepts True == 1.  The schema
        # admission must reject this historical JSON type change explicitly.
        self.assertEqual(retained,modified)
        actual_read=adapter.read
        def substitute(path):
            return copy.deepcopy(modified) if Path(path)==target else actual_read(path)
        with patch.object(np,'load',side_effect=AssertionError('No heavy array reads in schema preflight')):
            with patch.object(adapter,'read',side_effect=substitute),self.assertRaisesRegex(AssertionError,'type'):
                adapter.preflight_saved_schema(RETURN,REPAIR,SOURCE,p,old)

    def test_original_exact_reduction_failure_remains_false(self):
        row=read(RETURN/'outputs/decoder_activation_v1/probe_1_16/objective_components.json')['rows'][0]
        scalar=copy.deepcopy(row['finite_arithmetic']);changed=copy.deepcopy(scalar)
        changed['objective']=math.nextafter(scalar['objective'],math.inf)
        proof=adapter.prior.compare_finite_objective_only(changed,scalar,1e-10)
        self.assertIs(proof['original_exact_objective_check_passed'],False)
        self.assertFalse(proof['new_or_wider_bound'])
        with self.assertRaises(AssertionError):original.close_tree(changed,scalar,atol=0.,rtol=0.)

    def test_original_source_not_rewritten_and_only_frozen_adapter_scope(self):
        norm,objective,delta=[],[],[]
        module,derived=adapter.load_adapter_module(norm,objective,delta)
        self.assertIsInstance(derived,str);ast.parse(derived,feature_version=(3,10))
        self.assertIsNot(module,original)
        self.assertIsNotNone(ast.parse(Path(original.__file__).read_text(encoding='utf-8')))
        self.assertIn("direction_record['norm_L2'] == float(np.linalg.norm(direction)) <= p['common_radius_L2']",Path(original.__file__).read_text(encoding='utf-8'))
        self.assertIn("close_tree(finite_saved,row['finite_arithmetic'],atol=0,rtol=0)",Path(original.__file__).read_text(encoding='utf-8'))
        self.assertEqual(digest(Path(original.__file__)),adapter.prior.ORIGINAL_SHA)
        tree=ast.parse(derived)
        counts={'_r1_verify_exact_direction_radius':0,'_r1_compare_finite_objective_only':0,
                '_r2_verify_saved_delta':0,'_r2_typed_record_compare':0}
        class UndoDeclaredChanges(ast.NodeTransformer):
            def visit_Assert(self,node):
                if isinstance(node.test,ast.Call) and isinstance(node.test.func,ast.Name):
                    name=node.test.func.id
                    if name=='_r1_verify_exact_direction_radius':
                        counts[name]+=1;node.test=ast.parse(adapter.prior.NORM_ASSERTION,mode='eval').body
                    elif name=='_r2_verify_saved_delta':
                        counts[name]+=1;node.test=ast.parse(adapter.DELTA_ASSERTION,mode='eval').body
                return self.generic_visit(node)
            def visit_Call(self,node):
                if isinstance(node.func,ast.Name):
                    if node.func.id=='_r1_compare_finite_objective_only':
                        counts[node.func.id]+=1
                        return ast.parse(adapter.prior.FINITE_CALL,mode='eval').body
                    if node.func.id=='_r2_typed_record_compare':
                        counts[node.func.id]+=1;node.func.id='close_tree'
                return self.generic_visit(node)
        restored=UndoDeclaredChanges().visit(tree)
        original_tree=ast.parse(Path(original.__file__).read_text(encoding='utf-8'))
        self.assertEqual(ast.dump(restored,include_attributes=False),ast.dump(original_tree,include_attributes=False))
        self.assertEqual(counts,{'_r1_verify_exact_direction_radius':1,'_r1_compare_finite_objective_only':1,
                                 '_r2_verify_saved_delta':1,'_r2_typed_record_compare':2})
        typed_calls=[n for n in ast.walk(ast.parse(derived)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_r2_typed_record_compare']
        self.assertEqual(sorted(ast.unparse(n) for n in typed_calls),sorted([adapter.GATE_CALL.replace('close_tree','_r2_typed_record_compare'),adapter.EVENT_CALL.replace('close_tree','_r2_typed_record_compare')]))


def main():
    global adapter
    started=time.monotonic();log=io.StringIO()
    red='--red-original' in sys.argv
    if red:
        adapter=types.SimpleNamespace(typed_record_compare=original.close_tree)
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(TypedRecords)
    else:
        assert ADAPTER.is_file(),'Wait for the separately reviewed local R2 adapter'
        spec=importlib.util.spec_from_file_location('activation_schema_R2_under_test',ADAPTER)
        adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
        suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    folder=ANALYSIS/'r2_schema_tests';folder.mkdir(exist_ok=True)
    n=1
    while (folder/f'run_{n:03d}.json').exists():n+=1
    text=Path(__file__).read_text(encoding='utf-8');ast.parse(text,feature_version=(3,10));compile(text,str(__file__),'exec')
    files=[Path(__file__),Path(original.__file__)]+([] if red else [ADAPTER])
    evidence=dict(complete=result.wasSuccessful(),mode='RED_original_container_comparator' if red else 'R2_metadata_schema',
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),seconds=time.monotonic()-started,
        sources_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in files},
        log_sha256=hashlib.sha256(log.getvalue().encode('utf-8')).hexdigest(),image_decodes=0,model_forwards=0,
        gradient_queries=0,optimizer_updates=0,VM_connections=0,heavy_array_reads=0,scientific_gates_changed=False)
    for path,value in [(folder/f'run_{n:03d}.txt',log.getvalue()),(folder/f'run_{n:03d}.json',json.dumps(evidence,indent=2)+'\n')]:
        with path.open('x',encoding='utf-8',newline='\n') as stream:stream.write(value)
    print(log.getvalue(),end='');print(json.dumps(evidence))
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())
