"""Mock graph/source tests only; never import Torch or execute a neural model."""
import ast
import copy
import gc
import importlib.abc
import inspect
import json
from pathlib import Path
import sys
import unittest
import weakref


class FrameworkBlocker(importlib.abc.MetaPathFinder):
    attempted = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
                'torch', 'torchvision', 'torchaudio', 'numpy', 'PIL',
                'tensorflow', 'jax', 'jaxlib', 'keras'} or fullname.endswith('.worker'):
            self.attempted.append(fullname)
            raise AssertionError('No framework/model/worker in mock graph tests: ' + fullname)
        return None


BLOCKER = FrameworkBlocker()
sys.meta_path.insert(0, BLOCKER)
from research.codeformer_matched_feature_learning_v1 import graph as G

ROOT = Path(__file__).resolve().parents[1]


class Node:
    def __init__(self, *children):
        self.next_functions = tuple((child, 0) for child in children)


class AccumulateGrad(Node):
    def __init__(self, variable):
        super().__init__()
        self.variable = variable


def parameters():
    return {name: object() for name in G.ADAPTER_PARAMETER_NAMES}


def connected(values):
    return Node(*(AccumulateGrad(value) for value in values.values()))


class AdapterGraphReachability(unittest.TestCase):
    def setUp(self):
        self.parameters = parameters()

    def walk(self, root, expected=True, **caps):
        return G.inspect_adapter_reachability(
            root, self.parameters, expected_present=expected, **caps)

    def test_B_requires_all_six_exact_current_parameter_objects(self):
        result = self.walk(connected(self.parameters))
        self.assertTrue(result['pass'])
        self.assertEqual(result['current_adapter_leaves_reached'], 6)
        self.assertEqual(result['reachable_parameter_names'], sorted(self.parameters))
        self.assertEqual(result['visited_nodes'], 7)
        self.assertEqual(result['next_function_edges'], 6)
        self.assertFalse(result['nonzero_or_finite_branch_gradient_proven'])
        self.assertEqual(result['autograd_queries_or_backward_calls'], 0)

    def test_each_missing_parameter_fails_instead_of_accepting_a_partial_route(self):
        for excluded in self.parameters:
            values = {name: value for name, value in self.parameters.items() if name != excluded}
            with self.subTest(excluded=excluded), self.assertRaisesRegex(ValueError, 'All six'):
                self.walk(connected(values))

    def test_same_names_with_different_objects_are_not_current_adapter_leaves(self):
        with self.assertRaisesRegex(ValueError, 'All six'):
            self.walk(connected(parameters()))

    def test_A_requires_absence_and_accepts_detached_None_root(self):
        self.assertTrue(self.walk(None, expected=False)['pass'])
        result = self.walk(connected(parameters()), expected=False)
        self.assertEqual(result['current_adapter_leaves_reached'], 0)
        for name, value in self.parameters.items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'must be absent'):
                self.walk(Node(AccumulateGrad(value)), expected=False)
        with self.assertRaisesRegex(ValueError, 'All six'):
            self.walk(None)

    def test_shared_DAG_and_None_edges_are_legal_and_counted_once_per_node(self):
        shared = connected(self.parameters)
        root = Node(Node(shared, None), Node(shared), shared)
        result = self.walk(root)
        self.assertEqual(result['visited_nodes'], 10)
        self.assertEqual(result['next_function_edges'], 12)
        self.assertEqual(result['all_accumulate_grad_leaves'], 6)

    def test_unrelated_head_leaves_do_not_replace_or_hide_current_adapters(self):
        root = Node(connected(self.parameters), AccumulateGrad(object()))
        result = self.walk(root)
        self.assertEqual(result['current_adapter_leaves_reached'], 6)
        self.assertEqual(result['all_accumulate_grad_leaves'], 7)

    def test_a_cycle_is_rejected_but_a_completed_shared_node_is_not_a_cycle(self):
        root = Node()
        root.next_functions = ((root, 0),)
        with self.assertRaisesRegex(ValueError, 'Cycle'):
            self.walk(root, expected=False)
        a, b = Node(), Node()
        a.next_functions = ((b, 0),)
        b.next_functions = ((a, 0),)
        with self.assertRaisesRegex(ValueError, 'Cycle'):
            self.walk(a, expected=False)

    def test_node_and_edge_caps_fail_closed(self):
        root = connected(self.parameters)
        with self.assertRaisesRegex(ValueError, 'node cap'):
            self.walk(root, max_nodes=6)
        with self.assertRaisesRegex(ValueError, 'edge cap'):
            self.walk(root, max_edges=5)
        self.assertTrue(self.walk(root, max_nodes=7, max_edges=6)['pass'])

    def test_caps_and_policy_require_positive_exact_bounded_integers_and_bool(self):
        for field, value in [('max_nodes', 0), ('max_nodes', True), ('max_nodes', 1.0),
                             ('max_nodes', G.MAX_NODES + 1), ('max_edges', 0),
                             ('max_edges', False), ('max_edges', G.MAX_EDGES + 1)]:
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, 'caps'):
                self.walk(None, expected=False, **{field: value})
        for policy in (0, 1, None, 'B'):
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, 'presence policy'):
                self.walk(None, expected=policy)

    def test_six_distinct_parameter_objects_and_exact_names_are_mandatory(self):
        for values in ({}, dict(self.parameters, extra=object()),
                       dict.fromkeys(G.ADAPTER_PARAMETER_NAMES, object()),
                       {**self.parameters, G.ADAPTER_PARAMETER_NAMES[0]: None}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                G.inspect_adapter_reachability(None, values, expected_present=False)

    def test_malformed_next_functions_entries_and_child_nodes_are_rejected(self):
        for following in ([], ((None,),), ((None, True),), ((None, -1),),
                          ((None, 0.0),), ([None, 0],), ((123, 0),)):
            root = Node()
            root.next_functions = following
            with self.subTest(following=following), self.assertRaises(ValueError):
                self.walk(root, expected=False)

    def test_malformed_accumulate_grad_and_impostor_variable_nodes_are_rejected(self):
        missing = AccumulateGrad(object())
        del missing.variable
        outgoing = AccumulateGrad(object())
        outgoing.next_functions = ((Node(), 0),)
        impostor = Node()
        impostor.variable = self.parameters[G.ADAPTER_PARAMETER_NAMES[0]]
        for node in (missing, outgoing, impostor, AccumulateGrad(None)):
            with self.subTest(node=type(node).__name__), self.assertRaises(ValueError):
                self.walk(node, expected=False)

    def test_unreadable_node_properties_are_a_bounded_failure(self):
        class Broken:
            @property
            def next_functions(self):
                raise RuntimeError('mock unreadable graph')
        with self.assertRaisesRegex(ValueError, 'Unreadable'):
            self.walk(Broken(), expected=False)

    def test_iterative_walk_handles_depth_above_Python_recursion_limit(self):
        root = connected(self.parameters)
        for _ in range(3000):
            root = Node(root)
        self.assertEqual(self.walk(root)['visited_nodes'], 3007)

    def test_ephemeral_node_wrappers_remain_alive_during_walk_and_are_released(self):
        wrappers = []

        class Ephemeral:
            def __init__(self, depth):
                self.depth = depth
                wrappers.append(weakref.ref(self))

            @property
            def next_functions(self):
                # Every prior wrapper must still be held strongly by the walker.
                if any(reference() is None for reference in wrappers):
                    raise RuntimeError('node-wrapper reference lost before traversal end')
                return ((Ephemeral(self.depth - 1), 0),) if self.depth else ()

        root = Ephemeral(100)
        result = self.walk(root, expected=False)
        self.assertEqual(result['visited_nodes'], 101)
        gc.collect()
        self.assertIsNotNone(wrappers[0]())
        self.assertTrue(all(reference() is None for reference in wrappers[1:]))

    def test_returned_metadata_has_no_node_or_parameter_references(self):
        root = connected(self.parameters)
        result = self.walk(root)
        encoded = json.dumps(result, allow_nan=False)
        self.assertIn('structural_reachability_only', encoded)

        def inspect_value(value):
            self.assertTrue(type(value) in (dict, list, str, bool, int))
            if type(value) is dict:
                for key, item in value.items():
                    self.assertIs(type(key), str)
                    inspect_value(item)
            elif type(value) is list:
                for item in value:
                    inspect_value(item)
        inspect_value(result)

    def test_structural_paths_do_not_claim_numerically_nonzero_gradients(self):
        # No operation/gradient values occur in this mock graph at all.
        result = self.walk(connected(self.parameters))
        self.assertTrue(result['structural_reachability_only'])
        self.assertFalse(result['nonzero_or_finite_branch_gradient_proven'])

    def test_graph_module_and_model_instrumentation_are_source_only(self):
        graph_tree = ast.parse(inspect.getsource(G), feature_version=(3, 10))
        self.assertFalse(any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(graph_tree)))
        graph_calls = {node.func.attr for node in ast.walk(graph_tree)
                       if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertFalse({'backward', 'grad', 'forward', 'clone', 'detach'} & graph_calls)
        source = (ROOT / 'research/codeformer_matched_feature_learning_v1/model.py').read_text()
        tree = ast.parse(source, feature_version=(3, 10))
        factory = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'build_restorer')
        head = next(node for node in factory.body if isinstance(node, ast.ClassDef) and node.name == 'SpatialHead')
        forward = next(node for node in head.body if isinstance(node, ast.FunctionDef) and node.name == 'forward')
        encoder_calls = [node for node in ast.walk(forward) if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Attribute) and node.func.attr == 'encoder']
        self.assertEqual(len(encoder_calls), 1)
        self.assertIn('encoded = self.encoder(observed)', source)
        self.assertIn('encoder_graph_witness(encoded.grad_fn)', source)
        self.assertIn("witness_branch(site, 'stock_SFT', fused.grad_fn)", source)
        self.assertIn('UNWITNESSED_NO_GRAD', source)
        self.assertFalse({'torch', 'numpy', 'PIL'} & set(sys.modules))
        self.assertEqual(BLOCKER.attempted, [])

    def test_inlining_existing_encoder_temporary_preserves_original_A_math_AST(self):
        def head_forward(relative):
            tree = ast.parse((ROOT / relative).read_text(), feature_version=(3, 10))
            factory = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'build_restorer')
            head = next(node for node in factory.body if isinstance(node, ast.ClassDef) and node.name == 'SpatialHead')
            return next(node for node in head.body if isinstance(node, ast.FunctionDef) and node.name == 'forward')

        old = head_forward('research/codeformer_u96_postfusion_v1/model.py')
        new = head_forward('research/codeformer_matched_feature_learning_v1/model.py')
        encoder_call = next(node.value for node in new.body if isinstance(node, ast.Assign)
                            and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'encoded')

        class InlineA(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id == 'encoded':
                    return self.visit(copy.deepcopy(encoder_call))
                if node.id == 'observed':
                    return ast.parse('used.detach()', mode='eval').body
                return node

        def equations(forward):
            return [node for node in forward.body if isinstance(node, ast.Return) or
                    isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and
                    node.targets[0].id in {'context', 'mixed', 'logits'}]

        expected = [ast.dump(node, include_attributes=False) for node in equations(old)]
        actual = [ast.dump(InlineA().visit(copy.deepcopy(node)), include_attributes=False)
                  for node in equations(new)]
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
