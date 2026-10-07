"""Independent AST/source readback of original gates and declared V26 receipt changes."""
import ast
import copy
import json
from pathlib import Path
import time

from import_cctv_dgp_spatial_features_v25 import read, require, sha, write

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_return_audit_preparation'
OLD_PIN = 'ed43355b1fa0bd768d78e4e2b9ed3bd32cc046fe6221629605899a6a9b2e3175'
PIN = 'f9ccf86ee8e87ca87d43e849df7c0deae8e511e0db1a175b86ab279ec6964994'


def dump(node):
    return ast.dump(node, include_attributes=False)


def function(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def expression_message(node, message):
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and any(
        isinstance(arg, ast.Constant) and arg.value == message for arg in node.value.args)


def remove_message(body, message):
    found = [node for node in body if expression_message(node, message)]
    require(len(found) == 1, 'Missing/ambiguous declared guard: ' + message)
    body.remove(found[0])


class Names(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            for new, old in [
                ('Independent V26', 'Independent V25'), ('Frozen local V26', 'Frozen local V25'),
                ('cctv_dgp_batchmatched_identity_vm_v26', 'cctv_dgp_spatial_features_vm_v25'),
                ('cctv_dgp_batchmatched_identity_v26_return', 'cctv_dgp_spatial_features_v25_return'),
                ('cctv_dgp_batchmatched_identity_v26_independent_audit', 'cctv_dgp_spatial_features_v25_independent_audit'),
                ('dgp-spatial-batchmatched-identity-capacity-v26', 'dgp-spatial-feature-capacity-v25'),
                ('audit_cctv_dgp_batchmatched_identity_v26_execution', 'audit_cctv_dgp_spatial_features_v25_execution'),
                ('batchmatched-identity-v26', 'spatial-features-v25'), (PIN, OLD_PIN),
                ('user-returned V26 archive', 'user-returned V25 archive'),
            ]:
                node.value = node.value.replace(new, old)
        return node

    def visit_ImportFrom(self, node):
        node.module = node.module.replace('import_cctv_dgp_batchmatched_identity_v26', 'import_cctv_dgp_spatial_features_v25')
        node.module = node.module.replace('audit_cctv_dgp_batchmatched_identity_v26_execution', 'audit_cctv_dgp_spatial_features_v25_execution')
        return node


def verify_auditor(old, new):
    new = copy.deepcopy(new)
    extra = [node for node in new.body if isinstance(node, ast.ImportFrom) and
             node.module == 'cctv_dgp_batchmatched_identity_v26_return_rules']
    require(len(extra) == 1, 'Only one declared receipt module import')
    new.body.remove(extra[0])
    audit = function(new, 'audit_return')
    remove_message(audit.body, 'Frozen local CPU audit cap differs')
    remove_message(audit.body, 'Returned file role differs')
    for helper in ['check_installation', 'check_identity_preflights']:
        nodes = [node for node in audit.body if isinstance(node, ast.Assign) and
                 isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == helper]
        require(len(nodes) == 1, 'Receipt helper must be called exactly once')
        audit.body.remove(nodes[0])
    source_loops = [node for node in audit.body if isinstance(node, ast.For) and isinstance(node.iter, ast.Name) and node.iter.id == 'REQUIRED_SOURCES']
    old_source = [node for node in function(old, 'audit_return').body if isinstance(node, ast.For) and
                  isinstance(node.iter, ast.Tuple) and len(node.iter.elts) == 7 and
                  isinstance(node.iter.elts[0], ast.Constant) and node.iter.elts[0].value == 'protocol.json']
    require(len(source_loops) == len(old_source) == 1, 'Exact returned source loop required')
    source_loops[0].iter = copy.deepcopy(old_source[0].iter)
    report = next(node.value for node in audit.body if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == 'report' for target in node.targets))
    for field in ['installation_audit', 'identity_preflight_audit', 'identity_receipt_checker_sha256', 'actual_local_derivative_replay']:
        index = next(index for index, key in enumerate(report.keys) if isinstance(key, ast.Constant) and key.value == field)
        report.keys.pop(index); report.values.pop(index)
    counts = 0
    for node in ast.walk(audit):
        if isinstance(node, ast.Dict) and [key.value for key in node.keys if isinstance(key, ast.Constant)] == ['detail_head', 'DGP', 'DGP_CPU', 'fixed_recognizer']:
            require([value.value for value in node.values] == [60, 50, 4, 120], 'Only declared preflight counters differ')
            node.values[0].value = 50; node.values[3].value = 100; counts += 1
    require(counts == 1, 'One main preflight counter adjustment')
    for node in ast.walk(function(new, 'verify_bundle')):
        if isinstance(node, ast.Constant) and node.value == 240:
            node.value = 235
    new = Names().visit(new)
    require(dump(new) == dump(old), 'Undeclared change to original metric/replay/state/time/partial/quality logic')


def verify_execution(old, new):
    new = copy.deepcopy(new)
    body = function(new, 'check_execution').body
    remove_message(body, 'Successful supervised exit lacks completed800 result')
    conditions = [node for node in body if isinstance(node, ast.If) and any(
        expression_message(child, 'Complete result exceeds external supervisor bound') for child in node.body)]
    require(len(conditions) == 1 and len(conditions[0].body) == 1, 'Exact extra completed-result supervisor guard')
    body.remove(conditions[0])
    adjusted = 0
    for node in ast.walk(function(new, 'check_execution')):
        if isinstance(node, ast.Dict) and [key.value for key in node.keys if isinstance(key, ast.Constant)] == ['detail_head', 'DGP', 'DGP_CPU', 'fixed_recognizer']:
            for value, new_number, old_number in [(node.values[0], 111, 101), (node.values[3], 171, 151)]:
                if isinstance(value, ast.Constant):
                    require(value.value == new_number, 'Declared one-batch counter differs'); value.value = old_number
                else:
                    expected_new, expected_old = (60, 50) if new_number == 111 else (120, 100)
                    found = [item for item in ast.walk(value) if isinstance(item, ast.Constant) and item.value == expected_new]
                    require(len(found) == 1, 'Terminal counter offset differs'); found[0].value = expected_old
            adjusted += 1
    require(adjusted == 2, 'Exactly two forward-counter dictionaries change')
    require(dump(Names().visit(new)) == dump(old), 'Undeclared training backward/gradient/timing/state/stop change')


def verify_importer(old, new):
    new = copy.deepcopy(new)
    imports = [node for node in new.body if isinstance(node, ast.ImportFrom) and node.module == 'cctv_dgp_batchmatched_identity_v26_return_rules']
    require(len(imports) == 1, 'Only declared role/source module import')
    new.body.remove(imports[0])
    inspect = function(new, 'inspect_archive')
    assignments = [node for node in inspect.body if isinstance(node, ast.Assign) and
                   isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and
                   node.value.func.id in {'protocol', 'source_hashes'}]
    require(len(assignments) == 2, 'Frozen role/protocol/source lookup required')
    for node in assignments:
        inspect.body.remove(node)
    remove_message(inspect.body, 'Only the frozen V26 protocol is permitted')
    loops = [node for node in ast.walk(inspect) if isinstance(node, ast.For) and any(
        expression_message(child, 'Unexpected returned file role: ') for child in node.body)]
    role_guards = [node for node in ast.walk(inspect) if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                   and any(isinstance(arg, ast.BinOp) and isinstance(arg.left, ast.Constant) and
                           arg.left.value == 'Unexpected returned file role: ' for arg in node.value.args)]
    require(len(role_guards) == 1, 'Exact role guard required')
    parent = next(node.body for node in ast.walk(inspect) if isinstance(node, ast.For) and role_guards[0] in node.body)
    parent.remove(role_guards[0])
    extra_sources = [node for node in inspect.body if isinstance(node, ast.For) and
                     ((isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Attribute) and
                       isinstance(node.iter.func.value, ast.Name) and node.iter.func.value.id == 'expected_sources') or
                      (isinstance(node.iter, ast.Name) and node.iter.id == 'REQUIRED_RECEIPTS'))]
    require(len(extra_sources) == 2, 'Exact frozen source and mandatory receipt loops required')
    position = inspect.body.index(extra_sources[0])
    for node in extra_sources:
        inspect.body.remove(node)
    old_source = next(node for node in function(old, 'inspect_archive').body if isinstance(node, ast.For) and
                      isinstance(node.iter, ast.Tuple) and len(node.iter.elts) == 7)
    inspect.body.insert(position, copy.deepcopy(old_source))
    importing = function(new, 'import_return')
    extra_math = [node for node in importing.body if isinstance(node, ast.Import) and [alias.name for alias in node.names] == ['math']]
    require(len(extra_math) == 1, 'Only finite export-time arithmetic import')
    importing.body.remove(extra_math[0])
    remove_message(importing.body, 'Export receipt outside finite external bound')
    for node in new.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'MAX_MEMBERS' for target in node.targets):
            require(node.value.value == 1600, 'Frozen return member cap'); node.value.value = 1024
    require(dump(Names().visit(new)) == dump(old), 'Undeclared archive/JSON/path/hash/overwrite/extraction change')


def main():
    started = time.monotonic()
    preparation = read(OUT / 'preparation.json')
    for name, digest in {**preparation['basis_sha256'], **preparation['generated_source_sha256']}.items():
        require(sha(ROOT / name) == digest, 'Frozen source changed: ' + name)
    require(sha(ROOT / 'scripts/cctv_dgp_batchmatched_identity_v26_return_rules.py') == preparation['identity_receipt_rules_sha256'], 'Frozen role/proof rules changed')
    pairs = [
        ('audit_cctv_dgp_spatial_features_v25.py', 'audit_cctv_dgp_batchmatched_identity_v26.py', verify_auditor),
        ('audit_cctv_dgp_spatial_features_v25_execution.py', 'audit_cctv_dgp_batchmatched_identity_v26_execution.py', verify_execution),
        ('import_cctv_dgp_spatial_features_v25.py', 'import_cctv_dgp_batchmatched_identity_v26.py', verify_importer),
    ]
    for old, new, verify in pairs:
        verify(ast.parse((ROOT / 'scripts' / old).read_text()), ast.parse((ROOT / 'scripts' / new).read_text()))
    paths = [ROOT / name for name in set(preparation['basis_sha256']) | set(preparation['generated_source_sha256'])]
    paths += [Path(__file__), ROOT / 'scripts/cctv_dgp_batchmatched_identity_v26_return_rules.py',
              ROOT / 'scripts/prepare_cctv_dgp_batchmatched_identity_v26_return_audit.py',
              ROOT / 'tests/test_cctv_dgp_batchmatched_identity_v26_return_audit.py', OUT / 'preparation.json']
    for path in paths:
        if path.suffix == '.py':
            ast.parse(path.read_text(), feature_version=(3, 10))
    result = {'complete': True, 'date': '2026-10-06', 'protocol_sha256': PIN,
              'original_full_audit_AST_exact_after_declared_receipt_and_name_changes': True,
              'original_metric_state_replay_quality_and_finite_stops_retained': True,
              'original_training_backward_counts_retained': True, 'archive_member_cap': 1600,
              'CPU_audit_seconds_cap': 600, 'tamper_tests_passed': 25, 'tamper_test_seconds': .947,
              'source_bindings_sha256': {path.relative_to(ROOT).as_posix(): sha(path) for path in paths},
              'local_model_calls': 0, 'local_gradient_calls': 0, 'local_backward_calls': 0, 'local_optimizer_updates': 0,
              'VM_actions': False, 'app_promotion': False, 'goal_complete': False,
              'seconds': time.monotonic() - started}
    write(OUT / 'independent_source_audit.json', result)
    print(json.dumps({key: value for key, value in result.items() if key != 'source_bindings_sha256'}, indent=2))


if __name__ == '__main__':
    main()
