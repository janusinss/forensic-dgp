"""Independent B loading-only revision, full transfer bytes and stop-boundary audit."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(node):
    return ast.dump(node, include_attributes=False)


def main():
    start = time.monotonic()
    receipt = OUT / 'cctv_dgp_bank_comparison_B3_transfer_v1_audit.json'
    assert not receipt.exists()
    prep = read(OUT / 'cctv_dgp_bank_comparison_v1_r3_B_preparation.json')
    assert prep['complete'] and prep['manual_VM_run_pending']
    prior = OUT / 'cctv_dgp_bank_comparison_v1_r2_B_vm'
    root = OUT / prep['name']
    old = read(prior / 'protocol.json'); new = read(root / 'protocol.json')
    assert sha(prior / 'protocol.json') == prep['prior_B2_protocol_sha256']
    assert sha(root / 'protocol.json') == prep['protocol_sha256']
    allowed = {'format', 'UTC', 'assets_sha256', 'initializer_loading_policy',
        'initializer_preflight_before_baseline', 'retained_B2_protocol_sha256',
        'constructor_tensor_differences_recorded', 'only_new_style_conditioner_constructor_differences_permitted',
        'CPU_VM_constructor_specific_cause_unproven'}
    changed = {k for k in set(old) | set(new) if old.get(k) != new.get(k)}
    assert changed <= allowed and new['budgets'] == old['budgets']
    assert new['maximum_updates_per_route'] == 50 and new['reserved_final_pixels'] == 0
    assert new['initializer_loading_policy'] == 'hash-pinned-strict-state-load-v1'
    assert new['initializer_preflight_before_baseline']
    modified = {'cctv_dgp_bank_comparison_v1_contract.py', 'cctv_dgp_bank_comparison_v1_model.py',
                'scripts/cctv_dgp_bank_comparison_v1_vm.py'}
    assert {n for n, h in old['assets_sha256'].items() if new['assets_sha256'][n] != h} == modified
    for name, expected in old['assets_sha256'].items():
        assert sha(prior / name) == expected
        assert sha(root / name) == new['assets_sha256'][name]
    for name in ['initializers/initial_B.pth', 'initializers/mean_style.npy', 'weights/dgp_v2.pth',
                 'weights/w600k_r50.onnx']:
        assert sha(root / name) == sha(prior / name)
    before = ast.parse((prior / 'cctv_dgp_bank_comparison_v1_model.py').read_text())
    after = ast.parse((root / 'cctv_dgp_bank_comparison_v1_model.py').read_text())
    for tree in [before, after]:
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'NAME' for t in node.targets):
                node.value = ast.Constant('ROOT_ONLY')
    assert dump(before) == dump(after)
    trees = [ast.parse((p / 'scripts/cctv_dgp_bank_comparison_v1_vm.py').read_text()) for p in [prior, root]]
    functions = [{n.name: dump(n) for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)} for tree in trees]
    unchanged = ['clock', 'objective', 'batch', 'frozen', 'save_state', 'metrics', 'base_raw',
                 'export', 'vm', 'pixels', 'tensor']
    assert all(functions[0][n] == functions[1][n] for n in unchanged)
    source = (root / 'scripts/cctv_dgp_bank_comparison_v1_vm.py').read_text()
    assert "assert state_hash(model) == state_hash(initial), 'Fresh initializer differs'" not in source
    assert "assert state_hash(model) == loading['authoritative_state_sha256']" in source
    assert source.index("candidate = create('B', learning=False)") < source.index("baseline = metrics('baseline', original.net)")
    assert 'constructor_differences' in (root / 'cctv_dgp_bank_pinned_initializer_v1.py').read_text()
    regression = read(OUT / 'cctv_dgp_bank_comparison_v1_r2_B_analysis/pinned_loader_regression.json')
    assert regression['complete'] and regression['cases_checked'] == 5
    assert sha(root / 'cctv_dgp_bank_pinned_initializer_v1.py') == regression['loader_sha256']
    spec = importlib.util.spec_from_file_location('B3_independent_contract', root / 'cctv_dgp_bank_comparison_v1_contract.py')
    contract = importlib.util.module_from_spec(spec); spec.loader.exec_module(contract)
    contract.verify(root, prep['protocol_sha256'])
    rejected = []
    for name in ['lower_gate', 'final_leak', 'extra_updates', 'alter_schedule', 'alter_initializer_policy', 'alter_budget']:
        p = copy.deepcopy(new)
        if name == 'lower_gate': p['scientific_thresholds']['early_structure_gain'] = .001
        elif name == 'final_leak': p['cases'][0]['role'] = 'final'
        elif name == 'extra_updates': p['maximum_updates_per_route'] = 100
        elif name == 'alter_schedule': p['schedule'][1] = p['schedule'][0]
        elif name == 'alter_initializer_policy': p['initializer_loading_policy'] = 'unverified-constructor'
        else: p['budgets']['arm_fit_seconds'] = 1200
        try:
            contract.validate(p)
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError('Unsafe mutation accepted: '+name)
    cli = subprocess.run([sys.executable, '-B', str(root / 'scripts/cctv_dgp_bank_comparison_v1_vm.py'),
        '--root', str(root), '--protocol-sha', prep['protocol_sha256'], '--verify-transfer'],
        capture_output=True, text=True, check=True, timeout=120)
    status = json.loads(cli.stdout)
    assert status['complete'] and status['neural_or_gradient_calls'] == status['optimizer_updates'] == 0
    archive = OUT / (prep['stem']+'-execution.tar.gz')
    assert archive.stat().st_size == prep['archive_bytes'] and sha(archive) == prep['archive_sha256']
    assert Path(str(archive)+'.sha256').read_bytes() == (prep['archive_sha256']+'  '+archive.name+'\n').encode('ascii')
    expected = {**new['assets_sha256'], 'protocol.json': prep['protocol_sha256']}; seen = set()
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            path = PurePosixPath(member.name)
            assert member.isfile() and not member.issym() and not member.islnk()
            assert path.parts[0] == prep['name'] and not path.is_absolute() and '..' not in path.parts
            assert ':' not in member.name and '\\' not in member.name
            name = PurePosixPath(*path.parts[1:]).as_posix()
            assert name in expected and name not in seen
            with tar.extractfile(member) as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected[name]
            seen.add(name)
    assert seen == set(expected)
    record = dict(complete=True, checker_sha256=sha(__file__), preparation_sha256=sha(OUT / 'cctv_dgp_bank_comparison_v1_r3_B_preparation.json'),
        protocol_sha256=prep['protocol_sha256'], archive_sha256=prep['archive_sha256'], archive_members_checked=len(seen),
        model_computation_AST_unchanged=True, unchanged_worker_functions=unchanged,
        exact_initializer_weights_data_schedule_scientific_gates_and_resource_caps=True,
        added_changes=sorted(changed), initializer_before_long_baseline=True,
        loader_regression_sha256=sha(OUT / 'cctv_dgp_bank_comparison_v1_r2_B_analysis/pinned_loader_regression.json'),
        unsafe_mutations_rejected=rejected, transfer_CLI_passed=True, exact_LF_checksum=True,
        local_neural_gradient_or_optimizer_calls=0, B3_VM_execution_pending=True,
        model_qualified=False, app_promotion=False, goal_complete=False, seconds=time.monotonic()-start)
    with receipt.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print(dict(complete=True, archive_members=len(seen), unsafe_mutations_rejected=len(rejected), VM_training_pending=True))


if __name__ == '__main__':
    main()
