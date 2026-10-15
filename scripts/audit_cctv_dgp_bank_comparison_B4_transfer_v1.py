"""Independently check hashing-only B4 revision, complete archive and reuse transfer."""
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


def sha_stream(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024**2), b''):
        digest.update(block)
    return digest.hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return sha_stream(stream)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(node):
    return ast.dump(node, include_attributes=False)


def archive_check(path, prefix, expected, digest, size):
    assert path.stat().st_size == size and sha(path) == digest
    assert Path(str(path)+'.sha256').read_bytes() == (digest+'  '+path.name+'\n').encode('ascii')
    seen = set()
    with tarfile.open(path, 'r|gz') as tar:
        for member in tar:
            value = PurePosixPath(member.name)
            assert member.isfile() and not member.issym() and not member.islnk()
            assert value.parts[0] == prefix and not value.is_absolute() and '..' not in value.parts
            assert ':' not in member.name and '\\' not in member.name and value.as_posix() == member.name
            name = PurePosixPath(*value.parts[1:]).as_posix()
            assert name in expected and name not in seen
            with tar.extractfile(member) as stream:
                assert sha_stream(stream) == expected[name]
            seen.add(name)
    assert seen == set(expected)
    return len(seen)


def main():
    started = time.monotonic()
    receipt = OUT / 'cctv_dgp_bank_comparison_B4_transfer_v1_audit.json'
    assert not receipt.exists()
    prep_path = OUT / 'cctv_dgp_bank_comparison_v1_r4_B_preparation.json'
    prep = read(prep_path)
    assert prep['complete'] and prep['manual_VM_run_pending']
    prior = OUT / 'cctv_dgp_bank_comparison_v1_r3_B_vm'
    root, patch = OUT / prep['name'], OUT / prep['transfer_name']
    old, new = read(prior / 'protocol.json'), read(root / 'protocol.json')
    assert sha(prior / 'protocol.json') == prep['prior_B3_protocol_sha256']
    assert sha(root / 'protocol.json') == prep['protocol_sha256']
    allowed = {'format', 'UTC', 'assets_sha256', 'retained_B3_protocol_sha256',
               'loader_hash_policy', 'loader_hash_runtime_check_in_transfer_CLI'}
    changed = {name for name in set(old)|set(new) if old.get(name) != new.get(name)}
    assert changed <= allowed and old['budgets'] == new['budgets']
    assert old['initializer_loading_policy'] == new['initializer_loading_policy']
    modified = {name for name, expected in old['assets_sha256'].items()
                if name in new['assets_sha256'] and new['assets_sha256'][name] != expected}
    assert modified == {'cctv_dgp_bank_comparison_v1_contract.py',
                        'cctv_dgp_bank_comparison_v1_model.py', 'scripts/cctv_dgp_bank_comparison_v1_vm.py'}
    assert set(old['assets_sha256']) - set(new['assets_sha256']) == {'cctv_dgp_bank_pinned_initializer_v1.py'}
    for name, expected in new['assets_sha256'].items():
        assert sha(root / name) == expected
        if name in old['assets_sha256'] and name not in modified:
            assert sha(prior / name) == expected
    model_trees = [ast.parse((p / 'cctv_dgp_bank_comparison_v1_model.py').read_text()) for p in [prior, root]]
    for tree in model_trees:
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'NAME' for t in node.targets):
                node.value = ast.Constant('ROOT_ONLY')
    assert dump(model_trees[0]) == dump(model_trees[1])
    workers = [ast.parse((p / 'scripts/cctv_dgp_bank_comparison_v1_vm.py').read_text()) for p in [prior, root]]
    functions = [{node.name: dump(node) for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)} for tree in workers]
    unchanged = ['clock', 'objective', 'batch', 'frozen', 'save_state', 'metrics', 'base_raw',
                 'export', 'vm', 'pixels', 'tensor']
    assert all(functions[0][name] == functions[1][name] for name in unchanged)
    old_loader = ast.parse((prior / 'cctv_dgp_bank_pinned_initializer_v1.py').read_text())
    new_loader = ast.parse((root / 'cctv_dgp_bank_pinned_initializer_v2.py').read_text())
    loader_functions = [{node.name: dump(node) for node in tree.body if isinstance(node, ast.FunctionDef)}
                        for tree in [old_loader, new_loader]]
    assert {name for name in loader_functions[0] if loader_functions[0][name] != loader_functions[1][name]} == {'sha'}
    python_files = []
    for name in new['assets_sha256']:
        if not name.endswith('.py'):
            continue
        tree = ast.parse((root / name).read_text(encoding='utf-8-sig'), feature_version=(3, 10))
        assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                       and node.func.attr == 'file_digest' for node in ast.walk(tree)), name
        python_files.append(name)
    source = (root / 'scripts/cctv_dgp_bank_comparison_v1_vm.py').read_text()
    assert "from cctv_dgp_bank_pinned_initializer_v2 import load_pinned_initializer" in source
    assert source.index("candidate = create('B', learning=False)") < source.index("baseline = metrics('baseline', original.net)")
    assert "assert loader_sha(root / initializer_name) == p['assets_sha256'][initializer_name]" in source
    regression_path = OUT / 'cctv_dgp_bank_comparison_v1_r3_B_analysis/hash_compat_loader_regression.json'
    regression = read(regression_path)
    assert regression['complete'] and regression['cases_checked'] == 5
    assert regression['hashlib_file_digest_unavailable_in_checked_module']
    assert regression['loader_sha256'] == sha(root / 'cctv_dgp_bank_pinned_initializer_v2.py')
    assert sha(regression_path) == prep['hash_compat_loader_regression_sha256']
    spec = importlib.util.spec_from_file_location('independent_B4_contract', root / 'cctv_dgp_bank_comparison_v1_contract.py')
    contract = importlib.util.module_from_spec(spec); spec.loader.exec_module(contract)
    contract.verify(root, prep['protocol_sha256'])
    rejected = []
    for name in ['lower_gate', 'final_leak', 'extra_updates', 'alter_schedule', 'alter_hash_policy', 'alter_budget']:
        value = copy.deepcopy(new)
        if name == 'lower_gate': value['scientific_thresholds']['early_structure_gain'] = .001
        elif name == 'final_leak': value['cases'][0]['role'] = 'final'
        elif name == 'extra_updates': value['maximum_updates_per_route'] = 100
        elif name == 'alter_schedule': value['schedule'][1] = value['schedule'][0]
        elif name == 'alter_hash_policy': value['loader_hash_policy'] = 'skip-hashes'
        else: value['budgets']['arm_fit_seconds'] = 1200
        try:
            contract.validate(value)
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError('Unsafe protocol accepted: ' + name)
    # Exercise the real transfer CLI and changed hash routine with only that API missing.
    code = """import hashlib,importlib,runpy,sys,types
root,pin=sys.argv[1:]
sys.path.insert(0,root)
loader=importlib.import_module('cctv_dgp_bank_pinned_initializer_v2')
loader.hashlib=types.SimpleNamespace(sha256=hashlib.sha256)
sys.argv=[root+'/scripts/cctv_dgp_bank_comparison_v1_vm.py','--root',root,'--protocol-sha',pin,'--verify-transfer']
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    cli = subprocess.run([sys.executable, '-B', '-c', code, str(root), prep['protocol_sha256']],
                         capture_output=True, text=True, check=True, timeout=120)
    cli_status = json.loads(cli.stdout)
    assert cli_status['complete'] and cli_status['loader_hash_runtime_checked']
    assert cli_status['optimizer_updates'] == cli_status['neural_or_gradient_calls'] == 0
    overlay = read(patch / 'transfer.json')
    assert sha(patch / 'transfer.json') == prep['transfer_manifest_sha256']
    assert overlay['protocol_sha256'] == prep['protocol_sha256']
    assert overlay['source_protocol_sha256'] == prep['prior_B3_protocol_sha256']
    assert sha(patch / 'source_protocol.json') == prep['prior_B3_protocol_sha256']
    assert len(overlay['overlay_assets']) == prep['overlay_assets']
    assert set(overlay['overlay_assets']) == {
        name for name, expected in new['assets_sha256'].items() if old['assets_sha256'].get(name) != expected}
    reuse = set(new['assets_sha256']) - set(overlay['overlay_assets'])
    assert len(reuse) == prep['reused_assets']
    for name, expected in new['assets_sha256'].items():
        origin = patch / 'overlay' / name if name in overlay['overlay_assets'] else prior / name
        assert sha(origin) == sha(root / name) == expected
    installer = patch / 'install_packet.py'
    assert sha(installer) == overlay['installer_sha256'] == sha(root / 'scripts/install_packet_from_B3.py')
    installer_check = subprocess.run([sys.executable, '-B', str(installer), '--protocol-sha', prep['protocol_sha256'], '--verify-transfer'],
        capture_output=True, text=True, check=True, timeout=30)
    installer_status = json.loads(installer_check.stdout)
    assert installer_status['complete'] and installer_status['model_gradient_or_training_calls'] == 0
    installer_tree = ast.parse(installer.read_text(encoding='utf-8'), feature_version=(3, 10))
    assert not any(isinstance(node, ast.Import) and any(value.name in {'torch','numpy'} for value in node.names)
                   for node in ast.walk(installer_tree))
    assert 'signal.alarm(300)' in installer.read_text() and "target.mkdir()" in installer.read_text()
    complete_members = archive_check(OUT / (prep['stem']+'-execution.tar.gz'), prep['name'],
        {**new['assets_sha256'], 'protocol.json': prep['protocol_sha256']}, prep['archive_sha256'], prep['archive_bytes'])
    patch_members = {path.relative_to(patch).as_posix(): sha(path) for path in patch.rglob('*') if path.is_file()}
    checked_patch = archive_check(OUT / (prep['transfer_stem']+'.tar.gz'), prep['transfer_name'], patch_members,
                                 prep['transfer_archive_sha256'], prep['transfer_archive_bytes'])
    assert not (root / 'outputs').exists()
    result = dict(complete=True, checker_sha256=sha(__file__), preparation_sha256=sha(prep_path),
        protocol_sha256=prep['protocol_sha256'], archive_sha256=prep['archive_sha256'],
        archive_members_checked=complete_members, transfer_archive_sha256=prep['transfer_archive_sha256'],
        transfer_members_checked=checked_patch, transfer_archive_bytes=prep['transfer_archive_bytes'],
        exactly_reconstructed_reused_assets=len(reuse), overlay_assets=len(overlay['overlay_assets']),
        full_and_reuse_transfers_match_exact_packet=True, full_and_reuse_checksum_LF=True,
        Python310_syntax_files_checked=len(python_files), file_digest_runtime_calls_in_packet=0,
        real_transfer_CLI_passed_with_loader_file_digest_absent=True,
        actual_local_Python310_interpreter_used=False, VM_runtime_validation_pending=True,
        loader_only_function_changed='sha', model_computation_AST_unchanged=True,
        unchanged_worker_functions=unchanged, exact_initializer_weights_data_schedule_scientific_gates_and_resource_caps=True,
        loader_regression_sha256=sha(regression_path), unsafe_mutations_rejected=rejected,
        installer_transfer_only_CLI_passed=True, source_delete_calls=0, installer_cap_seconds=300,
        local_neural_gradient_or_optimizer_calls=0, B4_VM_execution_pending=True,
        model_qualified=False, app_promotion=False, goal_complete=False, seconds=time.monotonic()-started)
    with receipt.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(dict(complete=True, full_members=complete_members, reuse_transfer_bytes=prep['transfer_archive_bytes'],
               reused_assets=len(reuse), VM_training_pending=True), flush=True)


if __name__ == '__main__':
    main()
