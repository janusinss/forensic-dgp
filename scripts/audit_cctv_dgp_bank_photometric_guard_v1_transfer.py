"""Independent full-payload/reuse-transfer check of one new TRAIN loss term."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

from audit_cctv_dgp_bank_comparison_B4_transfer_v1 import archive_check, sha, read

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs'
SOURCE_PIN = '8c906e8b1a3b9534872313414321a2c780560d9ea5ab50fec3e34232813cad85'
NAME = 'cctv_dgp_bank_photometric_guard_v1_vm'


def tree(path): return ast.parse(Path(path).read_text(encoding='utf-8-sig'),feature_version=(3,10))


def functions(path):
    return {node.name:ast.dump(node,include_attributes=False) for node in ast.walk(tree(path)) if isinstance(node,ast.FunctionDef)}


def main():
    start = time.monotonic()
    receipt = OUT/'cctv_dgp_bank_photometric_guard_v1_transfer_audit.json'; assert not receipt.exists()
    prep_path = OUT/'cctv_dgp_bank_photometric_guard_v1_preparation.json'; prep = read(prep_path)
    assert prep['complete'] and prep['manual_VM_run_pending'] and prep['start_not_failed_B50']
    prior = OUT/'cctv_dgp_bank_comparison_v1_r4_B_vm'; root = OUT/NAME
    old, new = read(prior/'protocol.json'), read(root/'protocol.json')
    assert sha(prior/'protocol.json') == prep['source_B4_protocol_sha256'] == SOURCE_PIN
    assert sha(root/'protocol.json') == prep['protocol_sha256']
    allowed = {'format','UTC','assets_sha256','retained_B4_protocol_sha256','photometric_guard',
        'start_from_unchanged_initial_B_not_failed_B50','bank_feature_DC_removal','single_training_objective_change_only'}
    changed = {key for key in set(old)|set(new) if old.get(key) != new.get(key)}
    assert changed == allowed
    modified = {name for name,expected in old['assets_sha256'].items() if new['assets_sha256'].get(name) != expected}
    assert modified == {'cctv_dgp_bank_comparison_v1_contract.py','cctv_dgp_bank_comparison_v1_model.py','scripts/cctv_dgp_bank_comparison_v1_vm.py'}
    assert not (set(old['assets_sha256'])-set(new['assets_sha256']))
    for name,expected in new['assets_sha256'].items(): assert sha(root/name) == expected
    trees = [tree(folder/'cctv_dgp_bank_comparison_v1_model.py') for folder in [prior,root]]
    for value in trees:
        for node in value.body:
            if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id == 'NAME' for target in node.targets): node.value = ast.Constant('ROOT_ONLY')
    assert ast.dump(trees[0],include_attributes=False) == ast.dump(trees[1],include_attributes=False)
    worker = 'scripts/cctv_dgp_bank_comparison_v1_vm.py'
    before, after = functions(prior/worker), functions(root/worker)
    assert set(before) == set(after)
    changed_functions = [name for name in before if before[name] != after[name]]
    assert set(changed_functions) == {'run','objective'}
    helper = root/'cctv_dgp_bank_photometric_guard_v1_loss.py'
    math_path = OUT/'cctv_dgp_bank_photometric_guard_v1_math_audit.json'; math = read(math_path)
    assert math['complete'] and math['saved_TRAIN_cases'] == math['exact_baseline_zero_penalty_cases'] == 100
    assert math['helper_sha256'] == sha(helper) and prep['math_audit_sha256'] == sha(math_path)
    assert math['gradient_queries'] == math['optimizer_updates'] == math['backwards'] == 0
    assert not any(isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in {'backward','grad','step','finite_difference'} for node in ast.walk(tree(helper)))
    text = (root/worker).read_text(encoding='utf-8')
    assert "+ anchors + barrier + photo" in text
    assert "p['photometric_guard']['coefficient'], p['photometric_guard']['fraction_maximum']" in text
    assert "history.append({'update': update, 'objective': objective_mean, 'photometric_guard': photo_mean" in text
    assert "fit_seconds_" in text and "'includes_post_update1_gradient_checks': True" in text
    assert "candidate = create('B', learning=False)" in text and "load_pinned_initializer" in text
    spec = importlib.util.spec_from_file_location('independent_guard_contract',root/'cctv_dgp_bank_comparison_v1_contract.py')
    contract = importlib.util.module_from_spec(spec); spec.loader.exec_module(contract)
    contract.verify(root,prep['protocol_sha256'])
    rejected = []
    for name in ['lower_gate','extra_updates','final_leak','alter_schedule','change_guard','disable_guard','failed_resume','change_budget']:
        value = copy.deepcopy(new)
        if name == 'lower_gate': value['scientific_thresholds']['early_structure_gain'] = .001
        elif name == 'extra_updates': value['maximum_updates_per_route'] = 100
        elif name == 'final_leak': value['cases'][0]['role'] = 'final'
        elif name == 'alter_schedule': value['schedule'][1] = value['schedule'][0]
        elif name == 'change_guard': value['photometric_guard']['fraction_maximum'] = .4
        elif name == 'disable_guard': value['photometric_guard']['coefficient'] = 0
        elif name == 'failed_resume': value['start_from_unchanged_initial_B_not_failed_B50'] = False
        else: value['budgets']['arm_fit_seconds'] = 1200
        try: contract.validate(value)
        except AssertionError: rejected.append(name)
        else: raise AssertionError('Unsafe protocol accepted: '+name)
    python_files = []
    for name in new['assets_sha256']:
        if name.endswith('.py'):
            value = tree(root/name)
            assert not any(isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr == 'file_digest' for node in ast.walk(value))
            python_files.append(name)
    cli = subprocess.run([sys.executable,'-B',str(root/worker),'--root',str(root),'--protocol-sha',prep['protocol_sha256'],'--verify-transfer'],capture_output=True,text=True,check=True,timeout=120)
    status = json.loads(cli.stdout)
    assert status['complete'] and status['loader_hash_runtime_checked'] and status['optimizer_updates'] == status['neural_or_gradient_calls'] == 0
    patch = OUT/prep['transfer_name']; transfer = read(patch/'transfer.json')
    assert sha(patch/'transfer.json') == prep['transfer_manifest_sha256']
    assert transfer['source_protocol_sha256'] == sha(patch/'source_protocol.json') == SOURCE_PIN
    expected_overlay = {name:expected for name,expected in new['assets_sha256'].items() if old['assets_sha256'].get(name) != expected}
    assert transfer['overlay_assets'] == expected_overlay and len(expected_overlay) == prep['overlay_assets']
    reuse = set(new['assets_sha256'])-set(expected_overlay)
    assert len(reuse) == prep['reused_assets']
    for name,expected in new['assets_sha256'].items():
        origin = patch/'overlay'/name if name in expected_overlay else prior/name
        assert sha(origin) == expected
    installer = patch/'install_packet.py'
    assert sha(installer) == transfer['installer_sha256'] == sha(root/'scripts/install_packet_from_B4.py')
    install_text = installer.read_text(encoding='utf-8')
    assert "'quality_gate_failure'" in install_text and "supervisor['optimizer_updates'] == 50" in install_text
    assert "signal.alarm(300)" in install_text and "target.mkdir()" in install_text and '.unlink(' not in install_text
    assert not any(isinstance(node,ast.Import) and any(value.name in {'torch','numpy'} for value in node.names) for node in ast.walk(tree(installer)))
    install = subprocess.run([sys.executable,'-B',str(installer),'--protocol-sha',prep['protocol_sha256'],'--verify-transfer'],capture_output=True,text=True,check=True,timeout=30)
    assert json.loads(install.stdout)['model_gradient_or_training_calls'] == 0
    full_count = archive_check(OUT/(prep['stem']+'-execution.tar.gz'),NAME,{**new['assets_sha256'],'protocol.json':prep['protocol_sha256']},prep['archive_sha256'],prep['archive_bytes'])
    patch_files = {file.relative_to(patch).as_posix():sha(file) for file in patch.rglob('*') if file.is_file()}
    patch_count = archive_check(OUT/(prep['transfer_stem']+'.tar.gz'),prep['transfer_name'],patch_files,prep['transfer_archive_sha256'],prep['transfer_archive_bytes'])
    assert not (root/'outputs').exists()
    result = dict(complete=True,checker_sha256=sha(__file__),preparation_sha256=sha(prep_path),
        protocol_sha256=prep['protocol_sha256'],archive_sha256=prep['archive_sha256'],archive_members_checked=full_count,
        transfer_archive_sha256=prep['transfer_archive_sha256'],transfer_archive_bytes=prep['transfer_archive_bytes'],
        transfer_members_checked=patch_count,overlay_assets=len(expected_overlay),exactly_reconstructed_reused_assets=len(reuse),
        full_and_reuse_transfers_match_exact_packet=True,full_and_reuse_checksum_LF=True,
        model_computation_AST_unchanged=True,changed_worker_functions=changed_functions,
        scientific_change='additional degraded TRAIN brightness-gain hinge; no inference/feature/DC change',
        initializer_weights_frozen_assets_optimizer_data_schedule_gates_caps_unchanged=True,
        safe_protocol_mutations_rejected=rejected,Python310_syntax_files_checked=len(python_files),
        file_digest_runtime_calls=0,real_transfer_CLI_passed=True,installer_transfer_only_CLI_passed=True,
        local_neural_gradient_or_optimizer_calls=0,actual_VM_gradient_training_and_quality_pending=True,
        math_audit_sha256=sha(math_path),VM_storage_recheck_required=True,
        model_qualified=False,app_promotion=False,goal_complete=False,seconds=time.monotonic()-start)
    with receipt.open('x',encoding='utf-8',newline='\n') as stream: stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(dict(complete=True,full_members=full_count,transfer_members=patch_count,transfer_bytes=prep['transfer_archive_bytes'],local_optimizer_updates=0,seconds=result['seconds']),flush=True)


if __name__ == '__main__': main()
