"""Independent prospective source/packet invariants. No model or VM execution."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'outputs/cctv_dgp_batchmatched_identity_vm_v26'
NEW = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'
OUT = ROOT / 'outputs/cctv_dgp_feature_skips_v27_preparation'
PIN = '22f76701e317b38d945838a6767239c5636cff534b61ddfbf248cc42ba08a82e'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def same(left, right, label):
    assert ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False), label


def tree(text):
    return ast.parse(text, feature_version=(3, 10))


def function(module, name):
    return next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name)


def target_name(statement):
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
        return ast.unparse(statement.targets[0])
    return None


def head_contract(text, original=None):
    current = tree(text)
    original = tree(original if original is not None else (OLD / 'cctv_dgp_spatial_features_v25.py').read_text(encoding='utf-8'))
    cls = next(node for node in current.body if isinstance(node, ast.ClassDef))
    init = function(cls, '__init__')
    assert target_name(init.body[-2]) == 'self.feature_skips' and isinstance(init.body[-1], ast.For)
    assert ast.unparse(init.body[-2].value) == 'nn.ModuleList([nn.Conv2d(channels, 3, 1) for channels in [64, 128, 128, 128, 128]])'
    assert ast.unparse(init.body[-1]) == 'for layer in self.feature_skips:\n    nn.init.zeros_(layer.weight)\n    nn.init.zeros_(layer.bias)'
    init.body = init.body[:-2]
    forward = function(cls, 'forward')
    index = next(i for i, node in enumerate(forward.body) if target_name(node) == 'residuals')
    additions = forward.body[index:index + 3]
    assert target_name(additions[0]) == 'residuals' and ast.unparse(additions[0].value) == '[]'
    assert isinstance(additions[1], ast.For) and ast.unparse(additions[1].iter) == 'zip(self.feature_skips, fpn)'
    expected = tree('''for layer, feature in zip(self.feature_skips, fpn):
    support = F.interpolate(mask, size=feature.shape[-2:], mode='area')
    mass = support.sum((2, 3), keepdim=True)
    centered = feature - (feature * support).sum((2, 3), keepdim=True) / mass
    scale = ((centered.square() * support).sum((2, 3), keepdim=True) / mass).sqrt().clamp_min(.05)
    residuals.append(F.interpolate(layer(centered / scale), size=(256, 256), mode='bilinear', align_corners=False))
''').body[0]
    same(additions[1], expected, 'Input-only normalized five-scale shortcut assembly differs')
    assert target_name(additions[2]) == 'feature_skip' and ast.unparse(additions[2].value) == 'sum(residuals) / 5 ** 0.5'
    del forward.body[index:index + 3]
    q = next(node for node in forward.body if target_name(node) == 'q')
    argument = q.value.right.args[0]
    assert isinstance(argument, ast.BinOp) and isinstance(argument.op, ast.Add) and ast.unparse(argument.right) == 'feature_skip'
    q.value.right.args[0] = argument.left
    same(current, original, 'Original head/encoder helpers changed beyond five direct feature skips')
    return True


def worker_contract(text):
    for new, old in [
        ('dgp-direct-feature-skips-capacity-v27', 'dgp-spatial-batchmatched-identity-capacity-v26'),
        ('scripts/cctv_dgp_feature_skips_v27_vm.py', 'scripts/cctv_dgp_batchmatched_identity_v26_vm.py'),
        ('scripts/run_v27.sh', 'scripts/run_v26.sh'),
        ('cctv_dgp_feature_skips_v27_return', 'cctv_dgp_batchmatched_identity_v26_return'),
        ('cctv-dgp-feature-skips-v27', 'cctv-dgp-batchmatched-identity-v26'),
        ('cctv_dgp_feature_skips_v27_prepare', 'cctv_dgp_spatial_features_v25_prepare'),
        ('cctv_dgp_feature_skips_v27_preflight', 'cctv_dgp_batchmatched_identity_v26_preflight'),
        ('cctv_dgp_feature_skips_v27.py', 'cctv_dgp_spatial_features_v25.py'),
        ('scripts/install_v27.py', 'scripts/install_v26.py'), ('V27 update', 'V26 update')]:
        text = text.replace(new, old)
    current = tree(text)
    run = function(current, 'run')
    fitting = next(node for node in run.body if isinstance(node, ast.Try))
    index = next(i for i, node in enumerate(fitting.body) if target_name(node) == 'skip_gradients')
    assert ast.unparse(fitting.body[index].value) == '{str(index): float(layer.weight.grad.double().square().sum()) for index, layer in enumerate(head.feature_skips)}'
    expected = tree("assert all(value > 0 for value in skip_gradients.values()), 'All five direct DGP-feature readouts must have nonzero gradients before any optimizer'").body[0]
    same(fitting.body[index + 1], expected, 'Pre-optimizer five-scale gradient assertion differs')
    del fitting.body[index:index + 2]
    dictionaries = [node for node in ast.walk(fitting) if isinstance(node, ast.Dict)]
    matches = [(node, i) for node in dictionaries for i, key in enumerate(node.keys)
               if isinstance(key, ast.Constant) and key.value == 'per_feature_skip_weight_gradient_sum_squares']
    assert len(matches) == 1
    node, key_index = matches[0]
    assert ast.unparse(node.values[key_index]) == 'skip_gradients'
    del node.keys[key_index]; del node.values[key_index]
    same(current, tree((OLD / 'scripts/cctv_dgp_batchmatched_identity_v26_vm.py').read_text(encoding='utf-8')),
        'Original full trainer, objective, scientific stops, finite timing, export and supervision changed')
    return True


def preparation_contract(text):
    text = text.replace('from cctv_dgp_feature_skips_v27 import', 'from cctv_dgp_spatial_features_v25 import').replace('55524', '53781')
    current = tree(text); prepare = function(current, 'prepare_features')
    index = next(i for i, node in enumerate(prepare.body) if target_name(node) == 'prior')
    parts = prepare.body[index:index + 6]
    assert [target_name(node) for node in parts[:3]] == ['prior', 'shared', 'extra']
    assert all(isinstance(node, ast.Assert) for node in parts[3:])
    assert 'weights_only=True' in ast.unparse(parts[0]) and 'outputs/update0/head.pth' in ast.unparse(parts[0])
    assert 'torch.equal' in ast.unparse(parts[4]) and 'torch.count_nonzero' in ast.unparse(parts[5])
    del prepare.body[index:index + 6]
    dictionary = prepare.body[-1].value.elts[-1]
    assert isinstance(dictionary, ast.Dict)
    assert [key.value for key in dictionary.keys[-2:]] == ['shared_V26_initial_tensors_exact', 'new_feature_skip_tensors_exact_zero']
    dictionary.keys = dictionary.keys[:-2]; dictionary.values = dictionary.values[:-2]
    same(current, tree((OLD / 'cctv_dgp_spatial_features_v25_prepare.py').read_text(encoding='utf-8')),
        'Original inference-only feature preparation changed beyond head/count and exact shared-state proof')
    return True


def main():
    started = time.monotonic()
    assert sha(NEW / 'protocol.json') == PIN
    p, old, receipt = read(NEW / 'protocol.json'), read(OLD / 'protocol.json'), read(OUT / 'preparation.json')
    assert p['format'] == 'dgp-direct-feature-skips-capacity-v27'
    assert p['closed_V26_protocol_sha256'] == sha(OLD / 'protocol.json')
    assert p['cases'] == old['cases'] and p['references'] == old['references']
    assert p['budgets'] == old['budgets'] and sha(NEW / 'schedule.json') == sha(OLD / 'schedule.json')
    design = copy.deepcopy(p['design'])
    assert design.pop('direct_feature_skip_parameters') == 1743
    assert design.pop('direct_feature_skip_channel_RMS_floor') == .05
    assert design.pop('direct_feature_skip_scale') == '1/sqrt(5)'
    assert design['trainable_parameters'] == 55524; design['trainable_parameters'] = 53781
    assert design == old['design']
    proof = copy.deepcopy(p['identity_preflight_policy'])
    assert proof.pop('all36_matched_parameter_gradients_exactly_zero') is True
    proof['all26_matched_parameter_gradients_exactly_zero'] = True
    assert proof == old['identity_preflight_policy']
    assert len(p['assets_sha256']) == 246 and len(p['inherited_assets']) == 240 and len(p['transfer_assets_sha256']) == 6
    for name, digest in p['assets_sha256'].items():
        assert sha(NEW / name) == digest, name
    for name, row in p['inherited_assets'].items():
        assert name == row['source'] and row['sha256'] == old['assets_sha256'][name] == p['assets_sha256'][name]
        assert sha(OLD / name) == row['sha256']
    for name, digest in p['local_basis_sha256'].items():
        assert sha(ROOT / name) == digest
    head_contract((NEW / 'cctv_dgp_feature_skips_v27.py').read_text(encoding='utf-8'))
    worker_contract((NEW / 'scripts/cctv_dgp_feature_skips_v27_vm.py').read_text(encoding='utf-8'))
    preparation_contract((NEW / 'cctv_dgp_feature_skips_v27_prepare.py').read_text(encoding='utf-8'))
    proof_source = (NEW / 'cctv_dgp_feature_skips_v27_preflight.py').read_text(encoding='utf-8')
    proof_source = proof_source.replace('len(parameters)==36', 'len(parameters)==26').replace('torch.zeros(55524', 'torch.zeros(53781').replace('all36_', 'all26_')
    same(tree(proof_source), tree((OLD / 'cctv_dgp_batchmatched_identity_v26_preflight.py').read_text(encoding='utf-8')),
         'Original exact-zero matched identity proof changed beyond parameter count/names')
    shell = (NEW / 'scripts/run_v27.sh').read_text(encoding='utf-8').replace('scripts/cctv_dgp_feature_skips_v27_vm.py', 'scripts/cctv_dgp_batchmatched_identity_v26_vm.py')
    assert shell == (OLD / 'scripts/run_v26.sh').read_text(encoding='utf-8')
    archive = ROOT / 'outputs/cctv-dgp-feature-skips-v27-execution.tar.gz'
    assert receipt['archive_sha256'] == sha(archive) and receipt['archive_bytes'] == archive.stat().st_size
    assert archive.with_name(archive.name + '.sha256').read_text(encoding='ascii') == sha(archive) + '  ' + archive.name + '\n'
    expected = {'protocol.json', *p['transfer_assets_sha256']}
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        assert len(members) == 7 and all(member.isfile() and not member.issym() and not member.islnk() for member in members)
        assert {member.name for member in members} == {NEW.name + '/' + name for name in expected}
        for member in members:
            name = member.name[len(NEW.name) + 1:]
            assert hashlib.sha256(tar.extractfile(member).read()).hexdigest() == sha(NEW / name)
    for name in p['transfer_assets_sha256']:
        if name.endswith('.py'):
            tree((NEW / name).read_text(encoding='utf-8'))
    result = {'complete': True, 'date': '2026-10-06', 'checker_sha256': sha(Path(__file__)),
        'protocol_sha256': PIN, 'archive_sha256': sha(archive), 'archive_bytes': archive.stat().st_size,
        'original240_assets_retained': True, 'original_full_trainer_ast_retained_except_declared_names_and_skip_gradient_assertion': True,
        'original_full_head_ast_retained_except_five_normalized_shortcuts': True,
        'original_identity_proof_and_feature_guards_retained': True,
        'original_shell_supervisor_and_all_limits_retained': True,
        'same50_cases_schedule_loss_optimizer_and_scientific_stops': True,
        'new_feature_skip_trainable_parameters': 1743, 'trainable_parameters': 55524,
        'parameters_groups': 36, 'regular_archive_members': 7, 'local_neural_calls': 0,
        'local_gradient_calls': 0, 'optimizer_updates': 0, 'VM_actions': False,
        'app_promotion': False, 'actual_L4_pilot_pending': True, 'goal_complete': False,
        'seconds': time.monotonic() - started}
    with (OUT / 'independent_packet_audit.json').open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
