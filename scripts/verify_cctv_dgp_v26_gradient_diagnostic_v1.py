"""Independent thin-packet, source-delta and Windows-guard verification."""
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import time

from verify_cctv_dgp_v25_gradient_diagnostic_v1 import expected_layout

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs/cctv_dgp_v26_gradient_diagnostic_v1_vm'
OUT = ROOT / 'outputs/cctv_dgp_v26_gradient_diagnostic_v1_preparation'
NAME = 'cctv-dgp-v26-gradient-diagnostic-v1'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def inspect_archive(archive):
    expected = ['protocol.json', 'protocol.sha256', 'scripts/cctv_dgp_v26_gradient_diagnostic_v1_vm.py', 'scripts/run_gradient.sh']
    payload = {}
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar.getmembers():
            path = PurePosixPath(member.name)
            assert member.isreg() and not member.issym() and not member.islnk() and not member.sparse
            assert not path.is_absolute() and '..' not in path.parts and '\\' not in member.name and ':' not in member.name
            assert path.parts[0] == BUNDLE.name and path.as_posix() == member.name
            relative = '/'.join(path.parts[1:])
            assert relative in expected and relative not in payload
            assert 0 <= member.size <= 128 * 1024
            payload[relative] = tar.extractfile(member).read()
    assert sorted(payload) == sorted(expected)
    return payload


def source_contract(source):
    """Reverse only declared differences, then compare the entire legacy AST."""
    tree = ast.parse(source, feature_version=(3, 10))
    attributes = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not attributes.intersection({'backward', 'step', 'Adam', 'AdamW', 'SGD'})
    assert 'torch.save' not in {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert 'optim' not in {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'grad']
    assert len(calls) == 1 and ast.unparse(calls[0].func) == 'torch.autograd.grad'
    assert {v.arg: ast.unparse(v.value) for v in calls[0].keywords} == {
        'retain_graph': 'i < 6', 'create_graph': 'False', 'allow_unused': 'False'}
    assert source.index("assert sys.platform=='linux'") < source.index('out.mkdir()')
    proof = "    proof=read(parent/p['identity_proof_file'])\n    assert proof['complete'] and proof['cases']==50 and proof['gradient_calls']==20\n    assert proof['batchmatched_component_value']==proof['batchmatched_component_gradient_norm']==0\n    assert len(proof['rows'])==10 and all(r['all26_matched_gradient_tensors_exactly_zero'] and r['exact_reference_prediction_cosines'] for r in proof['rows'])\n"
    exact = "\n                    if update==0 and name=='ArcFace_regression':\n                        assert float(scalar.detach())==0 and all(torch.count_nonzero(g)==0 for g in pieces),'Initial corrected identity value/all26 gradients must stay exactly zero'"
    selection = "        functions += [next(n for n in helper.body if isinstance(n,ast.FunctionDef) and n.name=='assemble_terms')]\n        corrected=ast.parse((parent/'cctv_dgp_batchmatched_identity_v26.py').read_text())\n        functions += [next(n for n in corrected.body if isinstance(n,ast.FunctionDef) and n.name==name) for name in ['batchmatched_scores','objective_terms']]"
    assert source.count(proof) == source.count(exact) == source.count(selection) == 1
    normalized = source.replace(proof, '').replace(exact, '')
    normalized = normalized.replace(selection, "        functions += [next(n for n in helper.body if isinstance(n,ast.FunctionDef) and n.name==name) for name in ['assemble_terms','objective_terms']]")
    target_line = "                item['truth']=identity.embedding(item['target'],item['mask'],item['grid'])\n"
    assert normalized.count(target_line) == 1
    normalized = normalized.replace(target_line, target_line + "                item['base_cosine']=(identity.embedding(item['base']*item['mask']+item['x']*(1-item['mask']),item['mask'],item['grid'])*item['truth']).sum(1)\n")
    normalized = normalized.replace("assert progress['recognizer_forwards']==50", "assert progress['recognizer_forwards']==100")
    normalized = normalized.replace("'grid','truth','degraded_weight'", "'grid','truth','base_cosine','degraded_weight'")
    reverse = [
        ('cctv_dgp_v26_gradient_diagnostic_v1', 'cctv_dgp_v25_gradient_diagnostic_v1'),
        ('cctv-dgp-v26-gradient-diagnostic-v1', 'cctv-dgp-v25-gradient-diagnostic-v1'),
        ('closed_V26', 'closed_V25'), ('own-DGP-V26', 'own-DGP-V25'),
        ('scripts/cctv_dgp_batchmatched_identity_v26_vm.py', 'scripts/cctv_dgp_spatial_features_v25_vm.py'),
        ('cctv_dgp_batchmatched_identity_vm_v26', 'cctv_dgp_spatial_features_vm_v25'),
        ("base['format']=='dgp-spatial-batchmatched-identity-capacity-v26'", "base['format']=='dgp-spatial-feature-capacity-v25'"),
        ('V26', 'V25'), ("'recognizer_forwards':70", "'recognizer_forwards':120"),
    ]
    for before, after in reverse:
        assert before in normalized, before
        normalized = normalized.replace(before, after)
    original = (ROOT / 'scripts/cctv_dgp_v25_gradient_diagnostic_v1_vm.py').read_text(encoding='utf-8')
    assert ast.dump(ast.parse(normalized), include_attributes=False) == ast.dump(ast.parse(original), include_attributes=False)
    assert "for update in [0,50]" in source and 'range(0,50,5)' in source
    assert "assert progress=={'head_batches':20,'component_gradient_calls':140,'recognizer_forwards':70,'optimizer_updates':0}" in source
    return tree


def return_audit_contract():
    """The safe importer and full matrix auditor keep their original checks."""
    common = [('v26_gradient_diagnostic_v1', 'v25_gradient_diagnostic_v1'),
        ('cctv-dgp-v26-gradient', 'cctv-dgp-v25-gradient'), ('closed_V26', 'closed_V25'),
        ('cctv_dgp_batchmatched_identity_vm_v26', 'cctv_dgp_spatial_features_vm_v25'),
        ('cctv_dgp_batchmatched_identity_v26_return', 'cctv_dgp_spatial_features_v25_return'),
        ('cctv_dgp_batchmatched_identity_v26_loss_audit_v1', 'cctv_dgp_spatial_features_v25_loss_audit_v1'),
        ('V26', 'V25')]
    for prefix in ['import_cctv_dgp_', 'audit_cctv_dgp_']:
        suffix = 'gradient_diagnostic_v1.py' if prefix.startswith('import') else 'gradient_diagnostic_v1_return.py'
        new = ROOT / 'scripts' / (prefix + 'v26_' + suffix)
        old = ROOT / 'scripts' / (prefix + 'v25_' + suffix)
        normalized = new.read_text(encoding='utf-8')
        if prefix.startswith('import'):
            normalized = normalized.replace('cbbb9898dfdc6bc7b3409a21b26254b7286131d032e757047da9712c66b551f1',
                '884412f5e3dab571e630cdadb4f962046c8da309ebe6b62eac4d5af0e29d281a')
        else:
            initial = "    # Corrected reference must stay exactly zero in every initial batch and array.\n    require(rows[0]['component_values'][6]==0 and rows[0]['component_norms'][6]==0 and exact==50,'Initial corrected identity proof differs')\n    require(np.count_nonzero(arrays[0][6])==0 and all(b['cohort_weighted_component_values'][6]==0 and b['cohort_weighted_component_gradient_norms'][6]==0 for b in r['snapshots'][0]['batches']),'Initial corrected identity batch/gradient must be exactly zero')"
            assert initial in normalized
            normalized = normalized.replace(initial, "    # At zero tails the old comparison reports numerical identity cost on the same image.\n    require(rows[0]['component_values'][6]>0 and rows[0]['component_norms'][6]>0 and exact==50,'Demonstrated baseline identity-loss discrepancy')")
            normalized = normalized.replace("'recognizer_forwards':70", "'recognizer_forwards':120")
            normalized = normalized.replace("'corrected_initial_identity_exact_zero_verified':True", "'identity_baseline_numerical_discrepancy_confirmed':True")
        for before, after in common:
            normalized = normalized.replace(before, after)
        assert ast.dump(ast.parse(normalized), include_attributes=False) == ast.dump(ast.parse(old.read_text(encoding='utf-8')), include_attributes=False)


def main(receipt_name='independent_packet_audit.json'):
    started = time.monotonic()
    prep = read(OUT / 'preparation.json')
    archive = ROOT / 'outputs' / (NAME + '-execution.tar.gz')
    assert sha(archive) == prep['archive_sha256'] and archive.stat().st_size == prep['archive_bytes']
    assert Path(str(archive) + '.sha256').read_text(encoding='ascii').strip() == sha(archive) + '  ' + archive.name
    payload = inspect_archive(archive)
    for name, value in payload.items():
        assert value == (BUNDLE / name).read_bytes()
    p = read(BUNDLE / 'protocol.json')
    pin = sha(BUNDLE / 'protocol.json')
    assert pin == prep['protocol_sha256']
    assert (BUNDLE / 'protocol.sha256').read_text(encoding='ascii').strip() == pin + '  protocol.json'
    assert p['format'] == 'own-DGP-V26-saved-state-gradient-diagnostic-v1'
    assert p['snapshots'] == [0, 50] and p['head_batches'] == 20 and p['component_gradient_calls'] == 140
    assert p['recognizer_forwards'] == 70 and p['DGP_forwards'] == p['optimizer_updates'] == 0
    assert p['initial_corrected_identity_exact_zero_each_batch_and_all26_gradients']
    assert p['timing_limits'] == {'worker_seconds': 420, 'external_seconds': 480, 'external_kill_grace_seconds': 30,
        'export_seconds': 30, 'external_export_seconds': 60, 'export_kill_grace_seconds': 10,
        'free_disk_bytes': 1024 ** 3, 'allocated_VRAM_bytes': 20 * 1024 ** 3}
    for name, value in p['assets_sha256'].items():
        assert sha(BUNDLE / name) == value
    for name, value in p['local_basis_sha256'].items():
        assert sha(ROOT / name) == value
    parent = ROOT / 'outputs/cctv_dgp_batchmatched_identity_vm_v26'
    returned = ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_return'
    assert sha(parent / 'protocol.json') == p['closed_V26_protocol_sha256'] == sha(returned / 'protocol.json')
    original = read(parent / 'protocol.json')
    assert original['format'] == 'dgp-spatial-batchmatched-identity-capacity-v26'
    for name, value in original['assets_sha256'].items():
        assert sha(parent / name) == value
    for name, value in p['closed_V26_inputs_sha256'].items():
        assert sha(returned / name) == value
    assert len(p['closed_V26_inputs_sha256']) == 118
    cache = read(returned / 'outputs/frozen_DGP_features.json')
    assert len(cache['files_sha256']) == 250
    for name, value in cache['files_sha256'].items():
        assert sha(returned / 'outputs/frozen_DGP_features' / name) == value
    worker = BUNDLE / 'scripts/cctv_dgp_v26_gradient_diagnostic_v1_vm.py'
    source_contract(worker.read_text(encoding='utf-8'))
    return_audit_contract()
    assert sha(worker) == sha(ROOT / 'scripts/cctv_dgp_v26_gradient_diagnostic_v1_vm.py')
    shell = (BUNDLE / 'scripts/run_gradient.sh').read_text(encoding='utf-8')
    original_shell = (ROOT / 'outputs/cctv_dgp_v25_gradient_diagnostic_v1_vm/scripts/run_gradient.sh').read_text(encoding='utf-8')
    assert shell.replace('cctv_dgp_v26_gradient_diagnostic_v1', 'cctv_dgp_v25_gradient_diagnostic_v1') == original_shell
    transfer = subprocess.run([sys.executable, '-B', str(worker), '--root', str(BUNDLE),
        '--protocol-sha', pin, '--verify-transfer'], capture_output=True, text=True, timeout=20)
    assert transfer.returncode == 0, transfer.stderr
    guard = subprocess.run([sys.executable, '-B', str(worker), '--root', str(BUNDLE),
        '--protocol-sha', pin, '--run'], capture_output=True, text=True, timeout=20)
    assert sys.platform == 'win32' and guard.returncode != 0 and 'Existing Linux VM only' in guard.stderr
    assert not (BUNDLE / 'outputs').exists(), 'Local host rejection precedes output creation/model imports'
    assert receipt_name in ['independent_packet_audit.json', 'independent_packet_and_return_audit.json']
    receipt = {'complete': True, 'date': '2026-10-06', 'checker_sha256': sha(Path(__file__)),
        'protocol_sha256': pin, 'archive_sha256': sha(archive), 'archive_bytes': archive.stat().st_size,
        'regular_members': 4, 'original_assets_verified': len(original['assets_sha256']),
        'closed_saved_inputs_verified': 118, 'frozen_feature_arrays_verified': 250,
        'original_worker_AST_retained_except_declared_corrected_objective_proof_counter_and_names': True,
        'original_supervisor_unchanged_except_worker_name': True,
        'actual_Windows_transfer_and_neural_gradient_rejection_passed': True,
        'original_safe_import_and_full_matrix_audit_retained_except_declared_reference_and_name_changes': True,
        'neural_calls': 0, 'local_gradient_calls': 0, 'optimizer_updates': 0,
        'VM_actions': False, 'new_training_recipe': False, 'actual_L4_diagnostic_pending': True,
        'app_promotion': False, 'goal_complete': False, 'seconds': time.monotonic() - started}
    with (OUT / receipt_name).open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt-name', default='independent_packet_audit.json',
                        choices=['independent_packet_audit.json', 'independent_packet_and_return_audit.json'])
    main(parser.parse_args().receipt_name)
