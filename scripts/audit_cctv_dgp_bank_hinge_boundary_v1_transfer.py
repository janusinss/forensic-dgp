"""Audit sealed manual boundary diagnostic and unchanged computation limits."""
import ast
from datetime import datetime, timezone
import hashlib
import subprocess
import sys
import tarfile

from import_cctv_dgp_bank_loss_alignment_v1_r1_return import ROOT, OUT, RETURN, ANALYSIS, PARENT, PIN, read, sha, write


def main():
    receipt = OUT / 'cctv_dgp_bank_hinge_boundary_v1_transfer_audit.json'
    assert not receipt.exists()
    prep_path = OUT / 'cctv_dgp_bank_hinge_boundary_v1_preparation.json'
    prep = read(prep_path)
    dest = OUT / prep['name']
    p, preceding = read(dest / 'protocol.json'), read(RETURN / 'protocol.json')
    assert prep['complete'] and sha(dest / 'protocol.json') == prep['protocol_sha256']
    assert p['diagnostic_parent_protocol_sha256'] == PIN
    for key, value in preceding.items():
        if key not in {'name', 'format', 'assets_sha256', 'evidence_sha256', 'loss_and_optimizer_direction_configuration_unchanged'}:
            assert p[key] == value, key
    assert p['scalar_loss_values_weights_and_optimizer_direction_configuration_unchanged']
    assert not p['loss_and_optimizer_direction_configuration_unchanged']
    assert p['initial_zero_guard_gradient_required'] and p['exact_scalar_guard_parity_required']
    assert p['optimizer_updates'] == p['backwards'] == p['full_epochs'] == p['native_pixels_used'] == p['final_identity_pixels'] == 0
    assert p['maximum_gradient_queries'] == 360 and p['model_forward_cap'] == 80 and p['recognizer_forward_cap'] == 40
    assert p['minimum_free_bytes'] >= 2*p['output_byte_cap']+p['disk_reserve_bytes']+1024**2
    assert p['combined_gradient_and_forward_payload_upper_bound_bytes'] < p['output_byte_cap']
    for root, files in [(dest, p['assets_sha256']), (PARENT, p['parent_required_sha256'])]:
        for name, expected in files.items():
            path = root / name
            assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root)
            assert sha(path) == expected
    for name, expected in p['evidence_sha256'].items():
        assert sha(ROOT / name) == expected and read(ROOT / name)['complete']
    assert (dest / 'cctv_dgp_bank_loss_alignment_v1.py').read_bytes() == (RETURN / 'cctv_dgp_bank_loss_alignment_v1.py').read_bytes()
    helper = dest / 'cctv_dgp_bank_hinge_boundary_v1_loss.py'
    loss_audit = read(OUT / 'cctv_dgp_bank_hinge_boundary_v1_loss_audit.json')
    assert loss_audit['exact_scalar_and_metadata_parity_all40'] and sha(helper) == loss_audit['corrected_helper_sha256']
    worker = dest / 'scripts/cctv_dgp_bank_hinge_boundary_v1_vm.py'
    source = worker.read_text()
    tree = ast.parse(source)
    old_tree = ast.parse((RETURN / 'scripts/cctv_dgp_bank_loss_alignment_v1_r1_vm.py').read_text())
    functions = {n.name:n for n in tree.body if isinstance(n, ast.FunctionDef)}
    old_functions = {n.name:n for n in old_tree.body if isinstance(n, ast.FunctionDef)}
    for key in functions.keys() - {'run'}:
        assert ast.dump(functions[key]) == ast.dump(old_functions[key]), key
    assert functions['run'].body[0].value.func.id == 'require_vm'
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) and 'torch' in ast.unparse(n) for n in tree.body)
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr in ['step', 'backward', 'Adam', 'AdamW', 'SGD'] for n in ast.walk(tree))
    assert len([n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'grad']) == 1
    for called in ['components', 'hypothetical_adam_direction', 'summarize']:
        get = lambda t:[ast.dump(n) for n in ast.walk(t) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == called]
        assert get(tree) == get(old_tree)
    get_tensor = lambda t:ast.dump(next(n for n in ast.walk(t) if isinstance(n, ast.FunctionDef) and n.name == 'tensor'))
    assert get_tensor(tree) == get_tensor(old_tree)
    assert "assert exact,'Same L4 frozen raw control differs'" in source and 'error<=1e-6' in source
    assert "all(torch.count_nonzero(g).item()==0 for g in gradient_rows[6])" in source
    assert 'torch.equal(photo.detach(),retained_photo) and photo_terms==retained_terms' in source
    assert 'candidate.enable_vm_learning(parent)' in source
    assert 'state_hash(candidate)==before and buffer_hash(candidate)==buffers' in source
    assert 'file_digest' not in source
    shell = dest / 'scripts/run_boundary.sh'
    shell_text = shell.read_text()
    assert b'\r' not in shell.read_bytes() and shell_text.count(worker.name) == 3
    assert '930 python' in shell_text and '330 python' in shell_text and '--kill-after=30' in shell_text
    syntax = subprocess.run(['C:/Program Files/Git/bin/bash.exe', '-n', str(shell)], capture_output=True, text=True, timeout=30)
    assert syntax.returncode == 0, syntax.stderr
    archive = OUT / (prep['stem'] + '-execution.tar.gz')
    assert sha(archive) == prep['archive_sha256']
    assert archive.stat().st_size == prep['archive_bytes']
    assert (OUT / (archive.name + '.sha256')).read_bytes() == (prep['archive_sha256']+'  '+archive.name+'\n').encode('ascii')
    with tarfile.open(archive, 'r:gz') as tar:
        expected = {prep['name']+'/'+name:digest for name, digest in {**p['assets_sha256'], 'protocol.json':prep['protocol_sha256']}.items()}
        members = tar.getmembers()
        assert len(members) == len({m.name for m in members}) == len(expected) == 5
        for member in members:
            assert member.isfile() and member.name in expected
            assert hashlib.sha256(tar.extractfile(member).read()).hexdigest() == expected[member.name]
    assert not (dest / 'outputs').exists()
    negative = subprocess.run([sys.executable, '-B', str(worker), '--root', str(dest), '--parent-root', str(PARENT),
        '--protocol-sha', prep['protocol_sha256'], '--run'], capture_output=True, text=True, timeout=120)
    assert negative.returncode != 0 and "platform.system()=='Linux'" in negative.stderr
    assert not (dest / 'outputs').exists()
    write(OUT / 'cctv_dgp_bank_hinge_boundary_v1_windows_rejection.json',
          dict(returncode=negative.returncode, stdout=negative.stdout, stderr=negative.stderr, pre_neural_rejection=True))
    for name, expected in read(ANALYSIS / 'local_protected_before.json')['files_sha256'].items():
        assert sha(ROOT / name) == expected
    write(receipt, dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), checker_sha256=sha(__file__),
        preparation_sha256=sha(prep_path), protocol_sha256=prep['protocol_sha256'], archive_sha256=sha(archive),
        packet_assets_checked=4, archive_members_checked=5, parent_assets_checked=5852,
        LF_checksum_and_readonly_Bash_syntax_pass=True, Python310_hashing_compatible=True,
        corrected_input_layout_reused_and_exact_replay_retained=True,
        scalar_loss_parity_audit_sha256=sha(OUT / 'cctv_dgp_bank_hinge_boundary_v1_loss_audit.json'),
        single_declared_hinge_derivative_change=True,
        loss_weights_and_Adam_math_cases_states_gates_resource_limits_preserved=True,
        Windows_rejected_before_neural_or_gradient_imports=True,
        local_neural_calls=0, local_gradient_queries=0, local_optimizer_updates=0,
        manual_L4_gradient_measurements_and_new_behavior_unrun=True,
        no_agent_VM_launch=True, model_qualified=False, app_promotion=False, goal_complete=False))
    print(dict(complete=True, packet_assets=4, archive_members=5, parent_assets=5852,
               local_gradient_queries=0, manual_gradient_diagnostic_pending=True))


if __name__ == '__main__':
    main()
