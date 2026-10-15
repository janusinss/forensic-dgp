"""Independent R1 archive, layout regression, retained evidence, and launch guards."""
import ast
import copy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
from PIL import Image
import torch

from import_cctv_dgp_bank_loss_alignment_v1_return import ROOT, OUT, RETURN, ANALYSIS, PARENT, PIN, read, sha, write

DEST = OUT / 'cctv_dgp_bank_loss_alignment_v1_r1_vm'


def cpu_tensor_from_source(source):
    tree = ast.parse(source)
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'tensor')
    function = copy.deepcopy(function)

    class CPUOnly(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'cuda':
                assert not node.args and not node.keywords
                return node.func.value
            return node

    function = CPUOnly().visit(function)
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    environment = dict(torch=torch, np=np)
    exec(compile(module, '<isolated CPU tensor constructor>', 'exec'), environment)
    return environment['tensor']


def main():
    receipt = OUT / 'cctv_dgp_bank_loss_alignment_v1_r1_transfer_audit.json'
    assert not receipt.exists()
    prep_path = OUT / 'cctv_dgp_bank_loss_alignment_v1_r1_preparation.json'
    prep = read(prep_path)
    p = read(DEST / 'protocol.json')
    original = read(RETURN / 'protocol.json')
    old = read(PARENT / 'protocol.json')
    assert prep['complete'] and sha(DEST / 'protocol.json') == prep['protocol_sha256']
    assert sha(RETURN / 'protocol.json') == PIN
    for key, value in original.items():
        if key not in {'name', 'format', 'assets_sha256', 'evidence_sha256'}:
            assert p[key] == value, 'Unexpected protocol change: ' + key
    assert p['bit_exact_control_check_retained'] and p['frozen_raw_forward_blocks_retained'] == 40
    assert p['frozen_raw_forward_payload_bytes'] == 157286400
    assert p['combined_gradient_and_forward_payload_upper_bound_bytes'] == 712406496 < p['output_byte_cap']
    assert p['minimum_free_bytes'] >= 2 * p['output_byte_cap'] + p['disk_reserve_bytes'] + 1024**2
    assert p['optimizer_updates'] == p['backwards'] == p['full_epochs'] == p['native_pixels_used'] == p['final_identity_pixels'] == 0
    assert p['maximum_gradient_queries'] == 360 and p['model_forward_cap'] == 80 and p['recognizer_forward_cap'] == 40
    for root, files in [(DEST, p['assets_sha256']), (PARENT, p['parent_required_sha256'])]:
        for name, expected in files.items():
            path = root / name
            assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root)
            assert sha(path) == expected, name
    for name, expected in p['evidence_sha256'].items():
        assert sha(ROOT / name) == expected and read(ROOT / name)['complete']
    helper = 'cctv_dgp_bank_loss_alignment_v1.py'
    assert (DEST / helper).read_bytes() == (RETURN / helper).read_bytes()
    worker = DEST / 'scripts/cctv_dgp_bank_loss_alignment_v1_r1_vm.py'
    source = worker.read_text(encoding='utf-8')
    failed_source = (RETURN / 'scripts/cctv_dgp_bank_loss_alignment_v1_vm.py').read_text(encoding='utf-8')
    tree, old_tree = ast.parse(source), ast.parse(failed_source)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    old_functions = {n.name: n for n in old_tree.body if isinstance(n, ast.FunctionDef)}
    assert functions.keys() == old_functions.keys()
    for label in functions.keys() - {'run'}:
        assert ast.dump(functions[label]) == ast.dump(old_functions[label]), label
    run = functions['run']
    assert run.body[0].value.func.id == 'require_vm'
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) and 'torch' in ast.unparse(n) for n in tree.body)
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr in ['step', 'backward', 'Adam', 'AdamW', 'SGD'] for n in ast.walk(tree))
    # No objective, autograd query, or analytical direction changes are permitted.
    for function_name in ['components', 'summarize', 'hypothetical_adam_direction']:
        def calls(t):
            return [ast.dump(n) for n in ast.walk(t) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == function_name]
        assert calls(tree) == calls(old_tree)
    assert "assert exact,'Same L4 frozen raw control differs'" in source
    assert 'error<=1e-6' in source
    assert 'candidate.enable_vm_learning(parent)' in source
    assert 'state_hash(candidate)==before and buffer_hash(candidate)==buffers' in source
    assert 'maximum_absolute_error=float(np.abs(actual_raw.astype(np.float64)-expected.astype(np.float64)).max())' in source

    # Meaningful regression on all frozen diagnostic inputs/targets; this executes
    # only isolated tensor constructors. It never imports a model or VM worker.
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    parent_source = (PARENT / 'scripts/cctv_dgp_bank_comparison_v1_vm.py').read_text(encoding='utf-8')
    parent_tensor = cpu_tensor_from_source(parent_source)
    broken_tensor = cpu_tensor_from_source(failed_source)
    fixed_tensor = cpu_tensor_from_source(source)
    refs = {r['id']: r for r in old['references']}
    selected = [i for i in range(0, 3905, 5) if old['cases'][i]['id'] in p['case_ids']]
    assert len(selected) == 20
    samples = 0
    stride_mismatches = 0
    for begin in selected:
        block = old['cases'][begin:begin + 5]
        for field in ['input', 'target']:
            paths = [PARENT / c['input'] if field == 'input' else PARENT / refs[c['source_person_or_reference']]['target'] for c in block]
            arrays = []
            for path in paths:
                with Image.open(path) as image:
                    assert image.size == (256, 256)
                    arrays.append(np.asarray(image.convert('RGB')).copy())
            parent_batch = torch.cat([parent_tensor(a) for a in arrays])
            fixed_batch, broken_batch = fixed_tensor(arrays), broken_tensor(arrays)
            assert torch.equal(parent_batch, fixed_batch) and parent_batch.stride() == fixed_batch.stride()
            assert fixed_batch.is_contiguous() and torch.equal(broken_batch, fixed_batch)
            assert broken_batch.stride() != parent_batch.stride() and not broken_batch.is_contiguous()
            samples += len(arrays)
            stride_mismatches += 1
    assert samples == 200 and stride_mismatches == 40
    probe_path = ANALYSIS / 'layout_reproduction/run_results.json'
    probe = read(probe_path)
    folder = probe_path.parent
    arrays = {}
    for name, expected in probe['output_sha256'].items():
        path = folder / (name + '.npy')
        assert sha(path) == expected
        arrays[name] = np.load(path, allow_pickle=False)
        assert arrays[name].dtype == np.float32 and arrays[name].shape == (5, 256, 256, 3)
        assert np.isfinite(arrays[name]).all()
    assert np.array_equal(arrays['initial_canonical'], arrays['original_canonical'])
    delta = np.abs(arrays['initial_legacy'].astype(np.float64) - arrays['initial_canonical'].astype(np.float64))
    reproduction = probe['differences']['initial_legacy_vs_initial_canonical']
    assert float(delta.max()) == reproduction['maximum_absolute_error'] > 0
    assert int(np.count_nonzero(delta)) == reproduction['changed_values'] == 616175

    shell = DEST / 'scripts/run_alignment.sh'
    assert b'\r' not in shell.read_bytes()
    shell_source = shell.read_text()
    assert shell_source.count(worker.name) == 3 and '930 python' in shell_source and '330 python' in shell_source
    syntax = subprocess.run(['C:/Program Files/Git/bin/bash.exe', '-n', str(shell)], capture_output=True, text=True, timeout=30)
    assert syntax.returncode == 0, syntax.stderr
    archive = OUT / (prep['stem'] + '-execution.tar.gz')
    assert sha(archive) == prep['archive_sha256']
    assert Path(str(archive) + '.sha256').read_bytes() == (prep['archive_sha256'] + '  ' + archive.name + '\n').encode('ascii')
    with tarfile.open(archive, 'r:gz') as tar:
        expected = {prep['name'] + '/' + n: h for n, h in {**p['assets_sha256'], 'protocol.json': prep['protocol_sha256']}.items()}
        members = tar.getmembers()
        assert len(members) == len({m.name for m in members}) == len(expected) == 4
        for member in members:
            assert member.isfile() and member.name in expected
            assert hashlib.sha256(tar.extractfile(member).read()).hexdigest() == expected[member.name]
    assert not (DEST / 'outputs').exists()
    negative = subprocess.run([sys.executable, '-B', str(worker), '--root', str(DEST), '--parent-root', str(PARENT),
                               '--protocol-sha', prep['protocol_sha256'], '--run'], capture_output=True, text=True, timeout=120)
    assert negative.returncode != 0 and "platform.system()=='Linux'" in negative.stderr
    assert not (DEST / 'outputs').exists()
    write(OUT / 'cctv_dgp_bank_loss_alignment_v1_r1_windows_rejection.json',
          dict(returncode=negative.returncode, stdout=negative.stdout, stderr=negative.stderr,
               before_neural_and_gradient_imports=True))
    protected = read(ANALYSIS / 'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT / name) == expected for name, expected in protected.items())
    write(receipt, dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), checker_sha256=sha(__file__),
        preparation_sha256=sha(prep_path), protocol_sha256=prep['protocol_sha256'], archive_sha256=sha(archive),
        archive_members_checked=4, packet_assets_checked=3, parent_assets_checked=5852,
        CPU_input_and_target_case_tensors_matching_parent_values_and_strides=200,
        legacy_layout_regressions_detected_in_all40_batches=True,
        three_saved_frozen_CPU_output_arrays_independently_checked=True,
        frozen_CPU_layout_error_recomputed=float(delta.max()),
        loss_autograd_direction_source_case_lists_checkpoints_gates_and_limits_unchanged=True,
        bit_exact_control_replay_not_relaxed=True, all40_frozen_raw_blocks_will_be_retained=True,
        Bash_readonly_syntax_pass=True, LF_checksum_checked=True, Python310_hashing_compatible=True,
        Windows_rejected_before_neural_gradients_or_outputs=True, protected_bindings_unchanged=True,
        local_neural_calls=0, local_gradient_queries=0, local_optimizer_updates=0,
        actual_L4_repair_confirmation_and_gradients_unrun=True, no_agent_VM_launch=True,
        model_qualified=False, app_promotion=False, goal_complete=False))
    print(dict(complete=True, input_and_target_tensors_checked=samples,
               frozen_CPU_output_arrays_checked=3, raw_control_bit_equality_retained=True,
               actual_L4_confirmation_pending=True))


if __name__ == '__main__':
    main()
