"""Verify sealed names, retained math, transfer bytes and saved return fixtures."""
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import time

import numpy as np

from import_cctv_dgp_png_group_gradients_v1_r1_return import ROOT, OUT, NAME, STEM, PIN, PARENT, read, sha, write


def module(path, label):
    spec = importlib.util.spec_from_file_location(label, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def constants(tree, names):
    return {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and
            isinstance(node.targets[0], ast.Name) and node.targets[0].id in names}


def reject(call, *args):
    try:
        call(*args)
    except (AssertionError, ValueError):
        return True
    raise AssertionError('Invalid fixture was accepted')


def main():
    start = time.monotonic()
    receipt = OUT / 'cctv_dgp_png_group_gradients_v1_r1_transfer_audit.json'
    assert not receipt.exists()
    packet = OUT / NAME
    archive = OUT / (STEM + '-execution.tar.gz')
    prep = read(OUT / 'cctv_dgp_png_group_gradients_v1_r1_preparation.json')
    assert prep['complete'] and prep['protocol_sha256'] == PIN == sha(packet / 'protocol.json')
    assert sha(archive) == prep['archive_sha256']
    assert archive.stat().st_size == prep['archive_bytes']
    assert Path(str(archive) + '.sha256').read_bytes() == (sha(archive) + '  ' + archive.name + '\n').encode('ascii')
    p = read(packet / 'protocol.json')
    assert p['name'] == NAME and p['stem'] == STEM
    hashes = {**p['assets_sha256'], 'protocol.json': PIN}
    assert set(hashes) == {path.relative_to(packet).as_posix() for path in packet.rglob('*') if path.is_file()}
    assert len(hashes) == 7 and all(sha(packet / name) == digest for name, digest in hashes.items())
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        assert len(members) == 7
        seen = set()
        for member in members:
            path = PurePosixPath(member.name)
            assert member.isfile() and not member.islnk() and not member.issym()
            assert not path.is_absolute() and '..' not in path.parts and path.parts[0] == NAME
            assert path.as_posix() == member.name and ':' not in member.name and '\\' not in member.name
            name = PurePosixPath(*path.parts[1:]).as_posix()
            assert name in hashes and name.casefold() not in seen
            seen.add(name.casefold())
            with tar.extractfile(member) as stream:
                data = stream.read()
            assert len(data) == member.size == (packet / name).stat().st_size
            assert hashlib.sha256(data).hexdigest() == hashes[name]
        assert seen == {name.casefold() for name in hashes}

    old_root = OUT / 'cctv_dgp_png_group_gradients_v1_vm'
    old = read(old_root / 'protocol.json')
    assert all(sha(ROOT / name) == digest for name, digest in prep['original_files_sha256'].items())
    assert all(p[key] == value for key, value in old.items() if key not in ['name', 'UTC', 'assets_sha256'])
    for name in [name for name in old['assets_sha256'] if not name.startswith('scripts/')]:
        assert (packet / name).read_bytes() == (old_root / name).read_bytes()
    worker_path = packet / 'scripts' / (NAME + '.py')
    worker = worker_path.read_text(encoding='utf-8')
    tree = ast.parse(worker, feature_version=(3, 10))
    assert constants(tree, ['NAME', 'STEM']) == {'NAME': NAME, 'STEM': STEM}
    assert worker.replace(NAME, 'cctv_dgp_png_group_gradients_v1_vm').replace(
        STEM, 'cctv-dgp-bank-png-group-gradients-v1').replace(
        'outputs/png_group_gradients_v1_r1', 'outputs/png_group_gradients_v1') == (
        old_root / 'scripts/cctv_dgp_png_group_gradients_v1_vm.py').read_text(encoding='utf-8')
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
                   node.func.attr in ['step', 'backward', 'copy_', 'add_', 'sub_', 'Adam', 'AdamW', 'SGD']
                   for node in ast.walk(tree))
    assert sum(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
               node.func.attr == 'grad' for node in ast.walk(tree)) == 1
    assert worker.index('Initial delivered loss definition differs') < worker.index('torch.autograd.grad')
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
    assert isinstance(run.body[0], ast.Expr) and run.body[0].value.func.id == 'require_vm'
    main_fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    main_text = ast.unparse(main_fn)
    assert main_text.index('require_vm(root, parent)') < main_text.index('verify(root, parent, a.protocol_sha)')
    assert worker.index('require_vm(root,parent)') < worker.index('import torch')
    shell = (packet / 'scripts/run_png_groups.sh').read_text(encoding='utf-8')
    assert '\r' not in shell
    assert shell.count(NAME + '.py') == 3 and 'tee review.log' in shell and 'tee export.log' in shell
    assert 'timeout --signal=TERM --kill-after=30 930' in shell and 'timeout --signal=TERM --kill-after=30 330' in shell
    assert shell.count('statuses=("${PIPESTATUS[@]}")') == 2
    assert shell.replace(NAME + '.py', 'cctv_dgp_png_group_gradients_v1_vm.py') == (
        old_root / 'scripts/run_png_groups.sh').read_text(encoding='utf-8')

    worker_module = module(worker_path, 'sealed_worker_hash_verify_only')
    verified, parent = worker_module.verify(packet.resolve(), PARENT.resolve(), PIN)
    assert verified == p and 'torch' not in sys.modules
    result = subprocess.run([sys.executable, '-B', str(worker_path), '--root', str(packet),
        '--parent-root', str(PARENT), '--protocol-sha', PIN, '--run'], capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and 'require_vm' in result.stderr and 'AssertionError' in result.stderr
    assert not (packet / 'outputs').exists() and not (packet / 'review.log').exists()

    contract = module(packet / 'cctv_dgp_png_group_gradients_v1_contract.py', 'PNG_scope_contract')
    blocks, groups = contract.scopes(parent['cases'], p['case_ids'])
    assert blocks == p['reference_blocks'] and groups == p['gradient_groups']
    bindings = {group['key']: [] for group in groups}
    for block in blocks:
        for query in range(16):
            bindings[contract.query_key(block['source'], query)].append(block['reference'])
    assert sum(map(len, bindings.values())) == 320
    assert all(len(ids) == len(set(ids)) == 10 for ids in bindings.values())
    refs = {row['id']: row for row in parent['references']}
    cases = [row for row in parent['cases'] if row['id'] in p['case_ids']]
    assert len(contract.fixed_reference_matrices(cases, refs)) == 100
    invalid = []
    bad_refs = {cases[0]['source_person_or_reference']: {'matrix112': [[0., 0.]] * 5}}
    assert reject(contract.fixed_reference_matrices, cases[:1], bad_refs)
    invalid.append('landmarks_instead_of_affine')
    for durations in [[0, 1], [1], [float('nan'), 1]]:
        assert reject(contract.projection_seconds, durations)
    assert contract.projection_seconds([1., 2.]) == 60.
    invalid.extend(['zero_batch_duration', 'missing_batch_duration', 'nonfinite_batch_duration'])
    assert reject(contract.scopes, parent['cases'], p['case_ids'][:-1])
    invalid.append('missing_frozen_case')
    assert reject(contract.query_key, blocks[0]['source'], 16)
    invalid.append('query_out_of_range')

    checker_path = ROOT / 'scripts/audit_cctv_dgp_png_group_gradients_v1_r1_return.py'
    checker = module(checker_path, 'prospective_saved_return_checker')
    metrics = read(PARENT / 'outputs/bank_comparison_v1/baseline/metrics.json')['rows']
    selected = [row for row in metrics if row['id'] in p['case_ids']]
    assert len(metrics) == 3905 and len(selected) == 100
    forward = read(OUT / 'cctv_dgp_png_group_gradients_v1_forward_audit.json')
    assert forward['complete'] and forward['saved_forward_cases'] == 100
    assert forward['loss_helper_sha256'] == sha(packet / 'cctv_dgp_png_group_gradients_v1_losses.py')
    assert all(sha(PARENT / 'outputs/bank_comparison_v1/baseline/previews' / (r['case'] + '.png')) == r['PNG_sha256']
               for r in forward['PNG_bindings'])
    assert forward['zero_local_model_inference_autograd_or_optimizer_calls']
    for group in groups:
        value = checker.baseline_value(group, selected)
        assert np.isfinite(value)
        assert reject(checker.baseline_value, group, metrics)
    vector = np.array([1., -2., 3.], np.float64)
    directions = {'balanced': np.array([2., 1., -1.], np.float64)}
    record = dict(gradient_norm=float(np.linalg.norm(vector)), saved_direction_dot={'balanced': float(vector @ directions['balanced'])})
    assert checker.check_vector(vector, record, directions) == record['gradient_norm']
    assert reject(checker.check_vector, vector.astype(np.float32), record, directions)
    assert reject(checker.check_vector, np.array([np.nan, 0., 0.]), record, directions)
    assert reject(checker.check_vector, vector, dict(record, gradient_norm=0), directions)
    assert reject(checker.check_vector, vector, dict(record, saved_direction_dot={'balanced': 0.}), directions)
    invalid.extend(['gradient_dtype', 'nonfinite_gradient', 'gradient_norm_tamper', 'gradient_dot_tamper'])
    assert 'torch' not in sys.modules, 'No local neural/gradient library execution in transfer checker'
    with np.load(packet / 'directions.npz', allow_pickle=False) as saved:
        assert set(saved.files) == {f'balanced_p{i:02d}' for i in range(37)}
        assert sum(saved[f'balanced_p{i:02d}'].size for i in range(37)) == p['parameters']

    importer_text = (ROOT / 'scripts/import_cctv_dgp_png_group_gradients_v1_r1_return.py').read_text(encoding='utf-8')
    assert constants(ast.parse(importer_text), ['NAME', 'STEM', 'PIN']) == {'NAME': NAME, 'STEM': STEM, 'PIN': PIN}
    assert "cctv_dgp_png_group_gradients_v1_r1_transfer_audit.json" in importer_text
    original_checker = (OUT / 'cctv_dgp_png_group_gradients_v1_r1_return_checker_preparation/original_checker.py').read_text()
    current_checker = checker_path.read_text()
    before = "    baselines=read(PARENT/'outputs/bank_comparison_v1/baseline/metrics.json')['rows']"
    after = "    baselines=[row for row in read(PARENT/'outputs/bank_comparison_v1/baseline/metrics.json')['rows'] if row['id'] in p['case_ids']]\n    assert len(baselines)==100 and {row['id'] for row in baselines}==set(p['case_ids'])"
    assert current_checker.replace(after, before) == original_checker
    assert sha(OUT / 'cctv_dgp_png_group_gradients_v1_r1_return_checker_preparation/original_checker.py') == prep['prospective_return_auditor_sha256']
    protected = read(OUT / 'cctv_dgp_bank_group_step_check_v1_analysis/local_protected_before.json')['files_sha256']
    assert all(sha(ROOT / name) == digest for name, digest in protected.items())
    assert all(sha(ROOT / name) == digest for name, digest in prep['original_files_sha256'].items())
    checker_receipt = OUT / 'cctv_dgp_png_group_gradients_v1_r1_return_checker_preparation/verified.json'
    write(checker_receipt, dict(complete=True, protocol_sha256=PIN, checker_sha256=sha(checker_path),
        importer_sha256=sha(ROOT / 'scripts/import_cctv_dgp_png_group_gradients_v1_r1_return.py'),
        original_checker_retained=True, only_selected_baseline_filter_changed=True,
        selected_baseline_cases_checked=100, baseline_group_fixture_checks=32,
        full_parent_group_scope_rejected=32, invalid_vector_fixtures_rejected=4,
        execution_and_quality_counts_not_inferred_from_future_export=True,
        actual_return_not_yet_present=True, independent_return_audit_not_yet_performed=True,
        local_neural_gradient_optimizer_calls=0, model_qualified=False))
    assert time.monotonic() - start < 180
    write(receipt, dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), checker_sha256=sha(__file__),
        protocol_sha256=PIN, archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
        exact_regular_archive_files_verified=7, packet_assets_verified=6,
        parent_assets_verified=len(p['parent_required_sha256']), protected_local_files_verified=len(protected),
        matching_prepared_worker_importer_export_stems=True, corrected_result_stage_checked=True,
        original_V1_archive_and_sources_unchanged=True, only_packet_identity_changed=True,
        mathematical_helpers_and_all_scientific_gates_unchanged=True, checksum_is_exact_LF=True,
        actual_worker_hash_only_verify_passed=True, Windows_run_rejected_before_neural_import=True,
        all320_query_bindings_and32_PNG_groups_checked=True, frozen_affines_checked=100,
        invalid_fixtures_rejected=invalid, retained_saved_forward_value_audit_sha256=sha(OUT / 'cctv_dgp_png_group_gradients_v1_forward_audit.json'),
        saved_forward_cases_bound=100, unchanged_helper_receipt_reused_no_new_gradient=True,
        prospective_return_checker_sha256=sha(checker_path), return_checker_receipt_sha256=sha(checker_receipt),
        local_neural_model_gradient_or_optimizer_calls=0, no_VM_connection_or_launch=True,
        R1_only_local_and_unrun=True, original_VM_execution_unknown=True,
        model_qualified=False, more_epochs_qualified=False, goal_complete=False,
        seconds=time.monotonic() - start))
    print(dict(complete=True, archive_bytes=archive.stat().st_size, parent_assets=len(p['parent_required_sha256']),
        PNG_groups=32, future_gradient_queries=320, local_model_or_gradient_calls=0))


if __name__ == '__main__':
    main()
