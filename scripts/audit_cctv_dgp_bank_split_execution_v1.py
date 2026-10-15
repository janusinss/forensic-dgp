"""Independent finite-execution and migration-state audit; no neural calls."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import torch


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def finite(value):
    if isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all()
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            finite(item)
    elif isinstance(value, float):
        import math
        assert math.isfinite(value)


def audit(root, imported, receipt):
    start = time.monotonic()
    assert not receipt.exists()
    root = root.resolve()
    imp = read(imported)
    assert imp['complete'] and imp['every_member_path_and_byte_hash_checked']
    p = read(root / 'protocol.json')
    assert sha(root / 'protocol.json') == imp['protocol_sha256']
    assert sha(root / 'export_manifest.json') == imp['export_manifest_sha256']
    manifest = read(root / 'export_manifest.json')
    out = root / 'outputs/bank_comparison_v1'
    result = read(out / 'results.json')
    env = read(out / 'environment.json')
    assert result['complete'] and result['execution_finished']
    assert result['protocol_sha256'] == imp['protocol_sha256']
    assert not result['automatic_follow_on'] and not result['app_promotion'] and not result['goal_complete']
    assert result['full_epochs'] == 0 and not result['comparison_complete']
    assert env['GPU'] == 'NVIDIA L4' and env['host'].split('.')[0] == 'forensic-dgp-thesis'
    assert not env['AMP'] and not env['TF32'] and env['stored_normalization_unchanged']
    assert env['DGP_provenance']['weights_sha256'] == p['original_checkpoint_sha256']
    budget = p['budgets']
    assert 0 < result['seconds'] <= budget['worker_seconds']
    counts = result['counts']
    assert set(counts) == {'original', 'A', 'B', 'recognizer', 'prior_fixture', 'gradient_queries', 'backwards', 'optimizer_updates'}
    for name, count in counts.items():
        cap = budget[name + '_forward_calls'] if name in {'original', 'A', 'B', 'recognizer', 'prior_fixture'} else budget['maximum_' + name]
        assert isinstance(count, int) and 0 <= count <= cap
    assert [r['route'] for r in result['routes']] == p['selected_routes']
    by_case = p['cases']
    exposures = {}
    states = []
    for route in result['routes']:
        label, updates = route['route'], route['updates']
        assert updates == p['maximum_updates_per_route'] == 50
        history = read(out / ('history_' + label + '.json'))
        assert history['route'] == label and history['updates'] == updates
        assert [r['update'] for r in history['rows']] == list(range(1, updates + 1))
        assert [r['reference_starts'] for r in history['rows']] == p['schedule']
        finite(history)
        assert all(r['gradient_norm_before_clip'] > 0 for r in history['rows'])
        starts = [i for r in history['rows'] for i in r['reference_starts']]
        assert len(starts) == len(set(starts)) == 250
        assert all(i % 5 == 0 for i in starts)
        sources = Counter(by_case[i]['source'] for i in starts)
        assert len(sources) == 2 and set(sources.values()) == {125}
        profiles = Counter(by_case[i+j]['profile'] for i in starts for j in range(5))
        assert len(profiles) == 5 and set(profiles.values()) == {250}
        exposures[label] = dict(updates=updates, distinct_references=250, source_references=dict(sources),
                               profile_exposures=dict(profiles), full_epochs=0, fraction_of_781_reference_epoch=250/781)
        state_path = out / ('state_' + label + '50/full_state.pth')
        assert sha(state_path) == manifest['files'][state_path.relative_to(root).as_posix()]
        state = torch.load(state_path, map_location='cpu', weights_only=True)
        finite(state)
        assert state['protocol_sha256'] == imp['protocol_sha256'] and state['route'] == label
        assert state['updates'] == updates and state['schedule'] == p['schedule']
        assert not state['manual_failed_resume_allowed']
        assert state['counts']['optimizer_updates'] == updates
        assert state['counts']['backwards'] == updates * 5
        assert all(0 <= state['counts'][k] <= counts[k] for k in counts)
        assert state['CPU_rng'].dtype == torch.uint8 and state['CPU_rng'].ndim == 1 and state['CPU_rng'].numel() > 0
        assert len(state['CUDA_rng']) == 1
        assert all(v.dtype == torch.uint8 and v.ndim == 1 and v.numel() > 0 for v in state['CUDA_rng'])
        optimizer = state['optimizer']
        group, = optimizer['param_groups']
        assert len(group['params']) == len(optimizer['state']) == p['routes'][label]['tensors']
        for name in ['lr', 'eps', 'weight_decay']:
            assert group[name] == p['optimizer'][name]
        assert list(group['betas']) == p['optimizer']['betas']
        for value in optimizer['state'].values():
            assert set(value) == {'step', 'exp_avg', 'exp_avg_sq'}
            assert float(value['step']) == updates and value['exp_avg'].shape == value['exp_avg_sq'].shape
            assert (value['exp_avg_sq'] >= 0).all()
        scheduler = state['scheduler']
        assert scheduler['last_epoch'] == updates and scheduler['base_lrs'] == [p['optimizer']['lr']]
        assert scheduler['_last_lr'] == [p['optimizer']['lr']]
        states.append(dict(route=label, updates=updates, full_state_sha256=sha(state_path),
                           optimizer_and_scheduler_finite_and_matched=True, CPU_and_CUDA_RNG_retained=True,
                           failed_resume_allowed=False, exact_future_resume_guaranteed=False))
    assert counts['optimizer_updates'] == sum(r['updates'] for r in result['routes'])
    assert counts['backwards'] == counts['optimizer_updates'] * 5
    projection = read(out / 'storage_projection.json')
    assert projection['cap_bytes'] == budget['return_uncompressed_bytes']
    assert projection['projected_return_bytes'] <= projection['cap_bytes']
    output_bytes = sum(f.stat().st_size for f in out.rglob('*') if f.is_file())
    assert output_bytes <= budget['return_uncompressed_bytes']
    assert read(out / 'timing_projection.json')['projected_worker_seconds'] <= budget['worker_seconds']
    for route in p['selected_routes']:
        preflight = read(out / ('preflight_' + route + '.json'))
        assert preflight['optimizer_updates'] == 0
        assert preflight['fit_projection_seconds'] <= preflight['fit_cap_seconds'] == budget['arm_fit_seconds']
        assert preflight['snapshot_projection_seconds'] <= preflight['snapshot_cap_seconds'] == budget['snapshot_seconds']
    result_receipt = dict(complete=True, checker_sha256=sha(__file__), import_receipt_sha256=sha(imported),
        protocol_sha256=imp['protocol_sha256'], manifest_sha256=imp['export_manifest_sha256'],
        actual_optimizer_updates=counts['optimizer_updates'], counts=counts, exposures=exposures,
        states=states, environment_GPU=env['GPU'], worker_seconds=result['seconds'], output_bytes=output_bytes,
        enforced_fit_cap_seconds=budget['arm_fit_seconds'], actual_fit_seconds_separately_recorded=False,
        scientific_quality_audit_separate=True, local_neural_gradient_or_optimizer_calls=0,
        model_qualified=False, app_promotion=False, goal_complete=False, seconds=time.monotonic()-start)
    with receipt.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result_receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result_receipt[k] for k in ['complete', 'actual_optimizer_updates', 'worker_seconds', 'output_bytes', 'seconds']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--import-receipt', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.import_receipt, args.receipt)
