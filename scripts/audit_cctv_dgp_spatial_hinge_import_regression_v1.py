"""Trace the retained import regression and compare saved scalar values only.

No neural model, parameter installation, autograd, optimizer or VM execution.
"""
import ast
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import time

from cctv_dgp_spatial_path_replay_v1_contract import read, write, sha

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT/'outputs/cctv_dgp_spatial_readout_v1_return'
PARENT = ROOT/'outputs/cctv_dgp_bank_photometric_guard_v1_return'
REPLAY = ROOT/'outputs/cctv_dgp_spatial_path_replay_v1_return'
ANALYSIS = ROOT/'outputs/cctv_dgp_spatial_path_replay_v1_analysis'


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    import numpy as np
    from PIL import Image
    import torch
    start = time.monotonic()
    destination = ANALYSIS/'hinge_import_regression_audit.json'
    assert not destination.exists(), 'Retain prior audit'
    assert read(ANALYSIS/'independent_audit.json')['complete']
    source = read(SOURCE/'protocol.json')
    worker = SOURCE/'scripts/cctv_dgp_spatial_readout_v1_vm.py'
    legacy = PARENT/'cctv_dgp_bank_photometric_guard_v1_loss.py'
    corrected = ROOT/'scripts/cctv_dgp_bank_hinge_boundary_v1_loss.py'
    historical = ROOT/'outputs/cctv_dgp_bank_hinge_boundary_v1_analysis/independent_audit.json'
    historical_loss = ROOT/'outputs/cctv_dgp_bank_hinge_boundary_v1_vm/cctv_dgp_bank_hinge_boundary_v1_loss.py'
    assert sha(worker) == source['assets_sha256']['scripts/cctv_dgp_spatial_readout_v1_vm.py']
    assert sha(legacy) == source['parent_required_sha256'][legacy.name] == '7474a7688872a130bf60ced51eff37e25017f295f73a459b06d082c23d4ca393'
    assert sha(corrected) == sha(historical_loss) == '8368f97c13b4703834c6ea249a85a6970322dce45416bfecd23d8a299f6706f8'
    tree = ast.parse(worker.read_text(encoding='utf-8'))
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and any(a.name == 'brightness_guard' for a in n.names)]
    assert len(imports) == 1 and imports[0].module == legacy.stem
    loader = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_dependencies')
    assert "sys.path.insert(1, str(parent))" in ast.unparse(loader)
    assert corrected.stem not in worker.read_text(encoding='utf-8')
    old_f = next(n for n in ast.parse(legacy.read_text()).body if isinstance(n, ast.FunctionDef))
    new_f = next(n for n in ast.parse(corrected.read_text()).body if isinstance(n, ast.FunctionDef))
    assert ast.dump(old_f.args) == ast.dump(new_f.args) and len(old_f.body) == len(new_f.body)
    differences = []
    for before, after in zip(old_f.body, new_f.body):
        if ast.dump(before) == ast.dump(after):
            continue
        assert isinstance(before, ast.Assign) and isinstance(after, ast.Assign)
        name = before.targets[0].id
        assert name in ['mean_gain', 'actual_gain', 'excess'] and ast.dump(before.targets[0]) == ast.dump(after.targets[0])
        assert before.value.func.attr == 'clamp_min' and ast.literal_eval(before.value.args[0]) == 0
        assert after.value.func.attr == 'relu' and ast.dump(before.value.func.value) == ast.dump(after.value.args[0])
        differences.append(name)
    assert differences == ['mean_gain', 'actual_gain', 'excess']
    old_audit = read(historical)
    assert old_audit['complete'] and old_audit['inactive_initial_guard_zero_gradient_all20_independently_checked']
    assert old_audit['counts_recomputed']['optimizer_updates'] == 0
    alignment_path = ROOT/'outputs/cctv_dgp_bank_loss_alignment_v1_r1_analysis/loss_direction_review.json'
    alignment = read(alignment_path)
    assert alignment['group_summaries']['all']['initial']['scalar_zero_brightness_penalty_with_nonzero_gradient'] == 20
    initial = {}
    for route in ['A', 'B']:
        rows = read(SOURCE/f'outputs/spatial_readout_v1/preflight_{route}.json')['rows']
        assert len(rows) == 2 and all(row['terms']['photometric_guard'] == 0 for row in rows)
        initial[route] = dict(saved_preflight_zero_penalties=2, component_parameter_gradients_not_separately_saved=True)
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    old_module, new_module = module(legacy, 'saved_legacy_brightness_scalar'), module(corrected, 'saved_corrected_brightness_scalar')
    data = read(PARENT/'protocol.json')
    refs = {r['id']:r for r in data['references']}
    cases = {c['id']:c for c in data['cases']}
    p = read(REPLAY/'protocol.json')
    rows = []
    for i, block in enumerate(p['reference_blocks']):
        reference = refs[block['reference']]
        with Image.open(PARENT/reference['target']) as image:
            target = np.asarray(image.convert('RGB')).copy()
        with Image.open(PARENT/reference['observed']) as image:
            observed = np.asarray(image.convert('L')) > 0
        target = torch.from_numpy(np.stack([target]*5).astype(np.float32)/np.float32(255)).permute(0,3,1,2).contiguous()
        mask = torch.from_numpy(np.stack([observed]*5).astype(np.float32))[:,None]
        stage = REPLAY/'outputs/spatial_path_replay_v1'
        with np.load(stage/'baseline/packs'/f'r{i:02d}.npz', allow_pickle=False) as pack:
            baseline_array = pack['raw'].copy()
        baseline = torch.from_numpy(baseline_array).permute(0,3,1,2).contiguous()
        for label in ['A_u000', 'A_u001', 'B_u001', 'A_u050', 'B_u050']:
            with np.load(stage/label/'packs'/f'r{i:02d}.npz', allow_pickle=False) as pack:
                assert list(pack['ids']) == block['case_ids']
                prediction = torch.from_numpy(pack['raw'].copy()).permute(0,3,1,2).contiguous()
            old_value, old_terms = old_module.brightness_guard(prediction, baseline, target, mask, source['normalizers'][0])
            new_value, new_terms = new_module.brightness_guard(prediction, baseline, target, mask, source['normalizers'][0])
            assert torch.equal(old_value, new_value) and old_terms == new_terms
            assert not new_value.requires_grad and new_value.grad_fn is None
            if label == 'A_u000':
                assert float(new_value) == 0
            rows.append(dict(reference=block['reference'], snapshot=label, penalty=float(new_value), scalar_and_metadata_exact=True))
    write(destination, dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        worker_sha256=sha(worker), legacy_helper_sha256=sha(legacy), corrected_helper_sha256=sha(corrected),
        historical_hinge_audit_sha256=sha(historical), historical_alignment_review_sha256=sha(alignment_path),
        direct_import_module=imports[0].module, import_line=imports[0].lineno,
        parent_inserted_at_sys_path_index=1, corrected_module_not_imported=True,
        legacy_helper_bound_in_source_protocol=True, both_routes_share_same_objective_import=True,
        helper_changes_only_three_zero_boundary_subgradients=differences,
        historical_initial_B_zero_penalty_nonzero_legacy_gradient_references=20,
        historical_corrected_initial_B_zero_gradient_references=20,
        historical_optimizer_updates=0, fresh_A_guard_component_gradient_measurement_pending=True,
        source_initial_preflights=initial, scalar_rows_checked=len(rows), scalar_case_compositions=len(rows)*5,
        exact_scalar_and_metadata_parity=True, rows=rows,
        whole_failure_cause_not_uniquely_proven=True, learning_rate_cause_not_isolated=True,
        local_neural_calls=0, local_gradient_queries=0, local_optimizer_updates=0,
        model_qualified=False, seconds=time.monotonic()-start,
        primary_source='https://github.com/pytorch/pytorch/blob/v2.9.1/tools/autograd/derivatives.yaml'))
    print(json.dumps(dict(complete=True, import_regression_confirmed=True, scalar_rows=len(rows), local_gradient_queries=0)))


if __name__ == '__main__':
    main()
