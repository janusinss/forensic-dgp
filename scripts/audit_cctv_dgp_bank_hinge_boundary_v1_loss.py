"""Check scalar-equivalent hinge repair on saved arrays; no local autograd."""
import ast
from datetime import datetime, timezone
import importlib.util

import numpy as np
from PIL import Image
import torch

from import_cctv_dgp_bank_loss_alignment_v1_r1_return import ROOT, OUT, RETURN, ANALYSIS, PARENT, read, sha, write


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    receipt = OUT / 'cctv_dgp_bank_hinge_boundary_v1_loss_audit.json'
    assert not receipt.exists()
    assert read(ANALYSIS / 'independent_audit.json')['complete']
    review = read(ANALYSIS / 'loss_direction_review.json')
    assert review['complete'] and review['group_summaries']['all']['initial']['scalar_zero_brightness_penalty_with_nonzero_gradient'] == 20
    old_path = PARENT / 'cctv_dgp_bank_photometric_guard_v1_loss.py'
    new_path = ROOT / 'scripts/cctv_dgp_bank_hinge_boundary_v1_loss.py'
    p = read(RETURN / 'protocol.json')
    assert sha(old_path) == p['parent_required_sha256'][old_path.name]
    old_tree, new_tree = ast.parse(old_path.read_text()), ast.parse(new_path.read_text())
    old_function = next(n for n in old_tree.body if isinstance(n, ast.FunctionDef))
    new_function = next(n for n in new_tree.body if isinstance(n, ast.FunctionDef))
    assert ast.dump(old_function.args) == ast.dump(new_function.args)
    assert len(old_function.body) == len(new_function.body)
    differences = []
    for old_statement, new_statement in zip(old_function.body, new_function.body):
        if ast.dump(old_statement) != ast.dump(new_statement):
            assert isinstance(old_statement, ast.Assign) and isinstance(new_statement, ast.Assign)
            name = old_statement.targets[0].id
            assert ast.dump(old_statement.targets[0]) == ast.dump(new_statement.targets[0])
            assert name in ['mean_gain', 'actual_gain', 'excess']
            assert old_statement.value.func.attr == 'clamp_min' and ast.literal_eval(old_statement.value.args[0]) == 0
            assert new_statement.value.func.attr == 'relu'
            assert ast.dump(old_statement.value.func.value) == ast.dump(new_statement.value.args[0])
            differences.append(name)
    assert differences == ['mean_gain', 'actual_gain', 'excess']
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr in ['grad', 'backward', 'step', 'Adam', 'SGD'] for n in ast.walk(new_tree))
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    retained = module(old_path, 'retained_guard_scalar_only')
    corrected = module(new_path, 'boundary_guard_scalar_only')
    old = read(PARENT / 'protocol.json')
    refs = {r['id']:r for r in old['references']}
    norm = .022182490369457405
    rows = read(RETURN / 'outputs/loss_alignment_v1/run_results.json')['rows']
    checked = []
    for row in rows:
        begin, state = row['begin'], row['state']
        raw_path = RETURN / f'outputs/loss_alignment_v1/forward_replay/{state}_b{begin//5:03d}.npy'
        base_path = RETURN / f'outputs/loss_alignment_v1/forward_replay/initial_b{begin//5:03d}.npy'
        a, b = [torch.from_numpy(np.load(f, allow_pickle=False)).permute(0, 3, 1, 2).contiguous() for f in [raw_path, base_path]]
        targets, masks = [], []
        for case in old['cases'][begin:begin+5]:
            ref = refs[case['source_person_or_reference']]
            with Image.open(PARENT / ref['target']) as im:
                targets.append(np.asarray(im.convert('RGB')).copy())
            with Image.open(PARENT / ref['observed']) as im:
                masks.append(np.asarray(im.convert('L')) > 0)
        t = torch.from_numpy(np.stack(targets).astype(np.float32)/np.float32(255)).permute(0, 3, 1, 2).contiguous()
        m = torch.from_numpy(np.stack(masks).astype(np.float32))[:, None]
        before, old_terms = retained.brightness_guard(a, b, t, m, norm)
        after, new_terms = corrected.brightness_guard(a, b, t, m, norm)
        assert torch.equal(before, after) and old_terms == new_terms
        assert not after.requires_grad and after.grad_fn is None
        if state == 'initial':
            assert float(after) == 0
        checked.append(dict(state=state, begin=begin, penalty=float(after), exact_scalar_and_receipt_parity=True))
    # Exhaustive sign/boundary scalar-value equality independent of the face data.
    values = np.array([-1., -1e-12, -np.finfo(np.float64).tiny, -0., 0., np.finfo(np.float64).tiny, 1e-12, 1.])
    tensor = torch.from_numpy(values)
    assert torch.equal(tensor.clamp_min(0), torch.relu(tensor))
    # Declared analytical subgradient convention. This is not an autograd test.
    clamp_convention = (values >= 0).astype(int)
    relu_convention = (values > 0).astype(int)
    assert np.all(clamp_convention[values == 0] == 1) and np.all(relu_convention[values == 0] == 0)
    protected = read(ANALYSIS / 'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT / name) == expected for name, expected in protected.items())
    write(receipt, dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), checker_sha256=sha(__file__),
        retained_helper_sha256=sha(old_path), corrected_helper_sha256=sha(new_path),
        source_changes_limited_to_three_hinge_operators=True, scalar_rows_checked=40, scalar_case_compositions=200,
        exact_scalar_and_metadata_parity_all40=True, rows=checked,
        synthetic_boundary_scalar_values_checked=8, analytical_boundary_convention_not_actual_autograd=True,
        coefficient_fraction_masks_targets_normalization_and_scientific_gates_unchanged=True,
        no_model_imports_or_neural_calls=True, local_gradient_queries=0, local_optimizer_updates=0,
        actual_corrected_L4_gradient_behavior_unrun=True, model_qualified=False, goal_complete=False))
    print(dict(complete=True, scalar_rows_checked=40, exact_scalar_and_metadata_parity=True,
               local_gradient_queries=0, actual_corrected_L4_gradients_unrun=True))


if __name__ == '__main__':
    main()
