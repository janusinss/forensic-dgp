"""Data-only exact cross-composition audit; zero neural or gradient calls."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import torch

from cctv_dgp_bank_partition_v1 import validate_states, compose, VARIANTS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs'
PARENT = OUT/'cctv_dgp_bank_photometric_guard_v1_return'
ANALYSIS = OUT/'cctv_dgp_bank_path_ablation_v1_analysis'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    receipt = ANALYSIS/'partition_state_audit.json'
    assert not receipt.exists()
    assert read(ANALYSIS/'independent_audit.json')['complete']
    old = read(PARENT/'protocol.json')
    init_path = PARENT/'initializers/initial_B.pth'
    trained_path = PARENT/'outputs/bank_comparison_v1/state_B50/full_state.pth'
    assert sha(init_path) == old['assets_sha256']['initializers/initial_B.pth']
    manifest = read(PARENT/'export_manifest.json')['files']
    assert sha(trained_path) == manifest[trained_path.relative_to(PARENT).as_posix()]
    torch.set_num_threads(4); torch.set_grad_enabled(False)
    initial = torch.load(init_path, map_location='cpu', weights_only=True)
    stored = torch.load(trained_path, map_location='cpu', weights_only=True)
    assert stored['updates'] == 50 and not stored['manual_failed_resume_allowed']
    trained = stored['model']
    # Source-declared names, without constructing or forwarding any module.
    source = PARENT/'cctv_dgp_bank_comparison_v1_model.py'
    assert sha(source) == old['assets_sha256'][source.name]
    import ast
    tree = ast.parse(source.read_text(encoding='utf-8'))
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'CURRENT_NAMES' for t in n.targets))
    current = eval(compile(ast.Expression(assignment.value), str(source), 'eval'), {'__builtins__': {}}, {})
    names = ['current.'+n for n in current] + [n for n in initial
        if n.startswith(('style_conditioner.', 'spatial_conditioners.', 'fusions.'))]
    partitions = validate_states(initial, trained, names)
    sys.path.insert(0, str(PARENT))
    from cctv_dgp_bank_pinned_initializer_v2 import state_sha
    rows = {}
    for variant in VARIANTS:
        state = compose(initial, trained, partitions, variant)
        expected_changed = set()
        if variant in ('joint', 'decoder_only'): expected_changed |= partitions['decoder']
        if variant in ('joint', 'bank_only'): expected_changed |= partitions['bank']
        assert {n for n in state if not torch.equal(state[n], initial[n])} == expected_changed
        rows[variant] = dict(model_state=state_sha(state), changed_from_initial=sorted(expected_changed),
                             changed_tensors=len(expected_changed), frozen_tensors_exact=755)
    assert rows['joint']['model_state'] == read(PARENT/'outputs/bank_comparison_v1/state_B50/receipt.json')['model_state']
    protected = read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n) == expected for n, expected in protected.items())
    data = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), checker_sha256=sha(__file__),
        helper_sha256=sha(ROOT/'scripts/cctv_dgp_bank_partition_v1.py'),
        initial_file_sha256=sha(init_path), trained_file_sha256=sha(trained_path), variants=rows,
        partition_names={k:sorted(v) for k,v in partitions.items()},
        local_neural_calls=0, local_gradient_queries=0, local_optimizer_updates=0,
        counterfactual_not_independently_trained_arms=True, model_qualified=False)
    with receipt.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False)+'\n')
    print({k:r['changed_tensors'] for k,r in rows.items()})


if __name__ == '__main__': main()
