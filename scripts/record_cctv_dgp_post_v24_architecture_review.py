"""Saved evidence and source-only architecture review; no model execution or fixes."""
import ast
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import audit_cctv_dgp_degraded_detail_v24 as a


def main():
    started = time.monotonic()
    out = ROOT / 'outputs/cctv_dgp_post_v24_architecture_review_v1'
    out.mkdir()
    bindings = {}
    def bind(path):
        bindings[path.relative_to(ROOT).as_posix()] = a.sha(path)
    pilots = ['cctv_dgp_detail_prior_v22_r1', 'cctv_dgp_detail_skip_v23', 'cctv_dgp_degraded_detail_v24']
    table = []
    for name in pilots:
        audit_path = ROOT / 'outputs' / (name + '_independent_audit.json')
        trace_path = ROOT / 'outputs' / (name + '_diagnostic') / 'results.json'
        review_path = trace_path.with_name('visual_review.json')
        audit, trace, review = a.read(audit_path), a.read(trace_path), a.read(review_path)
        assert audit['complete'] and audit['complete_snapshot_updates'] == [0, 50]
        assert not audit['early_structure_stop']['pass'] and not audit['necessary_capacity_pass']
        assert review['complete']
        for path in (audit_path, trace_path, review_path):
            bind(path)
        degraded = trace['groups']['degraded']
        table.append({'pilot': name, 'delivered_feature_gain_percent': 100 * audit['early_structure_stop']['relative_feature_error_gain'],
            'required_early_percent': 1, 'stopped_update': 50,
            'degraded_median_observed_correction_RMS': degraded['median_saved_correction_RMS'],
            'degraded_median_observed_correction_RMS_in_byte_units': 255 * degraded['median_saved_correction_RMS'],
            'degraded_changed_PNG_pixels': degraded['changed_PNG_pixels'],
            'degraded_maximum_PNG_byte_change': degraded['maximum_PNG_byte_change'],
            'final800_executed': False})
    v24 = a.read(ROOT / 'outputs/cctv_dgp_degraded_detail_v24_diagnostic/results.json')
    loss_path = ROOT / 'outputs/cctv_dgp_degraded_detail_v24_loss_audit_v1/results.json'
    loss_check_path = loss_path.with_name('independent_saved_loss_audit.json')
    loss, loss_check = a.read(loss_path), a.read(loss_check_path)
    assert loss_check['complete'] and loss_check['results_sha256'] == a.sha(loss_path)
    assert loss['groups']['clear']['states']['0']['objective'] == 0
    assert all(row['states']['0']['objective'] == 0 for row in loss['rows'] if row['profile'] == 'clear')
    bind(loss_path); bind(loss_check_path)
    chosen = [row for row in v24['rows'] if row['profile'] != 'clear']
    ratios = [row['stages']['band_after_projection']['RMS'] / row['stages']['q_before_projection']['spatial_RMS'] for row in chosen]
    p = a.read(a.BUNDLE / 'protocol.json')
    bind(a.BUNDLE / 'protocol.json')
    v23_source = ROOT / 'outputs/cctv_dgp_detail_skip_vm_v23/scripts/cctv_dgp_detail_skip_v23.py'
    v24_source = a.BUNDLE / 'scripts/cctv_dgp_degraded_detail_v24.py'
    def head_tree(path):
        module = ast.parse(path.read_text(encoding='utf-8'))
        prepare = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'prepare')
        return ast.dump(next(node for node in prepare.body if isinstance(node, ast.ClassDef) and node.name == 'DetailHead'), include_attributes=False)
    assert head_tree(v23_source) == head_tree(v24_source)
    for path in (v23_source, v24_source, a.BUNDLE / 'cctv_dgp_degraded_objective_v24.py',
                 ROOT / 'models/dgp_synthesizer.py', ROOT / 'models/fpn_mobilenet.py',
                 ROOT / 'CCTV_DGP_STRUCTURE_TRACE_V21.md', ROOT / 'CCTV_DGP_STRUCTURE_V18_RESULTS.md',
                 ROOT / 'CCTV_DGP_INPUT_SELECTION_V19_R2_RESULTS.md', ROOT / 'CCTV_DGP_ARCHITECTURE_REVIEW.md',
                 ROOT / '.codex/skills/systematic-debugging/SKILL.md'):
        bind(path)
    positions = list(range(-6, 7))
    weights = [math.exp(-.5 * (position / 2) ** 2) for position in positions]
    total = sum(weights); kernel = [value / total for value in weights]
    gains = {str(period): 1 - sum(weight * math.cos(2 * math.pi * position / period) for position, weight in zip(positions, kernel))
             for period in [128, 64, 32, 16, 8]}
    result = {'complete': True, 'date': '2026-10-06', 'scope': 'Architecture discussion after three retained failures; saved evidence and pinned source only',
        'runner_sha256': a.sha(Path(__file__)), 'source_bindings_sha256': bindings,
        'three_pilots': table, 'V23_V24_head_AST_exact': True,
        'V24_changed_learned_tensor_count': sum(value['changed_values'] > 0 for name, value in v24['parameter_changes'].items() if name not in ['kernel', 'reflect_indices']),
        'V24_degraded_raw_feature_gain_percent': 100 * (1 - v24['groups']['degraded']['raw_update50_feature_MSE'] / v24['groups']['degraded']['raw_baseline_feature_MSE']),
        'V24_median_interior_band_RMS_over_preprojection_spatial_RMS': statistics.median(ratios),
        'ratio_limit': 'Same eroded-support fields within each saved-layer trace. Describes the stopped correction, not a causal gradient experiment or capacity bound.',
        'declared_Gaussian_1D_highpass_sinusoid_gain': gains,
        'filter_limit': 'Float64 normalized declared13-tap sigma2 filter, full support and interior/periodic sinusoid assumption. Actual head uses float32, normalized masked blur, mean removal and clamp; these values are explanatory, not measured restoration scores.',
        'corrected_loss': {'clear_reward_zero_verified': True,
            'clear_weighted_contribution_to_net_reduction': loss_check['clear_contribution'],
            'degraded_weighted_contribution_to_net_reduction': loss_check['degraded_contribution'],
            'net_objective_reduction': loss_check['overall_reduction'],
            'GPU_gradient_causality_proven': False},
        'current_early_feature_mask': 'Union of24x24 patches at two eyes, nose and two mouth corners intersected with6-pixel-eroded support; no dedicated face-outline score',
        'all_facial_features_required': ['eyes', 'nose', 'mouth', 'face outline', 'overall visible appearance'],
        'recommendation': 'Review a distinct own-DGP spatial/feature reconstruction path with full-resolution observation and multiscale context; keep appearance safeguards as measured constraints rather than stripping all broad correction through the failed final high-pass. Preserve original DGP and separate initial parity, capacity, generalization and acceptance.',
        'alternative': 'Explicitly declared frozen face prior plus our trained conditioning/structure path, previously conditionally authorized. Historical V10-V19 appearance/generalization failures still apply; this cannot be a pretrained-main relabel.',
        'architecture_choice_pending': True, 'fourth_recipe_prepared': False,
        'invalid_assumptions': ['A nonzero gradient or changed head means a useful spatial correction.', 'Removing the clear-target reward alone resolves visible whole-face reconstruction.', 'Tiny raw-to-PNG changes imply the only problem is export quantization.'],
        'limitations': 'Three update50 stops reject these finite recipes, not every possible head or architecture. Ten exposed photographs do not establish independent-identity, native CCTV or Zamboanga performance. No target-informed output, ablation, new candidate or model fix is produced.',
        'literature_checked': [
            {'url': 'https://xingangpan.github.io/projects/DGP.html', 'source': 'Author project page', 'fact': 'Published DGP uses a pretrained GAN and progressive image-specific generator adaptation; this retained FPN residual is a custom implementation.'},
            {'url': 'https://arxiv.org/abs/2111.09881', 'source': 'Authors paper v2,11March2022', 'fact': 'Restormer investigates efficient long-range restoration context.'},
            {'url': 'https://github.com/swz30/Restormer/blob/main/basicsr/models/archs/restormer_arch.py', 'source': 'Author implementation, read6October2026', 'fact': 'Four resolution levels, decoder feature concatenations and RGB residual output support a spatial/context design discussion; no weights or code are imported.'}],
        'seconds': time.monotonic() - started, 'neural_calls': 0, 'head_fixes': 0,
        'backward_calls': 0, 'optimizer_updates': 0, 'VM_actions': False,
        'native_or_reserved_accessed': False, 'app_promotion': False, 'goal_complete': False}
    a.write(out / 'review.json', result)
    print(json.dumps({'complete': True, 'three_pilots': table,
        'V24_projection_ratio': result['V24_median_interior_band_RMS_over_preprojection_spatial_RMS'],
        'fourth_recipe_prepared': False, 'neural_calls': 0}, indent=2))


if __name__ == '__main__':
    main()
