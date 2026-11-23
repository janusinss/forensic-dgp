"""Pure data/metric contracts for the unified CodeFormer R2 v3 completion training packet.
No Torch, model, VM, or filesystem side effects on import.
"""
from __future__ import annotations

VM_NAME = 'cctv_completion_codeformer_unified_r2_v3_vm'
STEM = 'cctv-completion-codeformer-unified-r2-v3'
FORMAT = 'completion-codeformer-unified-r2-v3'
OBJECTIVE_ID = (
    'unified_r2_v3_dual192fit_skip_casc3_ridge_routed_softkl_code_'
    'plus_moment_calibrated_v3r2_hybrid_area_damped_512_256_chroma_fusion'
)
ORIGINAL_SHA = 'b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797'
ORIGINAL_LOGICAL = '586a8854ed7b5e3f0e202c258addb7a87de63cd3068d153741b065379ff6668f'
ORIGINAL_BYTES = 370786611
PARENT_SHA = '0a6b4ad907c6ead86d8f26d3446666387493cdce81c6adc1ed78a8677d50fadd'
PARENT_BYTES = 701952165
PARENT_PROTOCOL_SHA = '8815ad168ab429bf4f30b8d719d3a6de6d1e2667fc38412c89c9c95e74e5ff3d'
PARENT_LOGICAL = '55feaa296f5164a6bd12995b487c9366c5fe7778f0e9dbe8912fc4ee7c4122d6'
TEACHER_INDEX_SHA = '0567a9ad54e6397390341a8dfc1ffa8ec50ceb9159da299605fee021ec678a69'
SUBSET_SHA = '0bb42287249693ab208fa1a609b263950cd3ebeec877480103d28f99b1f6c925'
BRANCH1_SUBSET_SHA = '8ec2d535cd6fb231af815d0ec833a1cd08f418458cbc4919c0ebd13f638f2a1d'
V4_PROTOCOL_SHA = '8bbb3dd611e95cfbce76e7e6426ffdadb31ce23dadb1dfdcd0246e501213069d'

KINDS = ('lower_face', 'eye_band', 'image_left_lens', 'cheek_object')
METRIC_KEYS = ('raw_hole_L1', 'raw_boundary_L1', 'PNG_hole_L1', 'PNG_boundary_L1')
GRAY_KEYS = ('GRAY_raw_mean_channel_spread', 'GRAY_PNG_mean_channel_spread')

TYPED_CAPS = {
    'parity': 8,
    'fit_code': 256,
    'fit_RGB': 128,
    'monitor': 128,
    'challenge': 64,
    'public_development': 56,
}

CONTRACT = {
    'task': 'Unified R2 v3 CodeFormer face mask completion co-adaptation from R1 terminal state',
    'continuation_from': 'cctv_completion_codeformer_finetune_v1_r1_vm/outputs/terminal_training_state.pth',
    'cumulative_paired_cycles_after_pilot': 64,
    'phase_paired_cycles': 32,
    'microbatch': 4,
    'selected_fit_cases': 128,
    'selected_fit_union_cases': 192,
    'selected_fit_clear_anchors': 64,
    'selected_fit_augmented_cases': 64,
    'retained_unused_fit_pool_cases': 256,
    'code_branch1_steps_newphase': 32,
    'code_branch2_steps_newphase': 32,
    'code_optimizer_steps_newphase': 64,
    'fusion_moment_calibration_steps': 64,
    'fusion_optimizer_steps_newphase': 32,
    'optimizer_steps_newphase': 96,
    'backward_calls_newphase': 384,
    'code_trainable_parameter_tensors': 278,
    'fusion_trainable_parameter_tensors': 54,
    'trainable_parameter_tensors': 332,
    'frozen_state_tensors_in_newphase': 165,
    'total_state_tensors': 497,
    'code_branch1_learning_rate_initial': 8.0e-6,
    'code_branch1_learning_rate_final': 2.0e-6,
    'code_learning_rate_initial': 4.0e-6,
    'code_learning_rate_final': 1.0e-6,
    'fusion_learning_rate_initial': 1.0e-5,
    'fusion_learning_rate_final': 3.0e-6,
    'learning_rate_schedule': (
        'decoupled_dual192fit_code_8e-6_to_2e-6_plus_4e-6_to_1e-6_casc3_ridge_softkl_'
        'moment_calibrated_fusion_1e-5_to_3e-6'
    ),
    'code_branch1_clear_anchor_role_weight': 1.25,
    'code_branch1_augmented_role_weight': 0.75,
    'code_clear_anchor_role_weight': 1.05,
    'code_augmented_role_weight': 0.95,
    'fusion_clear_anchor_role_weight': 1.20,
    'fusion_augmented_role_weight': 0.80,
    'code_branch1_trust_region_l2_coef': 1.0e-4,
    'code_trust_region_l2_coef': 5.0e-4,
    'fusion_trust_region_l2_coef': 8.0e-5,
    'code_branch1_latent_mse_weight': 1.00,
    'latent_mse_weight': 0.10,
    'skip_branch_weights': {'branch1_balanced128': 0.75, 'branch2_orthogonal128': 0.75},
    'spatial_token_weights_branch1': {
        'hole': 1.0,
        'ring_1cell': 0.50,
        'visible': 0.15,
    },
    'spatial_token_weights': {
        'inner_ring_1cell': 1.0,
        'outer_ring_1cell': 0.50,
        'hole_core': 0.10,
        'visible': 0.15,
    },
    'casc3_ridge_lambdas': {
        'encoder_block15_conv': 10.0,
        'encoder_block23_conv': 10.0,
        'feat_emb_linear': 3.0,
        'lower_face_endpoint_A_block23': 300.0,
    },
    'fusion_loss_weights': {
        'raw512_hole_L1': 1.00,
        'raw512_seam_L1': 1.00,
        'delivered256_hole_L1': 0.35,
        'delivered256_core_L1': 0.00,
        'delivered256_seam_L1': 0.20,
        'delivered256_grad_L1': 0.10,
        'chroma_RGB_weight': 0.10,
        'chroma_GRAY_weight': 0.25,
        'tiny_mask_area_tau256': 300.0,
    },
    'betas': [0.9, 0.99],
    'eps': 1e-8,
    'weight_decay': 0.0,
    'gradient_clip_L2': 1.0,
    'snapshot_cycles': [0, 8, 16, 32],
    'public_snapshot_cycles': [0, 32],
    'forward_Class_caps': TYPED_CAPS,
    'planned_network_forwards': 640,
    'hard_forward_cap': 668,
    'teacher_neural_calls': 0,
    'cached_teacher_sets': 80,
    'cached_fit_teacher_sets_used': 64,
    'cached_monitor_teacher_sets_diagnostic_only': 16,
    'worker_seconds': 1800,
    'cooperative_stop_seconds': 1680,
    'save_reserve_seconds': 120,
    'export_seconds': 900,
    'minimum_free_bytes': 8 * 1024 ** 3,
    'stop_free_bytes': 2 * 1024 ** 3,
    'output_bytes': 2560 * 1024 ** 2,
    'terminal_bytes': 1024 ** 3,
    'teacher_bytes': 24 * 1024 ** 2,
    'teacher_label_bytes': 24 * 1024 ** 2,
    'metadata_bytes': 8 * 1024 ** 2,
    'trainer_log_bytes': 4 * 1024 ** 2,
    'PNG_bytes_each': 256 * 1024,
    'PNG_cap_bytes': 256 * 1024,
    'CUDA_allocated_bytes_cap': 20 * 1024 ** 3,
    'export_raw_and_compressed_cap_bytes': 2816 * 1024 ** 2,
    'empty_bypasses': 168,
    'saved_eval_NPZ': 256,
    'saved_parity_NPZ': 4,
    'saved_fit_logit_NPZ': 128,
    'saved_generated_NPZ_max': 388,
    'retained_teacher_NPZ_copies': 80,
    'total_NPZ_cap': 468,
    'regular_member_cap_including_manifest': 1000,
    'relative_path_UTF8_cap': 240,
    'objective_id': OBJECTIVE_ID,
}


def need(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def skip_encoder_name(name: str) -> bool:
    return name.startswith('encoder.blocks.') and int(name.split('.')[2]) <= 14


def bottleneck_encoder_name(name: str) -> bool:
    return name.startswith('encoder.blocks.') and int(name.split('.')[2]) >= 15


def predictor_route_name(name: str) -> bool:
    return name == 'position_emb' or name.startswith(('feat_emb.', 'ft_layers.', 'idx_pred_layer.'))


def code_route_name(name: str) -> bool:
    return (
        name.startswith('encoder.')
        or name == 'position_emb'
        or name.startswith(('feat_emb.', 'ft_layers.', 'idx_pred_layer.'))
    )


def fusion_route_name(name: str) -> bool:
    return name.startswith('fuse_convs_dict.')


def trainable_name(name: str) -> bool:
    return code_route_name(name) or fusion_route_name(name)


def frozen_prior_name(name: str) -> bool:
    return name.startswith(('quantize.', 'generator.'))


def lr_for_cycle(
    cycle: int,
    total_cycles: int = 32,
    lr_max: float = 1.0e-5,
    lr_min: float = 3.0e-6,
) -> float:
    import math
    need(1 <= cycle <= total_cycles, f'Invalid cycle {cycle}')
    if total_cycles <= 1:
        return float(lr_max)
    progress = (cycle - 1) / float(total_cycles - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(lr_min + (lr_max - lr_min) * cosine)


def code_branch1_lr_for_cycle(cycle: int, total_cycles: int = 32) -> float:
    return lr_for_cycle(
        cycle,
        total_cycles=total_cycles,
        lr_max=CONTRACT['code_branch1_learning_rate_initial'],
        lr_min=CONTRACT['code_branch1_learning_rate_final'],
    )


def code_lr_for_cycle(cycle: int, total_cycles: int = 32) -> float:
    return lr_for_cycle(
        cycle,
        total_cycles=total_cycles,
        lr_max=CONTRACT['code_learning_rate_initial'],
        lr_min=CONTRACT['code_learning_rate_final'],
    )


def fusion_lr_for_cycle(cycle: int, total_cycles: int = 32) -> float:
    return lr_for_cycle(
        cycle,
        total_cycles=total_cycles,
        lr_max=CONTRACT['fusion_learning_rate_initial'],
        lr_min=CONTRACT['fusion_learning_rate_final'],
    )


def aggregate(rows, *, condition: bool = False):
    need(bool(rows), 'Empty metric rows')

    def mean(part):
        return {k: sum(r[k] for r in part) / len(part) for k in METRIC_KEYS}

    out = {
        'macro': mean(rows),
        'geometry': {
            kind: mean([r for r in rows if r['mask_kind'] == kind]) for kind in KINDS
        },
    }
    if condition:
        out['condition'] = {
            name: mean([r for r in rows if r['condition'] == name])
            for name in sorted({r['condition'] for r in rows})
        }
    return out


def compare(summary, baseline, *, conditions: bool = False):
    now = summary['metrics']
    old = baseline['metrics']
    groups = ['macro', 'geometry'] + (['condition'] if conditions else [])
    pairs = [(now['macro'], old['macro'])]
    for group in groups[1:]:
        need(set(now[group]) == set(old[group]), 'Compared subgroup membership differs')
        pairs.extend((now[group][key], old[group][key]) for key in now[group])
    nonworse = all(x[k] <= y[k] + 1e-12 for x, y in pairs for k in METRIC_KEYS)
    catastrophic = any(x[k] > y[k] * 1.10 for x, y in pairs for k in METRIC_KEYS)
    gain = (old['macro']['PNG_hole_L1'] - now['macro']['PNG_hole_L1']) / max(
        old['macro']['PNG_hole_L1'], 1e-12
    )
    return {
        'prospective_numeric_screen_pass': bool(gain >= 0.01 and nonworse),
        'PNG_macro_hole_gain': gain,
        'all_raw_PNG_hole_boundary_subgroups_nonworse': bool(nonworse),
        'catastrophic_regression_stop': bool(catastrophic),
        'visual_review_pending': True,
        'model_qualified': False,
    }


def gray_style(summary):
    rows = [
        r
        for r in summary['records']
        if not r.get('empty') and r.get('style') == 'GRAY'
    ]
    need(len(rows) == 8, 'Expected 8 strict-gray challenge records')
    for kind in KINDS:
        need(
            sum(r['mask_kind'] == kind for r in rows) == 2,
            'Expected 2 strict-gray records for ' + kind,
        )
    for row in rows:
        for k in GRAY_KEYS:
            need(
                k in row and isinstance(row[k], (int, float)) and row[k] >= 0,
                'Missing/invalid ' + k,
            )
    return {
        'cases': len(rows),
        'case_ids': [r['id'] for r in rows],
        'macro': {k: sum(r[k] for r in rows) / len(rows) for k in GRAY_KEYS},
        'geometry': {
            kind: {
                k: sum(r[k] for r in rows if r['mask_kind'] == kind)
                / sum(r['mask_kind'] == kind for r in rows)
                for k in GRAY_KEYS
            }
            for kind in KINDS
        },
        'records': {
            r['id']: {'mask_kind': r['mask_kind'], **{k: r[k] for k in GRAY_KEYS}}
            for r in rows
        },
    }


def compare_gray(summary, baseline):
    now = summary.get('GRAY_style') or gray_style(summary)
    old = baseline.get('GRAY_style') or gray_style(baseline)
    need(
        now['case_ids'] == old['case_ids'] and len(now['case_ids']) == 8,
        'Strict-gray case ordering/count differs',
    )
    failures = []
    for k in GRAY_KEYS:
        if now['macro'][k] > old['macro'][k] + 1e-12:
            failures.append(
                {
                    'group': 'macro',
                    'metric': k,
                    'current': now['macro'][k],
                    'baseline_R1': old['macro'][k],
                }
            )
    for kind in KINDS:
        for k in GRAY_KEYS:
            if now['geometry'][kind][k] > old['geometry'][kind][k] + 1e-12:
                failures.append(
                    {
                        'group': 'geometry/' + kind,
                        'metric': k,
                        'current': now['geometry'][kind][k],
                        'baseline_R1': old['geometry'][kind][k],
                    }
                )
    case_deltas = {
        cid: {
            k: now['records'][cid][k] - old['records'][cid][k]
            for k in GRAY_KEYS
        }
        for cid in now['case_ids']
    }
    return {
        'cases': 8,
        'all_GRAY_macro_geometry_chroma_nonworse': not failures,
        'nonworse_failures': failures,
        'macro_current': now['macro'],
        'macro_baseline_R1': old['macro'],
        'geometry_current': now['geometry'],
        'geometry_baseline_R1': old['geometry'],
        'case_deltas_vs_R1': case_deltas,
        'tolerance': 1e-12,
        'selection_gate_scope': 'Final cycle 32 strict-gray false-color non-worsening gate',
        'model_qualified': False,
    }
