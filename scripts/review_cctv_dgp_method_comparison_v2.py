"""Record the method decision and static inventories without training/gradients.

The selected bank's mechanics are verified. A conditioned restoration candidate,
matched executable training protocol and delivered quality remain pending.
"""
import argparse
import ast
import json
from pathlib import Path
import time

import torch


def run(root):
    workspace = Path(__file__).resolve().parents[1]
    import sys
    sys.path.insert(0, str(workspace))
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from dgp_face_restoration import sha
    start = time.monotonic()
    root = root.resolve()
    assert root.is_relative_to(workspace / 'outputs') and not root.exists()
    root.mkdir(parents=True)
    torch.set_num_threads(4)
    checkpoint = workspace / 'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth'
    expected = '646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'
    model, provenance = load_frozen_dgp_restorer(checkpoint, expected_sha256=expected)
    components = {name: {'unique_parameter_tensors': len(list(module.parameters())),
        'unique_parameter_elements': sum(p.numel() for p in module.parameters())}
        for name, module in model.net.named_children()}
    assert len(list(model.net.parameters())) == 181
    assert sum(p.numel() for p in model.net.parameters()) == 3312707
    assert len(model.net.state_dict()) == 622
    assert sum(v['unique_parameter_elements'] for v in components.values()) == 3312707
    prior_root = workspace / 'outputs/cctv_dgp_generative_bank_source_v1'
    probe_root = workspace / 'outputs/cctv_dgp_generative_bank_probe_v1'
    acquisition = json.loads((prior_root / 'acquisition.json').read_text())
    probe = json.loads((probe_root / 'results.json').read_text())
    audit = json.loads((workspace / 'outputs/cctv_dgp_generative_bank_probe_v1_independent_audit.json').read_text())
    assert audit['complete'] and audit['results_sha256'] == sha(probe_root / 'results.json')
    assert audit['checker_sha256'] == sha(workspace / 'scripts/audit_cctv_dgp_generative_bank_v1.py')
    assert audit['replay_max_error'] == 0 and audit['local_gradient_API_refused']
    forward_tree = ast.parse((workspace / 'models/dgp_synthesizer.py').read_text())
    cls = next(n for n in forward_tree.body if isinstance(n, ast.ClassDef) and n.name == 'DGPSynthesizer')
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    forward_calls = sorted({ast.unparse(n.func) for n in ast.walk(forward) if isinstance(n, ast.Call)})
    assert 'self.fpn' in forward_calls and 'self.final' in forward_calls
    assert not any('prior' in n.lower() or 'codebook' in n.lower() or 'generator' in n.lower() for n in forward_calls)
    anchor_root = workspace / 'outputs/cctv_dgp_multiscale_active_anchor_v1'
    anchor = json.loads((anchor_root / 'plan.json').read_text())
    anchor_audit = json.loads((anchor_root / 'independent_audit.json').read_text())
    assert anchor_audit['complete'] and anchor_audit['plan_sha256'] == sha(anchor_root / 'plan.json')
    scientific = json.loads((workspace / 'outputs/cctv_dgp_multiscale_calibration_vm_v1/protocol.json').read_text())
    decision = {
        'format': 'DGP-method-comparison-review-v2',
        'status': 'method_selected_and_prior_mechanics_verified_not_training_ready',
        'user_method_answer': 'apply the best approach here',
        'current_model': {'checkpoint': checkpoint.relative_to(workspace).as_posix(),
            'checkpoint_sha256': expected, 'unique_parameter_tensors': 181,
            'unique_parameter_elements': 3312707, 'state_entries_with_aliases_and_buffers': 622,
            'components': components, 'forward_calls': forward_calls,
            'method': 'fine-tuned DeblurGAN-v2-compatible MobileNet/FPN conditional restorer',
            'separate_generative_face_prior_in_executed_forward': False,
            'compatibility_class_name_retained': 'DGPSynthesizer',
            'starting_weights_retained': True, 'selected_extra_epochs': 2,
            'selected_extra_updates': 226, 'lifetime_epochs_confirmed': False,
            'weights_only_not_exact_optimizer_resume': True, 'loader_provenance': provenance},
        'selected_interpretation': 'own-trained input-conditioned hybrid generative-prior restoration',
        'comparison': {
            'A': {'method': 'corrected current-DGP copy with active clear/blur RGB reconstruction anchors',
                'repaired_deep_branch_retained': True,
                'reviewed_trainable_partition': 'decoder15;609219elements',
                'partition_is_split_repaired_graph_not_unmodified14tensor_decoder': True,
                'hypothesis': 'Always-active target reconstruction reduces the observed clear/blur appearance drift',
                'fixed_reconstruction_weights': [.2, 1, 1, 1],
                'extra_anchor_weights': [1, 1],
                'extra_anchor_normalizers': [anchor['baseline_clear_RGB_MSE'], anchor['baseline_blur_RGB_MSE']],
                'linear_contrasts_are_not_finite_validation': True},
            'B': {'method': 'retained observation features plus continuous frozen GAN bank and our trained reconstruction',
                'generator_origin': 'official standalone FFHQ StyleGAN2 release; external frozen params_ema',
                'generator_checkpoint_sha256': acquisition['bindings']['StyleGAN2_512_Cmul1_FFHQ_B12G4_scratch_800k.pth']['sha256'],
                'generator_trainable_elements': 0,
                'new_components_to_train': ['input-to-style conditioning', 'spatial feature conditioning/fusion'],
                'retained_components': ['current MobileNet/FPN observations', 'compatible current reconstruction weights'],
                'generator_feature_sizes_verified': [16, 32, 64, 128, 256],
                'final_restoration_size': 256, 'prior_native_size': 512,
                'pretrained_restoration_encoder_used': False,
                'pretrained_GLEAN_or_GFPGAN_restoration_checkpoint_used': False,
                'VQ_code_classification_used': False, 'V20_RGB_subtraction_used': False,
                'degraded_camera_conditioning_implemented_and_verified': False,
                'initial_conditioner_behavior_and_gradients_need_verification': True,
                'actual_GAN_contribution_requires_ablation_on_saved_outputs': True},
            'same_prepared_inputs_and_roles_required': True,
            'same_exposures_and_acceptance_required': True,
            'shared_active_reconstruction_objective_required_where_defined': True,
            'architecture_plus_objective_changes_not_unique_causal_proof': True,
            'fresh_GAN_from_random_weights_selected': False,
            'no_global_best_or_success_promise': True},
        'immutable_thresholds': scientific['scientific_thresholds'],
        'next_executable_prerequisite': {
            'action': 'Implement the input-conditioned bank/reconstruction copy and matched return checker',
            'checks': ['initial raw/PNG behavior against retained DGP',
                'strict copied/frozen/trainable source and tensor bindings',
                'finite manual L4 gradient/VRAM/runtime/storage preflight before optimizer',
                'identical TRAIN/DEV roles, exposures and original preservation/visual gates',
                'prior-enabled/disabled ablation to verify its contribution'],
            'training_packet_ready': False, 'manual_command_ready': False,
            'no_automatic_historical_pilot': True},
        'evidence_limits': {'component_inference_is_not_CCTV_restoration': True,
            'new_CCTV_inputs_read': 0, 'new_final_identity_pixels_read': 0,
            'known_FFHQ_corpus_source_overlap_with_own_FFHQ_TRAIN': True,
            'prior_per_identity_overlap_unexcluded': True,
            'FFHQ_prior_is_not_an_Asian_capture_source': True,
            'no_Zamboanga_or_ethnicity_inference': True,
            'completion_seven_families_separately_unqualified': True,
            'independent_final_review_pending': True},
        'current_DGP_forward_calls': 0, 'current_optimizer_updates': 0,
        'current_backwards': 0, 'current_gradient_queries': 0,
        'app_changed': False, 'model_promoted': False, 'goal_complete': False,
        'seconds': time.monotonic() - start}
    bound = [checkpoint, workspace / 'models/dgp_synthesizer.py', workspace / 'models/fpn_mobilenet.py',
        workspace / 'dgp_frozen_inference_v2.py', workspace / 'cctv_dgp_frozen_norm.py',
        workspace / 'scripts/cctv_dgp_multiscale_calibration_model_v1.py',
        workspace / 'CCTV_DGP_LATENT_DELTA_FEASIBILITY_V20.md',
        workspace / 'CCTV_DGP_BROADER_CODES_V16_R2_RESULTS.md',
        workspace / 'CCTV_DGP_MULTISCALE_CALIBRATION_V1_RESULTS.md',
        workspace / 'CCTV_DGP_POST_MULTISCALE_TRAINING_REVIEW.md',
        anchor_root / 'plan.json', anchor_root / 'independent_audit.json',
        prior_root / 'acquisition.json', probe_root / 'plan.json', probe_root / 'results.json',
        workspace / 'outputs/cctv_dgp_generative_bank_probe_v1_independent_audit.json',
        workspace / 'outputs/cctv_dgp_multiscale_calibration_vm_v1/protocol.json']
    decision['source_bindings'] = {p.relative_to(workspace).as_posix(): sha(p) for p in bound}
    with (root / 'selection.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(decision, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'complete': True, 'method_review': str(root / 'selection.json'),
        'training_packet_ready': False, 'current_model_unchanged': True,
        'current_neural_calls': 0, 'optimizer_updates': 0}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
