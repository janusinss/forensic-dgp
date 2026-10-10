"""Freeze one w=0 architecture ablation after the answered completion discussion."""
from collections import Counter
from datetime import datetime, timezone
import copy
import time
from completion_feature_fusion_off_v1_common import ROOT, BASE, OUT, sha, read, write, rgb, binary, verify_bindings


def main():
    started = time.monotonic(); assert not OUT.exists(), 'Retain each previous diagnostic; no automatic repeat'
    parent, result = read(BASE/'protocol.json'), read(BASE/'results.json')
    audit = read(BASE/'independent_saved_output_audit.json'); review = read(BASE/'visual_review.json')
    assert result['complete'] and audit['complete'] and review['all32_outputs_and_same_final_baselines_actually_viewed']
    assert result['state_before'] == result['state_after'] == parent['parent_model_state']
    verify_bindings(parent)
    bindings = dict(parent['sources_sha256']); cases = copy.deepcopy(parent['cases'])
    for name, digest in result['artifacts_sha256'].items():
        assert sha(BASE/name) == digest
        bindings[(BASE/name).relative_to(ROOT).as_posix()] = digest
    for name in ['protocol.json', 'results.json', 'external_receipt.json', 'independent_input_audit.json', 'input_visual_review.json',
                 'independent_saved_output_audit.json', 'visual_review.json']:
        bindings[(BASE/name).relative_to(ROOT).as_posix()] = sha(BASE/name)
    for c in cases:
        if c['rejected']: continue
        for key, folder, suffix in [('baseline_output', 'images', '.png'), ('baseline_stages', 'stages', '.npz'), ('baseline_metadata', 'metadata', '.json')]:
            c[key] = (BASE/folder/(c['id']+suffix)).relative_to(ROOT).as_posix()
        source = rgb(ROOT/c['input']); final = binary(ROOT/c['masks']['removal']); union = binary(ROOT/c['conditioning'])
        assert not (final & ~union).any() and not (final & binary(ROOT/c['masks']['protected'])).any()
        assert float(union.mean()) < .85
        if not final.any(): assert not union.any()
    for name in ['scripts/completion_feature_fusion_off_v1_common.py', 'scripts/prepare_completion_feature_fusion_off_v1.py',
                 'scripts/run_completion_feature_fusion_off_v1.py', 'scripts/supervise_completion_feature_fusion_off_v1.py',
                 'scripts/audit_completion_feature_fusion_off_v1.py', 'scripts/audit_completion_feature_fusion_off_v1_protocol.py',
                 'CCTV_DGP_COMPLETION_FEATURE_FUSION_V1_REVIEW.md', 'pretrained_completion.py', 'face_color_policy.py',
                 'third_party/codeformer/codeformer_arch.py', 'third_party/codeformer/vqgan_arch.py', 'checkpoints/codeformer_inpainting.pth']:
        bindings[name] = sha(ROOT/name)
    OUT.mkdir()
    write(OUT/'user_architecture_decision.json', {'complete': True, 'human_answer': 'apply the best approach',
          'question': 'Mask footprints, extra context and edge blending still leave central completion defects. The assumption that processing alone would solve them has failed. For the completion architecture review, which diagnostic should come next?',
          'selected_diagnostic': 'One frozen encoder-feature-fusion-disabled comparison; no mask or app modification',
          'authorization_scope': 'Local frozen inference and audits; actual training remains manual on the existing VM', 'UTC': datetime.now(timezone.utc).isoformat()})
    write(OUT/'input_review.json', {'complete': True, 'reviewer': 'Implementing assistant input-only development review; not independent final',
          'all9_existing_input_pages_actually_viewed_this_turn': True, 'input_pages': parent['input_pages'],
          'cases': 36, 'eligible': 32, 'empty_bypasses': 4, 'unchanged_input_only_exclusions': 4,
          'mask_changes': False, 'new_eligibility_labels': False, 'all132_input_cells': True,
          'observations': ['Lower-face mask/scarf/hand regions and obstructed eye regions retain visible face context.',
                           'Transparent glasses and uncovered controls retain empty removal areas.',
                           'The prior oblique masked face remains out of scope and nearly hidden face still needs less covering.',
                           'Union context hides some protected frames/hair only from the network; final source copying must preserve them.',
                           'Peripheral source objects and mixed source/estimate glare remain limits of this fixed footprint.'],
          'assistance_reused_for_synthetic_pairs': True, 'native_CCTV': False})
    p = {'format': 'completion-fixed-mask-encoder-feature-fusion-ablation-v1', 'UTC': datetime.now(timezone.utc).isoformat(),
         'hypothesis': 'Removing direct encoder-feature fusion can change regenerated covering patterns or central anatomy. The codebook prediction still uses observed context; performance may worsen.',
         'causal_limit': 'Cannot repair copied fragments outside final support; any change in estimated appearance remains a plausible unknown hidden estimate, not recovered identity.',
         'cases': cases, 'family_case_counts': dict(Counter(c['family'] for c in cases if not c['rejected'])),
         'comparison': 'Cached union-conditioned w=1 versus exactly one frozen w=0 estimate; same inputs/conditioning/final mask/seed/display',
         'intervention': {'parameter': 'network w', 'baseline': 1, 'candidate': 0, 'adain': False, 'default_inpainting_policy': 'Official w=1; w=0 is an exploratory nondefault setting'},
         'parity_case': parent['parity_case'], 'parity': 'One fresh w=1 union call must match cached input/raw512/completion256/PNG exactly before trials. Its first matched w=0 call must have exact input, encoder latent, logits, selected indices and quantized latent.',
         'sources_sha256': bindings, 'app_preservation_sha256': parent['app_preservation_sha256'],
         'parent_protocol_sha256': sha(BASE/'protocol.json'), 'parent_results_sha256': sha(BASE/'results.json'),
         'parent_model_state': parent['parent_model_state'], 'weights_sha256': 'b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797',
         'seed': parent['seed'], 'expected_trial_forwards': 28, 'max_forwards': {'completion': 29, 'internal512': 29, 'codebook': 29, 'DGP': 0, 'detector': 0},
         'expected_fusion_calls': {'32': 1, '64': 1, '128': 1}, 'expected_candidate_fusion_calls': {'32': 0, '64': 0, '128': 0},
         'requests': 37, 'expected_eligible': 32, 'expected_rejections': 4, 'expected_empty_bypasses': 4,
         'cap_seconds': 600, 'external_timeout_seconds': 630, 'artifact_cap_bytes': 268435456,
         'input_review_sha256': sha(OUT/'input_review.json'), 'user_decision_sha256': sha(OUT/'user_architecture_decision.json'),
         'conditioning_rule': parent['conditioning_rule'], 'delivered_rule': parent['delivered_rule'], 'quality_criteria': parent['quality_criteria'],
         'saved_stages': ['neural_input512', 'internal512', 'completion', 'logits', 'lq_feat', 'code_indices', 'quantized'],
         'display': parent['display'], 'device': 'cpu', 'torch': '2.13.0+cpu', 'restoration': 'off',
         'exposed_development_photographs': True, 'unknown_pretraining_overlap': True,
         'original_photo_mask_assistance_reused_for_synthetic_pairs': True, 'automatic_generated_outputs': 0,
         'hidden_ground_truth': None, 'hidden_metrics': None, 'native_or_reserved_used': False,
         'gradient_calls': 0, 'backward_calls': 0, 'optimizer_updates': 0, 'new_checkpoint': False,
         'automatic_quality_qualification': False, 'assisted_quality_qualification': False, 'app_adoption': False,
         'independent_final_review': False, 'goal_complete': False, 'seconds': time.monotonic()-started}
    write(OUT/'protocol.json', p)
    print({'prepared': True, 'cases': 36, 'protocol_sha256': sha(OUT/'protocol.json'), 'model_forwards': 0}, flush=True)


if __name__ == '__main__': main()
