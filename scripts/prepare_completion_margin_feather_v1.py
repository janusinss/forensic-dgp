"""Freeze a saved-output margin ablation without models, new masks or training."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'outputs/completion_conditioning_union_v1'
OUT = ROOT / 'outputs/completion_margin_feather_v1'


def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    assert not OUT.exists()
    p, result, audit, visual = [read(OLD / name) for name in ['protocol.json', 'results.json', 'independent_saved_output_audit.json', 'visual_review.json']]
    assert result['complete'] and audit['complete'] and visual['all32_outputs_and_same_final_baselines_actually_viewed']
    assert not visual['app_adoption'] and result['state_before'] == result['state_after']
    bindings = dict(p['sources_sha256'])
    for name, digest in result['artifacts_sha256'].items(): bindings[(OLD / name).relative_to(ROOT).as_posix()] = digest
    for name in ['protocol.json', 'results.json', 'independent_saved_output_audit.json', 'visual_review.json', 'pixel_support_diagnostic.json',
                 'input_visual_review.json', 'independent_input_audit.json']:
        bindings[(OLD / name).relative_to(ROOT).as_posix()] = sha(OLD / name)
    for name in ['completion_margin_feather_v1.py', 'prepare_completion_margin_feather_v1.py', 'run_completion_margin_feather_v1.py',
                 'audit_completion_margin_feather_v1.py', 'record_completion_margin_feather_v1_review.py']:
        bindings['scripts/' + name] = sha(ROOT / 'scripts' / name)
    bindings['outputs/cctv_dgp_delivered_margin_probe_v38_preparation/independent_packet_audit.json'] = sha(ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_preparation/independent_packet_audit.json')
    for name, digest in {**bindings, **p['app_preservation_sha256']}.items(): assert sha(ROOT / name) == digest, name
    protocol = {'format':'completion-saved-output-two-pixel-margin-feather-v1', 'UTC':datetime.now(timezone.utc).isoformat(),
                'cases':p['cases'], 'parent_protocol_sha256':sha(OLD / 'protocol.json'), 'parent_results_sha256':sha(OLD / 'results.json'),
                'sources_sha256':bindings, 'app_preservation_sha256':p['app_preservation_sha256'], 'family_case_counts':p['family_case_counts'],
                'quality_criteria':p['quality_criteria'], 'source_scope':p['source_scope'], 'exposed_development_photos':True,
                'hypothesis':'Two-pixel margin blending can soften a hard join without changing the core estimate, reviewed masks or neural context',
                'rule':'Core weight1; existing Euclidean margin weight1-distance_to_core/3 for distance<=2; outside final weight0; byte blend rounded floor(x+.5)',
                'comparison':'Existing union-conditioned PNG versus display-only feather on its same core/final support',
                'image_role':'Display processing only; saved raw neural outputs and their prior hard composition remain unchanged',
                'causal_limit':'Cannot remove copied objects outside final support, repair core anatomy or prove model quality; blending can reintroduce source covering within the existing margin',
                'expected_cases':36, 'expected_outputs':32, 'expected_rejections':4, 'expected_empty_bypasses':4,
                'cap_seconds':180, 'artifact_cap_bytes':64 * 1024**2, 'restoration':'Off', 'new_neural_or_detector_forwards':0,
                'model_forwards':0, 'gradient_calls':0, 'optimizer_updates':0, 'new_masks':False, 'mask_expansion':0,
                'automatic_outputs':0, 'assisted_original_photo_mask_reused_for_synthetic_pairs':True,
                'unknown_pretraining_overlap':True, 'hidden_ground_truth':None, 'hidden_metrics':None,
                'native_CCTV_or_reserved_final_used':False, 'app_changes':False, 'app_adoption':False,
                'automatic_quality_qualification':False, 'assisted_quality_qualification':False, 'independent_final_review':False, 'goal_complete':False}
    assert not any(name in sys.modules for name in ['torch', 'pretrained_completion', 'dgp_face_workflow_v3'])
    OUT.mkdir()
    with (OUT / 'protocol.json').open('x', encoding='utf-8', newline='\n') as stream: stream.write(json.dumps(protocol, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'complete':True, 'protocol_sha256':sha(OUT / 'protocol.json'), 'source_bindings':len(bindings), 'model_forwards':0}))


if __name__ == '__main__': main()
