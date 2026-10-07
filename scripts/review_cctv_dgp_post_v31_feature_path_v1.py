"""Static source/checkpoint review only; no model construction or differentiation."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'
RETURNED = ROOT / 'outputs/cctv_dgp_profile_batches_v31_return'
OUT = ROOT / 'outputs/cctv_dgp_post_v31_feature_path_review_v1'
FUSION = ['fpn.lateral' + str(i) + '.weight' for i in range(5)] + [
    'fpn.td' + str(i) + '.0.' + suffix for i in range(1, 4) for suffix in ['weight', 'bias']]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    start = time.monotonic()
    assert not OUT.exists(), 'Preserve prior review'
    p_path = ROOT / 'outputs/cctv_dgp_profile_batches_vm_v31/protocol.json'
    p = read(p_path)
    assert sha(p_path) == 'ae91d25066bcb532a8a88bdc48fe401c4b18d73e8d348b47b5d012d5df8e485e'
    audit_path = ROOT / 'outputs/cctv_dgp_profile_batches_v31_independent_audit.json'
    audit = read(audit_path)
    assert audit['complete'] and audit['VM_failure_retained'] and not audit['necessary_capacity_pass']
    visual_path = ROOT / 'outputs/cctv_dgp_profile_batches_v31_failure_review_v1/visual_review.json'
    visual = read(visual_path)
    assert visual['complete'] and visual['cases_reviewed'] == 50 and not visual['useful_whole_face_gain_established']
    original_path = PARENT / 'weights/dgp_v2.pth'
    stopped_path = RETURNED / 'outputs/update50/dgp_candidate_v31.pth'
    metrics_path = RETURNED / 'outputs/update50/metrics.json'
    assert sha(original_path) == p['original_checkpoint_sha256']
    assert sha(stopped_path) == read(metrics_path)['candidate_checkpoint_sha256']
    paths = [p_path, audit_path, visual_path, original_path, stopped_path, metrics_path,
             RETURNED / 'outputs/failure.json', RETURNED / 'outputs/gradient_preflight.json',
             ROOT / 'outputs/cctv_dgp_v30_sampling_gradient_v1_independent_audit.json',
             ROOT / 'outputs/cctv_dgp_post_v27_architecture_review_v1/user_architecture_decision.json',
             PARENT / 'models/dgp_synthesizer.py', PARENT / 'models/fpn_mobilenet.py',
             ROOT / 'scripts/cctv_dgp_mean_centered_decoder_v29.py', Path(__file__)]
    old = read(PARENT / 'protocol.json')
    for name in ['models/dgp_synthesizer.py', 'models/fpn_mobilenet.py']:
        assert sha(PARENT / name) == old['assets_sha256'][name]
        ast.parse((PARENT / name).read_text(encoding='utf-8'), feature_version=(3, 10))
    # Only safe tensor/state loading, with no model import, forward or autograd.
    import torch
    torch.set_num_threads(4)
    original = torch.load(original_path, map_location='cpu', weights_only=True)
    stopped = torch.load(stopped_path, map_location='cpu', weights_only=True)
    assert set(original) == set(stopped)
    decoder = {row['name'] for row in p['parameter_layout']}
    assert len(decoder) == 12 and sum(original[name].numel() for name in decoder) == 498627
    assert len(FUSION) == 11 and not decoder.intersection(FUSION)
    fusion = []
    for name in FUSION:
        left, right = original[name], stopped[name]
        assert left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
        fusion.append({'name': name, 'shape': list(left.shape), 'parameters': left.numel(),
                       'unchanged_at_stopped50': True, 'connectivity_gradient_measured': False})
    assert sum(row['parameters'] for row in fusion) == 479616
    changed = []
    frozen = 0
    for name, value in original.items():
        assert value.dtype == stopped[name].dtype and value.shape == stopped[name].shape
        assert bool(torch.isfinite(value).all()) and bool(torch.isfinite(stopped[name]).all())
        if torch.equal(value, stopped[name]):
            frozen += name not in decoder
        else:
            assert name in decoder, name
            changed.append(name)
    assert set(changed) == decoder
    assert all(torch.equal(original[name], stopped[name]) for name in original if name.startswith('fpn.'))
    record = {'complete': True, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'source_bindings_sha256': {path.relative_to(ROOT).as_posix(): sha(path) for path in paths},
              'reviewer_sha256': sha(Path(__file__)), 'original_checkpoint_sha256': sha(original_path),
              'stopped50_checkpoint_sha256': sha(stopped_path), 'original_state_entries': len(original),
              'frozen_state_entries_verified': frozen, 'changed_decoder_tensors': sorted(changed),
              'changed_decoder_parameters': 498627, 'feature_fusion_tensors': fusion,
              'feature_fusion_parameters': 479616,
              'all_FPN_state_entries_unchanged': True,
              'forward_path': ['prepared 256px RGB', 'frozen MobileNet multiscale backbone',
                               'five frozen lateral convolutions and three top-down convolutions',
                               'active reconstruction heads and RGB decoder', 'observed mean-centering and clip'],
              'next_hypothesis': 'Measure original feature-fusion gradient connectivity and objective tradeoffs before selecting new trainable weights.',
              'diagnostic_design': {'states': ['original', 'stopped50 V31 observation only'],
                                    'candidate_partition': FUSION,
                                    'decoder_partition_comparison': sorted(decoder),
                                    'normalization_and_backbone': 'frozen evaluation statistics and weights',
                                    'cohorts': 'input-metadata-selected approved TRAIN references; source/profile matched exposed and unexposed groups',
                                    'finite_updates': 0, 'epochs': 0, 'worker_cap_seconds': 600,
                                    'component_gradient_query_maximum': 280,
                                    'stop_rules': ['source/state mismatch', 'nonfinite output/loss/gradient',
                                                   'worker deadline', 'allocated VRAM over20GiB', 'unreviewed parameter partition'],
                                    'manual_existing_L4_only': True,
                                    'not_yet_executable_or_released': True},
              'head4': 'Retain previous inactive-head finding; do not repeat the all14 connected-gradient assumption.',
              'no_unique_cause_established': True, 'no_learning_rate_or_gate_change_selected': True,
              'V31_resume_permitted': False, 'new_training_recipe_selected': False,
              'training_packet_released': False, 'model_constructions': 0, 'neural_calls': 0,
              'gradient_calls': 0, 'backwards': 0, 'optimizer_updates': 0,
              'native_or_reserved_used': False, 'app_promotion': False, 'goal_complete': False,
              'seconds': time.monotonic() - start}
    assert record['seconds'] < 120
    OUT.mkdir()
    with (OUT / 'review.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps({key: record[key] for key in ['complete', 'changed_decoder_parameters',
                      'feature_fusion_parameters', 'all_FPN_state_entries_unchanged',
                      'neural_calls', 'gradient_calls', 'optimizer_updates', 'seconds']}))


if __name__ == '__main__':
    main()
