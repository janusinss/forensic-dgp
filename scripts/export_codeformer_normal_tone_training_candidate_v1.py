"""Export audited update50 tensor weights on CPU; no model construction/inference."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_normal_tone_training_v1 import contract as c

PROTO = 'd0c0b7a5be4577803bc156b611386c3be10068a3e78a47f9348125242f0c79d1'
MEMBER = 'outputs/normal_tone_training_v1/snapshot_0050/checkpoint.pt'


def tensor_digest(state, torch):
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str(value.dtype).encode('ascii'))
        digest.update(str(list(value.shape)).encode('ascii'))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def audited_tensor_digest(state, torch):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode()
        digest.update(len(header).to_bytes(8, 'big')); digest.update(header)
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    review = ROOT / 'outputs/codeformer_normal_tone_training_v1_return_review'
    imported = c.read_json(review / 'return_import_v1.json')
    state_review = c.read_json(review / 'state_v1/state_audit_v1.json')
    c.require(imported['complete'] is True and imported['protocol_sha256'] == PROTO and
              state_review['complete'] is True and state_review['protocol_sha256'] == PROTO and
              state_review['archive_sha256'] == imported['archive_sha256'] and
              state_review['evidence_closure']['import_receipt_sha256'] == c.sha256(review / 'return_import_v1.json'),
              'Matching closed import and independent CPU state audit required')
    c.require(all(state_review[key] is True for key in ('initializer_exact515', 'frozen_exact165',
              'changed_trainable350', 'update50_stopped_model_Adam_RNG_sampler_schedule_exact')),
              'Exact independent initial/frozen/committed state closure required')
    pixels = c.read_json(review / 'pixels_v1/audit.json')
    c.require(pixels['complete'] is True and pixels['protocol_sha256'] == PROTO and
              pixels['archive_sha256'] == imported['archive_sha256'] and
              pixels['import_receipt_sha256'] == c.sha256(review / 'return_import_v1.json') and
              pixels['new_RAW_PNG_pairs_verified'] == 812 and
              pixels['clear_guard_pass'] is False and pixels['blur_guard_pass'] is False,
              'Matching full saved-pixel audit and retained quality failures required')
    # Require the root's complete saved-output/visual review before packaging.
    visuals = c.read_json(review / 'root_visual_review_v1.json')
    c.require(visuals['complete'] is True and visuals['case_rows_reviewed'] == 406 and
              visuals['model_qualified'] is False and visuals['protocol_sha256'] == PROTO and
              visuals['archive_sha256'] == imported['archive_sha256'] and
              visuals['pixel_audit_sha256'] == c.sha256(review / 'pixels_v1/audit.json'),
              'Matching all406-case unqualified visual review required')
    c.require(visuals['personal_visual_review'] is True and len(visuals['sheets_reviewed']) == 34 and
              [{key: row[key] for key in ('path', 'sha256', 'bytes', 'size', 'case_ids')}
               for row in visuals['sheets_reviewed']] == pixels['visual_sheets'],
              'Every planned sheet and source-case binding must match root inspection')
    visual_cases = [case for row in visuals['sheets_reviewed'] for case in row['case_ids']]
    c.require(len(visual_cases) == len(set(visual_cases)) == 406 and
              all(row['viewed_by_root'] is True for row in visuals['sheets_reviewed']),
              'Exact personally inspected406 unique cases required')
    source = c.safe_path(Path(imported['import_root']), MEMBER)
    c.require(c.sha256(source) == imported['files_sha256'][MEMBER] and
              source.stat().st_size == imported['files_bytes'][MEMBER], 'Exact source checkpoint differs')
    destination = ROOT / 'outputs/codeformer_normal_tone_training_v1_candidate_unqualified'
    c.require(not destination.exists(), 'Preserve previous candidate export')
    import torch
    loaded = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
    c.require(loaded['protocol_sha256'] == PROTO and loaded['completed_updates'] == 50 and
              loaded['partial_microbatches_discarded'] == 0 and loaded['resume_safe'] is True and
              loaded['counts']['optimizer_updates'] == 50 and loaded['counts']['microbatches'] == 200 and
              loaded['schedule']['completed_updates'] == 50, 'Actual committed50 checkpoint required')
    state = loaded['model']
    c.require(len(state) == 515 and all(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and
              torch.isfinite(value).all().item() for value in state.values()), 'Finite515 CPU tensor state required')
    audited_digest = audited_tensor_digest(state, torch)
    c.require(audited_digest == state_review['checkpoint_readback']['update50']['model_sha256'],
              'Every source tensor must bind to the independently audited committed state')
    destination.mkdir()
    weights = destination / 'codeformer_u0050_unqualified.pth'
    temporary = destination / (weights.name + '.part')
    with temporary.open('xb') as stream:
        torch.save({'params_ema': state}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    c.require(temporary.stat().st_size <= 384 * 1024**2, 'Inference-weight cap exceeded')
    os.replace(temporary, weights)
    checked = torch.load(weights, map_location='cpu', weights_only=True, mmap=True)
    c.require(set(checked) == {'params_ema'} and list(checked['params_ema']) == list(state) and
              all(torch.equal(state[name], checked['params_ema'][name]) for name in state),
              'Every exported model tensor must exactly equal source checkpoint')
    sha = c.sha256(weights)
    with Path(str(weights) + '.sha256').open('xb') as stream:
        stream.write((sha + '  ' + weights.name + '\n').encode('ascii'))
    receipt = {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
        'status': 'TRAINED_RESEARCH_CANDIDATE_FAILED_PRESERVATION', 'protocol_sha256': PROTO,
        'source_checkpoint_member': MEMBER, 'source_checkpoint_sha256': imported['files_sha256'][MEMBER],
        'source_checkpoint_bytes': source.stat().st_size, 'committed_training_updates': 50,
        'exported_file': str(weights), 'exported_sha256': sha, 'exported_bytes': weights.stat().st_size,
        'model_tensor_count': 515, 'all_exported_tensors_bit_equal_source': True,
        'exported_model_tensor_digest': tensor_digest(state, torch),
        'audited_length_prefixed_model_tensor_digest': audited_digest,
        'params_ema_key_is_compatibility_only': True,
        'EMA_training_was_used': False, 'optimizer_or_RNG_in_export': False,
        'checkpoint_weights_only_CPU_deserialization': True,
        'model_construction_inference_autograd_optimizer_VM_calls': 0,
        'clear_preservation_pass': False, 'normal_blur_pass': False,
        'app_promoted': False, 'model_qualified': False,
        'credit': 'Our finite fine-tuned adaptation of the existing CodeFormer architecture and official initializer',
        'initializer_sha256': c.ASSETS['initializer']['sha256'],
        'state_audit_receipt_sha256': c.sha256(review / 'state_v1/state_audit_v1.json'),
        'pixel_audit_receipt_sha256': c.sha256(review / 'pixels_v1/audit.json'),
        'root_visual_review_sha256': c.sha256(review / 'root_visual_review_v1.json'),
        'exporter_source_sha256': c.sha256(__file__)}
    c.write_new_json(destination / 'provenance.json', receipt)
    readme = '''# Trained CodeFormer research candidate — not accepted

This file contains the genuine model weights after50 committed optimizer
updates in our manual NVIDIA L4 run. All515 tensors exactly match the retained
update50 checkpoint. It is our fine-tuned adaptation of CodeFormer, not a new
architecture trained from scratch. The`params_ema`key supports common loaders;
EMA was not used during this run.

This candidate FAILED clear-face and normal-blur preservation checks. Retain it
for research/review; it has not replaced the main app restorer and must not be
presented as an accepted forensic restoration model. No extra epochs or automatic
resume are authorized by this export. The official CodeFormer benchmark and all
original checkpoints remain intact. No exact recovered identity is claimed.

`provenance.json`binds the exact trained checkpoint, exported weight hash, update
count, independent state audit and complete visual review. The original full
checkpoint still retains optimizer/RNG/sampler/schedule and failure evidence.
No model inference, autograd, optimizer or training ran during CPU serialization.
'''
    with (destination / 'README.md').open('xb') as stream:
        stream.write(readme.encode('utf-8'))
    print({'complete': True, 'weights': str(weights), 'sha256': sha,
           'bytes': weights.stat().st_size, 'actual_training_updates': 50, 'model_qualified': False})


if __name__ == '__main__':
    main()
