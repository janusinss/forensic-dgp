"""Inert source/array audit of the R1 initializer stop; never imports Torch."""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'cctv-codeformer-role-preserving-training-v1-r1'
PROTOCOL = '4edacae4f1e61f06b21fb1308423398d5b9f842def160c02f74f8edc0975fc40'
ARCHIVE_SHA = '075d0fcc9b6dbd35e1ba65050c7837c00f1bd64afb44164a2a45ae991f1ab2e1'
ARCHIVE_BYTES = 34174817
OUTPUT = 'outputs/role_preserving_training_v1_r1'
STEM = OUTPUT + '/initializer_parity/train_u0003_slot1/'
NAMES = ('native', 'raw', 'logits', 'latent', 'tokens')
SHAPES = {'native': (1, 3, 512, 512), 'raw': (256, 256, 3),
          'logits': (1, 256, 1024), 'latent': (1, 256, 16, 16), 'tokens': (256,)}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def parse(payload):
    def unique(items):
        result = {}
        for name, value in items:
            need(name not in result, 'Duplicate metadata key')
            result[name] = value
        return result
    return json.loads(payload.decode('utf-8'), object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def main():
    start = time.monotonic()
    archive = ROOT / 'outputs' / (PREFIX + '-results.tar.gz')
    before = archive.stat()
    with archive.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    need(digest == ARCHIVE_SHA and before.st_size == ARCHIVE_BYTES, 'Actual new archive differs')
    selected = {}
    wanted = {OUTPUT + '/' + name for name in ('failure.json', 'factory.json', 'runtime.json', 'export_manifest.json')}
    wanted |= {STEM + name for name in ('record.json', 'context.json')}
    wanted |= {STEM + prefix + name + '.npy' for prefix in ('', 'parent/') for name in NAMES}
    sources = ['research/codeformer_role_preserving_training_v1_r1/' + name + '.py'
               for name in ('worker', 'model', 'contract')]
    wanted.update(sources)
    wanted.add('protocol.json')
    all_names = []
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            all_names.append(member.name)
            if member.name not in wanted:
                continue
            cap = 4 * 1024**2 if member.name.endswith('.npy') else 2 * 1024**2
            need(member.isfile() and member.size <= cap and member.name not in selected,
                 'Selected inert member type/cap/duplicate invalid')
            content = tar.extractfile(member).read()
            need(len(content) == member.size, 'Selected member short read')
            selected[member.name] = content
    need(set(selected) == wanted, 'Missing actual selected cause evidence')
    after = archive.stat()
    need((before.st_size, before.st_mtime_ns, before.st_ino) ==
         (after.st_size, after.st_mtime_ns, after.st_ino), 'Archive changed during read')
    need(len(all_names) == len(set(all_names)), 'Duplicate actual archive member')
    manifest = parse(selected[OUTPUT + '/export_manifest.json'])
    need(manifest['protocol_sha256'] == PROTOCOL and sha(selected['protocol.json']) == PROTOCOL,
         'Actual protocol identity differs')
    for name, content in selected.items():
        if name.endswith('/export_manifest.json'):
            continue
        need(manifest['files_sha256'][name] == sha(content) and
             manifest['files_bytes'][name] == len(content), 'Selected manifest binding differs')
    failure = parse(selected[OUTPUT + '/failure.json'])
    factory = parse(selected[OUTPUT + '/factory.json'])
    context = parse(selected[STEM + 'context.json'])
    record = parse(selected[STEM + 'record.json'])
    runtime = parse(selected[OUTPUT + '/runtime.json'])
    need(failure['completed_training_updates'] == failure['optimizer_constructions_completed'] ==
         failure['gradient_query_reservations'] == failure['backward_call_reservations'] == 0,
         'Actual activity differs from pre-Adam stop')
    need(failure['checkpoints'] == {} and failure['training_seconds'] == 0 and
         failure['reason'] == 'ValueError: Matched initializer inference parity failed; actual arrays and context retained',
         'Actual technical failure differs')
    contexts = {name: context[name + '_context'] for name in ('candidate', 'parent')}
    for name, value in contexts.items():
        need(value == record[name + '_context'] and
             all(row[1] is False for row in value['module_modes']) and
             all(value[key] is False for key in ('grad_enabled', 'inference_mode', 'autocast_enabled')),
             'Actual ordinary eval/no-grad FP32 context differs')
    need(contexts['candidate']['module_modes'] == contexts['parent']['module_modes'] and
         contexts['candidate']['requires_grad_tensors'] == 350 and
         contexts['parent']['requires_grad_tensors'] == 0 and context['context_error'] is None,
         'Actual matched module modes or explicit trainability differ')
    need(record['initial_RNG_sha256'] == record['after_candidate_RNG_sha256'], 'Actual saved RNG changed')
    arrays = {'candidate': {}, 'parent': {}}
    pins = {}
    for side, prefix, key in [('candidate', '', 'captured'), ('parent', 'parent/', 'parent_captured')]:
        for name in NAMES:
            member = STEM + prefix + name + '.npy'
            content = selected[member]
            need(record[key][name] == {'path': member, 'sha256': sha(content), 'bytes': len(content)},
                 'Actual saved array/record hash differs')
            value = np.load(io.BytesIO(content), allow_pickle=False, max_header_size=4096)
            need(value.shape == SHAPES[name] and value.dtype ==
                 (np.dtype('int64') if name == 'tokens' else np.dtype('float32')) and
                 np.isfinite(value).all(), 'Actual capture geometry/precision/finite condition differs')
            arrays[side][name] = value
            pins[member] = {'sha256': sha(content), 'bytes': len(content)}
    comparisons = {}
    for name in NAMES:
        a, b = arrays['candidate'][name], arrays['parent'][name]
        delta = a.astype(np.float64) - b.astype(np.float64)
        different = a != b
        locations = np.argwhere(different)
        maximum = tuple(int(i) for i in np.unravel_index(np.abs(delta).argmax(), delta.shape))
        row = {'shape': list(a.shape), 'dtype': str(a.dtype), 'elements': int(a.size),
            'exact_array_equal': bool(np.array_equal(a, b)),
            'exact_NPY_file_bytes_equal': selected[STEM + name + '.npy'] == selected[STEM + 'parent/' + name + '.npy'],
            'different_elements': int(np.count_nonzero(different)),
            'maximum_absolute_difference': float(np.abs(delta).max()),
            'mean_absolute_difference': float(np.abs(delta).mean()),
            'root_mean_square_difference': float(np.sqrt(np.square(delta).mean())),
            'mean_signed_candidate_minus_parent': float(delta.mean()),
            'first_difference_location': [int(i) for i in locations[0]] if len(locations) else None,
            'maximum_difference_location': list(maximum) if len(locations) else None,
            'candidate_value_at_maximum': float(a[maximum]) if len(locations) else None,
            'parent_value_at_maximum': float(b[maximum]) if len(locations) else None}
        recorded = context['differences'][name]
        need(recorded['exact'] == row['exact_array_equal'] and
             recorded['different_elements'] == row['different_elements'] and
             recorded['maximum_absolute_difference'] == row['maximum_absolute_difference'] and
             recorded['both_arrays_finite'] is True, 'Worker array-difference readout differs')
        comparisons[name] = row
    need(all(comparisons[name]['exact_array_equal'] for name in ('native', 'raw', 'latent', 'tokens')) and
         comparisons['logits']['different_elements'] == 223123 and
         comparisons['logits']['maximum_absolute_difference'] == 7.62939453125e-06,
         'Actual isolated logits-only mismatch differs')
    a, b = arrays['candidate']['logits'], arrays['parent']['logits']
    selected_a, selected_b = a.argmax(-1), b.argmax(-1)
    maximum_error_by_token = np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)), axis=-1)
    top2 = np.partition(b.astype(np.float64), -2, axis=-1)[..., -2:]
    margins = top2[..., 1] - top2[..., 0]
    selection = {'tokens': 256,
        'all_argmax_indices_equal': bool(np.array_equal(selected_a, selected_b)),
        'candidate_saved_tokens_equal_candidate_logit_argmax': bool(np.array_equal(selected_a.reshape(-1), arrays['candidate']['tokens'])),
        'parent_saved_tokens_equal_parent_logit_argmax': bool(np.array_equal(selected_b.reshape(-1), arrays['parent']['tokens'])),
        'minimum_parent_top2_margin': float(margins.min()),
        'all256_parent_top2_margins_exceed_twice_corresponding_maximum_logit_error': bool((margins > 2 * maximum_error_by_token).all()),
        'interpretation': 'For this one saved witness, the logits perturbation does not change any selected dictionary entry; exact native and delivered RAW arrays independently corroborate unchanged reconstructed pixels.'}
    source_pins = {}
    for name in sources:
        need((ROOT / name).read_bytes() == selected[name], 'Returned sealed source differs from reviewed local bytes')
        source_pins[name] = {'sha256': sha(selected[name]), 'bytes': len(selected[name])}
    model_digest = factory['canonical_tensor_state_sha256']
    need(model_digest['base'] == model_digest['candidate'] ==
         '92ac81750b86ef1da6f1f7c0480b9e64773fafe07b094e4a5ad7e88b3f7a0c8b',
         'Source-bound model identity differs')
    need(not any(name in sys.modules for name in ('torch', 'tensorflow', 'jax')),
         'No neural framework import permitted')
    report = {
        'format': 'codeformer-role-preserving-r1-independent-initializer-cause-audit-v1',
        'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
        'seconds': time.monotonic() - start,
        'archive_sha256': digest, 'archive_bytes': before.st_size,
        'actual_regular_archive_members': len(all_names), 'protocol_sha256': PROTOCOL,
        'manifest_sha256': sha(selected[OUTPUT + '/export_manifest.json']),
        'selected_source_pins': source_pins, 'selected_actual_array_files': pins,
        'actual_failure': {key: failure[key] for key in ('reason', 'seconds', 'phase_seconds', 'training_seconds',
            'completed_training_updates', 'optimizer_constructions_completed', 'optimizer_steps_completed',
            'gradient_query_reservations', 'backward_call_reservations', 'counts', 'checkpoints', 'resources')},
        'counters_are_reservations_not_completed_profiler_counts': True,
        'exact_initial_model_digest_source_bound': model_digest,
        'independent_checkpoint_tensor_replay_performed': False,
        'execution_context': {'all_module_mode_names_and_flags_match': True,
            'module_mode_count': len(contexts['candidate']['module_modes']),
            'all_modules_eval': True, 'ordinary_no_grad_both': True, 'autocast_disabled_both': True,
            'candidate_requires_grad_tensors': 350, 'parent_requires_grad_tensors': 0,
            'context_validator_passed': True,
            'saved_RNG_before_and_after_candidate_equal': True,
            'actual_RNG_sha256': record['initial_RNG_sha256']},
        'independent_array_comparisons': comparisons, 'dictionary_selection_evidence': selection,
        'recorded_worker_difference_metrics_independently_reproduced': True,
        'runtime': {key: value for key, value in runtime.items() if key != 'pip_freeze'},
        'demonstrated_cause': 'The prerequisite rejected non-bit-exact intermediate logits although the copied model tensor digest, evaluated input context, encoder latent, all256 selected dictionary indices, native512 output and delivered256 RAW output match. It stopped before Adam construction, gradients, training-mode references or quality evaluation.',
        'invalid_assumption': 'Same model values, eval/no-grad context and restored RNG were assumed sufficient for exact intermediate logits across two separately allocated objects; this actual return disproves that assumption for the saved witness.',
        'exact_low_level_operation_cause': 'NOT IDENTIFIED. The exported arrays locate divergence after the saved encoder latent and no later than predictor logits, but there are no intermediate predictor tensors, repeated same-object calls, fully matched requires_grad-flag calls, or CUDA operator/kernel traces.',
        'unproven_possible_causes': ['Remaining candidate350/parent0 parameter requires_grad flags or object/layout/backend dispatch differences.',
            'Backend execution variability; runtime does not enable deterministic algorithms.'],
        'unsupported_claims_avoided': ['No assertion that MHA native fastpath ran; architecture uses q/k distinct from value and batch_first=False.',
            'No assertion that one Torch/CUDA operation or version is defective.',
            'No arbitrary numerical epsilon or relaxed quality threshold.',
            'No learned quality improvement or failure from a zero-update run.'],
        'next_evidence_needed': 'A finite, zero-update manual VM diagnostic can separate same-object repeatability, separate-object parameter/layout/flag effects and first predictor-stage divergence; retain all contexts and compared arrays before gates. This report makes no source repair or training authorization.',
        'local_actions': 'Bounded inert TAR JSON/source inspection and allow_pickle=False NumPy array arithmetic only; no returned code executed, checkpoint loaded, model forward, autograd, optimizer or VM action.',
        'source_sha256': sha(Path(__file__).read_bytes()), 'new_model_created': False,
        'model_qualified': False, 'overall_goal_complete': False,
    }
    out = ROOT / 'outputs/codeformer_role_preserving_training_v1_r1_return_review/cause_v1'
    out.mkdir(parents=True, exist_ok=True)
    destination = out / 'audit.json'
    with destination.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    text = '''# R1 initializer stop: independent cause audit

Location: matched initializer parity, before Adam construction.

Cause demonstrated: only intermediate logits fail exact equality. Independent
readback confirms 223,123 of 262,144 logits differ, with maximum absolute error
0.00000762939453125. Encoder latent, all 256 selected dictionary entries,
native512 pixels and delivered256 RAW pixels are exactly equal, including NPY
file bytes. All 371 module modes match in evaluation mode, both calls use
ordinary no-grad FP32, and saved before/after RNG hashes match.

The exact low-level operation is unknown. Divergence is after the captured
encoder latent and no later than predictor logits. No predictor intermediates,
same-object repeats, fully matched requires-grad flag experiments or kernel
traces were exported. Candidate350/parent0 requires-grad flags remain different.
Nondeterministic algorithms remain permitted. Neither fact alone proves a cause.

This is a technical prerequisite stop: zero new updates, optimizer constructions,
gradient queries and backward calls; no new checkpoint or 406-case quality bank.
The source-bound model-value digests match. Training quality has not been tested.

Next evidence should distinguish repeatability, object/flag/backend effects and
first predictor-stage divergence in a bounded zero-update manual diagnostic.
No source repair, relaxed epsilon, training, neural work or VM operation was
performed by this audit. See audit.json for exact arrays, hashes and counters.
'''
    with (out / 'audit.md').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)
    print(json.dumps({'complete': True, 'report': str(destination), 'sha256': sha(destination.read_bytes()),
        'archive_sha256': digest, 'logits_different': 223123,
        'maximum_logit_difference': 7.62939453125e-06, 'other_four_arrays_exact': True,
        'new_updates': 0, 'exact_low_level_cause_identified': False}))


if __name__ == '__main__':
    main()
