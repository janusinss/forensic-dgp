"""Serialize an audited, unqualified composite on CPU; never construct/run a model.

This is an inference-only state/specification artifact, not an implemented app
loader. The complete source checkpoint (including head Adam/RNG) stays intact.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_tone_control_training_v1 import contract as c

PROTOCOL = 'f10ce6d52c2ec259632750af2d3afbdf97c5c925b99849fae2a2c4d9cd04a5b1'
ARCHIVE = '6a4386f84ebc27d6201ad188bfb19dc36bdb0a6fbab3c6f77c0662ecd4e547bf'
MEMBER = c.OUTPUTS + '/stopped_composite/checkpoint.pt'
SOURCE_SHA = 'f492d13cd63f9652407ca2eb4b90ae2b9a5fc49a528c4a62d552f47d1aec4a2f'
SOURCE_BYTES = 376734485
FORMAT = 'codeformer-tone-control-composite-inference-v1'
STATUS = 'TRAINED_RESEARCH_COMPOSITE_FAILED_ACCEPTANCE_GATES'
HEAD_SHAPES = {'hidden.weight': [32, 256], 'hidden.bias': [32],
               'output.weight': [1, 32], 'output.bias': [1]}
CAP = 384 * 1024**2
SHEET_KEYS = ('path', 'sha256', 'bytes', 'case_ids', 'case_count', 'known_TRAIN')


def pinned_json(path):
    path = c.no_links(path)
    before = path.stat(); value = c.read_json(path); digest = c.sha256(path); after = path.stat()
    c.require((before.st_size, before.st_mtime_ns, before.st_ino) ==
              (after.st_size, after.st_mtime_ns, after.st_ino), 'Audit receipt changed during read')
    return value, {'path': path.relative_to(ROOT).as_posix(), 'sha256': digest, 'bytes': after.st_size}


def require_identity(value, label):
    c.require(value.get('complete') is True and value.get('protocol_sha256') == PROTOCOL and
              value.get('archive_sha256') == ARCHIVE, 'Matching completed ' + label + ' required')


def preconditions():
    """Read small receipts and sheet bytes before the sole lazy Torch import."""
    review = ROOT / 'outputs/codeformer_tone_control_training_v1_return_review'
    names = {'import': 'return_import_v1.json', 'transport': 'independent_transport_v1.json',
        'state': 'state_v1/state_audit_v1.json', 'pixels': 'pixels_v1/audit.json',
        'visual_manifest': 'pixels_v1/visual_manifest.json', 'sheet_cells': 'pixels_v1/sheet_cells.json',
        'visual': 'root_visual_review_v1.json'}
    values, evidence = {}, {}
    for label, name in names.items(): values[label], evidence[label] = pinned_json(review / name)
    imported, transport, state, pixels, visual = [values[k] for k in ('import', 'transport', 'state', 'pixels', 'visual')]
    for label in ('import', 'transport', 'state', 'pixels', 'visual'): require_identity(values[label], label)
    expected_root = c.no_links(ROOT / 'outputs/codeformer_tone_control_training_v1_return').resolve(strict=True)
    c.require(c.no_links(Path(imported['import_root'])).resolve(strict=True) == expected_root and
              imported['regular_files'] == transport['regular_members'] == 8533 and
              imported['payload_bytes'] == transport['payload_bytes'] == 3209400295 and
              imported['files_sha256'][MEMBER] == SOURCE_SHA and imported['files_bytes'][MEMBER] == SOURCE_BYTES,
              'Exact retained safe import/source state required')
    c.require(transport['transport_integrity_pass'] is True and transport['transport_blockers'] == [] and
              transport['completed_head_updates_recorded'] == 50 and transport['restorer_updates_this_run_recorded'] == 0 and
              transport['operation_reservation_events'] == 3090 and transport['model_qualified'] is False,
              'Independent actual transport/chronology closure required')
    c.require(transport['member_bindings_freshly_hashed'] == {name: {'sha256': digest, 'bytes': imported['files_bytes'][name]}
              for name, digest in imported['files_sha256'].items()}, 'Import/transport member maps differ')
    protocol_path = c.safe_path(expected_root, 'protocol.json')
    c.require(c.sha256(protocol_path) == PROTOCOL, 'Imported protocol changed')
    p = c.read_json(protocol_path); c.verify_protocol(p)
    c.require(state['evidence_consistency_pass'] is True and state['model_qualified'] is False and
              state['evidence_binding']['import_receipt_sha256'] == evidence['import']['sha256'] and
              pixels['import_receipt_sha256'] == evidence['import']['sha256'], 'Independent audits bind a different import')
    cps = state['sections']['checkpoints']; stopped = cps['checkpoints']['stopped_composite']
    head50 = cps['checkpoints']['composite_head0050']
    c.require(set(cps['trained50_and_stopped_state_exact']) ==
              {'base_model', 'head', 'head_optimizer', 'rng', 'sampler', 'schedule'} and
              cps['all4head_tensors_changed'] is True and stopped['path'] == MEMBER and stopped['sha256'] == SOURCE_SHA and
              stopped['bytes'] == SOURCE_BYTES and stopped['completed_head_updates'] == 50 and
              stopped['restorer_updates_this_run'] == 0 and stopped['all515base_tensors_exact_parent'] is True and
              stopped['head_tensors'] == 4 and stopped['head_parameters'] == 8257 and stopped['head_Adam_states'] == 4 and
              stopped['partial_microbatches'] == 0 and stopped['inflight'] is False and
              stopped['base_model_tensor_sha256'] == head50['base_model_tensor_sha256'] ==
                  c.DESIGN['parents']['restorer']['model_state_sha256'] and
              stopped['head_tensor_sha256'] == head50['head_tensor_sha256'], 'Exact genuine trained composite audit required')
    c.require(state['sections']['operation_census']['counts'] == p['call_caps'] and
              state['sections']['operation_census']['trace_events'] == 3090, 'Final operation audit differs')
    required_counts = {'DEV_cases': 406, 'banks': 3, 'RAW_PNG_pairs': 1218, 'base_RAW_arrays': 812,
        'known_TRAIN_stress_pairs': 2, 'clear_cases': 61, 'normal_blur_cases': 53,
        'original_historical_parity_cases': 406, 'zero_head_historical_parity_cases': 406, 'base_fresh_repeat_comparisons': 406}
    c.require(pixels['audit_pass'] is True and pixels['model_qualified'] is False and
              all(type(pixels.get(key)) is int and pixels[key] == count for key, count in required_counts.items()) and
              all(pixels.get(key) == [] for key in ('parity_failures', 'saved_scalar_mismatches', 'gate_record_mismatches')) and
              all(pixels.get(key) is False for key in ('preservation_pass', 'dark_retention_pass', 'known_stress_pass',
                  'worker_overall_quality_requirements_pass')), 'Complete independent pixel checks and retained failures required')
    manifest = values['visual_manifest']; sheet_rows = pixels['visual_sheets']
    c.require(manifest['complete'] is True and manifest['DEV_sheets'] == 34 and manifest['TRAIN_stress_sheets'] == 1 and
              manifest['all406_DEV_rows_once'] is True and manifest['sheets'] == sheet_rows and len(sheet_rows) == 35,
              'Complete35-sheet source manifest required')
    c.require(visual['personal_visual_review'] is True and visual['model_qualified'] is False and visual['app_promoted'] is False and
              visual['import_receipt_sha256'] == evidence['import']['sha256'] and
              visual['pixel_audit_sha256'] == evidence['pixels']['sha256'] and
              visual['visual_manifest_sha256'] == evidence['visual_manifest']['sha256'] and
              visual['sheet_cells_sha256'] == evidence['sheet_cells']['sha256'] and
              visual['DEV_case_rows_reviewed'] == 406 and visual['known_TRAIN_case_rows_reviewed'] == 2 and
              len(visual['sheets_reviewed']) == 35 and
              [{key: row[key] for key in SHEET_KEYS} for row in visual['sheets_reviewed']] == sheet_rows and
              all(row['viewed_by_root'] is True for row in visual['sheets_reviewed']), 'Exact personal root35-sheet review required')
    dev, known, paths = [], [], set()
    for row in sheet_rows:
        c.require(type(row['known_TRAIN']) is bool and type(row['case_count']) is int and
                  row['case_count'] == len(row['case_ids']) and row['path'] not in paths, 'Unique exact sheet geometry/rows required')
        paths.add(row['path']); (known if row['known_TRAIN'] else dev).extend(row['case_ids'])
        sheet = c.safe_path(review / 'pixels_v1', row['path'])
        c.require(sheet.stat().st_size == row['bytes'] and c.sha256(sheet) == row['sha256'], 'Personally reviewed sheet changed')
    canonical = {name.split('/')[-2] for name in p['retained_output_dependencies'] if '/snapshot_0000/' in name}
    c.require(len(dev) == len(set(dev)) == 406 and set(dev) == canonical and len(known) == 2 and
              set(known) == set(c.DESIGN['evaluation']['known_overlap_stress']['cases']), 'Exact406DEV plus2knownTRAIN rows required')
    failure_path = c.safe_path(expected_root, c.OUTPUTS + '/failure.json')
    c.require(c.sha256(failure_path) == imported['files_sha256'][c.OUTPUTS + '/failure.json'], 'Authoritative failure changed')
    failure = c.read_json(failure_path)
    c.require(failure['protocol_sha256'] == PROTOCOL and failure['failure'] is True and failure['counts'] == p['call_caps'] and
              failure['completed_head_updates'] == 50 and failure['restorer_updates_this_run'] == 0 and
              failure['reason'] == transport['authoritative_quality_stop']['reason'], 'Retained actual quality stop required')
    source = c.safe_path(expected_root, MEMBER); before = source.stat()
    c.require(before.st_size == SOURCE_BYTES and c.sha256(source) == SOURCE_SHA, 'Exact stopped source hash required before Torch import')
    # The new model/data helpers and design are the inference contract, never
    # replaced with mutable app defaults or imported as a model here.
    sources = {}
    for name in ('model.py', 'data.py', 'design.json', 'worker.py'):
        relative = 'research/codeformer_tone_control_training_v1/' + name
        local = ROOT / relative
        c.require(c.sha256(local) == p['files_sha256'][relative], 'Frozen inference source changed')
        sources[relative] = p['files_sha256'][relative]
    return review, p, values, evidence, source, before, stopped, sources


def state_digest(state, torch, prefix=''):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        header = json.dumps([prefix + name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode()
        digest.update(len(header).to_bytes(8, 'big')); digest.update(header)
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tensor_manifest(state, torch):
    return {name: {'shape': list(value.shape), 'dtype': str(value.dtype), 'bytes': value.numel() * value.element_size(),
        'raw_sha256': hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
        for name, value in state.items()}


def validate_states(base, head, torch):
    c.require(isinstance(base, dict) and len(base) == 515 and isinstance(head, dict) and set(head) == set(HEAD_SHAPES) and
              all(isinstance(name, str) and name for name in base), 'Exact515base and4head tensor sections required')
    for state in (base, head):
        c.require(all(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and value.dtype == torch.float32 and
                      torch.isfinite(value).all().item() for value in state.values()), 'Finite FP32 CPU tensor values required')
    c.require(all(list(head[name].shape) == shape for name, shape in HEAD_SHAPES.items()) and
              sum(value.numel() for value in head.values()) == 8257 and
              sum(value.numel() for value in base.values()) == 94112707, 'Exact base/head tensor shapes/elements required')


def inference_specification(p, sources):
    return {'version': 1, 'controller': c.DESIGN['controller'],
        'composite_specification': c.DESIGN['composite_and_loss'], 'input_policy': c.DESIGN['input_policy'],
        'feature_policy': {'implementation': 'research.codeformer_tone_control_training_v1.data.features',
            'input': 'ExactRGB8/float32(255),RGB256HWC and explicit nonempty boolean observed256support',
            'grid': [8, 8], 'cell_pixels': [32, 32], 'sum_precision': 'float64', 'empty_cell_RGB': 0,
            'support_fraction_denominator': 1024, 'plane_order': ['R', 'G', 'B', 'observed_support_fraction'],
            'flatten': 'C-row-major,planes first', 'final_dtype': 'float32', 'features': 256,
            'support_is_not_covering_or_completion_mask': True},
        'restorer_specification': {'class': 'third_party.codeformer.codeformer_arch.CodeFormer',
            'kwargs': {'dim_embd': 512, 'n_head': 8, 'n_layers': 9, 'codebook_size': 1024,
                       'connect_list': ['32', '64', '128', '256'], 'fix_modules': ['quantize', 'generator']},
            'forward': {'w': 1.0, 'detach_16': True, 'adain': True},
            'input': 'bilinear256to512 align_cornersFalse,RGB*2-1',
            'delivery': 'clip((native512.float()+1)/2,0,1)BEFORE bilinear512to256 align_cornersFalse',
            'parameters_frozen': True, 'mode': 'eval', 'new_restorer_updates': 0},
        'controller_tensor_shapes': HEAD_SHAPES,
        'controller_application': 'Delivered256base plus common0.25*tanh scalar; clamp[0,1] then exact inputRGB outside observed support',
        'PNG': 'floor(clip(correctedRAW,0,1)*float32(255))uint8; restore exact inputRGB8 outside support',
        'precision': 'FP32;autocast/TF32disabled;retain recorded runtime/backend policy; crossruntime parity untested',
        'source_sha256': sources,
        'historical_implementation_bindings': {name: p['dependencies'][name] for name in
            ('research/codeformer_adaptation_v2/model.py', 'third_party/codeformer/codeformer_arch.py',
             'third_party/codeformer/vqgan_arch.py')},
        'loader_contract': {'required_distinct_format': FORMAT, 'implemented_by_this_exporter': False,
            'current_app_compatible': False, 'standard_CodeFormer_params_ema_compatible': False,
            'both_tensor_sections_required': True, 'strict_keys_shapes_dtypes_finite_and_digests_required': True,
            'base_only_fallback_permitted': False, 'zero_reinitialization_on_load_error_permitted': False,
            'inference_inputs': ['RGB256crop', 'explicit_observed_support_mask'],
            'forbidden_inputs': c.DESIGN['input_policy']['forbidden_inference_inputs'],
            'statement': 'Research-only composite weights require a dedicated versioned loader for both base_model and head plus the exact feature/support/delivery wrapper. The current app and standard CodeFormer loaders cannot reproduce this artifact.'}}


def main():
    review, p, values, evidence, source, before, audited, sources = preconditions()
    destination = c.no_links(ROOT / 'outputs/codeformer_tone_control_training_v1_candidate_unqualified')
    c.require(not destination.exists(), 'Preserve previous or partial candidate export')
    exporter_sha = c.sha256(__file__)
    import torch  # CPU tensor serialization only, after all audited visual/source gates.
    torch.set_num_threads(1)
    loaded = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
    c.require(loaded['format'] == 'codeformer-tone-control-composite-state-v1' and loaded['protocol_sha256'] == PROTOCOL and
              loaded['design_sha256'] == c.DESIGN_SHA256 and loaded['completed_head_updates'] == 50 and
              loaded['restorer_updates_this_run'] == 0 and loaded['historical_restorer_updates'] == 50 and
              loaded['partial_microbatches'] == 0 and loaded['head_optimizer_inflight'] is False and loaded['head_state_safe'] is True and
              loaded['counts'] == p['call_caps'] and loaded['sampler']['next_update'] == 51 and
              loaded['schedule']['completed_updates'] == 50 and c.equal(loaded['schedule']['training'], c.DESIGN['training']) and
              c.equal(loaded['controller'], c.DESIGN['controller']) and
              c.equal(loaded['composite_specification'], c.DESIGN['composite_and_loss']) and
              loaded['parent_checkpoint_sha256'] == c.ASSETS['base']['sha256'] and
              loaded['original_checkpoint_sha256'] == c.ASSETS['initializer']['sha256'], 'Actual audited final3090stopped state required')
    base, head = loaded['base_model'], loaded['head']; validate_states(base, head, torch)
    digests = {'base_model': state_digest(base, torch), 'head': state_digest(head, torch)}
    c.require(digests == {'base_model': audited['base_model_tensor_sha256'], 'head': audited['head_tensor_sha256']} and
              digests['base_model'] == loaded['base_model_tensor_digest'] and digests['head'] == loaded['head_tensor_digest'],
              'Every source tensor must match the independent state audit')
    # Inference export has no optimizer/RNG/sampler. The full immutable stopped
    # checkpoint retains them; none is constructed, restored or stepped here.
    base = {name: value.detach().cpu().contiguous().clone() for name, value in base.items()}
    head = {name: value.detach().cpu().contiguous().clone() for name, value in head.items()}
    del loaded
    combined = {**{'base_model/' + name: value for name, value in base.items()},
                **{'head/' + name: value for name, value in head.items()}}
    digests['namespaced_composite'] = state_digest(combined, torch)
    manifests = {'base_model': tensor_manifest(base, torch), 'head': tensor_manifest(head, torch)}
    specification = inference_specification(p, sources)
    provenance = {'status': STATUS, 'protocol_sha256': PROTOCOL, 'archive_sha256': ARCHIVE,
        'design_sha256': c.DESIGN_SHA256, 'source_checkpoint_member': MEMBER,
        'source_checkpoint_sha256': SOURCE_SHA, 'source_checkpoint_bytes': SOURCE_BYTES,
        'source_selection': 'stopped_composite,final3090reservations; head50weight-equivalence independently audited',
        'completed_head_updates': 50, 'restorer_updates_this_run': 0, 'historical_restorer_updates': 50,
        'operation_reservations': 3090, 'model_tensor_counts': {'base_model': 515, 'head': 4, 'total': 519},
        'tensor_parameters': {'base_model': 94112707, 'head': 8257},
        'tensor_digest_algorithm': 'Sort names;8byte-big-endian JSON[name,dtype,shape]length;header;contiguous raw bytes',
        'combined_digest_namespace': 'base_model/name or head/name,with slash; same length-prefixed algorithm',
        'tensor_digests': digests, 'audit_evidence': evidence,
        'source_checkpoint_full_optimizer_RNG_sampler_retained_separately': True,
        'quality_status': {key: False for key in ('clear_preservation_pass', 'normal_blur_pass', 'dark_retention_pass', 'known_stress_pass')},
        'authoritative_failure': values['transport']['authoritative_quality_stop'],
        'model_qualified': False, 'app_promoted': False, 'current_app_compatible': False,
        'credit': 'Existing CodeFormer and official initializer, retained full-coarse fine-tuned frozen restorer, and separately trained input-only scalar head. Not trained from scratch; no new geometry or identity recovery claim.',
        'upstream_license': 'third_party/codeformer/LICENSE (S-Lab License1.0); original provenance and terms retained',
        'exporter_source_sha256': exporter_sha}
    artifact = {'format': FORMAT, 'base_model': base, 'head': head,
        'controller': specification['controller'], 'composite_specification': specification['composite_specification'],
        'specification': specification, 'provenance': provenance, 'tensor_digests': digests,
        'inference_only': True, 'training_resumable': False, 'model_qualified': False,
        'current_app_compatible': False, 'status': STATUS}
    destination.mkdir()
    weights = destination / 'codeformer_tone_head0050_composite_unqualified.pth'
    temporary = weights.with_name(weights.name + '.part')
    class CappedWriter:
        def __init__(self, stream): self.stream, self.written = stream, 0
        def write(self, payload):
            c.require(self.written + len(payload) <= CAP, 'Inference composite384MiB cap exceeded')
            count = self.stream.write(payload); c.require(count == len(payload), 'Short inference-state write')
            self.written += count; return count
        def flush(self): return self.stream.flush()
        def tell(self): return self.stream.tell()
    with temporary.open('xb') as stream:
        torch.save(artifact, CappedWriter(stream)); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, weights)
    exported_sha = c.sha256(weights)  # Hash before the independent CPU readback load.
    checked = torch.load(weights, map_location='cpu', weights_only=True, mmap=True)
    validate_states(checked['base_model'], checked['head'], torch)
    c.require(set(checked) == set(artifact) and not {'params', 'params_ema'} & set(checked) and
              all(c.equal(checked[key], artifact[key]) for key in artifact if key not in ('base_model', 'head')) and
              tensor_manifest(checked['base_model'], torch) == manifests['base_model'] and
              tensor_manifest(checked['head'], torch) == manifests['head'] and
              state_digest(checked['base_model'], torch) == digests['base_model'] and
              state_digest(checked['head'], torch) == digests['head'], 'Every519exported tensor byte/spec must equal audited source')
    after = source.stat()
    c.require((before.st_size, before.st_mtime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino) and
              c.sha256(source) == SOURCE_SHA and c.sha256(weights) == exported_sha and c.sha256(__file__) == exporter_sha,
              'Source,exporter or exported bytes changed during serialization')
    for pin in evidence.values(): c.require(c.sha256(ROOT / pin['path']) == pin['sha256'], 'Audited evidence changed during export')
    with Path(str(weights) + '.sha256').open('xb') as stream:
        stream.write((exported_sha + '  ' + weights.name + '\n').encode('ascii')); stream.flush(); os.fsync(stream.fileno())
    receipt = {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(), **provenance,
        'format': FORMAT, 'exported_file': str(weights), 'exported_sha256': exported_sha,
        'exported_bytes': weights.stat().st_size, 'tensor_manifest': manifests,
        'all519exported_tensor_bytes_exact_source': True, 'source_weights_only_CPU_mmap': True,
        'exported_readback_weights_only_CPU_mmap': True, 'optimizer_RNG_sampler_in_export': False,
        'params_or_params_ema_alias': False, 'dedicated_inference_loader_implemented': False,
        'model_construction_inference_autograd_optimizer_training_VM_calls': 0}
    c.write_new_json(destination / 'provenance.json', receipt)
    text = '''# Trained composite research candidate — not accepted

This inference-only artifact contains the genuine frozen CodeFormer515 tensors
and the4 trained tone-controller tensors after50 head Adam updates. There were
zero new CodeFormer updates. All519 exported tensor byte values exactly match
the retained stopped checkpoint; its full optimizer/RNG/sampler state remains
in the immutable return. The independently audited head50 and stopped weights
agree. No extra epochs or automatic resume are authorized.

The run FAILED clear and normal-blur preservation, dark retention and the known
TRAIN stress requirements. Personal review of35 sheets covers406 DEV cases and
2 known TRAIN examples. Genuine weights and complete review do not qualify the
model, establish native CCTV ground truth or recover unsupported facial identity.

Format: `codeformer-tone-control-composite-inference-v1`. Tensor sections are
`base_model` and `head`; there is intentionally no `params_ema` or `params` alias.
This is incompatible with standard CodeFormer loaders and the current app.
A dedicated loader must strictly load both sections and reproduce the embedded
RGB256/input-support/feature/SiLU/0.25tanh/clamp/protected-padding/floorPNG policy.
No dedicated inference loader is implemented by this export. Never fall back to
base-only inference, silently reinitialize the head, substitute a covering mask,
apply the app's50percent blend/roundPNG flow, or pass clean/profile/role inputs.

`provenance.json` binds the exact source checkpoint, return archive/protocol,
all tensor digests, independent transport/state/pixel receipts and complete root
visual review. The file checksum uses an ASCII basename with one LF. Existing
CodeFormer architecture/official initialization and S-Lab License1.0 remain
credited; this is a frozen adapted restorer plus a new input-only tone head,
not a model trained from scratch. The app/defaults and all prior assets remain
unchanged. CPU serialization/readback constructed no model, ran no inference,
autograd, optimizer, training or VM workload.
'''
    with (destination / 'README.md').open('xb') as stream: stream.write(text.encode('utf-8'))
    print({'complete': True, 'weights': str(weights), 'sha256': exported_sha,
           'bytes': weights.stat().st_size, 'base_tensors': 515, 'head_tensors': 4, 'model_qualified': False})


if __name__ == '__main__': main()
