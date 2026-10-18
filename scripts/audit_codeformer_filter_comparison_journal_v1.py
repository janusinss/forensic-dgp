"""Independent inert final journal, lineage and sampled-resource audit.

Reads only a completed hash-bound frozen inference run. No framework, returned
code, model construction, checkpoint deserialization, image scoring or resume.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import stat
import time


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/codeformer_filter_comparison_v1'
WORKER_SHA = '75f96ffe5e7438beca734119c5e763b4ae2fe39ffc6181afc194a5e274252506'
DATA_SHA = '7823847ce7a709308fe1f4d35d35e2c14f752b2d5c303f5fa1cb4b7c8de8c3a8'
REVIEW_SHA = '51bff1593cd33ad2b369a7c20d2b5282ed67521df4b62e17a53f082941fc3b50'
CRITERIA_SHA = 'fd0ae5aa624f74775a65474957f42cf3cc69e0838febac4f1feb9d063340f903'
MODELS_SHA = '5a926fc23c58916374d4ad69bf68fc4066bc469c440d920a2046b09a2761a994'
ARMS = (('original', 'original0'), ('parent', 'parent50'), ('r2', 'candidate50'))
STATE_SHA = {
    'original': '6cb93cee937f67814edcf4625a68bae4bf8f1c2351f7306864ceefa63d4803bd',
    'parent': '92ac81750b86ef1da6f1f7c0480b9e64773fafe07b094e4a5ad7e88b3f7a0c8b',
    'r2': 'a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f'}
CAPS = {'forward_calls': 216, 'wall_seconds': 3600,
    'process_RSS_bytes': 3 * 1024**3, 'minimum_available_RAM_bytes': 768 * 1024**2,
    'minimum_disk_free_bytes': 1024**3, 'output_files': 660,
    'output_payload_bytes': 256 * 1024**2, 'journal_records': 700,
    'journal_bytes': 2 * 1024**2, 'watchdog_period_seconds': 1}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json(payload):
    def unique(items):
        result = {}
        for name, value in items:
            need(name not in result, 'Duplicate JSON key')
            result[name] = value
        return result
    return json.loads(payload.decode('utf-8'), object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def canonical(name):
    need(isinstance(name, str) and bool(name) and ':' not in name and '\\' not in name,
         'Canonical evidence name required')
    path = PurePosixPath(name)
    need(not path.is_absolute() and path.as_posix() == name and
         all(part not in ('', '.', '..') for part in path.parts), 'Escaping evidence name')
    return path


def regular(path, original_exception=False):
    path = Path(path).absolute()
    need(path.resolve(strict=True).is_relative_to(ROOT), 'Evidence outside workspace')
    for part in (path, *path.parents):
        if part == ROOT.parent:
            break
        info = part.lstat()
        need(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024,
             'Linked/reparse evidence')
    info = path.stat()
    allowed = original_exception and path == ROOT / 'outputs/codeformer_restoration_pretrained_v1/codeformer.pth'
    need(stat.S_ISREG(info.st_mode) and info.st_nlink == (3 if allowed else 1),
         'Exact ordinary evidence link count required')
    return path


def fingerprint(path, expected=None, size=None, original_exception=False, cap=2 * 1024**3):
    path = regular(path, original_exception)
    before = path.stat()
    need(before.st_size <= cap and (size is None or before.st_size == size), 'Evidence size differs')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    need((before.st_size, before.st_mtime_ns, before.st_ino) ==
         (after.st_size, after.st_mtime_ns, after.st_ino), 'Evidence changed while hashing')
    need(expected is None or digest == expected, 'Evidence SHA differs')
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': digest, 'bytes': before.st_size}


def resource_sample(value):
    need(set(value) == {'process_RSS_bytes', 'available_RAM_bytes', 'disk_free_bytes'} and
         all(type(n) is int and n > 0 for n in value.values()), 'Invalid saved resource sample')
    need(value['process_RSS_bytes'] <= CAPS['process_RSS_bytes'] and
         value['available_RAM_bytes'] >= CAPS['minimum_available_RAM_bytes'] and
         value['disk_free_bytes'] >= CAPS['minimum_disk_free_bytes'], 'Saved resource sample violates cap')
    return value


def finite_seconds(value):
    need(type(value) in (float, int) and math.isfinite(value) and 0 <= value <= CAPS['wall_seconds'],
         'Invalid bounded seconds')


def journal_closure(events, selected, cases, models):
    """Validate the exact 655-event one-model-at-a-time chronology."""
    need(len(events) == 655 and len(selected) == 72, 'Exact journal/selected census differs')
    position, call = 0, 0
    samples, seconds = [], []
    previous_reservation_seconds = -1.
    for arm, label in ARMS:
        loaded = events[position]; position += 1
        need(set(loaded) == {'event', 'arm', 'state_sha256', 'resources'} and
             loaded['event'] == 'model_loaded_frozen' and loaded['arm'] == arm and
             loaded['state_sha256'] == STATE_SHA[arm], 'Model load journal differs')
        samples.append(resource_sample(loaded['resources']))
        for case in selected:
            call += 1
            reserved, returned, committed = events[position:position + 3]; position += 3
            for event, name in ((reserved, 'forward_reserved'), (returned, 'forward_returned'),
                                (committed, 'output_committed')):
                need(event['event'] == name and event['call'] == call and event['arm'] == arm and
                     event['id'] == case['id'], 'Reserve/return/commit order or identity differs')
            need(set(reserved) == {'event', 'call', 'arm', 'id', 'input_sha256', 'seconds'} and
                 reserved['input_sha256'] == case['input']['sha256'], 'Reserved input pin differs')
            finite_seconds(reserved['seconds'])
            need(reserved['seconds'] >= previous_reservation_seconds, 'Reservation clock reversed')
            previous_reservation_seconds = reserved['seconds']
            need(set(returned) == {'event', 'call', 'arm', 'id', 'seconds', 'resources'},
                 'Return event schema differs')
            finite_seconds(returned['seconds']); seconds.append(returned['seconds'])
            samples.append(resource_sample(returned['resources']))
            need(set(committed) == {'event', 'call', 'arm', 'id', 'evidence'} and
                 committed['evidence'] == cases[case['id']]['outputs'][label]['evidence'],
                 'Committed evidence pin differs')
        finished = events[position]; position += 1
        need(finished == {'event': 'model_finished_frozen', 'arm': arm,
                          'state_sha256': STATE_SHA[arm]}, 'Model finish/state journal differs')
    complete = events[position]; position += 1
    need(position == len(events) and call == 216 and set(complete) == {'event', 'progress', 'resources'} and
         complete['event'] == 'complete' and complete['progress'] == {'reserved_forwards': 216,
         'returned_forwards': 216, 'committed_outputs': 216, 'in_flight': None, 'current_arm': 'r2'},
         'Final progress or terminal journal differs')
    samples.append(resource_sample(complete['resources']))
    return samples, seconds


def audit(manifest_sha, out):
    started = time.monotonic()
    need(len(manifest_sha) == 64 and all(c in '0123456789abcdef' for c in manifest_sha), 'Exact manifest SHA required')
    manifest_path = RUN / 'manifest.json'
    manifest_pin = fingerprint(manifest_path, manifest_sha, cap=8 * 1024**2)
    manifest_raw = manifest_path.read_bytes()
    need(hashlib.sha256(manifest_raw).hexdigest() == manifest_sha, 'Manifest changed after hashing')
    manifest = strict_json(manifest_raw)
    need(manifest['format'] == 'codeformer-filter-frozen-comparison-v1' and manifest['complete'] is True and
         manifest['source_sha256'] == WORKER_SHA and manifest['caps'] == CAPS, 'Final manifest contract differs')
    need(manifest['actual_forwards'] == manifest['committed_outputs'] == 216 and
         manifest['all96_cases_retained'] is True and manifest['input_insufficient_cases_skipped'] == 24 and
         manifest['readable_cases_inferred'] == 28 and manifest['borderline_cases_inferred'] == 44,
         'Final inference census differs')
    need(manifest['actual_training_updates'] == manifest['actual_optimizer_steps'] == manifest['actual_backward_calls'] == 0
         and manifest['paired_metrics_computed'] is False and manifest['quality_gates_changed'] is False and
         manifest['model_qualified'] is False and manifest['native_CCTV_or_identity_or_FINAL_claim'] is False,
         'Scope/quality claims differ')
    finite_seconds(manifest['seconds'])
    files = manifest['files']
    need(len(files) == 657 and sum(p['bytes'] for p in files.values()) + manifest_pin['bytes'] <= CAPS['output_payload_bytes'],
         'Exact bounded final file census differs')
    hashed = []
    for name, pin in files.items():
        canonical(name)
        need(set(pin) == {'path', 'sha256', 'bytes'} and pin['path'] == name, 'File pin schema differs')
        hashed.append(fingerprint(RUN / name, pin['sha256'], pin['bytes'], cap=CAPS['output_payload_bytes']))
    actual = {p.relative_to(RUN).as_posix() for p in RUN.rglob('*') if p.is_file()}
    need(actual == set(files) | {'manifest.json'}, 'Final output namespace not closed')
    for path in RUN.rglob('*'):
        if path.is_dir():
            need(not path.is_symlink() and not getattr(path.lstat(), 'st_file_attributes', 0) & 1024,
                 'Linked/reparse output directory')

    def saved(name, expected=None):
        need(name in files and (expected is None or files[name]['sha256'] == expected), 'Unbound metadata')
        need(files[name]['bytes'] <= 8 * 1024**2, 'Metadata cap exceeded')
        payload = (RUN / name).read_bytes()
        need(len(payload) == files[name]['bytes'] and hashlib.sha256(payload).hexdigest() == files[name]['sha256'],
             'Metadata changed after hashing')
        return strict_json(payload)

    data, review, criteria, model_doc = [saved(n, h) for n, h in (
        ('data_manifest.json', DATA_SHA), ('input_review.json', REVIEW_SHA),
        ('input_review_criteria.json', CRITERIA_SHA), ('model_binding.json', MODELS_SHA))]
    need(criteria['existing_quality_gates_changed'] is False and review['complete'] is True and model_doc['complete'] is True,
         'Prospective criteria/model binding incomplete')
    need(manifest['data_manifest_sha256'] == DATA_SHA and manifest['root_input_review_sha256'] == REVIEW_SHA and
         manifest['input_review_criteria_sha256'] == CRITERIA_SHA and manifest['model_binding_sha256'] == MODELS_SHA,
         'Manifest prospective pin joins differ')
    labels = {r['reference_id'] + '__' + r['profile']: r for r in review['cases']}
    case_map = {r['id']: r for r in manifest['cases']}
    need(len(labels) == len(case_map) == len(data['cases']) == len(manifest['cases']) == 96,
         'All96 reviewed cases required')
    need(Counter(r['decision'] for r in labels.values()) == {'readable': 28, 'borderline': 44, 'insufficient': 24},
         'Input review census differs')
    for source in data['cases']:
        case = case_map[source['id']]
        need(all(case[k] == v for k, v in source.items() if k != 'input_review') and
             case['input_review'] == labels[source['id']] and
             case['appearance_review_required'] == (case['reference_id'] == 'va_ffhq_08025') and
             case['inference_planned'] == (labels[source['id']]['decision'] != 'insufficient') and
             case['optimization_permitted_now'] is False, 'Prospective case/split/review changed')
        need(set(case['outputs']) == ({label for _, label in ARMS} if case['inference_planned'] else set()),
             'Skipped/inferred output membership differs')
    selected = [case_map[r['id']] for r in data['cases'] if case_map[r['id']]['inference_planned']]
    skipped = saved('skipped_cases.json')
    need(skipped == {'complete': True, 'cases': [case_map[r['id']] for r in data['cases'] if not case_map[r['id']]['inference_planned']],
         'refused_from_prospective_input_review': True, 'automatic_threshold_validated': False},
         'Explicit skipped24 records differ')
    models = {r['arm']: r for r in model_doc['models']}
    need(set(models) == {arm for arm, _ in ARMS} and len(model_doc['full515_tensor_schema']) == 515,
         'Exact three515-state models required')
    proofs = saved('model_state_proofs.json')
    need(proofs['complete'] is True and proofs['models'] == manifest['model_state_proofs'] and len(proofs['models']) == 3,
         'State proof document join differs')
    model_files = []
    for (arm, label), proof in zip(ARMS, proofs['models']):
        need(proof == {'arm': label, 'file': models[arm]['file'], 'before_state_sha256': STATE_SHA[arm],
             'after_state_sha256': STATE_SHA[arm], 'exact515_state_unchanged': True, 'forwards': 72},
             'Source-bound frozen515 state proof differs')
        pin = proof['file']; canonical(pin['path'])
        model_files.append(fingerprint(ROOT / pin['path'], pin['sha256'], pin['bytes'], original_exception=arm == 'original'))
        for case in selected:
            output = case['outputs'][label]
            for kind, filename in (('raw', 'raw.npy'), ('png', 'native.png'), ('evidence', 'evidence.json')):
                expected = 'outputs/' + label + '/' + case['id'] + '/' + filename
                need(output[kind] == files[expected], 'Model/case output pin differs')
            ev = saved(output['evidence']['path'])
            need(ev['id'] == case['id'] and ev['reference_id'] == case['reference_id'] and ev['split'] == case['split'] and
                 ev['profile'] == case['profile'] and ev['arm'] == label and ev['model_binding'] == models[arm] and
                 ev['canonical515_state_sha256'] == STATE_SHA[arm] and ev['input_review'] == case['input_review'] and
                 ev['input'] == case['input'] and ev['raw'] == output['raw'] and ev['png'] == output['png'],
                 'Per-output state/input/evidence join differs')
            need(ev['eval_all_modules'] is True and ev['parameters_require_grad'] is False and ev['no_grad'] is True and
                 ev['device'] == 'cpu' and ev['optimizer_updates'] == ev['backward_calls'] == 0 and
                 ev['paired_clean_metrics'] is None and ev['model_qualified'] is False, 'Per-output frozen scope differs')
    runtime = saved('runtime.json')
    need(runtime['device'] == 'cpu' and runtime['torch_threads'] == 4 and runtime['torch_interop_threads'] == 1,
         'Runtime execution policy differs')
    contract = saved('source_contract.json')
    need(contract['source_sha256'] == WORKER_SHA and contract['caps'] == CAPS and contract['inputs'] == 96 and
         contract['skip_insufficient'] == 24 and contract['models'] == dict(ARMS) and contract['scoring'] == 'none' and
         contract['no_completion'] is True and contract['training_updates'] == contract['backward_calls'] == 0 and
         contract['partial_resume_or_automatic_retry'] is False, 'Source contract scope differs')
    fingerprint(ROOT / 'scripts/run_codeformer_filter_comparison_v1.py', WORKER_SHA)
    architecture = [fingerprint(ROOT / p['path'], p['sha256'], p['bytes']) for p in model_doc['architecture_sources']]
    journal_pin = files['journal.jsonl']
    need(journal_pin['bytes'] <= CAPS['journal_bytes'], 'Journal byte cap exceeded')
    journal_raw = (RUN / 'journal.jsonl').read_bytes()
    need(len(journal_raw) == journal_pin['bytes'] and hashlib.sha256(journal_raw).hexdigest() == journal_pin['sha256'],
         'Journal changed after hashing')
    need(journal_raw.endswith(b'\n') and b'\r' not in journal_raw, 'Canonical complete journal lines required')
    events = [strict_json(line) for line in journal_raw.splitlines()]
    samples, timings = journal_closure(events, selected, case_map, models)
    samples.append(resource_sample(runtime['initial_resources']))
    total = sum(p['bytes'] for p in hashed) + manifest_pin['bytes']
    report = {'format': 'frozen-filter-comparison-independent-journal-audit-v1', 'complete': True, 'pass': True,
        'UTC': datetime.now(timezone.utc).isoformat(), 'manifest': manifest_pin,
        'auditor_sha256': fingerprint(Path(__file__))['sha256'], 'journal_rows': len(events),
        'ordered_forward_reservations': 216, 'ordered_returns': 216, 'ordered_output_commits': 216,
        'all3_one_model_at_time_intervals_closed': True, 'unfinished_reserved_calls': 0,
        'all96_cases_retained': True, 'inputs_inferred': 72, 'insufficient_skipped': 24,
        'all3_saved_before_after515_proofs_match_expected': True, 'model_files_freshly_hashed': model_files,
        'architecture_files_freshly_hashed': architecture, 'all658_file_hash_size_namespace_closed': True,
        'output_files': len(hashed) + 1, 'output_payload_bytes': total, 'resources': {
            'persisted_samples': len(samples), 'maximum_sampled_process_RSS_bytes': max(s['process_RSS_bytes'] for s in samples),
            'minimum_sampled_available_RAM_bytes': min(s['available_RAM_bytes'] for s in samples),
            'minimum_sampled_disk_free_bytes': min(s['disk_free_bytes'] for s in samples),
            'samples_within_prospective_caps': True, 'continuous_peak_or_minimum_claim': False},
        'recorded_worker_preclosure_seconds': manifest['seconds'], 'recorded_forward_return_interval_sum_seconds': sum(timings),
        'forward_interval_limit': 'Includes input conversion, native forward, delivery and saved resource check, not pure model profiler timing.',
        'state_limit': 'Saved before/after canonical state proofs are source-bound worker evidence; this audit freshly hashes original files but does not replay the live model state or neural forward.',
        'resource_limit': 'Samples persisted at model loads, forward returns, runtime and completion; background watchdog samples are not all logged and transient peaks may be missed.',
        'pixel_metrics_audit': 'Separate independent saved RAW/PNG/target audit; no image scoring here.',
        'training_updates': 0, 'neural_gradient_optimizer_VM_calls_by_auditor': 0,
        'checkpoint_deserialization_by_auditor': False, 'returned_code_executed': False,
        'model_qualified': False, 'goal_complete': False, 'audit_seconds': time.monotonic() - started}
    out = Path(out).absolute()
    need(out.parent.resolve(strict=True).is_relative_to(ROOT / 'outputs') and not out.exists(), 'New exclusive audit output required')
    for parent in (out.parent, *out.parent.parents):
        if parent == ROOT.parent:
            break
        info = parent.lstat()
        need(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024,
             'Linked audit output parent')
    out.mkdir()
    with (out / 'journal_audit_v1.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    return fingerprint(out / 'journal_audit_v1.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-sha', required=True)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.manifest_sha, args.out)), flush=True)
