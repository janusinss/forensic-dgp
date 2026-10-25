"""Publish the audited research-only fusion checkpoint after actual browser closure.

Inert without --publish. No model, network, VM, training or deletion operation.
All four existing documents receive complete exclusive backups before edits.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'outputs/codeformer_filter_fusion_return_review_20261019_v1'
MODEL = ROOT / 'outputs/codeformer_filter_fusion_training_v1_model'
PUBLICATION = REVIEW / 'publication_v1'
CURRENT = ('PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md',
           'PRACTICAL_OUTPUT_SCOPE.md', 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md')
REPORT = 'CCTV_CODEFORMER_FILTER_FUSION_TRAINING_V1_RETURN_REVIEW.md'
GUIDE = 'CCTV_CODEFORMER_FILTER_FUSION_LOCAL_TEST_GUIDE.md'
PROTOCOL = 'c1f7d8386856bd95ab3af3a8623f33fcecddd4b7e08f0696ea7694a0542c1949'
ARCHIVE = '3d6feaf30d4477fe84bd51c57d560cb10d7b4cd6509db627fc61e03accb35d16'
ARMS = ('original', 'r2', 'fusion50')


def need(value, message):
    if not value:
        raise ValueError(message)


def safe(path, exists=True):
    path = Path(path).absolute()
    need(path.resolve(strict=False).is_relative_to(ROOT), 'Exact workspace path required')
    for item in (path, *path.parents):
        if item.exists() or item.is_symlink():
            row = item.lstat()
            need(not stat.S_ISLNK(row.st_mode) and not getattr(row, 'st_file_attributes', 0) & 1024,
                 'Linked/reparse publication path refused')
    if exists:
        row = path.stat()
        need(stat.S_ISREG(row.st_mode) and row.st_nlink == 1, 'Singly-linked regular file required')
    return path


def pin(path):
    path = safe(path); before = path.stat()
    with path.open('rb') as stream:
        value = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    need((before.st_size, before.st_mtime_ns, before.st_ino) ==
         (after.st_size, after.st_mtime_ns, after.st_ino), 'Evidence changed while hashing')
    return {'bytes': before.st_size, 'sha256': value}


def load(path):
    path = safe(path); need(path.stat().st_size <= 8 * 1024**2, 'Bounded publication JSON required')
    return json.loads(path.read_bytes())


def new(path, data):
    path = safe(path, False); need(not path.exists(), 'Preserve earlier/partial publication')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        need(stream.write(data) == len(data), 'Short exclusive publication write')
        stream.flush(); os.fsync(stream.fileno())


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def link(path, label):
    return '[' + label + '](<' + str(Path(path)).replace('\\', '/') + '>)'


def browser_download_evidence(browser, exported, export_pin):
    """Bind measured comparisons/refusals/downloads; a layout-only receipt fails."""
    for key, expected in (
            ('model_export_receipt_sha256', export_pin['sha256']),
            ('model_weights_sha256', exported['export_sha256']),
            ('model_tensor_sha256', exported['model_tensor_sha256'])):
        need(browser.get(key) == expected, 'Measured browser model binding differs: ' + key)
    calls = browser.get('comparison_calls')
    need(isinstance(calls, list) and calls and
         len({row['case_id'] for row in calls}) == len(calls), 'Distinct actual browser comparison calls required')
    outputs = 0
    for row in calls:
        selection = row['selection']
        need(selection in ('all', *ARMS), 'Actual comparison selection differs')
        expected = ARMS if selection == 'all' else (selection,)
        need(row['status'] == 200 and isinstance(row['outputs'], list) and
             len(row['outputs']) == len(expected) and set(row['outputs']) == set(expected) and
             type(row['neural_forward_calls']) is int and row['neural_forward_calls'] == len(expected),
             'Actual success outputs and measured neural-call delta differ')
        outputs += len(expected)
    need(any(row['selection'] == 'all' for row in calls) and
         type(browser.get('actual_neural_forward_calls')) is int and
         browser['actual_neural_forward_calls'] == outputs, 'All three models and exact measured call total required')
    refusals = browser.get('refusal_calls')
    need(isinstance(refusals, list) and
         {'needs_clearer', 'out_of_scope', 'invalid'} <= {row['input_review'] for row in refusals} and
         all(row['status'] == 422 and type(row['neural_forward_calls']) is int and
             row['neural_forward_calls'] == 0 for row in refusals),
         'Insufficient, out-of-scope and invalid input refusals must make zero neural calls')
    need(browser.get('downloads_verified') is True, 'Physical PNG/bundle downloads must be independently checked')
    record = browser.get('download_audit')
    need(isinstance(record, dict), 'Actual independent download audit binding required')
    download_path = safe(ROOT / record['path'])
    need(pin(download_path) == {key: record[key] for key in ('bytes', 'sha256')}, 'Independent download audit changed')
    downloaded = load(download_path)
    need(downloaded['complete'] is True and downloaded['audit_pass'] is True and
         downloaded['url'] == browser['url'] and
         all(downloaded[key] == browser[key] for key in (
             'model_export_receipt_sha256', 'model_weights_sha256', 'model_tensor_sha256')) and
         downloaded['downloaded_PNGs_checked'] == outputs and
         downloaded['downloaded_bundles_checked'] == len(calls) and
         downloaded['saved_bytes_match_response_hashes'] is True and
         downloaded['RAW_PNG_separate'] is True and
         downloaded['outside_observed_support_RAW_and_PNG_match_input_exact'] is True,
         'Independent downloaded PNG/bundle proof differs from actual comparisons')
    physical = downloaded['physical_files']
    need(isinstance(physical, list) and len({row['path'] for row in physical}) == len(physical) and
         sum(Path(row['path']).suffix.lower() == '.png' for row in physical) == outputs and
         sum(Path(row['path']).suffix.lower() == '.zip' for row in physical) == len(calls),
         'Exact downloaded PNG and bundle files required')
    artifacts = {str(download_path.relative_to(ROOT)): pin(download_path)}
    for row in physical:
        path = safe(ROOT / row['path'])
        value = pin(path)
        need(value == {key: row[key] for key in ('bytes', 'sha256')}, 'Physical downloaded artifact changed')
        artifacts[str(path.relative_to(ROOT))] = value
    raw_browser = downloaded['source_browser_evidence']
    source = safe(ROOT / raw_browser['path'])
    need(source != download_path and
         pin(source) == {key: raw_browser[key] for key in ('bytes', 'sha256')},
         'Separate raw browser evidence binding required; avoid a circular closure hash')
    artifacts[str(source.relative_to(ROOT))] = pin(source)
    source_pins = downloaded['source_pins']
    need(isinstance(source_pins, dict) and
         {'research_filter_restoration_web_v1.py', 'static/research_filter_restoration_v1.js'} <= set(source_pins),
         'Actual comparison route and UI source bindings required')
    for name, value in source_pins.items():
        path = safe(ROOT / name)
        need(pin(path) == value, 'Browser/download source changed: ' + name)
        artifacts[str(path.relative_to(ROOT))] = value
    return artifacts


def evidence(browser_path):
    paths = {'import': REVIEW / 'return_import_v1.json', 'state': REVIEW / 'state_v1/state_audit_v1.json',
        'pixels': REVIEW / 'pixels_v1/audit.json', 'visual': REVIEW / 'root_visual_review_v1.json',
        'export': MODEL / 'model_export_v1.json', 'browser': safe(browser_path)}
    pins = {key: pin(path) for key, path in paths.items()}
    docs = {key: load(path) for key, path in paths.items()}
    for name in ('import', 'state', 'pixels', 'visual', 'export'):
        row = docs[name]
        need(row['complete'] is True and row['archive_sha256'] == ARCHIVE and
             row['protocol_sha256'] == PROTOCOL and row['model_qualified'] is False,
             'Exact completed audit/export graph required: ' + name)
    state, pixels, visual, exported, browser = (docs[k] for k in ('state', 'pixels', 'visual', 'export', 'browser'))
    need(docs['import']['regular_files'] == 9081 and state['audit_pass'] is True and
         state['all443_frozen_tensors_exact'] is True and state['actual_changed_fusion_tensor_count'] == 72 and
         state['saved_checkpoint50_and_stopped_exact_committed_state'] is True and
         state['import_receipt_sha256'] == pins['import']['sha256'], 'Actual50-update state lineage required')
    need(pixels['audit_pass'] is True and pixels['comparison_RAW_PNG_pairs_checked'] == 2390 and
         pixels['TRAIN_guard_RAW_PNG_pairs_checked'] == 48 and pixels['native_unpaired'] == 48 and
         pixels['FINAL_used'] is False and pixels['import_receipt_sha256'] == pins['import']['sha256'],
         'Full saved-pixel audit, native boundary and unused FINAL required')
    need(pixels['preservation_pass'] is False and pixels['parent_dark_retention_pass'] is True and
         pixels['R2_dark_retention_pass'] is False and pixels['additional_interim_stop_pass'] is False and
         pixels['recorded_failure'] == state['history']['original_stop_retained'],
         'Actual stopped quality status must agree across state and pixel audits')
    need(visual['root_personal_review_complete'] is True and visual['reviewed_sheets'] == 63 and
         len(visual['sheets']) == len({r['path'] for r in visual['sheets']}) == 63 and
         visual['pixel_audit_sha256'] == pins['pixels']['sha256'], 'Actual root review of63 sheets required')
    for row in visual['sheets']:
        need(pin(ROOT / row['path']) == {k: row[k] for k in ('bytes', 'sha256')}, 'Reviewed visual sheet changed')
    model_path = safe(Path(exported['model_path']))
    need(model_path == MODEL / 'codeformer_filter_fusion_v1_u50.pth' and
         pin(model_path) == {'bytes': exported['export_bytes'], 'sha256': exported['export_sha256']} and
         exported['source_training_updates'] == 50 and exported['all515_tensor_values_exact'] is True and
         exported['model_tensor_sha256'] == state['checkpoint_readback']['update50']['model_tensor_sha256'] and
         exported['app_default_changed'] is False and exported['params_ema_key_is_loader_compatibility_not_new_EMA'] is True,
         'Actual exact515-tensor research export required')
    need(exported['quality_stop_retained'] == pixels['recorded_failure'], 'Export must retain the actual quality failure')
    for key in ('import', 'state', 'pixels', 'visual'):
        need(exported['review_pins'][key] == pins[key], 'Export audit binding differs: ' + key)
    url = urlparse(browser['url'])
    need(browser['complete'] is True and browser['model_qualified'] is False and
         browser['main_app_default_changed'] is False and browser['layout_widths'] == [375, 768, 1280] and
         browser['unexpected_page_or_console_errors'] == [] and browser['overflow_widths'] == [] and
         url.scheme == 'http' and url.hostname == '127.0.0.1' and url.port is not None and
         url.path == '/restoration-filter-test' and not url.query and not url.fragment,
         'Measured loopback preview, browser/download and responsive closure required')
    need(isinstance(browser['evidence_paths'], list) and browser['evidence_paths'], 'Actual browser artifacts required')
    extra = {str(safe(ROOT / name).relative_to(ROOT)): pin(ROOT / name) for name in browser['evidence_paths']}
    measured = browser_download_evidence(browser, exported, pins['export'])
    need(all(name in extra and extra[name] == value for name, value in measured.items()),
         'Browser closure must bind every physical download, raw browser artifact and serving source')
    return docs, paths, pins, extra


def texts(docs):
    state, pixels, exported, browser = (docs[k] for k in ('state', 'pixels', 'export', 'browser'))
    date = str(datetime.now(timezone(timedelta(hours=8))).date())
    model = link(exported['model_path'], 'exact inference-only model')
    gates = '\n'.join('| ' + name.replace('_', ' ') + ' | ' + ('pass' if pixels[name] else 'FAIL') + ' |'
        for name in ('preservation_pass', 'parent_dark_retention_pass', 'R2_dark_retention_pass', 'additional_interim_stop_pass'))
    report = f'''# Filter-fusion V1: trained research model ready for comparison; quality stopped

{date}: The VM committed 50 Adam updates across 200 microbatches. The {model} contains exactly 515 trained-state tensors. All 72 existing fusion tensors changed; 443 other tensors stayed exact to R2. Snapshot 0 matches R2, and snapshot 50/stopped retain the same model/Adam/sampler/RNG values. These are optimizer updates, not 50 epochs. The finite 151-update experiment stopped at 50; it did not complete 151.

This is our fine-tuned adaptation of CodeFormer, initialized from retained R2. Untouched pretrained CodeFormer remains the benchmark. `params_ema` is a loader compatibility key; no new EMA or architecture is claimed. Model file SHA256: `{exported['export_sha256']}`. Canonical 515-tensor digest: `{exported['model_tensor_sha256']}`.

## Audited result and limits

The strict import covers 9,081 files. The independent CPU state audit passes; the separate saved-pixel audit recomputed 2,390 comparison RAW/PNG pairs plus 48 TRAIN-guard pairs. All 63 planned comparison/refusal sheets were personally reviewed by root. The saved training stop remains `{pixels['recorded_failure']}`.

| Quality requirement | Result |
|---|---|
{gates}

An audit passing proves evidence consistency; it does not reverse a failed quality requirement. The 48 native CCTV observations remain unpaired with no clean-reference scores. The 406 historical DEV slots and 72 mixed filter probes (36 TRAIN/36 DEV) remain separate. Reserved FINAL was not used. User wild1–wild5 have no clean originals and remain visual stress examples. No recovered identity, ethnicity or Zamboanga performance claim is established.

## Local research test

Open [{browser['url']}]({browser['url']}) and follow [{GUIDE}]({GUIDE}). The downloaded 256×256 outputs and comparison bundle were checked on the actual preview route. Layouts 375/768/1280 passed with no unexpected browser errors or horizontal overflow. This proves the functional preview; it is not independent FINAL review, main Auto qualification or user acceptance. The main application default was not changed. A rejected research checkpoint remains available only by explicit selection.

Original checkpoints, immutable training state/Adam/RNG/splits/provenance/failures and earlier storage recovery paths remain retained. Actual new training stays manual and finite on the L4/tmux workflow. Do not rerun or extend this failed recipe unchanged. Completion/detection model development remains deferred until restoration acceptance. The overall goal remains active.

Evidence: {link(REVIEW / 'state_v1/state_audit_v1.json', 'state audit')}, {link(REVIEW / 'pixels_v1/audit.json', 'saved-pixel audit')}, {link(REVIEW / 'root_visual_review_v1.json', 'root visual review')}, {link(MODEL / 'model_export_v1.json', 'research export')}. Browser closure and exact artifact pins are retained in the publication receipt.
'''
    guide = f'''# Test the experimental filter-fusion restoration model

The model is created for explicit research testing. It completed 50 updates and failed preservation/R2-dark retention. Keep this result separate from an accepted improvement. [Audited result and limits]({REPORT}).

1. Open [{browser['url']}]({browser['url']}).
2. Upload one cropped frontal or mildly turned face.
3. Review whether usable facial structure remains before generating a comparison.
4. Compare the explicitly selected experimental output with untouched CodeFormer and retained R2.
5. Download the 256×256 PNG or comparison bundle.

Supply a clearer or less-covered input when the face is block-like, unreadable, too strongly turned or heavily covered. The input review is not a calibrated automatic sufficiency threshold. This restoration test preserves visible appearance; covering detection and hidden-region completion remain separate and deferred.

Exact model: {model}. SHA256: `{exported['export_sha256']}`. The original training checkpoint/Adam/RNG state and failed gates remain preserved. Useful-looking sharpness cannot verify missing identity details. This opt-in route does not replace the main application default. The measured local URL can change after a server restart; use the current launcher receipt rather than guessing a port. No new training command or recipe is authorized by this guide.
'''
    header = f'''# Filter-fusion return audited; experimental model test available — {date}

The actual model is exported for explicit research comparison at [{browser['url']}]({browser['url']}). 50 new Adam updates changed 72 fusion tensors while 443 frozen tensors remained exact. State/import integrity passed; 2,390 comparison RAW/PNG pairs and 48 guard pairs were independently checked, and all 63 planned sheets reviewed. The finite run stopped on quality at update 50: preservation={pixels['preservation_pass']}, R2-dark retention={pixels['R2_dark_retention_pass']}, parent-dark retention={pixels['parent_dark_retention_pass']}. A functional preview is not an accepted improved model, main Auto qualification, user acceptance or independent FINAL review.

[{REPORT}]({REPORT}) and [{GUIDE}]({GUIDE}) retain the failure, model hashes and test limits. This is our CodeFormer adaptation; untouched CodeFormer remains the benchmark. The main app default and original checkpoints/splits/failure evidence remain preserved. Native CCTV stays unpaired; no ethnicity/Zamboanga/identity claim. Training remains manual finite L4/tmux; no unchanged failed rerun. Completion development stays deferred until restoration acceptance. The full goal remains active. Earlier storage cleanup is closed and is not counted again.

Earlier status below remains historical evidence.

---

'''
    return report.encode(), guide.encode(), header.encode()


def publish(browser):
    docs, paths, pins, extra = evidence(browser)
    need(not PUBLICATION.exists() and not any((ROOT / n).exists() for n in (REPORT, GUIDE)), 'Preserve prior publication')
    originals = {name: safe(ROOT / name).read_bytes() for name in CURRENT}
    before = {name: pin(ROOT / name) for name in CURRENT}
    need(all(hashlib.sha256(raw).hexdigest() == before[name]['sha256'] for name, raw in originals.items()), 'Source changed during capture')
    report, guide, header = texts(docs)
    backups = {}; staged = {}
    for name, raw in originals.items():
        backup = PUBLICATION / 'source_backups' / (name + '.before')
        new(backup, raw); need(pin(backup) == before[name], 'Full original backup differs')
        backups[name] = str(backup.relative_to(ROOT))
    for name, raw in originals.items():
        temporary = ROOT / ('.filter_fusion_return_v1.' + name + '.tmp')
        new(temporary, header + raw); staged[name] = temporary
    need(all(pin(path) == pins[key] for key, path in paths.items()) and
         all(pin(ROOT / name) == value for name, value in extra.items()) and
         all(pin(ROOT / name) == before[name] for name in CURRENT), 'Evidence or document changed before publication')
    new(ROOT / REPORT, report); new(ROOT / GUIDE, guide)
    journal = PUBLICATION / 'journal.jsonl'
    records = {}
    for name in CURRENT:
        with safe(journal, False).open('ab') as stream:
            stream.write((json.dumps({'event': 'replace_intent', 'document': name, 'before': before[name]}) + '\n').encode()); stream.flush(); os.fsync(stream.fileno())
        need(pin(ROOT / name) == before[name], 'Document changed immediately before replacement')
        os.replace(staged[name], ROOT / name)
        actual = (ROOT / name).read_bytes()
        need(actual == header + originals[name] and actual[len(header):] == originals[name], 'Original document suffix changed')
        records[name] = {'before': before[name], 'backup': backups[name], 'after': pin(ROOT / name), 'suffix_offset_bytes': len(header)}
        with journal.open('ab') as stream:
            stream.write((json.dumps({'event': 'replace_complete', 'document': name, **records[name]}) + '\n').encode()); stream.flush(); os.fsync(stream.fileno())
    result = {'complete': True, 'protocol_sha256': PROTOCOL, 'archive_sha256': ARCHIVE,
        'source_sha256': pin(__file__)['sha256'], 'evidence_pins': {str(paths[k].relative_to(ROOT)): v for k, v in pins.items()},
        'browser_artifact_pins': extra, 'documents': records, 'new_report': pin(ROOT / REPORT), 'new_guide': pin(ROOT / GUIDE),
        'model_qualified': False, 'main_app_default_changed': False, 'goal_complete': False, 'training_launched': False,
        'all_four_original_documents_fully_backed_up_before_any_replacement': True}
    new(PUBLICATION / 'publication.json', json_bytes(result))
    return {'complete': True, 'publication_receipt': str(PUBLICATION / 'publication.json'), 'model_qualified': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish', action='store_true', required=True)
    parser.add_argument('--browser-closure', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.browser_closure)), flush=True)
