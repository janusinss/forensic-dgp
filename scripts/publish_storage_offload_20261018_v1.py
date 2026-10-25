"""Explicit, once-only storage documentation publication after all three closures.

Inert unless --publish and the independently supplied pass-3 closure SHA256 are
both provided. No network, model, training, research-payload hashing or deletion.
Only the four named root Markdown documents may be replaced, after all four
exact original byte strings have exclusive, fsynced, hash-verified backups.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import time
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT = Path(r'C:\xampp\htdocs\YEAR 4\Testing')
OUT = 'outputs/storage_offload_20261018_v1'
PUBLICATION = OUT + '/publication_v1'
REPORT = OUT + '/STORAGE_OFFLOAD_REPORT.md'
PASS3_CLOSURE = OUT + '/pass3/independent_local_closure_v1.json'
PROTECTED = OUT + '/protected_before_v1.json'
PROTECTED_SHA = '030c95120218678865fcdd11aff5e69bcd2b874fc6aa9cd197c1d6ce747b5f96'
DOCUMENTS = {
    'PROJECT_HANDOFF.md': (923348, '30284d319a5f378ae400a0b4b75729a2dc496624198351c7798ae511296b54d1'),
    'SYSTEM_WORKFLOW_AND_GOAL.md': (421922, '788b2a0b4b588778fe82a05bfa71d04abdb9ed47e7a5bf126b13d9eadc83e61a'),
    'PRACTICAL_OUTPUT_SCOPE.md': (354813, '1a52d4c39ab20d703061bbcea8a9f3e88f33ece21e22bf042be031e16243e443'),
    'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md': (102815, '816237ed47ca141fb75c8a938563e6c237f5379edf1c128b734f4f18b6cf2730'),
}
PASS1_CLOSURE = OUT + '/independent_local_closure_v1.json'
PASS1_SHA = '29311fde6ef93e44e540a9935790b8f92ca71eb528049ccb0527c80e3f6049bb'
VM_CLOSURE = 'outputs/vm_archive_offload_20261018_v1/postapply_recovery_r1_independent_final_v1.json'
VM_SHA = 'c58f304b50adfaadc5025715d139ee58fe5bd6e98a460f564635611e3f885870'
VM_FULL_SHA = 'a13b585022e21e2c1e130ce99018bcae783784b2d6cac0698103da9d10cfae2c'
VM_PROBE = 'outputs/vm_archive_offload_20261018_v1/final_readonly_probe_v1/receipt.json'
VM_PROBE_SHA = '6359367470b058589cfdd3d564e0090504d81508d6069908f2ec290ee5ce028e'
RETURN = 'outputs/cctv-codeformer-filter-fusion-training-v1-results.tar.gz'
RETURN_SHA = '3d6feaf30d4477fe84bd51c57d560cb10d7b4cd6509db627fc61e03accb35d16'
PASS1_PATHS = {
    'outputs/cctv-dgp-bank-comparison-v1-r3-A-results.tar.gz',
    'outputs/cctv-dgp-bank-comparison-v1-r4-B-results.tar.gz',
    'outputs/cctv-dgp-bank-photometric-guard-v1-results.tar.gz',
}
PASS3_PATHS = {
    'outputs/cctv-dgp-bank-comparison-v1-results.tar.gz',
    'outputs/cctv-dgp-bank-comparison-v1-r2-B-results.tar.gz',
    'outputs/cctv-dgp-bank-comparison-v1-r3-B-results.tar.gz',
}
JSON_CAP = 8 * 1024**2


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def bounded(relative, *, existing=True):
    require(isinstance(relative, str) and '\\' not in relative, 'Portable relative path required')
    parts = PurePosixPath(relative).parts
    require(parts and not PurePosixPath(relative).is_absolute() and all(p not in ('', '.', '..') for p in parts),
            'Root-bounded relative path required')
    path = ROOT.joinpath(*parts)
    require(path.parent.resolve().is_relative_to(ROOT), 'Parent outside exact workspace')
    for parent in (ROOT, *path.relative_to(ROOT).parents):
        if parent == Path('.'):
            continue
        candidate = parent if parent.is_absolute() else ROOT / parent
        if candidate.exists():
            info = candidate.lstat()
            require(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 0x400,
                    'Linked/reparse ancestor refused')
    if existing:
        info = path.lstat()
        require(stat.S_ISREG(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 0x400,
                'Ordinary non-reparse file required')
        require(info.st_nlink == 1, 'Exactly one regular-file link required')
    return path


def metadata(path):
    info = path.lstat()
    return {'bytes': info.st_size, 'mtime_ns': info.st_mtime_ns, 'inode': info.st_ino,
            'nlink': info.st_nlink, 'dev': info.st_dev}


def read(relative, expected_sha=None, cap=JSON_CAP):
    path = bounded(relative)
    before = metadata(path)
    require(before['bytes'] <= cap, 'Bounded documentation/evidence read cap')
    raw = path.read_bytes()
    require(metadata(path) == before, 'File changed during evidence read')
    actual = digest(raw)
    require(expected_sha is None or actual == expected_sha, 'Evidence SHA256 mismatch: ' + relative)
    return raw, {'path': relative, 'bytes': len(raw), 'sha256': actual}, before


def read_json(relative, expected_sha=None):
    raw, pin, _ = read(relative, expected_sha)
    value = json.loads(raw.decode('utf-8-sig'))
    require(isinstance(value, dict), 'JSON evidence object required')
    return value, pin


def follow(pin):
    value, actual = read_json(pin['path'], pin['sha256'])
    require(actual['bytes'] == pin['bytes'], 'Evidence byte count mismatch')
    return value, actual


def protected_closed(apply):
    for key in ('protected_preapply', 'protected_postapply'):
        row = apply[key]
        require(row.get('complete') is True and row.get('files') == 3630 and
                row.get('logical_bytes_hashed') == 48535469284 and row.get('exact_before_pins_matched') is True,
                'Completed full protected before/after hash checks required')
    require(apply['evidence']['protected_before']['sha256'] == PROTECTED_SHA,
            'Original protected manifest binding required')
    require(apply.get('VM_calls') == 0 and apply.get('training_calls') == 0,
            'Storage-only local apply required')


def closed_evidence(pass3_sha):
    require(len(pass3_sha) == 64 and all(c in '0123456789abcdef' for c in pass3_sha),
            'Lowercase exact pass-3 closure SHA256 required')
    protected, protected_pin = read_json(PROTECTED, PROTECTED_SHA)
    require(protected.get('complete') is True and protected.get('files') == 3630 and
            len(protected['protected']) == 3630 and protected.get('latest_return_full_hash_verified') is True,
            'Exact full protected manifest required')
    first, first_pin = read_json(PASS1_CLOSURE, PASS1_SHA)
    third, third_pin = read_json(PASS3_CLOSURE, pass3_sha)
    for closure, count, size in ((first, 3, 15463424100), (third, 3, 6661664658)):
        require(closure.get('complete') is True and closure.get('independently_verified') is True and
                closure.get('files_removed') == count and closure.get('logical_bytes_removed') == size,
                'Exact independently completed local phase required')
    first_apply, first_apply_pin = follow(first['apply_receipt'])
    third_apply, third_apply_pin = follow(third['apply_receipt'])
    require(third_apply_pin['path'] == OUT + '/pass3/apply_receipt_v1.json', 'Exact pass-3 actual apply required')
    for closure, applied, paths, size in ((first, first_apply, PASS1_PATHS, 15463424100),
                                        (third, third_apply, PASS1_PATHS | PASS3_PATHS, 6661664658)):
        require(applied.get('complete') is True and applied.get('files_removed') == 3 and
                applied.get('logical_bytes_removed') == size, 'Actual completed local apply totals required')
        require(applied['observed_free_increase_bytes'] == applied['free_bytes_after'] - applied['free_bytes_before'] and
                closure['observed_free_increase_bytes'] == applied['observed_free_increase_bytes'],
                'Actual local measured space mismatch')
        require(applied['namespace_after']['complete'] is True and
                set(applied['namespace_after']['exact_missing']) == paths and
                not applied['namespace_after']['retained_metadata_changes'], 'Exact closed namespace required')
        protected_closed(applied)
    require(first_apply['observed_free_increase_bytes'] == 15431499776, 'Exact pass-1 measurement mismatch')
    require(third_apply['evidence']['plan_sha256'] == 'ec5b59e1b537c61f0a485ea1a911bf86af551db6bf1dd55d78a051d96d054e2e',
            'Exact pass-3 actual plan binding required')
    for applied, expected in ((first_apply, 15463424100), (third_apply, 6661664658)):
        verify, _ = follow(applied['evidence']['streamed_verify_receipt'])
        require(verify.get('complete') is True and verify.get('mode') == 'verify' and
                verify.get('streamed_downloaded_SHA256_verified') is True and verify.get('child_terminal') is True and
                verify.get('rclone_exit_code') == 0 and verify.get('expected_archives') == 3 and
                verify.get('expected_bytes') == expected, 'Terminal downloaded-content SHA256 proof required')
    vm, vm_pin = read_json(VM_CLOSURE, VM_SHA)
    vm_full, vm_full_pin = follow(vm['actual_completed_independent_full_audit'])
    probe, probe_pin = read_json(VM_PROBE, VM_PROBE_SHA)
    require(vm_full_pin['sha256'] == VM_FULL_SHA and vm.get('complete') is True and
            vm.get('independently_verified') is True and
            vm.get('all_actual_payload_ledger_guest_receipt_workload_and_final_readonly_probe_closed') is True and
            not vm.get('unresolved_findings') and vm.get('apply_replayed') is False,
            'Exact independent recovered VM closure required')
    require(vm.get('files_removed') == 6 and vm.get('logical_bytes_removed') == 16692148382 and
            vm.get('observed_VM_free_increase_bytes') == 16692150272 and
            vm.get('free_bytes_at_final_probe') == 31808192512 and
            vm_full.get('protected_hash_and_identity_records_exact_before_after_equal') == 47753 and
            vm_full.get('namespace_entries_exact_before_after_equal') == 506684 and
            probe.get('complete') is True and probe['actual_probe']['disk']['free_bytes'] == 31808192512,
            'Actual completed VM protection and measurement mismatch')
    for name, (size, sha) in DOCUMENTS.items():
        require(protected['protected'][name]['bytes'] == size and protected['protected'][name]['sha256'] == sha,
                'Protected original document differs')
    archive = protected['protected'][RETURN]
    require(archive['bytes'] == 4408715848 and archive['sha256'] == RETURN_SHA and
            metadata(bounded(RETURN)) == {k: archive[k] for k in ('bytes', 'mtime_ns', 'inode', 'nlink', 'dev')},
            'Latest archive protected identity differs; no new payload hash is performed')
    return {'protected': protected_pin, 'first': first, 'first_pin': first_pin, 'first_apply': first_apply,
            'first_apply_pin': first_apply_pin, 'third': third, 'third_pin': third_pin,
            'third_apply': third_apply, 'third_apply_pin': third_apply_pin, 'vm': vm,
            'vm_pin': vm_pin, 'vm_full_pin': vm_full_pin, 'probe_pin': probe_pin}


def header(e, newline='\n'):
    third = e['third_apply']
    text = f'''# Storage cleanup closed — 18 October 2026

| Completed phase | Exact compressed copies removed | Observed additional free space |
|---|---:|---:|
| Local pass 1 | 3; 15,463,424,100 bytes | 15,431,499,776 bytes |
| Existing L4 VM | 6; 16,692,148,382 bytes | 16,692,150,272 bytes |
| Local pass 3 | 3; 6,661,664,658 bytes | {third['observed_free_increase_bytes']:,} bytes |

The two local passes removed **22,125,088,758 logical bytes** in 6 archived copies; C: free space measured **{third['free_bytes_after']:,} bytes** after pass 3. VM free space measured **31,808,192,512 bytes** at its final probe. Earlier 52 GB/12 GB offloads are excluded. [Storage report and exact recovery paths](outputs/storage_offload_20261018_v1/STORAGE_OFFLOAD_REPORT.md).

Original checkpoints, trained states, research caches, splits, provenance, failures, migration backups, unpacked research and current R2/filter-fusion assets remain preserved under the completed protection audits. The **4,408,715,848-byte filter-fusion return** (SHA256 `{RETURN_SHA}`) reports a **failure after 50 completed updates**; independent state/output/visual review is pending. This cleanup changed no model, app default or ML behavior and launched no new training. The restoration goal remains active; actual training stays manual on the VM. Continue with the independent return audit before choosing another training experiment.

Earlier status below is retained as historical evidence.

---

'''
    return text.replace('\n', newline).encode('utf-8')


def report(e):
    first, third, vm = e['first_apply'], e['third_apply'], e['vm']
    total_observed = first['observed_free_increase_bytes'] + third['observed_free_increase_bytes']
    return f'''# Local → VM → local storage offload — 18 October 2026

All three ordered storage phases are independently closed. Only exact inactive compressed result copies were removed. This is storage completion; model quality and the thesis goal remain unqualified by these operations.

| Phase | Copies removed | Logical bytes removed | Observed additional free space | Free space after apply |
|---|---:|---:|---:|---:|
| Local pass 1 | 3 | 15,463,424,100 | 15,431,499,776 | 37,515,943,936 |
| Existing L4 VM | 6 | 16,692,148,382 | 16,692,150,272 | 31,823,777,792 |
| Local pass 3 | 3 | 6,661,664,658 | {third['observed_free_increase_bytes']:,} | {third['free_bytes_after']:,} |

The two new local passes removed **22,125,088,758 logical bytes**; their separate measured free-space increases total **{total_observed:,} bytes**. Local free space immediately before pass 3 was **{third['free_bytes_before']:,} bytes**. These timestamped measurements differ from logical sizes because other writes continue. The VM's final read-only probe measured **31,808,192,512 bytes free**; its allocated bytes removed were **16,692,187,136**. Windows and VM capacity are separate disks. Previous roughly 52 GB and 12 GB Drive offloads are excluded from these new totals.

Each local pass retained all unpacked research members, verified streamed Drive content SHA256 hashes, and completed before/after hashes for **3,630 protected files**, including all **1,022 model files** and the newest return. The actual namespace checks found only the exact six intended local archive paths missing and no retained metadata changes. The VM's recovered independent audit compared **506,684 research namespace entries** and **47,753 protected hash/identity records**. The later VM probe checked research metadata and absent HOME paths; it did not repeat the protected payload hashes established by the separate apply audit.

The VM SSH transport dropped after the six unlink operations. The guest completed the operation and retained durable evidence. Read-only downloads recovered its receipt, payload, ledger and workload records. Independent review closed those actual results without replaying the apply. Original failed checks, the prior report-cap revision and the SSH failure/recovery evidence remain retained separately.

The six removed VM HOME copies have full hash-matched Windows backups: the AdaIN gradient-isolation, tone-control training, normal-tone training, FP32 comparison, adaptation V2 and adaptation V1 result archives. Those Windows copies were not selected for local pass 3 removal. All research roots, original/trained checkpoints, research and training caches, splits, provenance, failure records, migration backups and current R2/filter-fusion assets remain protected. No model source, trained weight, app restoration default or evaluation split changed; no new inference, gradient, optimizer or training activity was initiated by this cleanup. New maintenance helpers and evidence files document storage operations.

The latest filter-fusion return remains local: **4,408,715,848 bytes**, SHA256 `{RETURN_SHA}`. Its export/supervisor records report **50 completed updates and a trainer failure**, with state checkpoints retained. Independent archive/state/output and visual audits remain pending; export completion is not restoration acceptance. Training remains restricted to the user's finite manual VM/tmux workflow. The next research action is the independent returned-result audit. Completion-model development stays deferred until restoration acceptance.

## Closed evidence

1. [First local independent closure](independent_local_closure_v1.json), SHA256 `{e['first_pin']['sha256']}`; actual apply SHA256 `{e['first_apply_pin']['sha256']}`.
2. [VM independent final closure](../vm_archive_offload_20261018_v1/postapply_recovery_r1_independent_final_v1.json), SHA256 `{e['vm_pin']['sha256']}`; full independent recovered audit SHA256 `{e['vm_full_pin']['sha256']}`.
3. [VM final read-only probe](../vm_archive_offload_20261018_v1/final_readonly_probe_v1/receipt.json), SHA256 `{e['probe_pin']['sha256']}`.
4. [Final local independent closure](pass3/independent_local_closure_v1.json), SHA256 `{e['third_pin']['sha256']}`; actual apply SHA256 `{e['third_apply_pin']['sha256']}`.

## Archive and restore locations

Pass 1 Drive folder: `thesis_archive:FORENSIC_DGP_ARCHIVE/Testing/20261018_v1/pass1/outputs/` — bank comparison r3-A, r4-B and photometric-guard result TAR copies.

Pass 3 Drive folder: `thesis_archive:FORENSIC_DGP_ARCHIVE/Testing/20261018_v1/pass3/outputs/` — bank comparison V1, r2-B and r3-B result TAR copies. Exact names, sizes and hashes are preserved in each pass's `plan_v1.json`. The VM Windows backup paths are preserved in `../vm_archive_offload_20261018_v1/windows_backup_proof_v1/scope.json`.

Future PowerShell restore pattern (documentary; not executed by publication):

```powershell
& 'C:\\Users\\janus\\AppData\\Local\\Programs\\rclone\\rclone.exe' copyto 'thesis_archive:FORENSIC_DGP_ARCHIVE/Testing/20261018_v1/pass1/outputs/cctv-dgp-bank-comparison-v1-r3-A-results.tar.gz' 'C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs\\cctv-dgp-bank-comparison-v1-r3-A-results.tar.gz' --config 'C:\\Users\\janus\\AppData\\Roaming\\rclone\\forensic-dgp-drive-manual.conf' --immutable --checksum
```

After restore, independently hash the full archive against its plan. This example's expected SHA256 is `2720c5838e6a5da748783ae0e476c2bf2210693434f8b7d25f7e7404dbb26222`.

Intentional documentation prepends happen only after all completed deletion protection checks. `publication_v1/publication.json` records all four exact original-byte backups, new header offsets, suffix hashes and post-publication file hashes separately. Storage completion does not complete the model or overall goal.
'''.encode('utf-8')


def exclusive(path, raw):
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    require(path.read_bytes() == raw, 'Exclusive written bytes differ')


def publish(pass3_sha):
    start = time.monotonic()
    require(ROOT == EXPECTED_ROOT.resolve() and ROOT.is_dir(), 'Exact authorized Windows workspace required')
    evidence = closed_evidence(pass3_sha)
    require(not bounded(PUBLICATION, existing=False).exists() and not bounded(REPORT, existing=False).exists(),
            'Existing publication retained; never replay automatically')
    originals = {}
    for name, (size, sha) in DOCUMENTS.items():
        raw, pin, info = read(name, sha)
        require(len(raw) == size, 'Exact original document size required')
        originals[name] = (raw, pin, info)
    publication = bounded(PUBLICATION, existing=False)
    publication.mkdir(exist_ok=False)
    backups = publication / 'source_backups'
    backups.mkdir(exist_ok=False)
    journal_path = publication / 'journal.jsonl'
    with journal_path.open('x', encoding='utf-8', newline='\n') as journal:
        def event(data):
            require(time.monotonic() - start <= 600, 'Finite ten-minute documentation publication cap')
            journal.write(json.dumps(data, sort_keys=True, separators=(',', ':')) + '\n')
            journal.flush()
            os.fsync(journal.fileno())
        event({'event': 'closed_evidence_validated', 'UTC': datetime.now(timezone.utc).isoformat(),
               'pass3_closure': evidence['third_pin'], 'intentional_documents': list(DOCUMENTS),
               'no_research_training_or_model_calls': True})
        backup_pins = {}
        for name, (raw, pin, info) in originals.items():
            require(metadata(bounded(name)) == info, 'Original document changed before backup')
            path = backups / (name + '.before')
            exclusive(path, raw)
            require(digest(path.read_bytes()) == pin['sha256'], 'Exact backup hash mismatch')
            backup_pins[name] = {'path': path.relative_to(ROOT).as_posix(), **{k: pin[k] for k in ('bytes', 'sha256')}}
            event({'event': 'original_backup_complete', 'source': pin, 'backup': backup_pins[name]})
        require(len(backup_pins) == 4 and all((backups / (name + '.before')).read_bytes() == value[0]
                                           for name, value in originals.items()), 'All four exact backups required before any root edit')
        event({'event': 'all_four_backups_fsynced_hash_verified_before_any_document_edit'})
        report_raw = report(evidence)
        exclusive(bounded(REPORT, existing=False), report_raw)
        report_pin = {'path': REPORT, 'bytes': len(report_raw), 'sha256': digest(report_raw)}
        staged = {}
        for name, (raw, pin, info) in originals.items():
            newline = '\r\n' if b'\r\n' in raw else '\n'
            prepend = header(evidence, newline)
            body = prepend + raw
            temp = bounded('.storage_offload_20261018_v1.' + name + '.tmp', existing=False)
            require(not temp.exists(), 'Exclusive publication temporary path already exists')
            exclusive(temp, body)
            staged[name] = (temp, body, len(prepend))
        # Every current original and every complete backup must still match before
        # the first replacement; the journal records any partial interruption.
        for name, (raw, pin, info) in originals.items():
            current, _, actual = read(name, pin['sha256'])
            require(current == raw and actual == info, 'Original document changed before publish')
        published = {}
        for name, (temp, body, offset) in staged.items():
            old, pin, info = originals[name]
            require(metadata(bounded(name)) == info, 'Concurrent protected-document mutation refused')
            event({'event': 'intentional_document_prepend_intent', 'path': name, 'original': pin,
                   'backup': backup_pins[name], 'new_header_bytes': offset})
            target = bounded(name)
            require(target.name in DOCUMENTS and target.parent == ROOT, 'Exact four-document replacement scope')
            os.replace(temp, target)
            new = target.read_bytes()
            require(new == body and new[offset:] == old and digest(new[offset:]) == pin['sha256'],
                    'Full original-byte suffix preservation failed')
            row = {'path': name, 'bytes': len(new), 'sha256': digest(new), 'header_bytes': offset,
                   'preserved_suffix_bytes': len(old), 'preserved_suffix_sha256': digest(new[offset:]),
                   'backup': backup_pins[name], 'original_metadata': info, 'post_metadata': metadata(target),
                   'all_original_bytes_preserved_at_exact_offset': True}
            published[name] = row
            event({'event': 'intentional_document_prepend_complete', **row})
        result = {'format': 'storage-offload-four-document-publication-v1', 'complete': True,
                  'UTC': datetime.now(timezone.utc).isoformat(), 'root': str(ROOT),
                  'source': {'path': Path(__file__).relative_to(ROOT).as_posix(),
                             'sha256': digest(Path(__file__).read_bytes())},
                  'closed_evidence': {k: evidence[k] for k in ('protected', 'first_pin', 'first_apply_pin',
                                      'third_pin', 'third_apply_pin', 'vm_pin', 'vm_full_pin', 'probe_pin')},
                  'report': report_pin, 'documents': published, 'all_four_backups_preceded_all_edits': True,
                  'intentional_doc_changes_after_completed_deletion_protection_checks': True,
                  'local_logical_bytes_removed': 22125088758, 'local_files_removed': 6,
                  'local_observed_free_increase_sum_bytes': evidence['first_apply']['observed_free_increase_bytes'] +
                                                          evidence['third_apply']['observed_free_increase_bytes'],
                  'free_C_bytes_at_publication': shutil.disk_usage(ROOT).free,
                  'VM_logical_bytes_removed': 16692148382, 'VM_files_removed': 6,
                  'network_calls': 0, 'research_payload_hash_calls': 0, 'model_or_training_calls': 0,
                  'model_or_overall_goal_completion': False, 'seconds': time.monotonic() - start}
        exclusive(publication / 'publication.json', (json.dumps(result, indent=2) + '\n').encode('utf-8'))
        event({'event': 'publication_complete', 'report': report_pin, 'documents': len(published)})
    print(json.dumps({'complete': True, 'documents': 4, 'publication': PUBLICATION + '/publication.json',
                      'report': report_pin, 'model_or_overall_goal_completion': False}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish', action='store_true', help='Explicit once-only four-document publication')
    parser.add_argument('--pass3-closure-sha256')
    args = parser.parse_args()
    if not args.publish or not args.pass3_closure_sha256:
        parser.error('Inert without both --publish and --pass3-closure-sha256; no publication performed')
    publish(args.pass3_closure_sha256)


if __name__ == '__main__':
    main()
