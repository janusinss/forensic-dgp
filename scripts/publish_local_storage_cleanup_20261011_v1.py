"""Publish completed local cleanup with recoverable payload paths and old handoff."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/local_storage_cleanup_20261011_v1'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def link(name,label=None):
    return '['+(label or Path(name).name)+'](<' +(ROOT/name).as_posix()+ '>)'


def main():
    receipt=OUT/'publication.json'
    assert not receipt.exists()
    audit=read(OUT/'independent_audit.json')
    plan=read(OUT/'plan.json')
    applied=read(OUT/'apply_receipt.json')
    inventory=read(OUT/'complete_inventory_before.json')
    assert audit['complete'] and plan['complete'] and applied['complete']
    assert audit['unexpected_lost_files']==audit['unrelated_metadata_changes']==0
    assert audit['exact_archive_deletions_checked']==applied['files_removed']==plan['exact_archive_count']
    count=applied['files_removed']
    net_free_increase=applied['observed_net_free_increase_bytes']/1024**3
    bytes_removed=applied['archive_logical_bytes_removed']
    free=shutil.disk_usage(ROOT).free/1024**3
    previous_allocated=inventory['unique_allocated_bytes']/1024**3
    now=datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8))).isoformat()
    report_name='LOCAL_STORAGE_CLEANUP_20261011_V1.md'
    assert not (ROOT/report_name).exists()
    handoff=ROOT/'PROJECT_HANDOFF.md'
    before=handoff.read_bytes()
    (OUT/'PROJECT_HANDOFF.before.md').write_bytes(before)
    report=f'''# Local storage cleanup - 11 October 2026

Completed {now}. Scope: `C:\\xampp\\htdocs\\YEAR 4\\Testing` on the Windows
computer. The VM was not contacted and no training was launched.

| Item | Verified result |
|---|---:|
| Unique local file storage before cleanup | {previous_allocated:.3f} GiB |
| Obsolete result archives removed | {count} |
| Compressed archive bytes removed | {bytes_removed/1024**3:.3f} GiB |
| Net C: free-space increase between recorded endpoints | {net_free_increase:.3f} GiB |
| C: free space at publication | {free:.3f} GiB |

The net free-space reading starts at the completed inventory snapshot and ends
at cleanup completion. Concurrent B-return downloads, unpacked results, B r3
preparation and other C: writes affect this value. It is separate from the exact
compressed bytes removed; no updated full project-size scan is claimed.
The initial sandbox inventory could not list 68 directories.
A read-only elevated completion inventoried all 614,838 files; the initial partial
record and access errors are retained. No filesystem permissions were changed.

## Removal and preserved evidence

Each removed `*-results.tar.gz` was obsolete and had a checksum-matched source
archive and a complete retained unpacked tree. Every regular tar member was
hashed and matched to a local file before selection; symlink/special-member,
missing-source and changed-byte candidates were excluded. Deletion rehashed the
entire retained tree and archive and used an exclusive archive handle while
holding retained files read-only. Exactly the listed regular archives were
deleted. No directory was removed and no unrelated local file was lost.

The independent audit rehashed {audit['full_retained_archive_payload_members_checked']:,}
archived payload members, checked {audit['retained_namespace_files_checked']:,}
retained namespace entries and {audit['critical_protected_file_hashes_checked']:,}
critical file hashes. Retained research-file metadata and immutable critical
hashes are unchanged. Any concurrent edits to the four shared handoff/goal documents are
recorded separately with their complete before/after bytes; cleanup keeps them.
The only other document change was the B VM guide: the other chat added a
521-byte superseded-command notice. Its exact prior bytes are retained in both
the current guide and a hash-matched backup. This exception was verified
separately; research-file checks and all removal/backup requirements remain.
Full checkpoints, training states, code, input/split data, output arrays/PNGs,
metrics, instructions, provenance, logs, research caches and failed gates remain
in the retained trees. Source/checksum/export and failure records stay.

The current A/B bank-comparison archives and new manual B r3 packet remain.
The 36.74 GiB research-cache migration backup and portable image-output recovery
archive remain. The running local app was not stopped or edited; no model was
adopted, retrained or replaced. Full goal status remains ACTIVE.

The first removal execution stopped after 15 verified archives when another
chat started B VM maintenance. The user clarified the separate scopes. A
bounded resume removed the other 37 archives; the original worker, first
15-row ledger and failed-stop records remain alongside the resumed receipt.
The resume's final protection check then stopped on the concurrent B guide
notice. A separate worker verified the exact old/new document bytes and closed
the completed removals with zero additional deletion. The independent audit
verified this execution history without relabeling either stopped removal
worker as successful.

The independent byte audit passed all 276,655 members before its namespace
check stopped on a Git loose-object timestamp. A separate namespace tail reused
that completed byte phase, freshly checked retained payload metadata and the
whole original file inventory, and verified the content IDs of every affected
Git object. Size, inode, device and link count of those objects are unchanged.
The original namespace stop and both checker sources/logs remain; no full byte
audit was repeated and no unexplained file change was accepted.

## Recovering a historical result

Use {link('outputs/local_storage_cleanup_20261011_v1/plan.json')} to find an
archive's `retained_tree` and `member_manifest`. The compressed member manifest
records every original member name, SHA256, source path, size and tar metadata.
The unpacked files are complete recovery copies, independently reverified.
The deleted gzip containers themselves are no longer retained; any later
repackaging gets a new checksum and must not be advertised as the old exact
compressed archive. Legacy download/import commands targeting removed archives
are historical; audit the retained tree or prepare a new recovery package.

Evidence: {link('outputs/local_storage_cleanup_20261011_v1/independent_audit.json')},
{link('outputs/local_storage_cleanup_20261011_v1/deletion_ledger.jsonl')} and
{link('outputs/local_storage_cleanup_20261011_v1/complete_inventory_before.json')}.
Files outside this project were not cleaned. No blanket MD-file, cache,
checkpoint, environment or dataset deletion occurred.

The native file-handle disposition used by the worker follows the
[Microsoft SetFileInformationByHandle documentation](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-setfileinformationbyhandle).
The audit receipts establish what this particular cleanup actually removed.

The user's B-return review, B r3 preparation and B VM offload maintenance were
observed concurrently and kept outside every local cleanup target. B quality
and the next manual commands belong to the other chat's separate review.
This report does not authorize or launch a training run. The VM-space
values in the previous handoff are prior guest snapshots and were not refreshed
by this local maintenance.
'''
    (ROOT/report_name).write_text(report,encoding='utf-8',newline='\n')
    prefix=f'''# Latest local maintenance - 11 October 2026

User-requested Windows project cleanup completed: {count} obsolete compressed
result archives removed after complete unpacked-payload/hash verification.
Compressed bytes removed: {bytes_removed/1024**3:.3f} GiB; C: free space at publication:
{free:.3f} GiB. All unique archived payloads remain in their recorded unpacked
trees, including full checkpoints/states, raw/PNG outputs, data/splits,
provenance, instructions, metrics/logs and failures. Current A/B bank transfers
and verified research-cache/image migration backups remain. No directory,
unrelated file or research cache was deleted. The running app was not stopped.

Historical result gzip containers listed in the cleanup ledger no longer exist
locally; retain their original checksums/metadata as historical evidence. A new
recovery archive must have a new checksum. Independent closure checked all
retained payload members and the 614k-file namespace, with critical hashes
unchanged. No VM connection, training or model adoption occurred. The full
goal remains ACTIVE. The observed B return, B r3 packet and other chat's VM
maintenance files are preserved for their separate workflow. The first stopped
15-file execution, subsequent 37-file resume and zero-deletion closure have
separate retained evidence. The B guide's concurrent superseded-command notice
retains its exact original bytes and is audited separately.
No additional training is launched by this cleanup.

Full removal/recovery map and evidence: {link(report_name)}.
All previous handoff bytes are backed up in the local cleanup receipt folder.
Older sections below retain their dated meaning.

---

'''.encode('utf-8')
    assert handoff.read_bytes()==before
    handoff.write_bytes(prefix+before)
    assert handoff.read_bytes()[len(prefix):]==before
    record=dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
        independent_audit_sha256=sha(OUT/'independent_audit.json'),report_sha256=sha(ROOT/report_name),
        old_handoff_sha256=hashlib.sha256(before).hexdigest(),new_handoff_sha256=sha(handoff),
        original_handoff_bytes_preserved=True,free_bytes_now=shutil.disk_usage(ROOT).free,
        goal_status='active',VM_connections_or_training_calls=0)
    with receipt.open('x',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(record,indent=2)+'\n')
    print(dict(complete=True,removed_archives=count,archive_bytes_removed_GiB=bytes_removed/1024**3,
        observed_net_free_increase_GiB=net_free_increase,free_GiB=free,
        report=report_name,goal_status='active'),flush=True)


if __name__=='__main__':
    main()
