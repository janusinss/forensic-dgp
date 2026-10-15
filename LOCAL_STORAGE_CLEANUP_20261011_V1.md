# Local storage cleanup - 11 October 2026

Completed 2026-10-11T03:42:01.631756+08:00. Scope: `C:\xampp\htdocs\YEAR 4\Testing` on the Windows
computer. The VM was not contacted and no training was launched.

| Item | Verified result |
|---|---:|
| Unique local file storage before cleanup | 192.684 GiB |
| Obsolete result archives removed | 52 |
| Compressed archive bytes removed | 31.971 GiB |
| Net C: free-space increase between recorded endpoints | 23.355 GiB |
| C: free space at publication | 58.571 GiB |

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

The independent audit rehashed 276,655
archived payload members, checked 614,786
retained namespace entries and 11,737
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

Use [plan.json](<C:/xampp/htdocs/YEAR 4/Testing/outputs/local_storage_cleanup_20261011_v1/plan.json>) to find an
archive's `retained_tree` and `member_manifest`. The compressed member manifest
records every original member name, SHA256, source path, size and tar metadata.
The unpacked files are complete recovery copies, independently reverified.
The deleted gzip containers themselves are no longer retained; any later
repackaging gets a new checksum and must not be advertised as the old exact
compressed archive. Legacy download/import commands targeting removed archives
are historical; audit the retained tree or prepare a new recovery package.

Evidence: [independent_audit.json](<C:/xampp/htdocs/YEAR 4/Testing/outputs/local_storage_cleanup_20261011_v1/independent_audit.json>),
[deletion_ledger.jsonl](<C:/xampp/htdocs/YEAR 4/Testing/outputs/local_storage_cleanup_20261011_v1/deletion_ledger.jsonl>) and
[complete_inventory_before.json](<C:/xampp/htdocs/YEAR 4/Testing/outputs/local_storage_cleanup_20261011_v1/complete_inventory_before.json>).
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
