"""Verify redundant historical result archives against complete retained trees.

No deletion, model import, inference, training, VM connection or backup removal.
"""
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/local_storage_cleanup_20261011_v1'
BLOCKED_PREFIXES = ('outputs/local_storage_cleanup_', 'outputs/cctv_dgp_local_research_cache_backup_',
                    'outputs/cctv_dgp_vm_migration_backup_', 'outputs/cctv_dgp_vm_output_offload_')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def bounded(value):
    p = ROOT / value
    assert p.resolve().is_relative_to(ROOT.resolve()) and p.resolve() != ROOT.resolve()
    assert not p.is_symlink()
    for parent in [p, *p.parents]:
        if parent == ROOT.parent:
            break
        assert not getattr(parent.lstat(), 'st_file_attributes', 0) & 0x400, str(parent)
    return p


def metadata(path):
    s = path.lstat()
    assert stat.S_ISREG(s.st_mode)
    return dict(bytes=s.st_size, dev=s.st_dev, inode=s.st_ino, nlink=s.st_nlink, mtime_ns=s.st_mtime_ns)


def main():
    assert not (OUT/'plan.json').exists() and not (OUT/'archive_checks').exists()
    inventory = json.loads((OUT/'complete_inventory_before.json').read_text())
    assert inventory['complete']
    with gzip.open(OUT/'complete_file_inventory_before.json.gz', 'rt', encoding='utf-8') as f:
        files = json.load(f)
    suffix_index = defaultdict(list)
    for row in files:
        if row['path'].startswith('outputs/') and not row['path'].startswith(BLOCKED_PREFIXES):
            suffix_index[(PurePosixPath(row['path']).name, row['bytes'])].append(row['path'])
    archives = [r for r in files if r['path'].startswith('outputs/') and r['path'].count('/') == 1
                and r['path'].endswith('-results.tar.gz') and r['bytes'] >= 64*1024**2
                and 'bank-comparison' not in r['path']]
    archives.sort(key=lambda r: (-r['bytes'], r['path']))
    (OUT/'archive_checks').mkdir()
    start = time.monotonic()
    digest_cache = {}
    candidates = []
    retained = []
    with (OUT/'checks_progress.jsonl').open('x', encoding='utf-8') as log:
        for number, row in enumerate(archives, 1):
            archive_start = time.monotonic()
            record = dict(archive=row['path'], verified=False, original_inventory=row)
            try:
                p = bounded(row['path'])
                before = metadata(p)
                assert all(before[k] == row[k] for k in before)
                assert before['nlink'] == 1, 'Archive already shares storage; retain'
                sidecar = p.with_name(p.name+'.sha256')
                assert sidecar.is_file(), 'Missing archive checksum receipt; retain'
                expected = sidecar.read_text(encoding='utf-8-sig').replace('\r','').strip().split()[0]
                assert len(expected) == 64 and sha(p) == expected, 'Archive checksum differs; retain'
                seen = set()
                possible = None
                members = []
                with tarfile.open(p, 'r|gz') as tar:
                    for member in tar:
                        rel = PurePosixPath(member.name)
                        assert not rel.is_absolute() and '..' not in rel.parts and '\\' not in member.name
                        assert member.name not in seen
                        seen.add(member.name)
                        if member.isdir():
                            continue
                        assert member.isfile(), 'Archive has a link/special member; retain'
                        assert member.size >= 0
                        if possible is None:
                            options = []
                            for local in suffix_index[(rel.name, member.size)]:
                                local_parts = PurePosixPath(local).parts
                                for strip in [0, 1]:
                                    parts = rel.parts[strip:]
                                    if not parts or len(parts) >= len(local_parts):
                                        continue
                                    if tuple(local_parts[-len(parts):]) == tuple(parts):
                                        base = PurePosixPath(*local_parts[:-len(parts)]).as_posix()
                                        if base.startswith('outputs/') and not base.startswith(BLOCKED_PREFIXES):
                                            options.append((base, strip))
                            possible = sorted(set(options), key=lambda q:(not ('return' in q[0]), q[1], q[0]))
                            assert possible, 'No complete retained-tree candidate; retain'
                        stream = tar.extractfile(member)
                        h = hashlib.sha256()
                        for block in iter(lambda:stream.read(1024**2), b''):
                            h.update(block)
                            assert time.monotonic()-archive_start < 900, 'Finite per-archive verification stop'
                        digest = h.hexdigest()
                        matches = []
                        for base, strip in possible:
                            relative = PurePosixPath(base, *rel.parts[strip:]).as_posix()
                            try:
                                local = bounded(relative)
                                details = metadata(local)
                                if details['bytes'] != member.size:
                                    continue
                                key = (relative, details['inode'], details['mtime_ns'], details['bytes'])
                                if key not in digest_cache:
                                    digest_cache[key] = sha(local)
                                if digest_cache[key] == digest:
                                    matches.append((base, strip))
                            except (OSError, AssertionError):
                                continue
                        possible = matches
                        assert possible, 'At least one archive file lacks a complete matching retained tree; retain'
                        members.append(dict(name=member.name, bytes=member.size, sha256=digest,
                            tar_mtime=member.mtime, tar_mode=member.mode, tar_uid=member.uid,
                            tar_gid=member.gid, tar_uname=member.uname, tar_gname=member.gname,
                            pax_headers=member.pax_headers))
                        assert time.monotonic()-start < 3600, 'Finite total preparation stop'
                assert members and metadata(p) == before
                base, strip = possible[0]
                for member in members:
                    relative = PurePosixPath(base, *PurePosixPath(member['name']).parts[strip:]).as_posix()
                    local = bounded(relative)
                    details = metadata(local)
                    key = (relative, details['inode'], details['mtime_ns'], details['bytes'])
                    assert digest_cache[key] == member['sha256']
                    member.update(retained_path=relative, retained_metadata=details)
                manifest = OUT/'archive_checks'/(p.name+'.members.json.gz')
                with gzip.open(manifest, 'xb', compresslevel=1) as f:
                    f.write(json.dumps(members, allow_nan=False).encode('utf-8'))
                record.update(verified=True, archive_sha256=expected, metadata=before,
                    retained_tree=base, prefix_parts_stripped=strip, complete_regular_members=len(members),
                    retained_payload_bytes=sum(m['bytes'] for m in members),
                    member_manifest=manifest.relative_to(ROOT).as_posix(), member_manifest_sha256=sha(manifest),
                    source_container_not_needed_for_local_payload_recovery=True,
                    original_compressed_container_will_not_be_reconstructible_as_identical_bytes=True)
                candidates.append(record)
            except (OSError, AssertionError, tarfile.TarError, EOFError) as e:
                record.update(retain_reason=str(e), exception_type=type(e).__name__)
                retained.append(record)
            record['seconds'] = time.monotonic()-archive_start
            write(OUT/'archive_checks'/(Path(row['path']).name+'.json'), record)
            log.write(json.dumps(record)+'\n')
            log.flush()
            os.fsync(log.fileno())
            print(dict(checked=number, of=len(archives), archive=Path(row['path']).name,
                verified=record['verified'], planned_GiB=sum(r['metadata']['bytes'] for r in candidates)/1024**3,
                seconds=round(record['seconds'], 1)), flush=True)
    assert candidates, 'No verified redundant archive candidates'
    plan = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), root=str(ROOT),
        inventory_sha256=sha(OUT/'complete_inventory_before.json'), candidates=candidates, retained=retained,
        requested_scope='local project unused/redundant files only; no VM maintenance',
        exact_archive_count=len(candidates), reclaim_logical_bytes=sum(c['metadata']['bytes'] for c in candidates),
        archive_member_payloads_all_retained=True, failed_gates_and_models_retained=True,
        current_bank_archives_preserved=True, portable_migration_backups_preserved=True,
        source_directories_will_not_be_removed=True, files_removed=0,
        neural_or_training_calls=0, seconds=time.monotonic()-start)
    write(OUT/'plan.json', plan)
    print(dict(complete=True, exact_archive_count=len(candidates), planned_GiB=plan['reclaim_logical_bytes']/1024**3,
        retained_archives=len(retained), files_removed=0),flush=True)


if __name__ == '__main__':
    main()
