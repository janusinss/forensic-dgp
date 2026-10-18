"""Strict local V2 return transport audit/import, separate from frozen training.

Stdlib only. Importing this module performs no I/O. No neural, gradient, optimizer,
VM, network, app or checkpoint execution occurs. Transport is not qualification.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tarfile
import time

PROTOCOL_SHA256 = '029291a2d58d7bc8b33fb99b86774e13db4e5c098bb8af5f508fdc5e24b20183'
OUTPUTS = 'outputs/adaptation_v2'
RESULT_ARCHIVE = 'cctv-codeformer-adaptation-v2-results.tar.gz'
EXPORT_RECEIPT = 'cctv-codeformer-adaptation-v2-export.json'
MAX_ARCHIVE_BYTES = 7623566951
MAX_PAYLOAD_BYTES = MAX_ARCHIVE_BYTES
MAX_FINAL_BYTES = 7 * 1024**3
MAX_TEMP_BYTES = 1024**3
MAX_METADATA_BYTES = 128 * 1024**2
RESERVE_BYTES = 4 * 1024**3
MAX_MEMBERS = 12000
MAX_SECONDS = 1800
HEX = re.compile(r'^[0-9a-f]{64}$')
DEVICE = re.compile(r'^(con|prn|aux|nul|conin\$|conout\$|com[1-9¹²³]|lpt[1-9¹²³])(?:\.|$)', re.I)
MANIFEST = re.compile(r'^outputs/adaptation_v2/export_manifest_\d{8}T\d{6}Z-[0-9a-f]{8}\.json$')
METADATA_SUFFIXES = {'.json', '.jsonl', '.log', '.txt', '.md', '.py', '.sh', '.sha256', '.csv', '.yaml', '.yml'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def preparation_status():
    return {'status': 'prepared_local_return_transport_auditor', 'actual_V2_return_audited': False,
            'protocol_sha256': PROTOCOL_SHA256, 'model_qualified': False, 'neural_calls': 0,
            'gradient_queries': 0, 'optimizer_updates': 0, 'VM_calls': 0}


def canonical_member(name):
    require(isinstance(name, str) and bool(name), 'Empty return path')
    p = PurePosixPath(name)
    require(bool(p.parts) and not p.is_absolute() and p.as_posix() == name and
            all(x not in ('', '.', '..') for x in p.parts), 'Escaped/noncanonical return path')
    require(not any(ch in name for ch in '\\:"<>|?*') and all(
        not DEVICE.match(x) and not x.endswith((' ', '.')) and len(x.encode('utf-16-le')) <= 510 and
        all(ord(ch) >= 32 for ch in x) for x in p.parts), 'Unsafe Windows return path')
    return name


def _json(payload):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, 'Duplicate JSON field')
            result[key] = value
        return result
    return json.loads(payload.decode('utf-8-sig'), object_pairs_hook=pairs,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def _no_links(path):
    path = Path(path).absolute()
    for ancestor in (path, *path.parents):
        if ancestor.exists() or ancestor.is_symlink():
            info = ancestor.lstat()
            require(not ancestor.is_symlink() and not getattr(info, 'st_file_attributes', 0) & 1024,
                    'Symlink/reparse path forbidden: ' + str(ancestor))
    return path


def _regular(path):
    path = _no_links(path)
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode), 'Expected regular file: ' + str(path))
    return info


def _identity(info):
    return info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino, info.st_dev


def _object_identity(info):
    # Windows Python lstat/fstat can expose different ctime semantics. Compare
    # their file identity/size/mtime, then bind each API's full identity to its
    # own initial reading rather than treating differing ctimes as corruption.
    return info.st_size, info.st_mtime_ns, info.st_ino, info.st_dev


def _hash(path, budget):
    before = _regular(path)
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024**2), b''):
            budget(); digest.update(block)
    require(_identity(_regular(path)) == _identity(before), 'Research input changed while hashing')
    return digest.hexdigest()


def _read_json(path, maximum):
    info = _regular(path)
    require(info.st_size <= maximum, 'JSON metadata exceeds finite bound')
    payload = Path(path).read_bytes()
    require(len(payload) == info.st_size and _identity(_regular(path)) == _identity(info), 'JSON source changed')
    return _json(payload)


def _write_json_new(path, value):
    path = _no_links(path)
    with path.open('x', encoding='utf-8', newline='\n') as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write('\n'); output.flush(); os.fsync(output.fileno())


def _member_regular(member):
    require(member.type in (tarfile.REGTYPE, tarfile.AREGTYPE) and member.isfile() and
            member.sparse is None and not any(k.startswith('GNU.sparse') for k in member.pax_headers),
            'Only nonsparse regular TAR members are accepted')


def _namespace(names):
    folded = set()
    for name in names:
        canonical_member(name)
        require(name.casefold() not in folded, 'Duplicate/case-colliding member')
        folded.add(name.casefold())
    for name in folded:
        parent = PurePosixPath(name).parent
        while parent.parts:
            require(parent.as_posix() not in folded, 'File/directory member collision')
            parent = parent.parent


def _metadata(name):
    p = PurePosixPath(name)
    return p.suffix.lower() in METADATA_SUFFIXES or p.name.upper() in {'LICENSE', 'COPYING', 'NOTICE', 'AUTHORS'}


def _incomplete(name):
    return any(x in PurePosixPath(name).name for x in ('.tmp-', '.partial-'))


def _schema(export, protocol, manifest, records, manifest_name):
    require(manifest.get('complete') is True and manifest.get('protocol_sha256') == PROTOCOL_SHA256 and
            manifest.get('model_qualified') is False and manifest.get('training_success_not_implied') is True and
            manifest.get('immutable_inputs_and_original_models_not_duplicated') is True,
            'Returned manifest scope differs')
    for flag in ('run_results_present', 'failure_present'):
        require(type(export.get(flag)) is bool and type(manifest.get(flag)) is bool and
                export[flag] == manifest[flag], 'Manifest/export status differs')
    expected = manifest.get('files_sha256')
    require(isinstance(expected, dict) and manifest_name not in expected, 'Invalid manifest hash namespace')
    expected = dict(expected, **{manifest_name: export['manifest_sha256']})
    require(set(expected) == set(records), 'Unlisted or missing returned member')
    _namespace(expected)
    mandatory = {'protocol.json', *protocol['export_files']}
    require(mandatory <= set(expected), 'Mandatory source/notice/manifest companion missing')
    allowed = mandatory | {'trainer.log', 'adaptation_v2.lock'}
    for name, fingerprint in expected.items():
        require(name.startswith(OUTPUTS + '/') or name in allowed, 'Unexpected return namespace: ' + name)
        require(isinstance(fingerprint, str) and HEX.fullmatch(fingerprint) and
                records[name]['sha256'] == fingerprint, 'Returned payload fingerprint differs: ' + name)
        if name in protocol['files_sha256']:
            require(fingerprint == protocol['files_sha256'][name], 'Returned frozen source/input changed')
    require(expected['protocol.json'] == PROTOCOL_SHA256, 'Returned protocol changed')
    require(('outputs/adaptation_v2/failure.json' in expected) == export['failure_present'] and
            ('outputs/adaptation_v2/run_results.json' in expected) == export['run_results_present'],
            'Actual returned run/failure flags differ')
    incomplete = manifest.get('incomplete_temp_files')
    actual_incomplete = {name for name in records if _incomplete(name)}
    require(isinstance(incomplete, dict) and set(incomplete) == actual_incomplete and
            manifest.get('incomplete_temp_files_are_not_resumable_checkpoints') is True,
            'Incomplete temporary evidence classification differs')
    for name, row in incomplete.items():
        require(isinstance(row, dict) and row.get('sha256') == records[name]['sha256'] and
                type(row.get('bytes')) is int and row['bytes'] == records[name]['bytes'] and
                row.get('resume_safe') is False and bool(row.get('reason')), 'Invalid incomplete file receipt')
    require(type(export.get('incomplete_temp_files')) is int and export['incomplete_temp_files'] == len(incomplete),
            'Incomplete file count differs')
    # Match the frozen worker exactly: '.tmp-' files are temporary allocations;
    # '.partial-' files are labelled incomplete but still count in final bytes.
    output_rows = {n: r for n, r in records.items() if n.startswith(OUTPUTS + '/')}
    temporary = sum(r['bytes'] for n, r in output_rows.items() if '.tmp-' in PurePosixPath(n).name)
    final = sum(r['bytes'] for n, r in output_rows.items() if '.tmp-' not in PurePosixPath(n).name)
    metadata = sum(r['bytes'] for n, r in output_rows.items()
                   if PurePosixPath(n).suffix.lower() in {'.json', '.jsonl', '.log', '.txt', '.sha256'})
    accounting = {'final_bytes': final, 'temporary_bytes': temporary, 'metadata_bytes': metadata}
    require(export.get('result_accounting') == accounting, 'Returned output accounting differs')
    total = sum(r['bytes'] for r in records.values())
    exported_metadata = sum(r['bytes'] for n, r in records.items() if _metadata(n))
    require(type(export.get('upper_payload_bytes')) is int and export['upper_payload_bytes'] == total and
            final <= MAX_FINAL_BYTES and temporary <= MAX_TEMP_BYTES and total - temporary <= MAX_FINAL_BYTES,
            'Final/temporary exported payload exceeds finite scope')
    require(export.get('exported_metadata_bytes') == exported_metadata and exported_metadata <= MAX_METADATA_BYTES,
            'Exported source/log metadata accounting differs')
    tar_upper = sum(((r['bytes'] + 511) // 512) * 512 + 4096 for r in records.values()) + 1024**2
    archive_upper = tar_upper + (tar_upper // 1000 + 1) + 1024**2
    require(export.get('conservative_archive_upper_bytes') == archive_upper and archive_upper <= MAX_ARCHIVE_BYTES,
            'Conservative TAR/gzip bound differs or exceeds7.1GiB')
    return expected, accounting, exported_metadata


def inspect_and_import(archive, sidecar, export_path, packet, destination, receipt_path):
    """Verify full bytes first; import only into an exclusive new directory.

    A rejected/partial import is retained with an exclusive failure receipt. No
    automatic retry, overwrite, recursive delete or checkpoint deserialization.
    """
    started = time.monotonic()
    archive, sidecar, export_path, packet, destination, receipt_path = map(
        lambda p: Path(p).absolute(), (archive, sidecar, export_path, packet, destination, receipt_path))
    journal = destination.parent / (destination.name + '_import_journal.jsonl')
    failure_path = receipt_path.with_name(receipt_path.stem + '_failure.json')
    for path in (destination, receipt_path, journal, failure_path):
        _no_links(path)
        require(not path.exists(), 'Retain prior import/evidence; choose a new explicit version: ' + str(path))
    require(destination.parent.is_dir() and receipt_path.parent.is_dir(), 'Create dedicated audit parents before import')
    _no_links(packet)
    require(packet.is_dir() and not destination.resolve().is_relative_to(packet.resolve()) and
            not receipt_path.resolve().is_relative_to(packet.resolve()), 'Do not write into the frozen packet')
    imported, phase = {}, 'preflight'
    def budget():
        require(time.monotonic() - started <= MAX_SECONDS, 'Local transport audit finite time bound reached')
    try:
        require(archive.name == RESULT_ARCHIVE and sidecar.name == RESULT_ARCHIVE + '.sha256' and
                export_path.name == EXPORT_RECEIPT, 'Unexpected V2 return filenames')
        before = _regular(archive)
        require(0 < before.st_size <= MAX_ARCHIVE_BYTES, 'Compressed return exceeds7.1GiB or is empty')
        _regular(sidecar)
        require(sidecar.stat().st_size <= 1024, 'Unbounded checksum sidecar')
        export = _read_json(export_path, 1024**2)
        require(export.get('complete') is True and export.get('model_qualified') is False and
                export.get('training_success_not_implied') is True and
                export.get('protocol_sha256') == PROTOCOL_SHA256, 'Unexpected V2 export receipt')
        require(sha := export.get('archive_sha256'), 'Missing compressed archive SHA256')
        require(isinstance(sha, str) and HEX.fullmatch(sha) and type(export.get('bytes')) is int and
                export['bytes'] == before.st_size, 'Malformed archive fingerprint/size')
        require(sidecar.read_bytes() == (sha + '  ' + archive.name + '\n').encode('ascii'),
                'Checksum sidecar differs or is not exact LF')
        protocol_path = packet / 'protocol.json'
        require(_hash(protocol_path, budget) == PROTOCOL_SHA256, 'Local frozen V2 protocol differs')
        protocol = _read_json(protocol_path, 8 * 1024**2)
        require(protocol.get('format') == 'codeformer-adaptation-finite-manual-v2' and
                protocol.get('outputs') == OUTPUTS and protocol.get('export_prefix') == 'cctv-codeformer-adaptation-v2',
                'Frozen protocol output scope differs')
        limits = protocol['limits']
        require([limits[k] for k in ('result_payload_bytes', 'checkpoint_temporary_bytes', 'archive_bytes',
                    'metadata_bytes', 'reserve_free_bytes')] ==
                [MAX_FINAL_BYTES, MAX_TEMP_BYTES, MAX_ARCHIVE_BYTES, MAX_METADATA_BYTES, RESERVE_BYTES],
                'Frozen protocol finite transport limits differ')
        require(isinstance(protocol.get('files_sha256'), dict) and isinstance(protocol.get('assets'), dict) and
                isinstance(protocol.get('export_files'), list), 'Invalid frozen packet schema')
        manifest_name = canonical_member(export.get('manifest_member'))
        require(MANIFEST.fullmatch(manifest_name) and HEX.fullmatch(export.get('manifest_sha256', '')),
                'Unexpected exported manifest name/hash')
        records, manifest_payload = {}, None
        with archive.open('rb') as compressed:
            handle_before = os.fstat(compressed.fileno())
            require(_object_identity(handle_before) == _object_identity(before), 'Archive identity changed before open')
            digest = hashlib.sha256()
            for block in iter(lambda: compressed.read(4 * 1024**2), b''):
                budget(); digest.update(block)
            require(digest.hexdigest() == sha, 'Compressed archive checksum mismatch')
            compressed.seek(0)
            phase = 'full_member_verification'
            payload_bytes = 0
            # ignore_zeros scans through padding to catch hidden concatenated TAR
            # members instead of declaring an early end marker an exact namespace.
            with tarfile.open(fileobj=compressed, mode='r:gz', ignore_zeros=True) as source:
                for member in source:
                    budget(); name = canonical_member(member.name); _member_regular(member)
                    require(name not in records and len(records) < MAX_MEMBERS and member.size >= 0,
                            'Duplicate or excessive member namespace')
                    payload_bytes += member.size
                    require(payload_bytes <= MAX_PAYLOAD_BYTES, 'Return payload exceeds7.1GiB')
                    if name == manifest_name:
                        require(member.size <= 8 * 1024**2, 'Manifest exceeds8MiB')
                    h, count, chunks = hashlib.sha256(), 0, [] if name == manifest_name else None
                    stream = source.extractfile(member)
                    for block in iter(lambda: stream.read(4 * 1024**2), b''):
                        budget(); h.update(block); count += len(block)
                        if chunks is not None:
                            chunks.append(block)
                    require(count == member.size, 'Short TAR member')
                    records[name] = {'sha256': h.hexdigest(), 'bytes': count}
                    if chunks is not None:
                        manifest_payload = b''.join(chunks)
            _namespace(records)
            require(manifest_payload is not None and hashlib.sha256(manifest_payload).hexdigest() ==
                    export['manifest_sha256'], 'Returned manifest fingerprint differs')
            manifest = _json(manifest_payload)
            expected, accounting, metadata_bytes = _schema(export, protocol, manifest, records, manifest_name)
            all_inputs = dict(protocol['files_sha256'])
            for asset in protocol['assets'].values():
                require(isinstance(asset, dict) and canonical_member(asset['path']) and HEX.fullmatch(asset['sha256']),
                        'Malformed original asset binding')
                require(asset['path'] not in all_inputs or all_inputs[asset['path']] == asset['sha256'],
                        'Original asset/source hashes conflict')
                all_inputs[asset['path']] = asset['sha256']
                require(type(asset.get('bytes')) is int and asset['bytes'] > 0 and
                        _regular(packet / asset['path']).st_size == asset['bytes'], 'Original model asset size differs')
            retained = {n: v for n, v in all_inputs.items() if n not in expected}
            require(manifest.get('retained_not_exported_sha256') == retained and
                    type(export.get('retained_not_exported_files')) is int and
                    export['retained_not_exported_files'] == len(retained), 'Retained original namespace differs')
            for name, fingerprint in all_inputs.items():
                canonical_member(name)
                require(isinstance(fingerprint, str) and HEX.fullmatch(fingerprint) and
                        _hash(packet / name, budget) == fingerprint, 'Original packet input changed: ' + name)
            require(_identity(_regular(archive)) == _identity(before), 'Archive changed during preflight')
            require(shutil.disk_usage(destination.parent).free >= payload_bytes + RESERVE_BYTES,
                    'Require complete return payload plus4GiB local reserve')
            destination.mkdir()
            phase = 'exclusive_import'
            compressed.seek(0)
            with journal.open('x', encoding='utf-8', newline='\n') as log, \
                    tarfile.open(fileobj=compressed, mode='r:gz', ignore_zeros=True) as source:
                for member in source:
                    budget(); name = canonical_member(member.name); _member_regular(member)
                    require(name in records and name not in imported and member.size == records[name]['bytes'],
                            'Archive namespace changed between verification and import')
                    target = destination.joinpath(*PurePosixPath(name).parts)
                    _no_links(target)
                    require(target.resolve().is_relative_to(destination.resolve()) and not target.exists(),
                            'Import target exists or escapes destination')
                    require(shutil.disk_usage(destination.parent).free >= member.size + RESERVE_BYTES,
                            'Local free reserve changed during import')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    log.write(json.dumps({'event': 'intent', 'file': name, **records[name]}) + '\n')
                    log.flush(); os.fsync(log.fileno())
                    h, count = hashlib.sha256(), 0
                    with target.open('xb') as output:
                        stream = source.extractfile(member)
                        for block in iter(lambda: stream.read(4 * 1024**2), b''):
                            budget(); output.write(block); h.update(block); count += len(block)
                        output.flush(); os.fsync(output.fileno())
                    require(count == records[name]['bytes'] and h.hexdigest() == expected[name],
                            'Imported payload changed: ' + name)
                    imported[name] = h.hexdigest()
                    log.write(json.dumps({'event': 'imported', 'file': name, 'bytes': count,
                                          'sha256': imported[name]}) + '\n'); log.flush(); os.fsync(log.fileno())
            require(imported == expected and _identity(_regular(archive)) == _identity(before) and
                    _identity(os.fstat(compressed.fileno())) == _identity(handle_before),
                    'Import namespace/archive identity differs')
        receipt = {'complete': True, 'audit_scope': 'actual local V2 return transport only, not training/quality audit',
            'archive_sha256': sha, 'archive_bytes': before.st_size, 'protocol_sha256': PROTOCOL_SHA256,
            'manifest_member': manifest_name, 'manifest_sha256': export['manifest_sha256'],
            'members': len(imported), 'payload_bytes': payload_bytes, 'retained_inputs_verified': len(retained),
            'all_original_packet_inputs_verified': len(all_inputs), 'all_members_hash_verified': True,
            'all_members_verified_before_destination_created': True, 'exact_namespace': True,
            'import_directory': str(destination), 'journal': str(journal),
            'run_results_present': export['run_results_present'], 'failure_present': export['failure_present'],
            'result_accounting': accounting, 'exported_metadata_bytes': metadata_bytes,
            'incomplete_temp_files': len(manifest['incomplete_temp_files']),
            'incomplete_files_preserved_not_resumable': True, 'model_qualified': False,
            'neural_calls': 0, 'gradient_queries': 0, 'optimizer_updates': 0, 'VM_calls': 0,
            'seconds': time.monotonic() - started}
        _write_json_new(receipt_path, receipt)
        return receipt
    except BaseException as error:
        # Exclusive write preserves prior failures. No cleanup deletes research
        # payloads, partial files, journals, downloaded archive or source inputs.
        if not failure_path.exists():
            _write_json_new(failure_path, {'complete': False, 'phase': phase, 'error_type': type(error).__name__,
                'error': str(error), 'imported': len(imported), 'protocol_sha256': PROTOCOL_SHA256,
                'original_archive_and_partial_evidence_retained': True, 'model_qualified': False,
                'neural_calls': 0, 'gradient_queries': 0, 'optimizer_updates': 0, 'VM_calls': 0})
        raise
