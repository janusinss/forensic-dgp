"""Strict CPU-only verification and exclusive import of AdaIN V1 return.

Treat every returned file as opaque bytes. Never execute returned code or load
checkpoints. All members are streamed and hashed before creating the destination.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import time

from research.codeformer_bounded_fusion_feasibility_v1.return_archive import (
    canonical, exact, identity, namespace, no_links, object_identity, parse,
    read, regular, require, sha, write_new, HEX, METADATA)

PROTOCOL_SHA = '4e2cc86857628617edbb313b6ef2911602a63ad4e1f9862afe07c57c0ba770c5'
ARCHIVE_SHA = '7f7fc6b5a1fd58ea1ae5801d5f1edc95e66d11fe5dbc0256e1b4fe88ab7fa633'
ARCHIVE_BYTES = 2864673701
MANIFEST_SHA = '0091134a5653825fb3a06d63e3907d1e57521f607e67da5d39ea856cdceff805'
OUTPUT = 'outputs/adain_gradient_isolation_v1'
MANIFEST = OUTPUT + '/export_manifest_20261016T170106Z-8446018b.json'
PREFIX = 'cctv-codeformer-adain-gradient-isolation-v1'
OPERATIONAL = {'dependency_install_receipt.json', 'dependency_install_journal.jsonl',
    'outputs/supervision/preflight.log', 'outputs/supervision/trainer.log',
    'outputs/supervision/run_receipt.json'}
CHECKPOINTS = {OUTPUT + '/snapshot_0000/checkpoint.pt': 384 * 1024**2,
    OUTPUT + '/snapshot_0050/checkpoint.pt': 832 * 1024**2,
    OUTPUT + '/stopped_checkpoint.pt': 832 * 1024**2}
MAX_FILES = 7000
MAX_SECONDS = 1200


def metadata(name):
    return PurePosixPath(name).suffix.lower() in METADATA or PurePosixPath(name).name.upper() in {'LICENSE', 'COPYING', 'NOTICE', 'AUTHORS'}


def inspect_and_import(workspace):
    base = no_links(Path(workspace).absolute())
    outputs = base / 'outputs'
    packet = outputs / 'cctv_codeformer_adain_gradient_isolation_v1_vm'
    dest = outputs / 'codeformer_adain_gradient_isolation_v1_return'
    review = outputs / 'codeformer_adain_gradient_isolation_v1_return_review'
    receipt = review / 'transport_v1.json'
    failure = review / 'transport_v1_attempt3_failure.json'
    journal = outputs / (dest.name + '_import_journal.jsonl')
    archive = outputs / (PREFIX + '-results.tar.gz')
    sidecar = outputs / (archive.name + '.sha256')
    export_path = outputs / (PREFIX + '-export.json')
    for p in (dest, receipt, failure, journal):
        no_links(p); require(not p.exists(), 'Prior/partial evidence must be retained: ' + str(p))
    review.mkdir(exist_ok=True)
    start = time.monotonic(); phase = 'preflight'; imported = {}
    def budget():
        require(time.monotonic() - start <= MAX_SECONDS, 'Finite local transport cap exceeded')
    try:
        before = regular(archive)
        require(before.st_size == ARCHIVE_BYTES, 'Pinned archive size differs')
        require(regular(sidecar).st_size <= 1024 and sidecar.read_bytes() ==
            (ARCHIVE_SHA + '  ' + archive.name + '\n').encode('ascii'), 'Exact LF checksum sidecar required')
        exported = read(export_path, 1024**2)
        require(exported.get('complete') is True and exported.get('archive_sha256') == ARCHIVE_SHA and
            type(exported.get('bytes')) is int and exported['bytes'] == ARCHIVE_BYTES and
            exported.get('protocol_sha256') == PROTOCOL_SHA and exported.get('manifest_member') == MANIFEST and
            exported.get('manifest_sha256') == MANIFEST_SHA and exported.get('training_success_not_implied') is True and
            exported.get('model_qualified') is False and exported.get('failure_present') is True and
            exported.get('run_results_present') is False, 'Pinned export identity/status differs')
        require(sha(archive, budget) == ARCHIVE_SHA, 'Compressed archive SHA256 differs')
        require(sha(packet / 'protocol.json', budget) == PROTOCOL_SHA, 'Frozen protocol SHA256 differs')
        p = read(packet / 'protocol.json', 4 * 1024**2)
        require(p.get('format') == 'codeformer-adain-gradient-isolation-finite-manual-v1' and
            p.get('outputs') == OUTPUT and p.get('export_prefix') == PREFIX and
            p['training']['max_updates'] == 50 and p['limits']['result_payload_bytes'] == 3758096384 and
            p['limits']['archive_bytes'] == 3865470566 and p['limits']['metadata_bytes'] == 134217728,
            'Frozen finite protocol scope differs')
        files = p['files_sha256']; namespace(files)
        require(all(isinstance(v, str) and HEX.fullmatch(v) for v in files.values()), 'Invalid immutable digest map')
        parents = (outputs / p['parent_dir'], outputs / 'cctv_codeformer_adaptation_v2_vm')
        # This is a local audit resolver only. The VM installer used its exact FP32
        # sibling and has no fallback. Every chosen local source is hash-bound.
        verified = {}; overlay = set()
        for folder, dirs, names in os.walk(packet, followlinks=False):
            for directory in dirs: no_links(Path(folder) / directory)
            for name in names:
                rel = (Path(folder) / name).relative_to(packet).as_posix()
                require(rel == 'protocol.json' or rel in files, 'Unlisted frozen overlay file')
                overlay.add(rel)
        for name, digest in files.items():
            sources = [packet / name, *(parent / name for parent in parents)]
            selected = next((f for f in sources if f.exists()), None)
            require(selected is not None, 'Missing immutable source: ' + name)
            require(sha(selected, budget) == digest, 'Immutable source differs: ' + name)
            verified[name] = {'sha256': digest, 'bytes': regular(selected).st_size,
                              'resolved_path': str(selected)}
        for name, spec in p['assets'].items():
            require(spec['path'] in verified and verified[spec['path']]['sha256'] == spec['sha256'] and
                verified[spec['path']]['bytes'] == spec['bytes'], 'Initializer/teacher asset differs: ' + name)
        phase = 'full_stream_verification'; records = {}; jsons = {}; payload = 0
        critical = {'protocol.json', MANIFEST, 'dependency_install_receipt.json',
            'outputs/supervision/run_receipt.json', OUTPUT + '/failure.json', OUTPUT + '/run_results.json'}
        with archive.open('rb') as raw:
            opened = os.fstat(raw.fileno())
            require(object_identity(opened) == object_identity(before), 'Archive object identity changed')
            with tarfile.open(fileobj=raw, mode='r|gz') as tar:
                for member in tar:
                    budget(); name = canonical(member.name)
                    require(member.type in (tarfile.REGTYPE, tarfile.AREGTYPE) and member.isfile() and
                        member.sparse is None and not member.linkname and
                        not any(k.startswith('GNU.sparse') for k in member.pax_headers), 'Only nonsparse regular members allowed')
                    require(name not in records and len(records) < MAX_FILES, 'Duplicate/file cap violation')
                    require(type(member.size) is int and 0 <= member.size <= p['limits']['result_payload_bytes'], 'Invalid member size')
                    payload += member.size
                    require(payload <= p['limits']['result_payload_bytes'], 'Total payload cap exceeded')
                    suffix = PurePosixPath(name).suffix.lower()
                    if name.startswith(OUTPUT + '/'):
                        require(suffix in {'.json', '.jsonl', '.npy', '.png', '.pt'}, 'Unexpected returned output extension')
                        if suffix == '.pt':
                            require(name in CHECKPOINTS and member.size <= CHECKPOINTS[name], 'Unexpected/oversized checkpoint')
                        elif suffix == '.png':
                            png_cap = (3 * 512 * 512 + 4096 if name == OUTPUT + '/first_training_example/clean512.png'
                                       else p['limits']['PNG_file_bytes'])
                            require(member.size <= png_cap, 'PNG cap exceeded: ' + name)
                        elif suffix == '.npy': require(member.size <= 4198400, 'Array cap exceeded')
                        elif name == MANIFEST: require(member.size <= 4 * 1024**2, 'Manifest cap exceeded')
                        else: require(member.size <= 16 * 1024**2, 'Output metadata cap exceeded')
                    else:
                        require(name == 'protocol.json' or name in p['export_files'] or name in OPERATIONAL or
                            name == 'adain_gradient_isolation_v1.lock', 'Outside exact output/companion namespace')
                    if name in OPERATIONAL and suffix == '.log': require(member.size <= 2 * 1024**2, 'Closed log cap exceeded')
                    require(suffix not in {'.pth', '.pickle', '.pkl', '.ckpt', '.safetensors'}, 'Unexpected weight/object extension')
                    h = hashlib.sha256(); count = 0; chunks = [] if name in critical else None
                    with tar.extractfile(member) as source:
                        for block in iter(lambda: source.read(4 * 1024**2), b''):
                            budget(); h.update(block); count += len(block)
                            if chunks is not None: chunks.append(block)
                    require(count == member.size, 'Truncated member')
                    records[name] = {'sha256': h.hexdigest(), 'bytes': count}
                    if chunks is not None: jsons[name] = parse(b''.join(chunks))
            require(identity(regular(archive)) == identity(before) and
                identity(os.fstat(raw.fileno())) == identity(opened), 'Archive changed during verification')
        namespace(records)
        require(records[MANIFEST]['sha256'] == MANIFEST_SHA and records['protocol.json']['sha256'] == PROTOCOL_SHA and
            exact(jsons['protocol.json'], p), 'Returned protocol/manifest differs')
        m = jsons[MANIFEST]; expected = m['files_sha256']; namespace(expected)
        require(m.get('complete') is True and m.get('protocol_sha256') == PROTOCOL_SHA and
            m.get('training_success_not_implied') is True and m.get('model_qualified') is False and
            m.get('failure_present') is True and m.get('run_results_present') is False and
            m.get('immutable_inputs_and_original_models_not_duplicated') is True and
            m.get('incomplete_temp_files') == {}, 'Manifest identity/status differs')
        require(set(records) == set(expected) | {MANIFEST}, 'Unlisted or missing archived members')
        for name, digest in expected.items():
            require(isinstance(digest, str) and HEX.fullmatch(digest) and records[name]['sha256'] == digest,
                    'Manifest payload SHA256 differs: ' + name)
        mandatory = {'protocol.json', *p['export_files'], *OPERATIONAL}
        require(mandatory <= set(expected), 'Required frozen source/closed companion missing')
        for name in mandatory - OPERATIONAL:
            require(expected[name] == (PROTOCOL_SHA if name == 'protocol.json' else files[name]), 'Returned immutable source differs')
        retained = {name: digest for name, digest in files.items() if name not in expected}
        retained.update({spec['path']: spec['sha256'] for spec in p['assets'].values() if spec['path'] not in expected})
        require(exact(m.get('retained_not_exported_sha256'), retained) and
            exported['retained_not_exported_files'] == len(retained), 'Retained immutable namespace differs')
        require(exact(m.get('closed_operational_companions_sha256'),
            {name: records[name]['sha256'] for name in OPERATIONAL}), 'Closed companions differ')
        metadata_bytes = sum(v['bytes'] for name, v in records.items() if metadata(name))
        require(metadata_bytes == exported['exported_metadata_bytes'] <= p['limits']['metadata_bytes'] and
            payload == exported['upper_payload_bytes'], 'Exported byte accounting differs')
        # Frozen exporter writes its manifest before output_accounting(out).
        # The manifest is consequently included in result-only accounting.
        out_records = {name: row for name, row in records.items() if name.startswith(OUTPUT + '/')}
        require({'final_bytes': sum(row['bytes'] for row in out_records.values()), 'temporary_bytes': 0,
            'metadata_bytes': sum(row['bytes'] for name, row in out_records.items() if metadata(name)),
            'regular_files': len(out_records)} == exported['result_accounting'], 'Result-only accounting differs')
        install = jsons['dependency_install_receipt.json']; supervision = jsons['outputs/supervision/run_receipt.json']
        require(install.get('complete') is True and install.get('protocol_sha256') == PROTOCOL_SHA, 'Installer receipt differs')
        require(supervision.get('protocol_sha256') == PROTOCOL_SHA and supervision.get('manual_tmux') is True and
            supervision.get('execution_max_updates') == 50 and supervision.get('no_automatic_retry_or_resume') is True and
            supervision.get('training_success_not_implied') is True and supervision.get('model_qualified') is False and
            supervision.get('automatic_app_promotion') is False, 'Manual finite supervisor boundary differs')
        stages = supervision['stages']
        require(len(stages) == 2, 'Two closed preflight/train stages required')
        for stage, flag, cap, grace in zip(stages, ('--verify-transfer', '--run'), (300, 3000), (30, 180)):
            require(stage.get('mode', stage.get('stage')) == flag and stage.get('external_cap_seconds') == cap and
                stage.get('grace_seconds') == grace and type(stage.get('exit_code')) is int and
                0 <= stage.get('seconds', -1) <= cap + grace + 15, 'Finite supervisor stage differs')
        require(stages[0]['exit_code'] == 0 and stages[1]['exit_code'] != 0 and
            stages[1]['exit_code'] == supervision['training_process_exit_code'] and
            OUTPUT + '/failure.json' in records and OUTPUT + '/run_results.json' not in records,
            'Supervisor and failure terminal status differ')
        failed = jsons[OUTPUT + '/failure.json']
        require(failed.get('complete') is False and failed.get('protocol_sha256') == PROTOCOL_SHA and
            failed.get('model_qualified') is False and type(failed.get('completed_updates')) is int and
            0 <= failed['completed_updates'] <= 50, 'Failed worker boundary differs')
        tar_upper = sum(((v['bytes'] + 511) // 512) * 512 + 4096 for v in records.values()) + 1024**2
        archive_upper = tar_upper + tar_upper // 1000 + 1 + 1024**2
        require(archive_upper == exported['conservative_archive_upper_bytes'] <= p['limits']['archive_bytes'], 'Archive upper-bound accounting differs')
        require(shutil.disk_usage(outputs).free >= payload + 1024**3, 'Insufficient exclusive import space plus reserve')
        phase = 'exclusive_import'; dest.mkdir(exist_ok=False)
        with journal.open('x', encoding='utf-8', newline='\n') as log:
            def note(value):
                log.write(json.dumps(value, allow_nan=False) + '\n'); log.flush(); os.fsync(log.fileno())
            note({'phase': 'all_members_verified_before_import', 'archive_sha256': ARCHIVE_SHA, 'members': len(records)})
            with archive.open('rb') as raw:
                opened = os.fstat(raw.fileno())
                require(object_identity(opened) == object_identity(before), 'Archive changed before import')
                with tarfile.open(fileobj=raw, mode='r|gz') as tar:
                    for member in tar:
                        budget(); name = canonical(member.name)
                        require(name in records and name not in imported and member.size == records[name]['bytes'], 'Import member differs')
                        target = no_links(dest / name); require(target.is_relative_to(dest), 'Import escape')
                        target.parent.mkdir(parents=True, exist_ok=True); no_links(target.parent)
                        h = hashlib.sha256(); count = 0
                        with target.open('xb') as target_stream, tar.extractfile(member) as source:
                            for block in iter(lambda: source.read(4 * 1024**2), b''):
                                budget(); count += len(block); require(count <= records[name]['bytes'], 'Import member grew')
                                target_stream.write(block); h.update(block)
                            target_stream.flush(); os.fsync(target_stream.fileno())
                        require({'sha256': h.hexdigest(), 'bytes': count} == records[name], 'Import bytes/hash differ')
                        imported[name] = records[name]; note({'member': name, **records[name]})
                require(identity(regular(archive)) == identity(before) and identity(os.fstat(raw.fileno())) == identity(opened),
                    'Archive changed during import')
            require(imported == records, 'Incomplete import'); note({'phase': 'import_complete', 'members': len(records)})
        result = {'format': 'codeformer-adain-gradient-isolation-strict-return-transport-v1', 'complete': True,
            'archive_sha256': ARCHIVE_SHA, 'archive_bytes': ARCHIVE_BYTES, 'protocol_sha256': PROTOCOL_SHA,
            'manifest_member': MANIFEST, 'manifest_sha256': MANIFEST_SHA, 'import_directory': str(dest),
            'exact_namespace_verified': True, 'whole_archive_and_all_member_hashes_verified_before_import': True,
            'files_sha256': {n: row['sha256'] for n, row in records.items()},
            'files_bytes': {n: row['bytes'] for n, row in records.items()}, 'member_maps_include_manifest': True,
            'members': len(records), 'payload_bytes': payload, 'metadata_bytes': metadata_bytes,
            'immutable_sources_and_assets_freshly_verified': verified, 'local_overlay_files': len(overlay),
            'retained_not_exported_sha256': retained, 'reported_completed_updates': failed['completed_updates'],
            'reported_failure': failed['error'], 'reported_training_exit_code': stages[1]['exit_code'],
            'export_complete_not_training_success': True, 'source_sha256': sha(__file__),
            'shared_boundary_source_sha256': sha(base / 'research/codeformer_bounded_fusion_feasibility_v1/return_archive.py'),
            'import_journal_sha256': sha(journal), 'seconds': time.monotonic() - start,
            'returned_code_executed': False, 'pickle_or_checkpoint_loaded': False, 'neural_calls': 0,
            'gradient_queries': 0, 'optimizer_updates': 0, 'VM_calls': 0, 'model_qualified': False,
            'training_authorized': False, 'classification': 'Transport audit only; scientific/visual audit remains separate.',
            'prior_checker_failure': {'receipt': 'transport_v1_failure.json',
                'receipt_sha256': sha(review / 'transport_v1_failure.json'),
                'source': 'transport_checker_attempt1.py', 'source_sha256': sha(review / 'transport_checker_attempt1.py'),
                'cause': 'Checker applied delivered256 PNG cap to one declared clean512 TRAIN example.',
                'fix': 'Only first_training_example/clean512.png uses the frozen worker 3*512*512+4096 cap; all other PNG caps unchanged.'},
            'second_checker_failure': {'receipt': 'transport_v1_attempt2_failure.json',
                'receipt_sha256': sha(review / 'transport_v1_attempt2_failure.json'),
                'source': 'transport_checker_attempt2.py', 'source_sha256': sha(review / 'transport_checker_attempt2.py'),
                'cause': 'Checker omitted the already-written export manifest from result-only accounting.',
                'fix': 'Include manifest exactly as frozen worker writes manifest before output_accounting(out); no payload/cap changes.'}}
        write_new(receipt, result)
        return result
    except BaseException as error:
        if not failure.exists(): write_new(failure, {'complete': False, 'phase': phase, 'error_type': type(error).__name__,
            'message': str(error), 'imported_members': len(imported), 'partial_evidence_retained': True,
            'automatic_retry': False, 'model_qualified': False, 'returned_code_executed': False})
        raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = inspect_and_import(args.workspace)
    print(json.dumps({k: result[k] for k in ('complete', 'members', 'payload_bytes', 'reported_completed_updates',
        'reported_failure', 'reported_training_exit_code', 'seconds', 'model_qualified')}, allow_nan=False))
