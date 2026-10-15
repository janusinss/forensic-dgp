"""Repair transfer/export identity in a fresh packet; retain sealed V1 bytes."""
import ast
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
import tarfile

from import_cctv_dgp_png_group_gradients_v1_return import ROOT, OUT, PARENT, read, sha, write

BASE = 'cctv_dgp_png_group_gradients_v1_r1'
NAME = BASE + '_vm'
STEM = 'cctv-dgp-png-group-gradients-v1-r1'
OLD_BASE = 'cctv_dgp_png_group_gradients_v1'
OLD_NAME = OLD_BASE + '_vm'
OLD_PIN = '57a2acef267a02a5ca464b6e16060dc3e7ff7741ba7f4aa0cf3ff0984b6f7b71'
OLD_ARCHIVE_SHA = 'c72f2d3b07f486666d7d9c08ea23a9cbb29c4830b005cc24129ee84ba70e02e9'


def replace_exact(text, old, new, count=1):
    assert text.count(old) == count, (old, text.count(old), count)
    return text.replace(old, new)


def new_text(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(value)


def constants(text):
    tree = ast.parse(text, feature_version=(3, 10))
    return {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and
            isinstance(node.targets[0], ast.Name) and node.targets[0].id in ['NAME', 'STEM', 'PIN']}


def main():
    old_root = OUT / OLD_NAME
    old_archive = OUT / 'cctv-dgp-png-group-gradients-v1-execution.tar.gz'
    old = read(old_root / 'protocol.json')
    assert sha(old_root / 'protocol.json') == OLD_PIN
    assert sha(old_archive) == OLD_ARCHIVE_SHA
    preserved = {str(path.relative_to(ROOT)).replace('\\', '/'): sha(path) for path in
                 [old_archive, Path(str(old_archive) + '.sha256'), old_root / 'protocol.json',
                  OUT / (OLD_BASE + '_preparation.json'), ROOT / 'scripts' / ('prepare_' + OLD_BASE + '.py'),
                  ROOT / 'scripts' / ('import_' + OLD_BASE + '_return.py'),
                  ROOT / 'scripts' / ('audit_' + OLD_BASE + '_return.py')]}
    for name, digest in old['assets_sha256'].items():
        assert sha(old_root / name) == digest
        preserved[(old_root / name).relative_to(ROOT).as_posix()] = digest
    source = old_root / 'scripts' / (OLD_NAME + '.py')
    original_worker = source.read_text(encoding='utf-8')
    assert constants(original_worker) == {'NAME': OLD_NAME, 'STEM': 'cctv-dgp-bank-png-group-gradients-v1'}
    worker = replace_exact(original_worker, "NAME = '" + OLD_NAME + "'", "NAME = '" + NAME + "'")
    worker = replace_exact(worker, "STEM = 'cctv-dgp-bank-png-group-gradients-v1'", "STEM = '" + STEM + "'")
    worker = replace_exact(worker, 'outputs/png_group_gradients_v1', 'outputs/png_group_gradients_v1_r1', 4)
    inverse = worker.replace(NAME, OLD_NAME).replace(STEM, 'cctv-dgp-bank-png-group-gradients-v1').replace(
        'outputs/png_group_gradients_v1_r1', 'outputs/png_group_gradients_v1')
    assert inverse == original_worker, 'Only the three identity fields may change'
    assert constants(worker) == {'NAME': NAME, 'STEM': STEM}
    shell_source = old_root / 'scripts/run_png_groups.sh'
    original_shell = shell_source.read_text(encoding='utf-8')
    shell = replace_exact(original_shell, OLD_NAME + '.py', NAME + '.py', 3)
    assert shell.replace(NAME + '.py', OLD_NAME + '.py') == original_shell
    assert '\r' not in worker + shell

    packet = OUT / NAME
    archive = OUT / (STEM + '-execution.tar.gz')
    receipt = OUT / (BASE + '_preparation.json')
    stop = OUT / (OLD_BASE + '_export_name_stop')
    assert not any(path.exists() for path in [packet, archive, receipt, stop])
    protected = read(OUT / 'cctv_dgp_bank_group_step_check_v1_analysis/local_protected_before.json')['files_sha256']
    assert all(sha(ROOT / name) == digest for name, digest in protected.items())
    forward_path = OUT / (OLD_BASE + '_forward_audit.json')
    forward = read(forward_path)
    assert forward['complete'] and forward['saved_forward_cases'] == 100
    assert forward['loss_helper_sha256'] == old['assets_sha256'][OLD_BASE + '_losses.py']

    stop.mkdir()
    shutil.copyfile(source, stop / 'original_worker.py')
    shutil.copyfile(ROOT / 'scripts' / ('prepare_' + OLD_BASE + '.py'), stop / 'original_preparer.py')
    write(stop / 'receipt.json', dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        location=source.relative_to(ROOT).as_posix() + ':20',
        cause='Worker export STEM has bank prefix; sealed archive and importer do not',
        fix='Fresh V1-R1 packet with one consistent root, export stem and result directory',
        original_packet_protocol_sha256=OLD_PIN, original_archive_sha256=OLD_ARCHIVE_SHA,
        sealed_original_files_sha256=preserved, original_not_approved_for_upload_or_run=True,
        original_human_VM_execution_unknown=True, no_agent_VM_connection_or_launch=True,
        human_reply='recheck again and give me command to upload just to be sure',
        original_packet_assets_and_archive_kept=True, scientific_recipe_and_limits_unchanged=True))

    packet.mkdir()
    (packet / 'scripts').mkdir()
    for name, digest in old['assets_sha256'].items():
        if name in ['scripts/' + OLD_NAME + '.py', 'scripts/run_png_groups.sh']:
            continue
        target = packet / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(old_root / name, target)
        assert sha(target) == digest
    new_text(packet / 'scripts' / (NAME + '.py'), worker)
    new_text(packet / 'scripts/run_png_groups.sh', shell)
    p = dict(old)
    p.update(name=NAME, stem=STEM, UTC=datetime.now(timezone.utc).isoformat(),
             identity_repair=dict(original_protocol_sha256=OLD_PIN,
                 original_archive_sha256=OLD_ARCHIVE_SHA, retained_name_stop_sha256=sha(stop / 'receipt.json'),
                 corrected_worker_export_stem=True, mathematical_source_unchanged=True,
                 source_inverse_comparison_exact=True, output_stage='outputs/png_group_gradients_v1_r1',
                 parent_cases_states_gates_and_all_execution_limits_unchanged=True))
    p['assets_sha256'] = {path.relative_to(packet).as_posix(): sha(path) for path in packet.rglob('*') if path.is_file()}
    assert len(p['assets_sha256']) == 6
    unchanged = {k: v for k, v in old.items() if k not in ['name', 'UTC', 'assets_sha256']}
    assert all(p[k] == v for k, v in unchanged.items())
    write(packet / 'protocol.json', p)
    pin = sha(packet / 'protocol.json')
    with tarfile.open(archive, 'x:gz', compresslevel=1) as tar:
        for file in sorted(path for path in packet.rglob('*') if path.is_file()):
            tar.add(file, arcname=NAME + '/' + file.relative_to(packet).as_posix(), recursive=False)
    digest = sha(archive)
    new_text(Path(str(archive) + '.sha256'), digest + '  ' + archive.name + '\n')

    importer_source = ROOT / 'scripts' / ('import_' + OLD_BASE + '_return.py')
    importer = importer_source.read_text(encoding='utf-8').replace(OLD_BASE, BASE).replace(
        'cctv-dgp-png-group-gradients-v1', STEM).replace(OLD_PIN, pin)
    assert constants(importer) == {'NAME': NAME, 'STEM': STEM, 'PIN': pin}
    importer_path = ROOT / 'scripts' / ('import_' + BASE + '_return.py')
    new_text(importer_path, importer)
    audit_source = ROOT / 'scripts' / ('audit_' + OLD_BASE + '_return.py')
    auditor = replace_exact(audit_source.read_text(encoding='utf-8'),
        'import_' + OLD_BASE + '_return', 'import_' + BASE + '_return')
    auditor = replace_exact(auditor, "out=RETURN/'outputs/png_group_gradients_v1'", "out=RETURN/'outputs/png_group_gradients_v1_r1'")
    ast.parse(auditor, feature_version=(3, 10))
    auditor_path = ROOT / 'scripts' / ('audit_' + BASE + '_return.py')
    new_text(auditor_path, auditor)
    assert all(sha(ROOT / name) == value for name, value in preserved.items())
    assert all(sha(ROOT / name) == value for name, value in protected.items())
    write(receipt, dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        preparer_sha256=sha(__file__), name=NAME, stem=STEM,
        protocol_sha256=pin, archive_sha256=digest, archive_bytes=archive.stat().st_size,
        original_name_stop_sha256=sha(stop / 'receipt.json'), original_files_sha256=preserved,
        worker_inverse_to_V1_exact_except_identity=True, shell_inverse_to_V1_exact_except_worker_filename=True,
        scientific_protocol_unchanged_except_identity_and_recovery_metadata=True,
        local_forward_value_audit_sha256=sha(forward_path), unchanged_local_saved_forward_cases=100,
        importer_sha256=sha(importer_path), prospective_return_auditor_sha256=sha(auditor_path),
        actual_manual_VM_diagnostic_unrun=True, original_user_execution_unknown=True,
        no_agent_VM_connection_or_launch=True, local_model_gradient_or_optimizer_calls=0,
        scientific_gates_and_original_checkpoint_preserved=True, transfer_audit_pending=True,
        model_qualified=False, goal_complete=False))
    print(dict(complete=True, name=NAME, protocol_sha256=pin, archive_sha256=digest, bytes=archive.stat().st_size))


if __name__ == '__main__':
    main()
