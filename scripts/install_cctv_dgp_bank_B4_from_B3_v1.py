"""Manual, finite, data-only installation using hash-verified retained B3 assets."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import signal
import stat
import subprocess
import time
import urllib.request

SOURCE_NAME = 'cctv_dgp_bank_comparison_v1_r3_B_vm'
TARGET_NAME = 'cctv_dgp_bank_comparison_v1_r4_B_vm'
TRANSFER_NAME = 'cctv_dgp_bank_comparison_B4_transfer'
SOURCE_PIN = 'f34b7b54cd322d26a1a2159cffa28aabb1258fcb484914fd781d70e3e7bc0c35'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def relative(name):
    value = PurePosixPath(name)
    assert not value.is_absolute() and '..' not in value.parts and value.parts
    assert value.as_posix() == name and ':' not in name and '\\' not in name
    return value


def checked(root, name, expected):
    relative(name)
    path = root.joinpath(*PurePosixPath(name).parts)
    assert path.resolve().is_relative_to(root.resolve())
    assert stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink()
    assert not any(parent.is_symlink() for parent in path.parents)
    assert sha(path) == expected, 'Source/transfer bytes differ: ' + name
    return path


def verify_transfer(patch, pin):
    transfer = read(patch / 'transfer.json')
    assert transfer['complete'] and transfer['protocol_sha256'] == pin
    assert transfer['source_protocol_sha256'] == SOURCE_PIN
    assert transfer['source_name'] == SOURCE_NAME and transfer['target_name'] == TARGET_NAME
    assert transfer['copy_cap_seconds'] == 300 and transfer['model_gradient_or_training_calls'] == 0
    assert sha(patch / 'protocol.json') == pin
    p = read(patch / 'protocol.json')
    assert p['reserved_final_pixels'] == 0 and p['maximum_updates_per_route'] == 50
    assert p['budgets']['minimum_free_disk_bytes'] == 16*1024**3
    assert p['retained_B3_protocol_sha256'] == SOURCE_PIN
    source = read(checked(patch, 'source_protocol.json', SOURCE_PIN))
    assert set(transfer['overlay_assets']) <= set(p['assets_sha256'])
    assert all(p['assets_sha256'][name] == expected for name, expected in transfer['overlay_assets'].items())
    for name, expected in transfer['overlay_assets'].items():
        checked(patch / 'overlay', name, expected)
    copied = set(p['assets_sha256']) - set(transfer['overlay_assets'])
    assert all(p['assets_sha256'][name] == source['assets_sha256'][name] for name in copied)
    assert sha(patch / 'install_packet.py') == transfer['installer_sha256']
    actual = {path.relative_to(patch).as_posix() for path in patch.rglob('*') if path.is_file()}
    wanted = {'protocol.json', 'source_protocol.json', 'transfer.json', 'install_packet.py'} | {
        'overlay/' + name for name in transfer['overlay_assets']}
    assert actual == wanted and not any(path.is_symlink() for path in patch.rglob('*'))
    return transfer, p, source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol-sha', required=True)
    parser.add_argument('--verify-transfer', action='store_true')
    a = parser.parse_args()
    patch = Path(__file__).resolve().parent
    transfer, p, old = verify_transfer(patch, a.protocol_sha)
    if a.verify_transfer:
        print(json.dumps(dict(complete=True, overlay_assets=len(transfer['overlay_assets']),
            source_copy_pending=True, model_gradient_or_training_calls=0)), flush=True)
        return
    start = time.monotonic()
    assert platform.system() == 'Linux' and platform.node().split('.')[0] == 'forensic-dgp-thesis'
    assert Path.home().resolve() == Path('/home/janusdominic0') and os.getuid() == 1001
    assert patch == Path.home() / TRANSFER_NAME
    request = urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/id',
                                     headers={'Metadata-Flavor': 'Google'})
    assert urllib.request.urlopen(request, timeout=5).read().decode() == '4410777042005672095'
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Finite 300-second installation stop; retain partial root')))
    signal.alarm(300)
    source = Path.home() / 'forensic-dgp' / SOURCE_NAME
    target = Path.home() / 'forensic-dgp' / TARGET_NAME
    assert source.resolve() == source and target.resolve() == target
    assert source.is_dir() and not target.exists()
    assert (Path.home() / 'forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python').is_file()
    assert sha(source / 'protocol.json') == SOURCE_PIN
    failure_path = source / 'outputs/bank_comparison_v1/failure.json'
    failure = read(failure_path)
    assert failure['message'] == "module 'hashlib' has no attribute 'file_digest'"
    assert all(value == 0 for value in failure['progress']['counts'].values())
    supervisor_path = source / 'supervisor_receipt.json'
    supervisor = read(supervisor_path)
    assert supervisor['complete'] and supervisor['optimizer_updates'] == 0 and supervisor['trainer_exit_code'] == 1
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                         capture_output=True, text=True, check=True, timeout=15)
    assert not gpu.stdout.strip(), 'Active GPU workload; retain source and defer install'
    for name, expected in old['assets_sha256'].items():
        checked(source, name, expected)
    preserved = {str(path): sha(path) for path in [source / 'protocol.json', failure_path, supervisor_path]}
    copy_bytes = sum((patch / 'overlay' / name if name in transfer['overlay_assets'] else source / name).stat().st_size
                     for name in p['assets_sha256'])
    copy_bytes += (patch / 'protocol.json').stat().st_size + 64*1024**2
    free_before = shutil.disk_usage(source).free
    assert free_before - copy_bytes >= p['budgets']['minimum_free_disk_bytes'], 'Need16GiB free after copied installation'
    target.mkdir()
    try:
        for name, expected in p['assets_sha256'].items():
            origin = checked(patch / 'overlay' if name in transfer['overlay_assets'] else source, name, expected)
            dest = target.joinpath(*relative(name).parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            assert not dest.exists()
            with origin.open('rb') as incoming, dest.open('xb') as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=1024**2)
            assert sha(dest) == expected
        with (target / 'protocol.json').open('xb') as stream:
            stream.write((patch / 'protocol.json').read_bytes())
        assert sha(target / 'protocol.json') == a.protocol_sha
        assert all(sha(Path(path)) == expected for path, expected in preserved.items())
        free_after = shutil.disk_usage(target).free
        assert free_after >= p['budgets']['minimum_free_disk_bytes']
        receipt = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
            protocol_sha256=a.protocol_sha, source_protocol_sha256=SOURCE_PIN,
            transfer_sha256=sha(patch / 'transfer.json'), installer_sha256=sha(__file__),
            packet_assets_copied_and_rehashed=len(p['assets_sha256']),
            retained_source_records_sha256=preserved, original_failure_retained=True,
            source_files_deleted=0, model_gradient_or_training_calls=0,
            free_before_bytes=free_before, free_after_bytes=free_after,
            install_cap_seconds=300, seconds=time.monotonic()-start)
        with (target / 'packet_install_receipt.json').open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
        print(json.dumps(dict(complete=True, assets=len(p['assets_sha256']),
            free_GiB=free_after/1024**3, training_launched=False)), flush=True)
    except BaseException:
        # Keep the source, stop and partial target for independent investigation.
        raise


if __name__ == '__main__':
    main()
