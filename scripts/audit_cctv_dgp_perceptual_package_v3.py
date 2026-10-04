"""Independently inspect the additive V3 archive and recipe; no model execution."""
import argparse
import ast
import hashlib
from pathlib import Path, PurePosixPath
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from cctv_dgp_pilot import read, sha, write
from cctv_dgp_perceptual_training_v3 import configure_paths, verify_protocol_v3, START_SHA
from derive_cctv_dgp_perceptual_vm_v3 import derive_sources


def audit(root, bundle_dir, parent_return, archive):
    root, bundle_dir, parent_return, archive = map(Path, (root, bundle_dir, parent_return, archive))
    configure_paths(bundle_dir, parent_return)
    protocol = verify_protocol_v3(root)
    recipe = read(bundle_dir / 'perceptual_protocol_v3.json')
    for name, content in derive_sources().items():
        if (ROOT / name).read_text(encoding='utf-8') != content:
            raise ValueError('Declared source derivation differs: ' + name)
    outer = Path(str(archive) + '.sha256').read_bytes()
    expected_checksum = (sha(archive) + '  ' + archive.name + '\n').encode('ascii')
    if outer != expected_checksum:
        raise ValueError('Archive checksum or LF filename differs')
    expected = set(recipe['new_assets_sha256']) | {'perceptual_protocol_v3.json', 'perceptual_protocol_v3.sha256'}
    norm = read(ROOT / 'outputs/cctv_dgp_normfix_v2/normfix_v2.json')
    protected = set(read(root / 'protocol.json')['assets_sha256']) | set(norm['assets_sha256']) | {'protocol.json', 'protocol.sha256', 'normfix_v2.json'}
    if expected & protected:
        raise ValueError('Historical parent/runtime overwrite in overlay')
    checked = set()
    bytes_total = 0
    python_files = 0
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar.getmembers():
            path = PurePosixPath(member.name)
            if (not member.isfile() or member.issym() or member.islnk() or path.is_absolute()
                    or '..' in path.parts or '\\' in member.name or ':' in member.name
                    or not path.parts or path.parts[0] != 'cctv_dgp_vm_bundle'):
                raise ValueError('Unsafe/nonfile package member')
            name = PurePosixPath(*path.parts[1:]).as_posix()
            if name not in expected or name in checked:
                raise ValueError('Unexpected/duplicate package member: ' + name)
            bytes_total += member.size
            if member.size > 8 * 1024**2 or bytes_total > 12 * 1024**2 or len(checked) >= 50:
                raise ValueError('Finite additive package limits exceeded')
            with tar.extractfile(member) as stream:
                payload = stream.read()
            if hashlib.sha256(payload).hexdigest() != sha(bundle_dir / name):
                raise ValueError('Archive payload differs: ' + name)
            if name in recipe['new_assets_sha256'] and hashlib.sha256(payload).hexdigest() != recipe['new_assets_sha256'][name]:
                raise ValueError('Declared new asset differs: ' + name)
            if name.endswith('.sh') and b'\r' in payload:
                raise ValueError('VM launcher has CRLF')
            if name.endswith('.py'):
                ast.parse(payload.decode('utf-8-sig'), feature_version=(3, 10))
                python_files += 1
            checked.add(name)
    if checked != expected:
        raise ValueError('Incomplete additive package inventory')
    return {'complete': True, 'archive_sha256': sha(archive), 'archive_bytes': archive.stat().st_size,
            'protocol_sha256': sha(bundle_dir / 'perceptual_protocol_v3.json'),
            'package_files_checked': len(checked), 'python310_files_checked': python_files,
            'starting_checkpoint_sha256': START_SHA, 'parent_assets_unchanged': True,
            'data_order_identity_guardrails_unchanged': True, 'expected_total_updates': protocol['expected_total_updates'],
            'runtime_cap_seconds': 5400, 'local_model_forwards': 0, 'local_backward_calls': 0,
            'local_optimizer_updates': 0, 'new_vm_preflight_pending': True, 'training_pending': True,
            'native_reserved_used': False, 'production_checkpoint_promoted': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'outputs/cctv_dgp_vm_bundle_v1')
    parser.add_argument('--bundle-dir', type=Path, default=ROOT / 'outputs/cctv_dgp_perceptual_vm_v3')
    parser.add_argument('--parent-return', type=Path, default=ROOT / 'outputs/cctv_dgp_normfix_return_v2/outputs/cctv_dgp_pilot')
    parser.add_argument('--archive', type=Path, default=ROOT / 'outputs/cctv-dgp-perceptual-v3.tar.gz')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    report = audit(args.root, args.bundle_dir, args.parent_return, args.archive)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        write(args.receipt, report)
    print(report)
