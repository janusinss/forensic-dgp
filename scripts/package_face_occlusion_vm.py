"""Package new pilot files only; preserve original coverage inputs and evidence."""
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import tarfile


def sha_stream(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024*1024), b''):
        digest.update(chunk)
    return digest.hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return sha_stream(stream)


def build_bundle(root, files, inventory_path, archive):
    root = Path(root).resolve()
    inventory_path, archive = Path(inventory_path).resolve(), Path(archive).resolve()
    checksum = archive.with_name(archive.name+'.sha256')
    if not all(p.is_relative_to(root) for p in (inventory_path, archive, checksum)):
        raise ValueError('Package destination escapes workspace')
    if any(p.exists() for p in (inventory_path, archive, checksum)):
        raise ValueError('Preserve existing package/inventory/checksum')
    if len(files) != len(set(files)) or not files:
        raise ValueError('Require unique nonempty member paths')
    inventory = {}
    for relative in sorted(files):
        path = PurePosixPath(relative)
        if path.is_absolute() or '..' in path.parts or '\\' in relative or PureWindowsPath(relative).drive:
            raise ValueError('Archive member must be a relative workspace path')
        source = (root/relative).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            raise ValueError('Missing member or path escapes workspace')
        inventory[relative] = sha(source)
    inventory_relative = inventory_path.relative_to(root).as_posix()
    if inventory_relative in inventory:
        raise ValueError('Inventory cannot hash itself')
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_bytes((json.dumps(inventory,indent=2)+'\n').encode())
    archive.parent.mkdir(parents=True, exist_ok=True)
    expected = {**inventory, inventory_relative: sha(inventory_path)}
    with archive.open('xb') as stream:
        with gzip.GzipFile(filename='', fileobj=stream, mode='wb', mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w') as tar:
                for relative in expected:
                    source = root/relative
                    info = tar.gettarinfo(str(source), arcname=relative)
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ''
                    info.mode = 0o644
                    with source.open('rb') as data:
                        tar.addfile(info, data)
    with tarfile.open(archive,'r:gz') as tar:
        members = tar.getmembers()
        if len(members) != len(expected) or {m.name for m in members} != set(expected):
            raise ValueError('Archive membership differs')
        for member in members:
            if not member.isfile() or sha_stream(tar.extractfile(member)) != expected[member.name]:
                raise ValueError(f'Archive bytes differ: {member.name}')
    digest = sha(archive)
    checksum.write_bytes((digest+'  '+archive.name+'\n').encode())
    return dict(files=len(expected), bytes=archive.stat().st_size,
                uncompressed_bytes=sum((root/p).stat().st_size for p in expected),
                sha256=digest, archive_verified=True, checksum_line_endings='LF')


def main():
    root = Path(__file__).resolve().parents[1]
    registry_path = root/'outputs/face_occlusion_initial_v1/manifest.json'
    if sha(registry_path) != '7fb3fe1f5e4c2fc8a5266e87dfcd9e84d1bf07728145ebaebe5d487a52047650':
        raise ValueError('Initial registry changed')
    registry = json.loads(registry_path.read_text())
    if registry['optimizer_updates'] != 0 or not registry['heads_equal']:
        raise ValueError('Require matched untrained initial states')
    for entry in registry['arms'].values():
        if sha(root/entry['path']) != entry['sha256']:
            raise ValueError('Exported initialization changed')
    for path, key in [('face_occlusion_adapter.py','adapter_sha256'),
                      ('scripts/export_face_occlusion_initial.py','exporter_sha256')]:
        if sha(root/path) != registry[key]:
            raise ValueError('Initialization implementation changed')
    old = root/'outputs/coverage_protocol_v1/vm_inventory.json'
    if sha(old) != 'c831635933c9b615b4849ad23811a3b82ebde1883aa30a5c2249b8879fa95623':
        raise ValueError('Original coverage inventory changed')
    files = ['face_occlusion_adapter.py', 'scripts/train_face_occlusion_vm.py',
             'scripts/export_face_occlusion_initial.py', 'scripts/setup_face_occlusion_vm.sh',
             'scripts/package_face_occlusion_vm.py', 'requirements_face_occlusion_vm.txt',
             'FACE_OCCLUSION_PILOT.md', 'FACE_OCCLUSION_VM.md', 'FACE_EXTRACTION_LICENSE.txt',
             'outputs/face_occlusion_initial_v1/manifest.json',
             'outputs/face_occlusion_initial_v1/pretrained.pth',
             'outputs/face_occlusion_initial_v1/random.pth',
             'tests/test_face_occlusion_adapter.py', 'tests/test_face_occlusion_vm.py',
             'tests/test_face_occlusion_package.py']
    if set(files) & set(json.loads(old.read_text())):
        raise ValueError('New package would overwrite original coverage inputs')
    shell = (root/'scripts/setup_face_occlusion_vm.sh').read_bytes()
    if b'\r' in shell or b'\r' in (root/'requirements_face_occlusion_vm.txt').read_bytes():
        raise ValueError('Use Linux line endings for shell and requirement files')
    inventory = root/'outputs/face_occlusion_bundle_v1/inventory.json'
    archive = root/'outputs/face-occlusion-vm-code.tar.gz'
    report = build_bundle(root, files, inventory, archive)
    (inventory.parent/'build.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
