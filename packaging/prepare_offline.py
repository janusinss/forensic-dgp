"""Prepare a pinned Windows x64 offline release payload; never run downloaded code.

Only the build machine resolves packages. End-user setup consumes these exact
wheel bytes with local indexes disabled and every requirement hash-pinned.
"""
from __future__ import annotations

import argparse
from email.parser import Parser
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import urllib.parse
import urllib.request
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PYTHON_SPEC = dict(version='3.13.9', path='python/python-3.13.9-amd64.zip', bytes=33724497,
                   sha256='b87eb2b5d7a31220b56f3ab19b1817065af3dd57997aff752132bc13e7832915')
PYTHON_URL = 'https://www.python.org/ftp/python/3.13.9/python-3.13.9-amd64.zip'
GROUPS = ('application', 'capture')
LOCK_SOURCES = {group: ROOT / 'implementation' / 'locks' / ('windows-cp313-' + group + '.txt') for group in GROUPS}
MAX_PAYLOAD_BYTES = 4 * 1024**3
MAX_EXPANDED_BYTES = 16 * 1024**3
MAX_FILES = 4096


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 240 or '\\' in value or ':' in value:
        raise ValueError('Unsafe payload relative path.')
    for part in value.split('/'):
        if (not part or part in ('.', '..') or part.endswith(('.', ' ')) or
                re.search(r'[\x00-\x1f<>"|?*]', part) or
                re.match(r'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)', part, re.I)):
            raise ValueError('Unsafe payload relative path.')
    return value


def ordinary(path: Path, kind: str = 'file', allow_missing: bool = False) -> Path:
    path = Path(os.path.abspath(path))
    for index, current in enumerate((path, *path.parents)):
        try:
            info = current.lstat()
        except FileNotFoundError:
            if index == 0 and not allow_missing:
                raise ValueError('Required payload path is missing: ' + str(path)) from None
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 1024:
            raise ValueError('Payload paths cannot contain links or reparse points.')
        expected_directory = index > 0 or kind == 'directory'
        if expected_directory and not stat.S_ISDIR(info.st_mode):
            raise ValueError('Payload ancestor is not an ordinary directory.')
        if index == 0 and kind == 'file' and not stat.S_ISREG(info.st_mode):
            raise ValueError('Payload member is not an ordinary file.')
    return path


def normalized_name(name: str) -> str:
    return re.sub(r'[-_.]+', '-', name).lower()


def validate_lock(text: str) -> dict[str, tuple[str, str]]:
    if not isinstance(text, str) or len(text) > 256 * 1024:
        raise ValueError('Offline lock is missing or oversized.')
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        match = re.fullmatch(r'([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9][A-Za-z0-9.!+_-]*) --hash=sha256:([0-9a-f]{64})', line)
        if not match:
            raise ValueError('Every offline lock entry must be exactly version and SHA-256 pinned; remote/options entries are forbidden.')
        name, version, digest = match.groups()
        name = normalized_name(name)
        if name in result:
            raise ValueError('Duplicate package in offline lock: ' + name)
        result[name] = (version, digest)
    if not result:
        raise ValueError('An offline lock cannot be empty.')
    return result


def wheel_metadata(wheel: Path) -> tuple[str, str, object]:
    ordinary(wheel)
    if wheel.suffix != '.whl':
        raise ValueError('Only wheels may appear in an offline wheelhouse.')
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if name.count('/') == 1 and name.endswith('.dist-info/METADATA')]
        if len(names) != 1:
            raise ValueError('Wheel must contain exactly one METADATA file.')
        info = archive.getinfo(names[0])
        if info.file_size > 1024 * 1024:
            raise ValueError('Wheel metadata is oversized.')
        metadata = Parser().parsestr(archive.read(info).decode('utf-8'))
        name, version = metadata.get('Name'), metadata.get('Version')
        if not name or not version or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', name) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.!+_-]*', version):
            raise ValueError('Wheel metadata has an invalid package name/version.')
        return normalized_name(name), version, metadata


def wheel_lock(wheelhouse: Path, required: dict[str, str]) -> str:
    ordinary(wheelhouse, 'directory')
    records = {}
    for wheel in sorted(wheelhouse.iterdir(), key=lambda p:p.name.lower()):
        name, version, _ = wheel_metadata(wheel)
        if name in records:
            raise ValueError('Duplicate wheel package: ' + name)
        records[name] = (version, file_sha(wheel))
    for name, version in required.items():
        record = records.get(normalized_name(name))
        if not record or record[0] != version:
            raise ValueError('A required wheel is missing or its version differs: ' + name)
    if not records:
        raise ValueError('The wheelhouse is empty.')
    return '# Windows x64 CPython 3.13; generated from verified wheel bytes.\n' + ''.join(
        f'{name}=={version} --hash=sha256:{digest}\n' for name,(version,digest) in sorted(records.items()))


def source_models() -> list[dict]:
    result = []
    for group, filename in (('restoration', 'models.json'), ('completion', 'completion_assets.json')):
        value = json.loads((ROOT / 'implementation' / filename).read_text(encoding='utf8'))
        for identity, entry in sorted(value['assets'].items()):
            result.append(dict(group=group,id=identity,filename=entry['filename'],size=entry['size'],sha256=entry['sha256']))
    return result


def _archive_expanded_size(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        total = 0
        if len(archive.infolist()) > 100000:
            raise ValueError('Payload archive has too many members.')
        seen = set()
        for entry in archive.infolist():
            name = entry.filename.rstrip('/')
            if not name:
                continue
            safe_relative(name)
            if name.lower() in seen:
                raise ValueError('Payload archive contains duplicate names.')
            seen.add(name.lower())
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError('Payload archive links are forbidden.')
            total += entry.file_size
            if total > MAX_EXPANDED_BYTES:
                raise ValueError('Payload archive expands beyond its bound.')
        return total


def verify_payload(root: Path, *, python_spec: dict | None = None, model_specs: list[dict] | None = None) -> dict:
    root = ordinary(root, 'directory')
    manifest_path = ordinary(root / 'manifest.json')
    if manifest_path.stat().st_size > 1024 * 1024:
        raise ValueError('Offline payload manifest is oversized.')
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or set(manifest) != {'schema','platform','python','locks','files','models'}:
        raise ValueError('Unsupported offline payload manifest schema.')
    if type(manifest['schema']) is not int or manifest['schema'] != 1 or manifest['platform'] != 'win_amd64':
        raise ValueError('Offline payload must target Windows x64 schema 1.')
    python_spec = PYTHON_SPEC if python_spec is None else python_spec
    if manifest['python'] != python_spec:
        raise ValueError('Offline Python descriptor differs from the pinned runtime.')
    if manifest['locks'] != {group:'locks/'+group+'.txt' for group in GROUPS}:
        raise ValueError('Offline lock locations differ.')
    entries = manifest['files']
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_FILES:
        raise ValueError('Offline payload inventory is invalid.')
    inventory, folded = {}, set()
    payload_bytes, expanded = len(raw), 0
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {'path','bytes','sha256'}:
            raise ValueError('Invalid offline inventory entry.')
        name = safe_relative(entry['path'])
        if name == 'manifest.json' or name.lower() in folded:
            raise ValueError('Duplicate/case-colliding offline payload member.')
        folded.add(name.lower())
        if type(entry['bytes']) is not int or not 0 < entry['bytes'] <= MAX_PAYLOAD_BYTES or not isinstance(entry['sha256'],str) or not re.fullmatch('[0-9a-f]{64}',entry['sha256']):
            raise ValueError('Offline payload size/hash descriptor is invalid.')
        path = ordinary(root / name)
        if path.stat().st_size != entry['bytes'] or file_sha(path) != entry['sha256']:
            raise ValueError('Offline payload hash/checksum differs: ' + name)
        inventory[name] = entry
        payload_bytes += entry['bytes']
        if payload_bytes > MAX_PAYLOAD_BYTES:
            raise ValueError('Offline payload exceeds its size limit.')
        expanded += _archive_expanded_size(path) if path.suffix in ('.zip','.whl') else entry['bytes']
        if expanded > MAX_EXPANDED_BYTES:
            raise ValueError('Offline payload expands beyond its size limit.')
    actual = set()
    for member in root.rglob('*'):
        ordinary(member, 'directory' if member.is_dir() else 'file')
        if member.is_file():
            actual.add(member.relative_to(root).as_posix())
    if actual != set(inventory) | {'manifest.json'}:
        raise ValueError('Offline payload inventory has missing or unexpected files.')
    python_entry = inventory.get(python_spec['path'])
    if python_entry != {key:python_spec[key] for key in ('path','bytes','sha256')}:
        raise ValueError('Offline Python archive is not inventory-bound.')
    for group in GROUPS:
        lock_path = manifest['locks'][group]
        if lock_path not in inventory:
            raise ValueError('Offline lock file missing from inventory.')
        lock = validate_lock((root / lock_path).read_text(encoding='utf8'))
        wheelhouse = root / 'wheels' / group
        ordinary(wheelhouse, 'directory')
        wheel_records = {}
        for wheel in wheelhouse.iterdir():
            name, version, _ = wheel_metadata(wheel)
            if name in wheel_records:
                raise ValueError('Duplicate wheel package.')
            wheel_records[name] = (version,file_sha(wheel))
        if wheel_records != lock:
            raise ValueError('Offline wheel inventory differs from its hashed lock: ' + group)
    specifications = source_models() if model_specs is None else model_specs
    expected = {(m['group'],m['id']):m for m in specifications}
    models = manifest['models']
    if not isinstance(models,list) or len(models) != len(expected):
        raise ValueError('Offline model inventory differs from the admitted models.')
    seen_models = set()
    for model in models:
        if not isinstance(model,dict) or set(model) != {'group','id','filename','path'}:
            raise ValueError('Invalid offline model entry.')
        key = (model['group'],model['id'])
        source = expected.get(key)
        if not source or key in seen_models or model['filename'] != source['filename'] or model['path'] != 'models/' + source['filename']:
            raise ValueError('Offline model binding differs.')
        seen_models.add(key)
        entry = inventory.get(model['path'])
        if not entry or entry['bytes'] != source['size'] or entry['sha256'] != source['sha256']:
            raise ValueError('Offline model hash or size differs from its admitted source.')
    return dict(manifest=manifest, manifest_sha256=hashlib.sha256(raw).hexdigest(),
                payload_bytes=payload_bytes,payload_expanded_bytes=expanded)


def stage_payload(source: Path, destination: Path, expected_locks: dict[str,bytes], **verification) -> dict:
    verified = verify_payload(source, **verification)
    for group in GROUPS:
        actual = (source / verified['manifest']['locks'][group]).read_bytes().replace(b'\r\n',b'\n')
        if actual != expected_locks[group].replace(b'\r\n',b'\n'):
            raise ValueError('Offline payload lock differs from the committed source: ' + group)
    ordinary(destination, 'directory', allow_missing=True)
    if destination.exists() and any(destination.iterdir()):
        cached = verify_payload(destination, **verification)
        if cached['manifest_sha256'] != verified['manifest_sha256']:
            raise ValueError('Offline staging contains a different verified payload; existing files preserved.')
        return cached
    destination.mkdir(parents=True,exist_ok=True)
    for relative in ['manifest.json'] + [entry['path'] for entry in verified['manifest']['files']]:
        origin = ordinary(source / relative)
        target = ordinary(destination / relative, allow_missing=True)
        target.parent.mkdir(parents=True,exist_ok=True)
        with origin.open('rb') as input_stream, target.open('xb') as output_stream:
            shutil.copyfileobj(input_stream,output_stream,1024*1024)
    copied = verify_payload(destination, **verification)
    if copied['manifest_sha256'] != verified['manifest_sha256']:
        raise ValueError('Offline manifest changed during staging.')
    return copied


def _package_environment() -> dict:
    result = {key:value for key,value in os.environ.items() if not key.upper().startswith(('PIP_','UV_'))}
    result['PIP_CONFIG_FILE'] = os.devnull
    return result


def _run_pip(arguments: list[str]) -> None:
    subprocess.run([sys.executable,'-I','-m','pip','--isolated',*arguments],
                   env=_package_environment(),check=True,timeout=1800)


def _required_application() -> dict[str,str]:
    required = {}
    for line in (ROOT / 'requirements.txt').read_text(encoding='utf8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        match = re.fullmatch(r'([A-Za-z0-9._-]+)==([A-Za-z0-9.!+_-]+)',line)
        if not match:
            raise ValueError('Application requirements must be pinned.')
        required[normalized_name(match[1])] = match[2]
    required['uv'] = '0.13.4'
    required['pip'] = '26.2.2'
    if required.get('torch') != '2.13.0+cpu':
        raise ValueError('Review the pinned CPU Torch policy before preparing a different runtime.')
    return required


def capture_requirements() -> dict[str,str]:
    return {'opencv-python-headless':'4.13.0.92','numpy':'2.5.2','pip':'26.2.2'}


def download_wheels(output: Path, group: str, required: dict[str,str], wheel_source: Path | None) -> None:
    output.mkdir(parents=True,exist_ok=False)
    origin, cached = None, {}
    if wheel_source and (wheel_source / group).exists():
        origin = ordinary(wheel_source / group,'directory')
        for wheel in origin.iterdir():
            name, version, _ = wheel_metadata(wheel)
            if name in cached:
                raise ValueError('Ambiguous duplicate package in the supplied wheel cache: '+name)
            cached[name] = (version,wheel)
    platform = ['--only-binary=:all:','--implementation','cp','--python-version','313','--abi','cp313','--platform','win_amd64']
    if group == 'application':
        torch = cached.get('torch')
        if torch and torch[0] == required['torch']:
            shutil.copyfile(torch[1],output / torch[1].name)
        else:
            _run_pip(['download',*platform,'--no-deps','--index-url','https://download.pytorch.org/whl/cpu',
                      '--dest',str(output),'torch=='+required['torch']])
    # Ask the resolver to copy only the required closure. Old direct pins and
    # unrelated cached wheels never become stale extras in the output payload.
    complete_roots = all(name in cached and cached[name][0] == version for name,version in required.items())
    index = ['--no-index'] if complete_roots else ['--index-url','https://pypi.org/simple']
    links = ['--find-links',str(output)]
    if origin:
        links += ['--find-links',str(origin)]
    requirements = [name+'=='+version for name,version in sorted(required.items())]
    _run_pip(['download',*platform,*index,*links,'--dest',str(output),*requirements])
    wheel_lock(output, required)


def write_notices(wheelhouse: Path, destination: Path, group: str) -> None:
    # Wheel license trees can exceed MAX_PATH when nested below a build folder.
    # Keep exact license bytes in short names and record their original paths.
    notices, index = [], []
    for wheel in sorted(wheelhouse.iterdir()):
        name, version, metadata = wheel_metadata(wheel)
        notices.append(name+' '+version+'\nLicense: '+str(metadata.get('License-Expression') or metadata.get('License') or 'See included wheel license files')+'\n')
        with zipfile.ZipFile(wheel) as archive:
            for entry in archive.infolist():
                filename = entry.filename
                if not ('.dist-info/' in filename and any(token in filename.lower().split('/')[-1] for token in ('license','copying','notice'))):
                    continue
                if entry.is_dir():
                    continue
                if entry.file_size > 8*1024*1024:
                    raise ValueError('License file exceeds the reviewed size bound: '+filename)
                safe_relative(filename)
                base = re.sub(r'[^A-Za-z0-9._-]','_',filename.rsplit('/',1)[-1]).strip('.')[:48] or 'LICENSE'
                short = hashlib.sha256(filename.encode('utf8')).hexdigest()[:20] + '-' + base
                relative = group + '/' + name + '/' + short
                target = destination / relative
                target.parent.mkdir(parents=True,exist_ok=True)
                data = archive.read(entry)
                with target.open('xb') as stream:
                    stream.write(data)
                index.append(dict(package=name,version=version,wheel=wheel.name,original_path=filename,
                                  saved_path=relative,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    destination.mkdir(parents=True,exist_ok=True)
    (destination / (group+'-NOTICES.txt')).write_text('\n'.join(notices),encoding='utf8',newline='\n')
    (destination / (group+'-INDEX.json')).write_text(json.dumps(dict(schema=1,files=index),indent=2)+'\n',encoding='utf8',newline='\n')


def acquire_python(destination: Path, existing: Path | None) -> None:
    destination.parent.mkdir(parents=True,exist_ok=True)
    if existing:
        ordinary(existing)
        if existing.stat().st_size != PYTHON_SPEC['bytes'] or file_sha(existing) != PYTHON_SPEC['sha256']:
            raise ValueError('Provided Python archive differs from its pin.')
        shutil.copyfile(existing,destination)
    else:
        request = urllib.request.Request(PYTHON_URL, headers={'User-Agent':'ForensicDGP-ReleaseBuilder/0.2'})
        with urllib.request.urlopen(request,timeout=60) as response, destination.open('xb') as output:
            final = urllib.parse.urlparse(response.url)
            if final.scheme != 'https' or final.hostname != 'www.python.org':
                raise ValueError('Python archive redirected outside its official HTTPS host.')
            count = 0
            while block := response.read(1024*1024):
                count += len(block)
                if count > PYTHON_SPEC['bytes']:
                    raise ValueError('Python archive exceeds its expected size.')
                output.write(block)
        if destination.stat().st_size != PYTHON_SPEC['bytes'] or file_sha(destination) != PYTHON_SPEC['sha256']:
            raise ValueError('Downloaded Python archive differs from its pin.')
    with zipfile.ZipFile(destination) as archive:
        license_name = next((name for name in archive.namelist() if name.lower() in ('license.txt','license')),None)
        if not license_name:
            raise ValueError('The private Python archive must include its license.')
        license_path = destination.parents[1] / 'licenses' / 'PYTHON-LICENSE.txt'
        license_path.parent.mkdir(parents=True,exist_ok=True)
        license_path.write_bytes(archive.read(license_name))


def copy_models(destination: Path, sources: list[Path]) -> list[dict]:
    models = []
    for entry in source_models():
        found = None
        for source in sources:
            ordinary(source,'directory')
            for candidate in (source / entry['filename'], source / entry['group'] / entry['filename'],source / 'implementation' / entry['filename']):
                if candidate.exists():
                    ordinary(candidate)
                    if candidate.stat().st_size == entry['size'] and file_sha(candidate) == entry['sha256']:
                        found = candidate
                        break
            if found:
                break
        if not found:
            raise ValueError('Supply an explicit model source containing exact admitted bytes: '+entry['filename'])
        relative = 'models/' + entry['filename']
        target = destination / relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(found,target)
        models.append({key:entry[key] for key in ('group','id','filename')} | {'path':relative})
    return models


def prepare_payload(output: Path, model_sources: list[Path], python_archive: Path | None = None,
                    wheel_source: Path | None = None, write_locks: bool = False) -> dict:
    output = ordinary(output,'directory',allow_missing=True)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Preparation output must be absent or empty; existing payloads are preserved.')
    output.mkdir(parents=True,exist_ok=True)
    acquire_python(output / PYTHON_SPEC['path'],python_archive)
    models = copy_models(output,model_sources)
    required = {'application':_required_application(),'capture':capture_requirements()}
    for group in GROUPS:
        print('[payload] Preparing '+group+' wheels',flush=True)
        wheelhouse = output / 'wheels' / group
        download_wheels(wheelhouse,group,required[group],wheel_source)
        lock = wheel_lock(wheelhouse,required[group])
        lock_path = output / 'locks' / (group+'.txt')
        lock_path.parent.mkdir(parents=True,exist_ok=True)
        lock_path.write_text(lock,encoding='utf8',newline='\n')
        source_lock = LOCK_SOURCES[group]
        if write_locks:
            source_lock.parent.mkdir(parents=True,exist_ok=True)
            source_lock.write_text(lock,encoding='utf8',newline='\n')
        elif not source_lock.is_file() or source_lock.read_bytes().replace(b'\r\n',b'\n') != lock.encode():
            raise ValueError('Generated lock differs from source; review and explicitly use --write-locks before committing a release.')
        _run_pip(['install','--dry-run','--ignore-installed','--no-index','--find-links',str(wheelhouse),
                  '--require-hashes','--only-binary=:all:','-r',str(lock_path)])
        write_notices(wheelhouse,output / 'licenses',group)
    notices = ROOT / 'third_party'
    for source in sorted(notices.rglob('*')):
        if source.is_file() and source.name.lower() in ('license','license.txt','notice.md','training_attribution.json'):
            ordinary(source)
            relative = source.relative_to(notices)
            target = output / 'licenses' / 'application-source' / relative
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
    inventory = []
    for member in sorted(output.rglob('*'),key=lambda p:p.relative_to(output).as_posix()):
        if member.is_file():
            inventory.append(dict(path=member.relative_to(output).as_posix(),bytes=member.stat().st_size,sha256=file_sha(member)))
    manifest = dict(schema=1,platform='win_amd64',python=PYTHON_SPEC,
                    locks={group:'locks/'+group+'.txt' for group in GROUPS},files=inventory,models=models)
    (output / 'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    return verify_payload(output)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--model-source',action='append',default=[],type=Path)
    parser.add_argument('--python-archive',type=Path)
    parser.add_argument('--wheel-source',type=Path,help='Reuse available application/capture wheel directories; missing groups download from the fixed indexes.')
    parser.add_argument('--write-locks',action='store_true',help='Explicitly refresh tracked locks for review before committing.')
    parser.add_argument('--verify-only',action='store_true')
    args = parser.parse_args(argv)
    if args.verify_only:
        result = verify_payload(args.output)
    else:
        if os.name != 'nt' or sys.version_info[:2] != (3,13):
            raise ValueError('Prepare Windows x64 CPython 3.13 payloads on a Windows/Python 3.13 build host.')
        result = prepare_payload(args.output,args.model_source,args.python_archive,args.wheel_source,args.write_locks)
    print(json.dumps({key:value for key,value in result.items() if key != 'manifest'},indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError,ValueError,RuntimeError,zipfile.BadZipFile,subprocess.SubprocessError) as problem:
        print('[ERROR] '+str(problem),file=sys.stderr)
        raise SystemExit(1)
