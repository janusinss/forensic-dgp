"""Validate an immutable offline release payload before using any bundled bytes.

The installer supplies the manifest fingerprint from its embedded release record.
This module never downloads files or resolves packages against an online index.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import shutil
import time

from .setup_events import check_cancel, event


class OfflinePayloadError(RuntimeError):
    """The offline installer payload is incomplete, changed, or unsafe."""


def _ordinary(path: Path, directory: bool = False) -> None:
    info = path.lstat()
    if (stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 1024
            or not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))
            or (not directory and info.st_nlink != 1)):
        raise OfflinePayloadError('Offline payload paths must be ordinary unlinked files and directories.')


def _relative(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 240 or '\\' in value:
        raise OfflinePayloadError('Invalid relative offline payload path.')
    if PurePosixPath(value).is_absolute():
        raise OfflinePayloadError('Offline payload path escapes its root.')
    for part in value.split('/'):
        if (part in ('', '.', '..') or part.endswith(('.', ' '))
                or re.search(r'[\x00-\x1f<>"|?*:]', part)
                or re.match(r'(?i)^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)', part)):
            raise OfflinePayloadError('Invalid relative offline payload path.')
    return value


def _identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_nlink


def _digest(path: Path, expected: dict) -> None:
    _ordinary(path)
    before = path.lstat()
    if before.st_size != expected['bytes']:
        raise OfflinePayloadError('Offline payload file size differs: ' + path.name)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        if _identity(os.fstat(stream.fileno())) != _identity(before):
            raise OfflinePayloadError('Offline payload file changed before reading.')
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            check_cancel()
            digest.update(block)
        if _identity(os.fstat(stream.fileno())) != _identity(before):
            raise OfflinePayloadError('Offline payload file changed while reading.')
    if _identity(path.lstat()) != _identity(before) or digest.hexdigest() != expected['sha256']:
        raise OfflinePayloadError('Offline payload file failed SHA-256 verification: ' + path.name)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise OfflinePayloadError('Duplicate offline manifest field.')
        result[key] = value
    return result


def _lock_requirements(text: str):
    text=text.replace('\\\n',' ')
    result=[]; names=set()
    for line in text.splitlines():
        line=line.strip()
        if not line or line.startswith('#'):
            continue
        match=re.fullmatch(r'([A-Za-z0-9][A-Za-z0-9_.-]*)==([A-Za-z0-9][A-Za-z0-9.!+_-]*)(\s+--hash=sha256:[0-9a-f]{64})+',line)
        if not match:
            raise OfflinePayloadError('Offline lock must contain only exact hashed wheel requirements.')
        name,version=match.group(1,2); normalized=re.sub(r'[-_.]+','-',name).lower()
        if normalized in names:
            raise OfflinePayloadError('Offline lock contains duplicate packages.')
        names.add(normalized)
        result.append((name,version,re.findall(r'--hash=sha256:([0-9a-f]{64})',line)))
    if not result:
        raise OfflinePayloadError('Offline lock is empty.')
    return result


def committed_versions(group: str, code_root: Path | None = None):
    """Read exact dependency versions from source already covered by its receipt.

    Legacy releases without lock files keep their prior direct-pin health check.
    A malformed present lock is a verification failure, never a fallback.
    """
    if group not in ('application','capture'):
        raise OfflinePayloadError('Unknown committed package lock group.')
    code_root=Path(__file__).resolve().parents[1] if code_root is None else code_root
    path=code_root/'implementation'/'locks'/('windows-cp313-'+group+'.txt')
    if not path.exists() and not path.is_symlink():
        return None
    for parent in (path.parent,*path.parent.parents):
        _ordinary(parent,directory=True)
        if parent==code_root:
            break
    _ordinary(path)
    if path.stat().st_size>1024*1024:
        raise OfflinePayloadError('Committed package lock exceeds its size limit.')
    return [(name,version) for name,version,_ in _lock_requirements(path.read_text(encoding='utf-8'))]


class OfflinePayload:
    def __init__(self, root: Path, manifest: dict):
        self.root = root
        self.manifest = manifest
        self.files = {}
        if type(manifest) is not dict or set(manifest) != {'schema','platform','python','locks','files','models'}:
            raise OfflinePayloadError('Unsupported offline manifest fields.')
        if type(manifest['schema']) is not int or manifest['schema'] != 1 or manifest['platform'] != 'win_amd64':
            raise OfflinePayloadError('Unsupported offline manifest platform/schema.')
        if type(manifest['files']) is not list or not 1 <= len(manifest['files']) <= 2048:
            raise OfflinePayloadError('Invalid offline file inventory.')
        total = 0
        for entry in manifest['files']:
            if type(entry) is not dict or set(entry) != {'path','bytes','sha256'}:
                raise OfflinePayloadError('Invalid offline file inventory entry.')
            name = _relative(entry['path'])
            key = name.casefold()
            if name == 'manifest.json' or key in self.files or type(entry['bytes']) is not int or not 0 < entry['bytes'] <= 2 * 1024**3 or not isinstance(entry['sha256'],str) or not re.fullmatch('[0-9a-f]{64}',entry['sha256']):
                raise OfflinePayloadError('Invalid or duplicate offline file inventory entry.')
            self.files[key] = entry
            total += entry['bytes']
        if total > 4 * 1024**3:
            raise OfflinePayloadError('Offline payload exceeds its size limit.')
        python = manifest['python']
        if (type(python) is not dict or set(python) != {'version','path','bytes','sha256'}
                or python['version'] != '3.13.9' or python['path'] != 'python/python-3.13.9-amd64.zip'
                or type(python['bytes']) is not int or not isinstance(python['sha256'],str)
                or self.files.get(python['path'].casefold()) != {key:python[key] for key in ('path','bytes','sha256')}):
            raise OfflinePayloadError('Offline Python manifest binding differs.')
        if manifest['locks'] != {'application':'locks/application.txt','capture':'locks/capture.txt'}:
            raise OfflinePayloadError('Offline package lock paths differ.')
        observed = set()
        for candidate in root.rglob('*'):
            _ordinary(candidate, directory=candidate.is_dir())
            if candidate.is_file():
                name = candidate.relative_to(root).as_posix()
                observed.add(name.casefold())
        if observed != set(self.files) | {'manifest.json'}:
            raise OfflinePayloadError('Offline payload inventory differs from its directory.')
        for entry in self.files.values():
            self.file(entry['path'])
        self._versions = {}
        for group in ('application','capture'):
            self._versions[group] = self._validate_lock(group)
        self.models = {}
        if type(manifest['models']) is not list or len(manifest['models']) > 32:
            raise OfflinePayloadError('Invalid offline model inventory.')
        for entry in manifest['models']:
            if (type(entry) is not dict or set(entry) != {'group','id','filename','path'}
                    or entry['group'] not in ('restoration','completion')
                    or not isinstance(entry['id'],str) or not re.fullmatch('[a-z0-9_]{1,80}',entry['id'])
                    or _relative(entry['filename']) != PurePosixPath(entry['filename']).name
                    or not _relative(entry['path']).startswith('models/')
                    or entry['path'].casefold() not in self.files):
                raise OfflinePayloadError('Invalid offline model entry.')
            key = entry['group'],entry['id']
            if key in self.models:
                raise OfflinePayloadError('Duplicate offline model entry.')
            self.models[key] = entry

    def file(self, relative: str) -> Path:
        relative = _relative(relative)
        entry = self.files.get(relative.casefold())
        if entry is None or entry['path'] != relative:
            raise OfflinePayloadError('File is absent from the offline inventory.')
        path = self.root.joinpath(*relative.split('/'))
        for parent in path.parents:
            _ordinary(parent, directory=True)
            if parent == self.root:
                break
        _digest(path, entry)
        return path

    def _validate_lock(self, group: str):
        text = self.file(self.manifest['locks'][group]).read_text(encoding='utf-8')
        requirements=_lock_requirements(text)
        group_hashes = {item['sha256'] for item in self.files.values()
                        if item['path'].startswith('wheels/' + group + '/') and item['path'].endswith('.whl')}
        for name,version,hashes in requirements:
            if not set(hashes).issubset(group_hashes):
                raise OfflinePayloadError('Offline lock hashes differ from its wheels.')
        return [(name,version) for name,version,_ in requirements]

    def locked_versions(self, group: str):
        if group not in self._versions:
            raise OfflinePayloadError('Unknown offline package group.')
        return list(self._versions[group])

    def package_command(self, python: Path, group: str):
        self.locked_versions(group)
        prefix = []
        if os.name == 'nt':
            executable = os.path.abspath(str(python))
            if not re.match(r'^[A-Za-z]:[\\/]',executable) or Path(executable).parent.name.lower() != 'scripts' or Path(executable).name.lower() != 'python.exe':
                raise OfflinePayloadError('Offline packages require the computed local environment Python.')
            environment = str(Path(executable).parent.parent)
            prefix = ['--prefix', '\\\\?\\' + environment]
        return [str(python),'-I','-m','pip','--isolated','install','--disable-pip-version-check',
                *prefix,'--no-index','--find-links',str(self.root/'wheels'/group),'--require-hashes',
                '--only-binary=:all:','--no-cache-dir','-r',str(self.file(self.manifest['locks'][group]))]

    def model_source(self, group: str, identifier: str, asset: dict) -> Path:
        entry = self.models.get((group,identifier))
        if entry is None or entry['filename'] != asset['filename']:
            raise OfflinePayloadError('Required model is absent from the offline manifest.')
        file = self.files[entry['path'].casefold()]
        if file['bytes'] != asset['size'] or file['sha256'] != asset['sha256']:
            raise OfflinePayloadError('Offline model differs from the admitted application asset.')
        return self.file(entry['path'])


def package_environment() -> dict:
    result = {key:value for key,value in os.environ.items()
              if not key.upper().startswith(('PIP_','UV_'))}
    result['PIP_CONFIG_FILE'] = os.devnull
    return result


_PAYLOAD_CACHE = {}


def _snapshot(root: Path):
    result=[]
    for path in root.rglob('*'):
        directory=path.is_dir()
        _ordinary(path,directory=directory)
        info=path.lstat()
        result.append((path.relative_to(root).as_posix(),directory,*_identity(info),info.st_ctime_ns))
        if len(result)>8192:
            raise OfflinePayloadError('Offline payload path inventory exceeds its bound.')
    return tuple(sorted(result))


def load_payload() -> OfflinePayload | None:
    selected = os.environ.get('DGP_OFFLINE_PAYLOAD')
    fingerprint = os.environ.get('DGP_OFFLINE_PAYLOAD_SHA256')
    if selected is None and fingerprint is None:
        return None
    try:
        if (not selected or not Path(selected).is_absolute() or not isinstance(fingerprint,str)
                or not re.fullmatch('[0-9a-f]{64}',fingerprint)):
            raise OfflinePayloadError('Offline payload root and pinned manifest hash are both required.')
        root = Path(os.path.abspath(selected))
        for parent in (root,*root.parents):
            _ordinary(parent,directory=True)
        manifest_path = root/'manifest.json'
        _ordinary(manifest_path)
        if manifest_path.stat().st_size > 2 * 1024**2:
            raise OfflinePayloadError('Offline manifest exceeds its size limit.')
        data = manifest_path.read_bytes()
        if hashlib.sha256(data).hexdigest() != fingerprint:
            raise OfflinePayloadError('Offline manifest failed its pinned SHA-256 check.')
        cache_key=(str(root),fingerprint)
        snapshot=_snapshot(root)
        cached=_PAYLOAD_CACHE.get(cache_key)
        if cached is not None and cached[1]==snapshot:
            return cached[0]
        manifest = json.loads(data.decode('utf-8-sig'),object_pairs_hook=_unique_object)
        result=OfflinePayload(root,manifest)
        after=_snapshot(root)
        if after != snapshot:
            raise OfflinePayloadError('Offline payload changed during complete verification.')
        if len(_PAYLOAD_CACHE)>=4:
            _PAYLOAD_CACHE.clear()
        _PAYLOAD_CACHE[cache_key]=(result,after)
        return result
    except (OSError,ValueError,TypeError,KeyError) as exc:
        raise OfflinePayloadError('Offline payload could not be validated: ' + str(exc)) from exc


PIP_VERSION = '26.2.2'
PIP_WHEEL = 'wheels/application/pip-26.2.2-py3-none-any.whl'


def _private_python_root(selected: Path, *, staged: bool) -> Path:
    from .managed_paths import managed_root
    managed=managed_root()
    if managed is None:
        raise OfflinePayloadError('Derived private Python requires a verified managed installation.')
    selected=Path(os.path.abspath(selected))
    base=managed/'python'
    actual_base=selected.parent.parent if staged else selected.parent
    same=os.path.normcase(str(actual_base))==os.path.normcase(str(base))
    if not same:
        try:
            same=os.path.samefile(actual_base,base)
        except OSError:
            same=False
    if (not same or (staged and (selected.name!='python' or not re.fullmatch('python-bootstrap-[0-9a-f]{32}',selected.parent.name)))
            or (not staged and selected.name!='python-3.13.9-amd64')):
        raise OfflinePayloadError('Derived Python operation must stay in its computed owned staging/runtime directory.')
    for parent in (selected,*selected.parents):
        _ordinary(parent,directory=True)
    return selected


def _pip_proof(payload):
    if ('pip',PIP_VERSION) not in payload.locked_versions('application'):
        raise OfflinePayloadError('The offline application lock must include the reviewed pip version.')
    payload.file(PIP_WHEEL)
    return {'schema':1,'python_archive_sha256':payload.manifest['python']['sha256'],
            'pip_wheel_sha256':payload.files[PIP_WHEEL]['sha256'],'pip_version':PIP_VERSION,
            'ensurepip_patch':'25.2-to-26.2.2'}


def _verify_private_pip(python: Path) -> None:
    probe="import sys; prefix=chr(92)*2+'?'+chr(92) if sys.platform=='win32' else ''; sys.path=[prefix+p if prefix and len(p)>2 and p[1]==':' and not p.startswith(prefix) else p for p in sys.path]; import importlib.metadata as m,ensurepip; assert m.version('pip')=='26.2.2' and ensurepip.version()=='26.2.2', 'Private pip/ensurepip differs'"
    subprocess.run([str(python),'-I','-B','-c',probe],env=package_environment(),check=True,timeout=30)


def verify_private_python(selected: Path) -> None:
    root=_private_python_root(selected,staged=False)
    payload=load_payload()
    if payload is None:
        raise OfflinePayloadError('Offline private Python validation requires its pinned payload.')
    proof=_pip_proof(payload)
    record=root/'.dgp-derived-runtime.json'
    _ordinary(record)
    if record.stat().st_size>8192 or json.loads(record.read_text(encoding='utf-8'))!=proof:
        raise OfflinePayloadError('Derived private Python receipt differs from this ZIP and pip wheel.')
    _verify_private_pip(root/'python.exe')


def _remove_obsolete_pip_metadata(root: Path) -> None:
    # pip can leave deep license directories behind when the Windows global
    # long-path policy is off. This helper is restricted to a new owned staging
    # tree and runs only after the replacement pip metadata has been installed.
    root=_private_python_root(root,staged=True)
    old=root/'Lib'/'site-packages'/'pip-25.2.dist-info'
    extended=Path((chr(92)*2+'?'+chr(92)+str(old)) if os.name=='nt' else str(old))
    if not extended.exists():
        return
    replacement=root/'Lib'/'site-packages'/'pip-26.2.2.dist-info'/'METADATA'
    replacement=Path((chr(92)*2+'?'+chr(92)+str(replacement)) if os.name=='nt' else str(replacement))
    _ordinary(replacement)
    metadata=replacement.read_text(encoding='utf-8')
    if '\nName: pip\n' not in '\n'+metadata or '\nVersion: 26.2.2\n' not in '\n'+metadata:
        raise OfflinePayloadError('Replacement pip metadata was not verified; previous staged files were retained.')
    _ordinary(extended,directory=True)
    for item in extended.rglob('*'):
        _ordinary(item,directory=item.is_dir())
    shutil.rmtree(extended)


def prepare_private_python(selected: Path) -> None:
    """Patch only a new owned staging tree, never an active/shared Python.

    The immutable official ZIP remains a release input. Its derived runtime is
    recorded as that exact ZIP plus a verified pip wheel and one known anchor
    replacement, with both installed pip and ensurepip checked before publish.
    """
    root=_private_python_root(selected,staged=True)
    payload=load_payload()
    if payload is None:
        raise OfflinePayloadError('Offline private Python preparation requires its pinned payload.')
    proof=_pip_proof(payload)
    source=payload.file(PIP_WHEEL)
    ensurepip=root/'Lib'/'ensurepip'
    bundled=ensurepip/'_bundled'
    for parent in (ensurepip,bundled):
        _ordinary(parent,directory=True)
    init=ensurepip/'__init__.py'
    old_wheel=bundled/'pip-25.2-py3-none-any.whl'
    _ordinary(init); _ordinary(old_wheel)
    anchor=b'_PIP_VERSION = "25.2"'
    content=init.read_bytes()
    if content.count(anchor)!=1:
        raise OfflinePayloadError('The official ensurepip version anchor differs; staged files were preserved.')
    new_wheel=bundled/Path(PIP_WHEEL).name
    with source.open('rb') as reader,new_wheel.open('xb') as writer:
        for block in iter(lambda: reader.read(1024*1024),b''):
            check_cancel(); writer.write(block)
        writer.flush(); os.fsync(writer.fileno())
    _digest(new_wheel,{'bytes':source.stat().st_size,'sha256':proof['pip_wheel_sha256']})
    init.write_bytes(content.replace(anchor,b'_PIP_VERSION = "26.2.2"'))
    # Exact ordinary file in a newly extracted, owned staging tree only.
    _ordinary(old_wheel)
    old_wheel.unlink()
    check_cancel()
    lock=root/'.dgp-base-pip-lock.txt'
    with lock.open('x',encoding='utf-8',newline='\n') as writer:
        writer.write('pip=='+PIP_VERSION+' --hash=sha256:'+proof['pip_wheel_sha256']+'\n')
    prefix=str(root)
    if os.name=='nt':
        prefix='\\\\?\\'+prefix
    # Execute the verified pip wheel directly so the obsolete bundled pip is
    # never used to install packages. An explicit extended prefix also supports
    # hosts where the global LongPathsEnabled setting is disabled.
    invoke="import runpy,sys; sys.path.insert(0,sys.argv[1]); sys.argv=['-m',*sys.argv[2:]]; runpy.run_module('pip',run_name='__main__',alter_sys=True)"
    subprocess.run([str(root/'python.exe'),'-I','-B','-c',invoke,str(source),
                    '--isolated','install','--disable-pip-version-check','--no-index',
                    '--find-links',str(source.parent),'--require-hashes','--only-binary=:all:',
                    '--no-cache-dir','--prefix',prefix,'-r',str(lock)],
                   env=package_environment(),check=True,timeout=120)
    _remove_obsolete_pip_metadata(root)
    _verify_private_pip(root/'python.exe')
    record=root/'.dgp-derived-runtime.json'
    with record.open('x',encoding='utf-8',newline='\n') as writer:
        writer.write(json.dumps(proof,sort_keys=True)+'\n')
        writer.flush(); os.fsync(writer.fileno())



def protect_active_model(destination: Path, group: str, replacement: dict) -> None:
    """A verified model of the active release cannot be replaced in place.

    Future model-changing releases must use relocation until model generations
    have their own transaction. Corrupt/missing models remain repairable.
    """
    from .managed_paths import managed_root
    managed=managed_root()
    if managed is None:
        return
    selected=Path(os.path.abspath(destination))
    family={'restoration':'implementation','completion':'completion'}.get(group)
    if family is None:
        raise OfflinePayloadError('Unknown active model group.')
    expected=managed/'data'/'models'/family/replacement['filename']
    if os.path.normcase(str(selected))!=os.path.normcase(str(expected)):
        raise OfflinePayloadError('Active model repair escaped its computed installation directory.')
    current_path=managed/'current.json'
    if not current_path.exists() and not current_path.is_symlink():
        return
    def record(path,maximum=16384):
        for parent in path.parents:
            _ordinary(parent,directory=True)
            if parent==managed:
                break
        _ordinary(path)
        if path.stat().st_size>maximum:
            raise OfflinePayloadError('Previous installation metadata is oversized; use Move installation.')
        value=json.loads(path.read_text(encoding='utf-8-sig'),object_pairs_hook=_unique_object)
        if type(value) is not dict:
            raise OfflinePayloadError('Previous installation metadata differs; use Move installation.')
        return value
    current=record(current_path)
    if set(current)!={'schema','commit'} or type(current['schema']) is not int or current['schema']!=1 or not isinstance(current['commit'],str) or not re.fullmatch('[0-9a-f]{40}',current['commit']):
        raise OfflinePayloadError('Previous active version differs; use Move installation.')
    commit=current['commit']; receipt=record(managed/'receipts'/(commit+'.json'),16*1024**2)
    if receipt.get('schema')!=1 or type(receipt.get('schema')) is not int or receipt.get('commit')!=commit or type(receipt.get('files')) is not dict:
        raise OfflinePayloadError('Previous model source receipt differs; use Move installation.')
    relative='implementation/'+('models.json' if group=='restoration' else 'completion_assets.json')
    fingerprint=receipt['files'].get(relative)
    if not isinstance(fingerprint,str) or not re.fullmatch('[0-9a-f]{64}',fingerprint):
        raise OfflinePayloadError('Previous model source receipt is missing; use Move installation.')
    source=managed/'versions'/commit/relative
    prior=record(source,2*1024**2)
    _digest(source,{'bytes':source.stat().st_size,'sha256':fingerprint})
    if prior.get('schema_version')!=1 or type(prior.get('schema_version')) is not int or type(prior.get('assets')) is not dict:
        raise OfflinePayloadError('Previous model manifest differs; use Move installation.')
    admitted=[value for value in prior['assets'].values() if type(value) is dict and value.get('filename')==replacement['filename']]
    if len(admitted)!=1:
        raise OfflinePayloadError('Existing model ownership is uncertain; use Move installation.')
    prior_asset=admitted[0]
    if type(prior_asset.get('size')) is not int or prior_asset['size']<1 or not isinstance(prior_asset.get('sha256'),str) or not re.fullmatch('[0-9a-f]{64}',prior_asset['sha256']):
        raise OfflinePayloadError('Previous admitted model fingerprint differs; use Move installation.')
    _ordinary(selected)
    before=selected.lstat()
    if before.st_size!=prior_asset['size']:
        return
    digest=hashlib.sha256()
    with selected.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            check_cancel(); digest.update(block)
        if _identity(os.fstat(stream.fileno()))!=_identity(before):
            raise OfflinePayloadError('Previous model changed while checking; files were preserved.')
    if _identity(selected.lstat())!=_identity(before):
        raise OfflinePayloadError('Previous model changed while checking; files were preserved.')
    if digest.hexdigest()==prior_asset['sha256'] and (prior_asset['sha256']!=replacement['sha256'] or prior_asset['size']!=replacement['size']):
        raise OfflinePayloadError('A valid active release model needs a different fingerprint. Use Move installation; the working model was preserved.')


class CopyProgress:
    """Throttle GUI callbacks without reducing per-block cancellation checks."""
    def __init__(self, name, total, *, emit=None, clock=None):
        self.name=name; self.total=total; self.emit=event if emit is None else emit
        self.clock=time.monotonic if clock is None else clock
        self.last_bytes=0; self.last_time=self.clock()
        self.emit('download:'+name,'copying','Copying verified offline '+name,current=0,total=total)

    def update(self, current):
        now=self.clock()
        if current==self.total or (current-self.last_bytes>=8*1024*1024 and now-self.last_time>=0.1):
            self.emit('download:'+self.name,'copying',self.name,current=current,total=self.total)
            self.last_bytes=current; self.last_time=now
