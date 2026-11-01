"""Model-set namespaces bound to admitted bytes, never to a mutable active-model pointer.

Older application versions retain their legacy paths. New managed versions select
one family-specific inventory; only the application current.json activates code.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import uuid

from .managed_paths import ManagedPathError, _ordinary_existing, data_path, data_root

MARKER = '.model-set.json'
MANIFESTS = {'implementation': 'models.json', 'completion': 'completion_assets.json'}
MAXIMUM_METADATA = 2 * 1024 ** 2


class ModelStoreError(ManagedPathError):
    """A model namespace cannot be proved safe for this release."""


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ModelStoreError('Duplicate model metadata field.')
        result[key] = value
    return result


def _read(path):
    _ordinary_existing(path, directory=False)
    if path.stat().st_nlink != 1 or path.stat().st_size > MAXIMUM_METADATA:
        raise ModelStoreError('Model-set metadata is aliased or oversized.')
    return json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=_unique)


def _filename(name):
    return (isinstance(name, str) and 0 < len(name) <= 100
            and name not in ('.', '..') and not name.endswith(('.', ' '))
            and not re.search(r'[\\/\x00-\x1f<>"|?*:]', name)
            and not re.match(r'(?i)^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)', name))


def _selection(code_root, family):
    if family not in MANIFESTS:
        raise ModelStoreError('Unknown model family.')
    code = Path(os.path.abspath(code_root))
    data = data_root(code)
    if 'DGP_MANAGED_ROOT' not in os.environ:
        return data_path(code_root, 'models', family), None
    if not re.fullmatch('[0-9a-f]{40}', code.name):
        raise ModelStoreError('Model sets require the selected managed source version.')
    versions = data.parent / 'versions'
    if (os.path.normcase(str(code.parent)) != os.path.normcase(str(versions))
            and not os.path.samefile(code.parent, versions)):
        raise ModelStoreError('Model sets require the selected managed source version.')
    for directory in (versions, code, code / 'implementation', *code.parents):
        _ordinary_existing(directory, directory=True)
    body = _read(code / 'implementation' / MANIFESTS[family])
    if (type(body) is not dict or type(body.get('schema_version')) is not int
            or body['schema_version'] != 1 or type(body.get('assets')) is not dict
            or not 1 <= len(body['assets']) <= 32):
        raise ModelStoreError('Invalid admitted model inventory.')
    assets = {}
    names = set()
    for key, value in body['assets'].items():
        if (not isinstance(key, str) or not re.fullmatch('[A-Za-z0-9_-]{1,128}', key)
                or type(value) is not dict or not _filename(value.get('filename'))
                or type(value.get('size')) is not int or not 0 < value['size'] <= 512 * 1024 ** 2
                or not isinstance(value.get('sha256'), str)
                or not re.fullmatch('[0-9a-f]{64}', value['sha256'])):
            raise ModelStoreError('Unsafe or invalid admitted model asset.')
        name = value['filename'].casefold()
        if name in names:
            raise ModelStoreError('Case-aliased model filenames are refused.')
        names.add(name)
        assets[key] = {field: value[field] for field in ('filename', 'size', 'sha256')}
    canonical = json.dumps(assets, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()
    fingerprint = hashlib.sha256(canonical).hexdigest()
    proof = {'schema': 1, 'family': family, 'fingerprint': fingerprint, 'assets': assets}
    folder = data_path(code, 'models', family, fingerprint[:32])
    if os.name == 'nt' and any(len(str(folder / value['filename'])) >= 260 for value in assets.values()):
        raise ModelStoreError('Model filenames exceed the supported Windows path budget.')
    return folder, proof


def _validate(folder, proof):
    if not folder.exists() and not folder.is_symlink():
        return False
    _ordinary_existing(folder, directory=True)
    marker = folder / MARKER
    if marker.exists() or marker.is_symlink():
        value = _read(marker)
        if (type(value) is not dict or value != proof or type(value.get('schema')) is not int
                or type(value.get('assets')) is not dict):
            raise ModelStoreError('Model-set binding differs; existing weights were preserved.')
        # Reject Python equality aliases such as true == 1 in nested metadata.
        if json.dumps(value, sort_keys=True) != json.dumps(proof, sort_keys=True):
            raise ModelStoreError('Model-set binding types differ.')
        return True
    if next(folder.iterdir(), None) is not None:
        raise ModelStoreError('Nonempty model namespace has no binding; files were preserved.')
    return False


def model_directory(code_root, family):
    """Resolve the release's computed model family without creating or activating it."""
    try:
        folder, proof = _selection(code_root, family)
        if proof is not None:
            _validate(folder, proof)
        return folder
    except ModelStoreError:
        raise
    except (OSError, ValueError, TypeError, ManagedPathError) as exc:
        raise ModelStoreError('Model-set directory or manifest could not be verified: ' + str(exc)) from exc


def prepare_model_directory(code_root, family):
    """Bind an empty namespace before setup writes any weights; never replace a marker."""
    temporary = None
    try:
        folder, proof = _selection(code_root, family)
        if proof is None or _validate(folder, proof):
            return folder
        folder.mkdir(parents=True, exist_ok=True)
        # Repeat ordinary-path and marker checks after directory creation.
        checked, current_proof = _selection(code_root, family)
        if checked != folder or current_proof != proof:
            raise ModelStoreError('Model inventory changed during preparation.')
        if _validate(folder, proof):
            return folder
        temporary = folder / ('.model-set-' + uuid.uuid4().hex + '.partial')
        with temporary.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(proof, stream, sort_keys=True, separators=(',', ':'))
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        marker = folder / MARKER
        if os.name == 'nt':
            temporary.rename(marker)
        else:
            os.link(temporary, marker)
            temporary.unlink()
        _validate(folder, proof)
        return folder
    except ModelStoreError:
        raise
    except (OSError, ValueError, TypeError, ManagedPathError) as exc:
        raise ModelStoreError('Model-set preparation failed; previous sets were preserved: ' + str(exc)) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def admitted_model_path(code_root, family, asset):
    """Repair only an exact asset admitted by this fully bound managed set."""
    try:
        folder, proof = _selection(code_root, family)
        if proof is None or not _validate(folder, proof):
            raise ModelStoreError('Model repair requires a bound managed set.')
        if (type(asset) is not dict or type(asset.get('size')) is not int
                or not isinstance(asset.get('filename'), str)
                or not isinstance(asset.get('sha256'), str)):
            raise ModelStoreError('Invalid replacement model inventory.')
        selected = {field: asset[field] for field in ('filename', 'size', 'sha256')}
        if not any(selected == value for value in proof['assets'].values()):
            raise ModelStoreError('Replacement bytes are outside the bound model inventory.')
        return folder / asset['filename']
    except ModelStoreError:
        raise
    except (OSError, ValueError, TypeError, ManagedPathError) as exc:
        raise ModelStoreError('Model repair binding could not be verified: ' + str(exc)) from exc


def require_model_directory(code_root, family, expected, *, prepare=False):
    """Reject a stale imported default before setup can write into the old set."""
    selected = model_directory(code_root, family)
    if os.path.normcase(str(selected)) != os.path.normcase(str(expected)):
        raise ModelStoreError('Model inventory changed after initialization; restart setup.')
    if prepare:
        prepared = prepare_model_directory(code_root, family)
        if os.path.normcase(str(prepared)) != os.path.normcase(str(expected)):
            raise ModelStoreError('Model inventory changed during preparation; previous sets were preserved.')
    return selected
