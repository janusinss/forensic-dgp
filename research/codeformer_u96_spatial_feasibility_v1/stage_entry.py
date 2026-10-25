"""Byte-bound staged bootstrap; no framework/model imports or automatic launch."""
import hashlib
import importlib
import json
from pathlib import Path
import stat
import sys

sys.dont_write_bytecode = True
PACKAGE = 'research/codeformer_u96_spatial_feasibility_v1'


def _ordinary(path):
    for current in (path, *path.parents):
        state = current.lstat()
        expected = stat.S_ISREG(state.st_mode) if current == path else stat.S_ISDIR(state.st_mode)
        if not expected or stat.S_ISLNK(state.st_mode) or getattr(state, 'st_file_attributes', 0) & 1024:
            raise ValueError('Ordinary unlinked bootstrap source required')
        if current == path and state.st_nlink != 1:
            raise ValueError('Single-link bootstrap source required')
    return path


def main():
    # --help imports only the inert reviewed CLI and does not touch models.
    root = Path(__file__).resolve().parents[2]
    if '--help' not in sys.argv[1:]:
        args = sys.argv[1:]
        if '--protocol-sha' not in args:
            raise ValueError('Explicit protocol SHA required before staged imports')
        index = args.index('--protocol-sha')
        if index + 1 >= len(args):
            raise ValueError('Missing protocol SHA')
        raw = _ordinary(root / 'protocol.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != args[index + 1]:
            raise ValueError('Exact supplied protocol SHA required')
        def pairs(items):
            answer = {}
            for key, value in items:
                if key in answer:
                    raise ValueError('Duplicate bootstrap JSON key')
                answer[key] = value
            return answer
        p = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs)
        for relative in (PACKAGE + '/__init__.py', PACKAGE + '/stage_entry.py',
                         PACKAGE + '/entry.py', PACKAGE + '/packet.py',
                         'scripts/run_u96_spatial_feasibility_v1.py'):
            path = _ordinary(root / relative)
            data = path.read_bytes()
            if p['files'][relative] != {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}:
                raise ValueError('Sealed bootstrap/helper differs: ' + relative)
    sys.path.insert(0, str(root))
    entry = importlib.import_module('research.codeformer_u96_spatial_feasibility_v1.entry')
    if Path(entry.__file__).resolve() != root / PACKAGE / 'entry.py':
        raise ValueError('Active CLI belongs to another stage')
    return entry.main()


if __name__ == '__main__':
    raise SystemExit(main())
