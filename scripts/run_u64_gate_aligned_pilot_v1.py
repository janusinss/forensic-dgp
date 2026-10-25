"""Sealed stdlib-only bridge; no diagnostic runs on import or --help."""
import hashlib
import importlib
import json
from pathlib import Path
import stat
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    package = 'research/codeformer_u64_gate_aligned_pilot_v1'
    if '--help' not in sys.argv[1:]:
        args = sys.argv[1:]
        if '--protocol-sha' not in args:
            raise ValueError('Explicit protocol SHA required before staged imports')
        index = args.index('--protocol-sha')
        if index+1 >= len(args):
            raise ValueError('Missing protocol SHA')
        def ordinary(path):
            for current in (path, *path.parents):
                info = current.lstat()
                if (not (stat.S_ISREG(info.st_mode) if current == path else stat.S_ISDIR(info.st_mode))
                    or stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 1024
                    or current == path and info.st_nlink != 1):
                    raise ValueError('Ordinary unlinked bootstrap source required')
            return path
        protocol_file = ordinary(root/'protocol.json')
        if protocol_file.stat().st_size > 4*1024**2:
            raise ValueError('Bounded protocol before read')
        raw = protocol_file.read_bytes()
        if len(raw) > 4*1024**2 or hashlib.sha256(raw).hexdigest() != args[index+1]:
            raise ValueError('Exact bounded supplied protocol required')
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError('Duplicate bootstrap JSON key')
                result[key] = value
            return result
        p = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs)
        for relative in ('research/__init__.py', package+'/__init__.py', package+'/runtime.py', 'scripts/run_u64_gate_aligned_pilot_v1.py'):
            data = ordinary(root/relative).read_bytes()
            if p['files'][relative] != {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}:
                raise ValueError('Sealed bootstrap source differs: '+relative)
    sys.dont_write_bytecode = True; sys.path.insert(0, str(root))
    runtime = importlib.import_module('research.codeformer_u64_gate_aligned_pilot_v1.runtime')
    if Path(runtime.__file__).resolve() != root/package/'runtime.py':
        raise ValueError('Use own staged runtime')
    return runtime.main()


if __name__ == '__main__':
    raise SystemExit(main())
