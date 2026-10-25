"""Standalone authenticated bootstrap; use -B in the manual transfer guide."""
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import stat
import sys

sys.dont_write_bytecode = True
PACKAGE = 'research/codeformer_conditioned_sft_learning_v1'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def ordinary(path, *, directory=False):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        info = item.lstat()
        expected = stat.S_ISDIR(info.st_mode) if item != path or directory else stat.S_ISREG(info.st_mode)
        need(expected and not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024,
             'Ordinary bootstrap ancestry required')
        if item == path and not directory:
            need(info.st_nlink == 1 and 0 < info.st_size <= 32 * 1024**2,
                 'Bounded single-link bootstrap source required')
    return path.resolve(strict=True)


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'Duplicate bootstrap JSON key refused')
            result[key] = value
        return result
    def reject(value):
        raise ValueError('Nonfinite bootstrap token: ' + value)
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)
    def finite(item, depth=0):
        need(depth <= 64, 'Bounded bootstrap JSON nesting required')
        if type(item) is float:
            need(math.isfinite(item), 'Finite bootstrap JSON required')
        elif type(item) is dict:
            for child in item.values(): finite(child, depth + 1)
        elif type(item) is list:
            for child in item: finite(child, depth + 1)
    finite(value)
    return value


def authenticate(root, protocol):
    """Bind own script, all package parents and direct imported components."""
    required = ('scripts/run_conditioned_sft_learning_v1.py', 'research/__init__.py',
                PACKAGE + '/__init__.py', PACKAGE + '/packet.py', PACKAGE + '/entry.py')
    for relative in required:
        path = ordinary(root / relative)
        raw = path.read_bytes()
        need(protocol['files'].get(relative) == {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()},
             'Sealed bootstrap source differs: ' + relative)


def main():
    if '--help' in sys.argv[1:] or '-h' in sys.argv[1:]:
        print('Manual conditioned-sft learning: --root PATH --protocol-sha SHA256 '
              '(--verify-transfer | --status | --run | --supervise | --export)')
        return 0
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--protocol-sha', required=True)
    args, _ = parser.parse_known_args()
    root = ordinary(args.root, directory=True)
    need(ordinary(__file__) == root / 'scripts/run_conditioned_sft_learning_v1.py',
         'Use bootstrap from the exact supplied stage')
    raw = ordinary(root / 'protocol.json').read_bytes()
    need(type(args.protocol_sha) is str and len(args.protocol_sha) == 64 and
         all(c in '0123456789abcdef' for c in args.protocol_sha) and
         hashlib.sha256(raw).hexdigest() == args.protocol_sha,
         'Exact supplied protocol SHA256 required before imports')
    protocol = strict_json(raw)
    authenticate(root, protocol)
    need('research' not in sys.modules, 'Fresh standalone bootstrap required')
    sys.path.insert(0, str(root))
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    module = importlib.import_module('research.codeformer_conditioned_sft_learning_v1.entry')
    need(Path(module.__file__).resolve() == root / PACKAGE / 'entry.py', 'CLI import origin differs')
    return module.main()


if __name__ == '__main__':
    raise SystemExit(main())
