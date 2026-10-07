"""Read-only remaining VM cleanup candidates; no removal, training or imports of models."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time

HOME = Path('/home/janusdominic0')
REPO = HOME / 'forensic-dgp'
START = time.monotonic()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            assert time.monotonic() - START < 240
            h.update(block)
    return h.hexdigest()


def command(args):
    r = subprocess.run(args, capture_output=True, text=True, timeout=25)
    return {'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr[-1500:]}


def row(path, info):
    return {'path': str(path), 'bytes': info.st_size, 'allocated_bytes': info.st_blocks * 512,
            'uid': info.st_uid, 'nlink': info.st_nlink}


def main():
    assert sys.platform == 'linux' and os.uname().nodename.split('.')[0] == 'forensic-dgp-thesis'
    assert Path.home().resolve() == HOME
    pinpath = HOME / 'archive-duplicates_plan_20261007_v1.json'
    assert sha(pinpath) == 'a7d1da403c0790ebc346235cfd7f1c19c6b31b780302fae58065196cdd638ea6'
    pinned = set(json.loads(pinpath.read_text())['protected_assets_sha256'])
    bytecode, markdown, errors = [], [], []
    directories = {}
    roots = (REPO, HOME / '.local', HOME / '.cache/pip', HOME / '.cache/torch', HOME / '.nv')
    for root in roots:
        if not root.is_dir() or root.is_symlink():
            continue
        for directory, names, files in os.walk(root, followlinks=False, onerror=lambda e: errors.append(str(e))):
            assert time.monotonic() - START < 240
            names[:] = [n for n in names if n != '.git' and not (Path(directory) / n).is_symlink()]
            dpath = Path(directory)
            info = dpath.stat()
            directories[str(dpath)] = {'files_direct': len(files), 'child_dirs': len(names),
                'allocated_bytes': info.st_blocks * 512, 'uid': info.st_uid}
            for name in files:
                path = dpath / name
                if path.is_symlink():
                    continue
                info = path.stat()
                if not stat.S_ISREG(info.st_mode):
                    continue
                if path.suffix == '.pyc' and path.parent.name == '__pycache__':
                    try:
                        source = Path(importlib.util.source_from_cache(str(path)))
                    except ValueError:
                        continue
                    if info.st_uid == os.getuid() and info.st_nlink == 1 and source.is_file() and not source.is_symlink() and str(path) not in pinned:
                        bytecode.append({**row(path, info), 'source_retained': str(source)})
                if path.suffix.lower() == '.md' and path.is_relative_to(REPO):
                    if any(part in {'.venv', 'venv', 'site-packages', 'node_modules'} for part in path.parts):
                        continue
                    markdown.append({**row(path, info), 'sha256': sha(path), 'current_dependency_pinned': str(path) in pinned})
    pip = HOME / '.cache/pip'
    pip_dirs = {name: value for name, value in directories.items() if Path(name).is_relative_to(pip)}
    usage = command(['du', '-x', '-B1', '--max-depth=2', str(HOME / '.cache')])
    apt = command(['du', '-x', '-B1', '--max-depth=1', '/var/cache/apt/archives'])
    apt_files = []
    for path in Path('/var/cache/apt/archives').glob('*.deb'):
        if path.is_file() and not path.is_symlink():
            apt_files.append(row(path, path.stat()))
    disk = shutil.disk_usage('/')
    groups = {}
    for item in bytecode:
        path = Path(item['path'])
        if path.is_relative_to(REPO):
            label = str(REPO / path.relative_to(REPO).parts[0])
        elif path.is_relative_to(HOME / '.local'):
            label = str(HOME / '.local')
        else:
            label = str(path.parent)
        g = groups.setdefault(label, {'files': 0, 'bytes': 0, 'allocated_bytes': 0})
        for key in ('bytes', 'allocated_bytes'):
            g[key] += item[key]
        g['files'] += 1
    print(json.dumps({'complete': True, 'hostname': os.uname().nodename, 'home': str(HOME),
        'current_uid': os.getuid(), 'disk': {'free_bytes': disk.free, 'used_bytes': disk.used, 'total_bytes': disk.total},
        'GPU': command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']),
        'tmux': command(['tmux', 'list-panes', '-a', '-F', '#{session_name}\t#{pane_current_command}\t#{pane_dead}']),
        'bytecode_with_retained_source': bytecode, 'bytecode_groups': groups,
        'research_tree_markdown': markdown,
        'pip_cache_directories': {'count': len(pip_dirs), 'direct_file_count': sum(r['files_direct'] for r in pip_dirs.values()),
            'directory_allocated_bytes': sum(r['allocated_bytes'] for r in pip_dirs.values()),
            'owned_by_current_user': all(r['uid'] == os.getuid() for r in pip_dirs.values())},
        'remaining_cache_usage': usage, 'APT_archive_usage': apt, 'APT_downloaded_deb_files': apt_files,
        'APT_clean_help': command(['apt-get', '--help']), 'read_errors': errors,
        'files_removed': 0, 'model_or_training_calls': 0, 'seconds': time.monotonic() - START}, indent=2))


if __name__ == '__main__':
    main()
