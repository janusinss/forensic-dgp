"""Read-only maintenance inventory for the user-authorized existing VM cleanup."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def command(args, timeout=60):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {'returncode': result.returncode, 'stdout': result.stdout,
                'stderr': result.stderr[-1200:]}
    except subprocess.TimeoutExpired:
        return {'timed_out': True, 'command': args}


def usage(path, depth, count=70):
    result = command(['du', '-x', '-B1', '--max-depth=' + str(depth), str(path)], 90)
    rows = []
    for line in result.pop('stdout', '').splitlines():
        value, name = line.split('\t', 1)
        rows.append({'bytes': int(value), 'path': name})
    result['largest'] = sorted(rows, key=lambda row: row['bytes'], reverse=True)[:count]
    return result


def inventory():
    if sys.platform != 'linux' or os.uname().nodename.split('.')[0] != 'forensic-dgp-thesis':
        raise RuntimeError('Read only the existing forensic-dgp-thesis Linux VM')
    home = Path.home().resolve()
    repo = home / 'forensic-dgp'
    disk = shutil.disk_usage('/')
    archives = []
    for directory in [home, repo] + [path for path in repo.iterdir() if path.is_dir() and not path.is_symlink()]:
        for path in directory.iterdir():
            if path.is_file() and not path.is_symlink() and path.name.endswith(('.tar.gz', '.zip')):
                info = path.stat()
                archives.append({'path': str(path), 'bytes': info.st_size,
                                 'mtime_ns': info.st_mtime_ns, 'inode': info.st_ino})
    cache_paths = [home / '.cache/pip', home / '.cache/uv', home / '.cache/torch',
                   Path('/var/cache/apt/archives'), Path('/var/log/journal'), Path('/tmp')]
    processes = command(['ps', '-eo', 'pid,comm,args'], 10)
    processes['stdout'] = '\n'.join(line for line in processes['stdout'].splitlines()
        if ('forensic-dgp' in line or 'train_' in line or 'supervise_' in line)
        and 'base64' not in line and 'inspect_cctv_dgp_vm_storage' not in line)
    detail_roots = [repo / 'expanded_feature_bundle/outputs',
                    repo / 'cctv_dgp_broader_codes_vm_v16_r2/outputs',
                    repo / 'coverage_vm_bundle/outputs']
    details = {}
    for directory in detail_roots:
        if not directory.is_dir():
            continue
        files = []
        types = {}
        for path in directory.rglob('*'):
            if path.is_file() and not path.is_symlink():
                size = path.stat().st_size
                key = path.suffix or '(no extension)'
                entry = types.setdefault(key, {'files': 0, 'bytes': 0})
                entry['files'] += 1
                entry['bytes'] += size
                if size >= 50 * 1024**2:
                    files.append({'path': str(path), 'bytes': size})
        details[str(directory)] = {'usage': usage(directory, 3, 35), 'file_types': types,
                                   'largest_files': sorted(files, key=lambda row: row['bytes'], reverse=True)[:25]}
    print(json.dumps({'complete': True, 'hostname': os.uname().nodename, 'home': str(home),
        'disk': {'total_bytes': disk.total, 'used_bytes': disk.used, 'free_bytes': disk.free},
        'home_usage': usage(home, 1, 40), 'repo_usage': usage(repo, 2, 85),
        'var_usage': usage('/var', 2, 25), 'cache_usage': {str(path): usage(path, 1, 15)
            for path in cache_paths if path.exists()},
        'archives': sorted(archives, key=lambda row: row['bytes'], reverse=True), 'output_details': details,
        'GPU': command(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], 15),
        'tmux': command(['tmux', 'list-panes', '-a', '-F', '#{session_name}\t#{pane_current_command}\t#{pane_dead}'], 10),
        'related_processes': processes, 'training_started': False, 'files_removed': 0}, indent=2))


if __name__ == '__main__':
    inventory()
