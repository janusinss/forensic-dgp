"""Read-only fingerprints of three historical VM feature-cache directories."""
import hashlib
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import time

HOME = Path('/home/janusdominic0')
REPO = HOME / 'forensic-dgp'
ROOTS = [REPO / 'expanded_feature_bundle/outputs/expanded_feature_training/anatomical_cache',
         REPO / 'expanded_feature_bundle/outputs/expanded_feature_training/fixed_cache',
         REPO / 'cctv_dgp_broader_codes_vm_v16_r2/outputs/broader_codes_v16_r2/cache']


def inspect():
    started = time.monotonic()
    if sys.platform != 'linux' or platform.node().split('.')[0] != 'forensic-dgp-thesis' or Path.home().resolve() != HOME:
        raise RuntimeError('Only the existing named VM/user may be inspected')
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                         capture_output=True, text=True, check=True, timeout=20)
    if gpu.stdout.strip():
        raise RuntimeError('Preserve active GPU work')
    files = []
    for root in ROOTS:
        if not root.is_dir() or root.is_symlink() or root.resolve() != root:
            raise ValueError('Missing/noncanonical cache directory: ' + str(root))
        for directory, dirs, names in os.walk(root, followlinks=False):
            if any((Path(directory) / name).is_symlink() for name in dirs):
                raise ValueError('Cache contains a symlink directory')
            for name in sorted(names):
                if time.monotonic() - started >= 1200:
                    raise TimeoutError('Read-only cache fingerprint cap1200 seconds')
                path = Path(directory) / name
                info = path.lstat()
                if path.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1 or path.resolve() != path:
                    raise ValueError('Require an owned canonical regular cache file')
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(4 * 1024**2), b''):
                        if time.monotonic() - started >= 1200:
                            raise TimeoutError('Read-only cache fingerprint deadline')
                        digest.update(block)
                after = path.stat()
                if (info.st_size, info.st_mtime_ns, info.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                    raise RuntimeError('Cache changed while fingerprinting')
                files.append({'path': str(path), 'root': str(root), 'relative': path.relative_to(root).as_posix(),
                    'bytes': info.st_size, 'sha256': digest.hexdigest(), 'inode': info.st_ino,
                    'mtime_ns': info.st_mtime_ns, 'allocated_bytes': info.st_blocks * 512,
                    'cleanup_candidate': path.suffix in {'.bin', '.npz'}})
                if len(files) % 500 == 0 or info.st_size > 1024**3:
                    print(json.dumps({'fingerprinted_files': len(files), 'bytes': sum(f['bytes'] for f in files),
                                      'seconds': time.monotonic() - started}), file=sys.stderr, flush=True)
    print(json.dumps({'complete': True, 'format': 'inactive-feature-cache-backup-manifest-r2',
        'roots': [str(root) for root in ROOTS], 'files': files,
        'candidate_bytes': sum(f['bytes'] for f in files if f['cleanup_candidate']),
        'all_backup_bytes': sum(f['bytes'] for f in files), 'seconds': time.monotonic() - started,
        'GPU_idle': True, 'files_removed': 0, 'training_started': False}, indent=2))


if __name__ == '__main__':
    inspect()
