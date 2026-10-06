"""Read-only final runtime/cache check; no model loading, training or deletion."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HOME = Path('/home/janusdominic0')
REPO = HOME / 'forensic-dgp'
PHASE = REPO / 'maintenance_storage_20261006_v1/archive-duplicates'
CACHE_ROOTS = (
    REPO / 'expanded_feature_bundle/outputs/expanded_feature_training/anatomical_cache',
    REPO / 'expanded_feature_bundle/outputs/expanded_feature_training/fixed_cache',
    REPO / 'cctv_dgp_broader_codes_vm_v16_r2/outputs/broader_codes_v16_r2/cache',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    require(sys.platform == 'linux' and os.uname().nodename.split('.')[0] == 'forensic-dgp-thesis'
            and Path.home().resolve() == HOME, 'Only existing named VM/user')
    verification = json.loads((PHASE / 'verification.json').read_text())
    protected_path = PHASE / 'protected_before.json'
    require(sha(protected_path) == verification['protected_snapshot_sha256'], 'Protected snapshot differs')
    protected = json.loads(protected_path.read_text())
    groups = []
    for root in CACHE_ROOTS:
        require(root.is_dir() and not root.is_symlink() and root.resolve() == root, 'Cache root differs')
        expected = {name for name in set(protected['sha256']) | set(protected['scientific_tensor_stamps'])
                    if Path(name).is_relative_to(root)}
        observed = {str(path) for path in root.rglob('*') if path.is_file() or path.is_symlink()}
        require(expected == observed, 'Retained cache file set differs: ' + str(root))
        total = 0
        for name in sorted(observed):
            path = Path(name)
            require(not path.is_symlink() and path.resolve() == path, 'Cache path differs')
            value = path.stat()
            total += value.st_size
            if name in protected['scientific_tensor_stamps']:
                observed_stamp = {'bytes': value.st_size, 'mtime_ns': value.st_mtime_ns,
                                  'inode': value.st_ino, 'allocated_bytes': value.st_blocks * 512}
                require(observed_stamp == protected['scientific_tensor_stamps'][name], 'Cache stamp differs: ' + name)
            if name in protected['sha256']:
                require(sha(path) == protected['sha256'][name], 'Cache metadata hash differs: ' + name)
        groups.append({'root': str(root), 'files': len(observed), 'bytes': total})
    require(sum(row['files'] for row in groups) == 4431
            and sum(row['bytes'] for row in groups) == 39448585279, 'Retained cache totals differ')
    python = REPO / 'cctv_dgp_vm_bundle/.venv/bin/python'
    require(python.is_file() and os.access(python, os.X_OK), 'Existing runtime executable missing')
    code = ('import json,sys,torch,numpy,cv2;'
            'print(json.dumps({"python":sys.version.split()[0],"torch":torch.__version__, '
            '"numpy":numpy.__version__,"opencv":cv2.__version__, '
            '"CUDA_available":torch.cuda.is_available()}))')
    runtime = json.loads(subprocess.check_output([str(python), '-B', '-c', code], text=True, timeout=60))
    require(runtime['CUDA_available'], 'Existing CUDA runtime unavailable')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True, timeout=20).strip()
    require(gpu == 'NVIDIA L4', 'GPU differs')
    compute = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                                      text=True, timeout=20).strip()
    require(not compute, 'GPU task present; no further maintenance')
    tmux = subprocess.run(['tmux', 'list-panes', '-a', '-F', '#{session_name} #{pane_current_command}'],
                          capture_output=True, text=True, timeout=20)
    require(tmux.returncode in (0, 1), 'tmux inspection failed')
    disk = shutil.disk_usage('/')
    print(json.dumps({'complete': True, 'hostname': os.uname().nodename, 'runtime': runtime,
                      'GPU': gpu, 'GPU_idle': True, 'tmux': tmux.stdout.strip(),
                      'retained_cache_groups': groups, 'retained_cache_files': 4431,
                      'retained_cache_bytes': 39448585279, 'free_bytes': disk.free,
                      'used_bytes': disk.used, 'total_bytes': disk.total,
                      'files_removed': 0, 'model_loaded': False, 'training_started': False,
                      'VM_stopped': False, 'seconds': time.monotonic() - started}, indent=2))


if __name__ == '__main__':
    main()
