"""Read-only runtime, removed-cache and retained-metadata checks after maintenance."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HOME = Path('/home/janusdominic0')
REPO = HOME / 'forensic-dgp'
MANIFEST_PIN = 'af7a583f06cbc9a245e9c93815231be53d53c76d4a529578b4551c8bc318e291'
BACKEND_PIN = 'a3cd8ebfc42872fd935627ab3702c30b22191e0d62dd5ebd856f4a83eadacdf4'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    started = time.monotonic()
    assert sys.platform == 'linux' and os.uname().nodename.split('.')[0] == 'forensic-dgp-thesis'
    assert Path.home().resolve() == HOME
    source = HOME / 'cache_manifest_for_cleanup_20261007_v2.json'
    assert sha(source) == MANIFEST_PIN
    manifest = json.loads(source.read_text())
    removed = [row for row in manifest['files'] if Path(row['path']).suffix in {'.bin', '.npz'}]
    retained = [row for row in manifest['files'] if Path(row['path']).suffix not in {'.bin', '.npz'}]
    assert len(removed) == 4429 and len(retained) == 2
    assert all(not os.path.lexists(row['path']) for row in removed)
    for row in retained:
        path = Path(row['path'])
        assert path.is_file() and not path.is_symlink() and path.stat().st_size == row['bytes']
        assert sha(path) == row['sha256']
    observed = {str(path) for root in manifest['roots'] for path in Path(root).rglob('*')
                if path.is_file() or path.is_symlink()}
    assert observed == {row['path'] for row in retained}
    python = REPO / 'cctv_dgp_vm_bundle/.venv/bin/python'
    assert python.is_file() and os.access(python, os.X_OK)
    program = ('import json,sys,torch,numpy,cv2;print(json.dumps({"python":sys.version.split()[0],'
               '"torch":torch.__version__,"numpy":numpy.__version__,"opencv":cv2.__version__,'
               '"CUDA_available":torch.cuda.is_available()}))')
    runtime = json.loads(subprocess.check_output([str(python), '-B', '-c', program], text=True, timeout=60))
    assert runtime['CUDA_available']
    backend = HOME / 'cleanup_cctv_dgp_vm_storage_20261007_v2_r1.py'
    assert sha(backend) == BACKEND_PIN
    namespace = {'__name__': 'read_only_maintenance_final', '__file__': str(backend)}
    exec(compile(backend.read_text(), str(backend), 'exec'), namespace)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        namespace['idle']()
    usage = json.loads(buffer.getvalue())['read_only_cache_usage_check']
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True, timeout=20).strip()
    assert gpu == 'NVIDIA L4'
    disk = shutil.disk_usage('/')
    assert disk.free >= 6 * 1024**3
    v30 = REPO / 'cctv_dgp_broader_mean_vm_v30'
    tmux = subprocess.run(['tmux', 'list-panes', '-a', '-F', '#{session_name} #{pane_current_command} #{pane_dead}'],
                          capture_output=True, text=True, timeout=20)
    assert tmux.returncode in (0, 1)
    print(json.dumps({'complete': True, 'hostname': os.uname().nodename, 'runtime': runtime,
                      'GPU': gpu, 'GPU_idle': True, 'active_cache_readers': usage['cache_readers'],
                      'processes_inspected': usage['processes_inspected'], 'tmux': tmux.stdout.strip(),
                      'source_cache_data_files_verified_absent': len(removed),
                      'retained_cache_metadata_files': len(retained),
                      'retained_cache_metadata_bytes': sum(row['bytes'] for row in retained),
                      'free_bytes': disk.free, 'used_bytes': disk.used, 'total_bytes': disk.total,
                      'V30_installed': v30.exists(), 'V30_outputs_present': (v30 / 'outputs').exists(),
                      'V30_execution_archive_present': (HOME / 'cctv-dgp-broader-mean-v30-execution.tar.gz').exists(),
                      'files_removed_by_this_check': 0, 'model_loaded': False,
                      'training_started': False, 'VM_stopped': False,
                      'seconds': time.monotonic() - started}, indent=2))


if __name__ == '__main__':
    main()
