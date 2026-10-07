"""Read-only exact inactive recognizer-copy and V30 dependency inventory."""
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time

HOME = Path('/home/janusdominic0')
REPO = HOME / 'forensic-dgp'
FOLDERS = ('cctv_dgp_detail_prior_vm_v22_r1', 'cctv_dgp_detail_skip_vm_v23',
           'cctv_dgp_degraded_detail_vm_v24', 'cctv_dgp_spatial_features_vm_v25',
           'cctv_dgp_batchmatched_identity_vm_v26')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    started = time.monotonic()
    assert sys.platform == 'linux' and os.uname().nodename.split('.')[0] == 'forensic-dgp-thesis'
    assert Path.home().resolve() == HOME
    rows = []
    for folder in FOLDERS:
        p = REPO / folder / 'weights/w600k_r50.onnx'
        assert p.is_file() and p.resolve() == p and not p.is_symlink()
        s = p.lstat()
        assert stat.S_ISREG(s.st_mode) and s.st_uid == os.getuid() and s.st_nlink == 1
        rows.append({'path': str(p), 'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns,
                     'inode': s.st_ino, 'allocated_bytes': s.st_blocks * 512, 'sha256': sha(p)})
    keep = REPO / 'cctv_dgp_feature_skips_vm_v27/weights/w600k_r50.onnx'
    assert keep.resolve() == keep and keep.is_file() and not keep.is_symlink()
    names = ('cctv_dgp_feature_skips_vm_v27', 'cctv_dgp_original_decoder_gradient_v1_r2_vm',
             'cctv_dgp_active_original_decoder_vm_v28', 'cctv_dgp_v28_preservation_diagnostic_v1_vm',
             'cctv_dgp_mean_centered_decoder_vm_v29', 'cctv_dgp_mixed_vm_v9_r2',
             'cctv_dgp_broader_mean_vm_v30')
    helpers = ('cleanup_cctv_dgp_vm_storage_20261007_v1.py',
               'independent_cctv_dgp_vm_storage_cleanup_20261007_v1.py',
               'archive-duplicates_plan_20261007_v1.json')
    assert not any((HOME / name).exists() for name in helpers), 'Fresh helper names must be unused'
    assert not (REPO / 'maintenance_storage_20261007_v1').exists(), 'Preserve existing maintenance evidence'
    print(json.dumps({'complete': True, 'hostname': os.uname().nodename,
        'duplicate_candidates': rows,
        'retained_VM_recognizer': {'path': str(keep), 'bytes': keep.stat().st_size, 'sha256': sha(keep)},
        'dependency_roots_present': {name: (REPO / name).is_dir() for name in names},
        'current_V30_archive_present': (HOME / 'cctv-dgp-broader-mean-v30-execution.tar.gz').is_file(),
        'fresh_maintenance_names_unused': True, 'files_removed': 0,
        'neural_or_training_calls': 0, 'seconds': time.monotonic() - started}, indent=2))


if __name__ == '__main__':
    main()
