"""Read-only validation of four possible backed inactive recognizer duplicates."""
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

HOME = Path('/home/janusdominic0')
REPO = HOME / 'forensic-dgp'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert sys.platform == 'linux' and os.uname().nodename.split('.')[0] == 'forensic-dgp-thesis'
    assert Path.home().resolve() == HOME
    planpath = HOME / 'archive-duplicates_plan_20261007_v1.json'
    assert sha(planpath) == 'a7d1da403c0790ebc346235cfd7f1c19c6b31b780302fae58065196cdd638ea6'
    pinned = set(json.loads(planpath.read_text())['protected_assets_sha256'])
    retained = REPO / 'cctv_dgp_feature_skips_vm_v27/weights/w600k_r50.onnx'
    assert sha(retained) == '4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43'
    rows = []
    for folder in ('cctv_dgp_targets_vm_v6', 'cctv_dgp_capacity_vm_v7',
                   'cctv_dgp_face_code_fit_vm_v12', 'cctv_dgp_face_code_fit_vm_v12_r2'):
        p = REPO / folder / 'weights/w600k_r50.onnx'
        s = p.lstat()
        assert stat.S_ISREG(s.st_mode) and not p.is_symlink() and p.resolve() == p
        rows.append({'path': str(p), 'bytes': s.st_size, 'allocated_bytes': s.st_blocks * 512,
                     'uid': s.st_uid, 'nlink': s.st_nlink, 'sha256': sha(p),
                     'current_dependency_pinned': str(p) in pinned})
    print(json.dumps({'complete': True, 'rows': rows, 'retained_VM_copy': str(retained),
                     'current_uid': os.getuid(), 'files_removed': 0, 'model_or_training_calls': 0}, indent=2))


if __name__ == '__main__':
    main()
