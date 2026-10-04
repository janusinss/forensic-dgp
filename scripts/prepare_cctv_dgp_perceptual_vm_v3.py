"""Prepare an additive matched VM pilot; no model forwards, gradients or training."""
import copy
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from cctv_dgp_pilot import read, sha, write, verify_bundle
from cctv_dgp_perceptual_training_v3 import (ARMS, AUDIT_SHA, EXISTING_RUNTIME_ASSETS,
    PARENT_SHA, RETURN_SHA, START_SHA, START_STATE, configure_paths, verify_protocol_v3)
from derive_cctv_dgp_perceptual_vm_v3 import derive_sources

PARENT = ROOT / 'outputs/cctv_dgp_vm_bundle_v1'
RETURN = ROOT / 'outputs/cctv_dgp_normfix_return_v2/outputs/cctv_dgp_pilot'
DEST = ROOT / 'outputs/cctv_dgp_perceptual_vm_v3'
ARCHIVE = ROOT / 'outputs/cctv-dgp-perceptual-v3.tar.gz'
FILES = [
    'cctv_dgp_perceptual_v3.py', 'cctv_dgp_perceptual_training_v3.py',
    'scripts/run_cctv_dgp_perceptual_vm_v3.py', 'scripts/audit_cctv_dgp_perceptual_results_v3.py',
    'scripts/run_cctv_dgp_perceptual_vm_v3.sh', 'scripts/derive_cctv_dgp_perceptual_vm_v3.py',
    'scripts/prepare_cctv_dgp_perceptual_vm_v3.py', 'scripts/audit_cctv_dgp_perceptual_package_v3.py',
    'tests/test_cctv_dgp_perceptual_v3.py', 'tests/test_cctv_dgp_perceptual_training_v3.py',
    'CCTV_DGP_PERCEPTUAL_V3.md',
]
CAPSULES = {
    'parent_v2_local_audit.json': ROOT / 'outputs/cctv_dgp_normfix_return_v2/local_independent_audit.json',
    'perceptual_scales_v3.json': ROOT / 'outputs/cctv_dgp_perceptual_calibration_v3/scales.json',
    'calibration_protocol_v3.json': ROOT / 'outputs/cctv_dgp_perceptual_calibration_v3/protocol.json',
}


def main():
    if DEST.exists() or ARCHIVE.exists() or Path(str(ARCHIVE) + '.sha256').exists():
        raise ValueError('Preserve the existing V3 preparation/package')
    parent = verify_bundle(PARENT)
    if sha(PARENT / 'protocol.json') != PARENT_SHA:
        raise ValueError('Frozen parent protocol differs')
    norm = read(ROOT / 'outputs/cctv_dgp_normfix_v2/normfix_v2.json')
    protected = set(parent['assets_sha256']) | set(norm['assets_sha256']) | {'protocol.json', 'protocol.sha256', 'normfix_v2.json'}
    if protected & (set(FILES) | set(CAPSULES) | {'perceptual_protocol_v3.json', 'perceptual_protocol_v3.sha256'}):
        raise ValueError('V3 would overwrite a historical asset')
    for name, expected in derive_sources().items():
        if (ROOT / name).read_text(encoding='utf-8') != expected:
            raise ValueError('V3 source is not the declared derivation: ' + name)
    if sha(CAPSULES['parent_v2_local_audit.json']) != AUDIT_SHA:
        raise ValueError('Audited parent receipt differs')
    DEST.mkdir()
    pins = {}
    for name in FILES:
        target = DEST / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith('.sh'):
            if b'\r' in (ROOT / name).read_bytes():
                raise ValueError('Use LF in the VM launcher source')
        shutil.copyfile(ROOT / name, target)
        pins[name] = sha(target)
    for name, source in CAPSULES.items():
        shutil.copyfile(source, DEST / name)
        pins[name] = sha(DEST / name)
    expected = copy.deepcopy(parent)
    expected['arms'] = copy.deepcopy(ARMS)
    expected['weights']['dgp'] = 'outputs/cctv_dgp_pilot/camera_identity/best.pth'
    write(DEST / 'perceptual_protocol_v3.json', {
        'format': 'cctv-calibrated-perceptual-pilot-v3', 'date': '2026-10-04',
        'parent_protocol_sha256': PARENT_SHA, 'starting_checkpoint_sha256': START_SHA,
        'starting_state_hash': START_STATE, 'parent_return_results_sha256': RETURN_SHA,
        'parent_independent_audit_sha256': AUDIT_SHA, 'runtime_cap_seconds': 5400,
        'declared_difference': 'Matched continued postactivation versus training-only calibrated signed preactivation VGG supervision; identity retained in both arms',
        'new_assets_sha256': pins, 'existing_runtime_assets_sha256': EXISTING_RUNTIME_ASSETS,
        'protocol': expected, 'native_reserved_used': False, 'local_optimizer_updates': 0,
        'cuda_execution_pending': True, 'production_checkpoint_promoted': False,
    })
    with (DEST / 'perceptual_protocol_v3.sha256').open('x', encoding='ascii', newline='\n') as stream:
        stream.write(sha(DEST / 'perceptual_protocol_v3.json') + '\n')
    configure_paths(DEST, RETURN)
    verify_protocol_v3(PARENT)
    with tarfile.open(ARCHIVE, 'x:gz', compresslevel=5) as archive:
        for path in sorted(DEST.rglob('*')):
            if path.is_file():
                info = tarfile.TarInfo('cctv_dgp_vm_bundle/' + path.relative_to(DEST).as_posix())
                info.size, info.mtime, info.mode = path.stat().st_size, 0, 0o644
                with path.open('rb') as stream:
                    archive.addfile(info, stream)
    with Path(str(ARCHIVE) + '.sha256').open('x', encoding='ascii', newline='\n') as stream:
        stream.write(sha(ARCHIVE) + '  ' + ARCHIVE.name + '\n')
    print({'complete': True, 'archive_bytes': ARCHIVE.stat().st_size, 'archive_sha256': sha(ARCHIVE),
           'protocol_sha256': sha(DEST / 'perceptual_protocol_v3.json'), 'new_asset_files': len(pins),
           'actual_training_pending': True, 'local_optimizer_updates': 0})


if __name__ == '__main__':
    main()
