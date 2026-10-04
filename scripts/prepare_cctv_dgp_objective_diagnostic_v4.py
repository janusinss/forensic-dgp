"""Build a new additive diagnostic overlay; no model execution or gradients."""
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cctv_dgp_pilot import read, sha, write
from cctv_dgp_perceptual_training_v3 import configure_paths, verify_protocol_v3, START_SHA, START_STATE
from cctv_dgp_objective_diagnostic_v4 import (training_groups, TERM_WEIGHTS, V3_RETURN_SHA,
    V3_AUDIT_SHA, verify_recipe)

PARENT = ROOT / 'outputs/cctv_dgp_vm_bundle_v1'
PERCEPTUAL = ROOT / 'outputs/cctv_dgp_perceptual_vm_v3'
RETURN = ROOT / 'outputs/cctv_dgp_normfix_return_v2/outputs/cctv_dgp_pilot'
V3_RETURN = ROOT / 'outputs/cctv_dgp_perceptual_return_v3/outputs/cctv_dgp_perceptual_v3'
DEST = ROOT / 'outputs/cctv_dgp_objective_vm_v4'
ARCHIVE = ROOT / 'outputs/cctv-dgp-objective-v4.tar.gz'
FILES = [
    'cctv_dgp_objective_diagnostic_v4.py',
    'scripts/run_cctv_dgp_objective_diagnostic_vm.py',
    'scripts/run_cctv_dgp_objective_diagnostic_vm.sh',
    'scripts/audit_cctv_dgp_objective_diagnostic_v4.py',
    'scripts/prepare_cctv_dgp_objective_diagnostic_v4.py',
    'tests/test_cctv_dgp_objective_diagnostic_v4.py',
    'CCTV_DGP_OBJECTIVE_DIAGNOSTIC_V4.md',
]
CAPSULE = ROOT / 'outputs/cctv_dgp_perceptual_return_v3/local_independent_audit.json'


def main():
    if DEST.exists() or ARCHIVE.exists() or Path(str(ARCHIVE) + '.sha256').exists():
        raise ValueError('Preserve existing diagnostic package/preparation')
    configure_paths(PERCEPTUAL, RETURN)
    protocol = verify_protocol_v3(PARENT)
    if sha(V3_RETURN / 'results.json') != V3_RETURN_SHA or sha(CAPSULE) != V3_AUDIT_SHA:
        raise ValueError('Independently audited V3 return differs')
    recipe = read(PERCEPTUAL / 'perceptual_protocol_v3.json')
    protected = (set(protocol['assets_sha256']) | set(recipe['new_assets_sha256'])
                 | set(recipe['existing_runtime_assets_sha256'])
                 | {'protocol.json', 'protocol.sha256', 'normfix_v2.json',
                    'perceptual_protocol_v3.json', 'perceptual_protocol_v3.sha256'})
    names = set(FILES) | {'v3_local_audit_for_diagnostic.json',
                         'objective_diagnostic_protocol_v4.json', 'objective_diagnostic_protocol_v4.sha256'}
    if names & protected:
        raise ValueError('Diagnostic would replace a historical asset')
    for name in FILES:
        if name.endswith('.sh') and b'\r' in (ROOT / name).read_bytes():
            raise ValueError('VM launcher requires LF')
    groups = training_groups(protocol)
    DEST.mkdir()
    for name in FILES:
        target = DEST / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    shutil.copyfile(CAPSULE, DEST / 'v3_local_audit_for_diagnostic.json')
    pins = {p.relative_to(DEST).as_posix(): sha(p) for p in sorted(DEST.rglob('*')) if p.is_file()}
    write(DEST / 'objective_diagnostic_protocol_v4.json', {
        'format': 'cctv-objective-zero-update-diagnostic-v4', 'date': '2026-10-04',
        'starting_checkpoint_sha256': START_SHA, 'starting_state_hash': START_STATE,
        'v3_return_results_sha256': V3_RETURN_SHA, 'v3_local_audit_sha256': V3_AUDIT_SHA,
        'perceptual_parent_protocol_sha256': sha(PERCEPTUAL / 'perceptual_protocol_v3.json'),
        'assets_sha256': pins, 'groups': groups, 'term_weights': TERM_WEIGHTS,
        'runtime_cap_seconds': 600, 'expected_autograd_grad_calls': 50,
        'optimizer_updates': 0, 'validation_references_used': 0, 'native_cases_used': 0,
        'native_reserved_used': False, 'vm_execution_pending': True,
        'purpose': 'Measure weighted objective gradient conflict on fixed training-only cases; no training/selection or quality claim.',
    })
    with (DEST / 'objective_diagnostic_protocol_v4.sha256').open('x', encoding='ascii', newline='\n') as stream:
        stream.write(sha(DEST / 'objective_diagnostic_protocol_v4.json') + '\n')
    verify_recipe(PARENT, DEST, PERCEPTUAL, RETURN, V3_RETURN)
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
           'protocol_sha256': sha(DEST / 'objective_diagnostic_protocol_v4.json'), 'new_files': len(names),
           'training_references': 40, 'optimizer_updates': 0, 'vm_execution_pending': True})


if __name__ == '__main__':
    main()
