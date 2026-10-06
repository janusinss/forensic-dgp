"""Freeze separate inference correction after audited L4 normalization evidence."""
import ast
from pathlib import Path
import shutil
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_input_selection_vm_v19_r2'
SOURCES = ['cctv_dgp_input_selection_v19_r2.py', 'dgp_input_normalization_v19_r2.py',
    'scripts/run_cctv_dgp_input_selection_v19_r2.py', 'scripts/audit_cctv_dgp_input_selection_v19_r2.py',
    'scripts/supervise_cctv_dgp_input_selection_v19_r2.py', 'scripts/launch_cctv_dgp_input_selection_v19_r2.py',
    'scripts/import_cctv_dgp_input_selection_v19_r2.py', 'scripts/prepare_cctv_dgp_input_selection_v19_r2.py',
    'tests/test_cctv_dgp_input_selection_v19_r2.py', 'tests/test_cctv_dgp_input_selection_v19_r2_runtime.py']


def prepare():
    start = time.monotonic()
    sys.path.insert(0, str(ROOT))
    import cctv_dgp_input_selection_v19_r2 as q
    old = ROOT / 'outputs/cctv_dgp_input_selection_vm_v19'
    failure = ROOT / 'outputs/cctv_dgp_input_selection_failure_return_v19'
    diag = ROOT / 'outputs/cctv_dgp_input_selection_v19_parity_diagnostic_return_r1'
    parent = ROOT / 'outputs/cctv_dgp_face_code_fit_vm_v12_r2'
    r2 = ROOT / 'outputs/cctv_dgp_broader_codes_vm_v16_r2'
    mixed = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'
    baseline = ROOT / 'outputs/cctv_dgp_generalization_return_v15/outputs/generalization_v15'
    hp, _, _ = q.original.verify(old, parent, r2, mixed, baseline, q.ORIGINAL_PIN)
    d = q.read(diag / 'results.json')
    audit = q.read(diag / 'local_independent_audit.json')
    q.require(q.sha(diag / 'results.json') == q.DIAGNOSTIC_RESULTS and d['complete']
              and d['normalization_hypothesis_confirmed'] and audit['complete']
              and audit['full_diagnostic_audit'] and audit['normalization_hypothesis_confirmed']
              and audit['archive_sha256'] == q.DIAGNOSTIC_ARCHIVE, 'Confirmed independently audited diagnostic required')
    q.require(not OUT.exists(), 'Preserve existing/partial R2; no regeneration')
    OUT.mkdir()
    def copy(name, source):
        q.require(time.monotonic() - start <= 300, 'R2 preparation exceeds300s; preserve partial bundle')
        target = q.safe(OUT, name); target.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith('.py'):
            ast.parse(source.read_text(encoding='utf-8'), feature_version=(3, 10))
        q.require(not target.exists(), 'No R2 overwrite')
        shutil.copyfile(source, target)
    for name in SOURCES:
        copy(name, ROOT / name)
    for name in sorted(set(hp['assets_sha256']) | {q.original.PLAN, 'protocol.sha256'}):
        copy('original_v19/' + name, old / name)
    # Keep the entire audited diagnostic, including both raw encodings and logs.
    for source in sorted(path for path in diag.rglob('*') if path.is_file()):
        copy('diagnostic/' + source.relative_to(diag).as_posix(), source)
    copy('lineage/diagnostic_export.json', ROOT / 'outputs/parity_diagnostic_export_v19_r1.json')
    copy('lineage/failed_execution.json', failure / 'outputs/input_selection_v19/execution.json')
    copy('lineage/original_failure.json', failure / 'supervisor_failure.json')
    copy('lineage/original_failure_audit.json', ROOT / 'outputs/cctv_dgp_input_selection_v19_failure_audit.json')
    assets = {path.relative_to(OUT).as_posix(): q.sha(path) for path in OUT.rglob('*') if path.is_file()}
    p = {**hp, 'format': q.FORMAT, 'design': q.DESIGN, 'assets_sha256': assets,
         'original_protocol_sha256': q.ORIGINAL_PIN, 'diagnostic_results_sha256': q.DIAGNOSTIC_RESULTS,
         'correction': 'Exact V15 canonical DGP comparison encoding plus unchanged V18 spatial encoding; no gate relaxation',
         'review_scope': 'Repeated photographic development validation; no independent final/native/Zamboanga claim'}
    q.write(OUT / q.PLAN, p)
    pin = q.sha(OUT / q.PLAN)
    with (OUT / 'protocol.sha256').open('x', encoding='ascii', newline='\n') as stream:
        stream.write(pin + '\n')
    q.verify(OUT, parent, r2, mixed, baseline, pin)
    names = sorted(set(assets) | {q.PLAN, 'protocol.sha256'})
    q.require(sum((OUT / name).stat().st_size for name in names) <= 128 * 1024**2,
              'R2 execution bundle exceeds128MiB')
    archive = ROOT / 'outputs/cctv-dgp-input-selection-v19-r2-execution.tar.gz'
    with tarfile.open(archive, 'x:gz', compresslevel=5) as stream:
        for name in names:
            q.require(time.monotonic() - start <= 300, 'R2 archive exceeds300s; preserve partial bundle')
            stream.add(OUT / name, arcname=name, recursive=False)
    digest = q.sha(archive)
    with Path(str(archive) + '.sha256').open('x', encoding='ascii', newline='\n') as stream:
        stream.write(digest + '  ' + archive.name + '\n')
    launcher = ROOT / 'outputs/launch_cctv_dgp_input_selection_v19_r2.py'
    q.require(not launcher.exists(), 'Preserve existing R2 external launcher')
    shutil.copyfile(OUT / 'scripts/launch_cctv_dgp_input_selection_v19_r2.py', launcher)
    q.write(ROOT / 'outputs/cctv_dgp_input_selection_v19_r2_preparation.json', {
        'complete': True, 'protocol_sha256': pin, 'original_protocol_sha256': q.ORIGINAL_PIN,
        'diagnostic_results_sha256': q.DIAGNOSTIC_RESULTS, 'archive_sha256': digest,
        'archive_bytes': archive.stat().st_size, 'members': len(names), 'source_assets': len(assets),
        'launcher_sha256': q.sha(launcher), 'validation_references': 104, 'validation_cases': 520,
        'training_parity_cases': 50, 'expected_neural_forwards': q.expected_counts(),
        'original_scientific_gates_changed': False, 'local_neural_forwards': 0,
        'backward_calls': 0, 'optimizer_updates': 0, 'VM_launched': False,
        'seconds': time.monotonic() - start, 'production_promoted': False})
    print((ROOT / 'outputs/cctv_dgp_input_selection_v19_r2_preparation.json').read_text(), flush=True)


if __name__ == '__main__':
    prepare()
