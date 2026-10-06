"""Freeze finite V19 development inference after positive training-only processing."""
import ast
from pathlib import Path
import shutil
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_input_selection_vm_v19'
SOURCES = ['cctv_dgp_input_selection_v19.py', 'dgp_input_selector_v19.py',
    'cctv_dgp_structure_audit_numeric_v18_r2.py',
    'scripts/run_cctv_dgp_input_selection_v19.py', 'scripts/audit_cctv_dgp_input_selection_v19.py',
    'scripts/supervise_cctv_dgp_input_selection_v19.py', 'scripts/launch_cctv_dgp_input_selection_v19.py',
    'scripts/import_cctv_dgp_input_selection_v19.py', 'scripts/prepare_cctv_dgp_input_selection_v19.py',
    'tests/test_dgp_input_selector_v19.py', 'tests/test_cctv_dgp_input_selection_v19.py']


def prepare():
    start = time.monotonic()
    sys.path.insert(0, str(ROOT))
    import cctv_dgp_input_selection_v19 as q
    v18 = ROOT / 'outputs/cctv_dgp_structure_vm_v18'
    returned = ROOT / 'outputs/cctv_dgp_structure_return_v18/outputs/structure_v18'
    parent = ROOT / 'outputs/cctv_dgp_face_code_fit_vm_v12_r2'
    r2 = ROOT / 'outputs/cctv_dgp_broader_codes_vm_v16_r2'
    mixed = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'
    baseline = ROOT / 'outputs/cctv_dgp_generalization_return_v15/outputs/generalization_v15'
    sys.path.insert(0, str(v18))
    import cctv_dgp_structure_v18 as old
    cp, legacy = old.verify(v18, parent, r2, mixed, baseline, q.V18_PIN)
    broad = q.read(r2 / legacy.PLAN)
    processing = ROOT / 'outputs/cctv_dgp_input_selector_v19'
    audit = q.read(processing / 'independent_processing_audit.json')
    q.require(audit['complete'] and audit['results_sha256'] == q.sha(processing / 'results.json')
              and audit['unchanged_quality_guard']['qualified_for_separate_generalization_protocol'], 'Positive independently audited training-only control required')
    q.require(not OUT.exists(), 'Preserve existing frozen V19; no regeneration')
    q.require(q.sha(ROOT / 'dgp_input_selector_v19.py') == q.SELECTOR_PIN, 'Already frozen selector changed')
    OUT.mkdir()
    def copy(name, source):
        q.require(time.monotonic() - start <= 300, 'Preparation exceeds300s; preserve partial package')
        target = q.safe(OUT, name); target.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith('.py'): ast.parse(source.read_text(encoding='utf-8'), feature_version=(3, 10))
        shutil.copyfile(source, target)
    for name in SOURCES: copy(name, ROOT / name)
    for name in sorted(set(cp['assets_sha256']) | {'structure_protocol_v18.json', 'protocol.sha256'}):
        copy('parent_v18/' + name, v18 / name)
    copy('dgp_structure_conditioner_v18.py', v18 / 'dgp_structure_conditioner_v18.py')
    for name, source in {
        'weights/structure_update600.pth': returned / 'update600/decoder.pth',
        'lineage/v18_results.json': returned / 'results.json',
        'lineage/v18_neural_receipt.json': returned / 'neural_execution_receipt.json',
        'lineage/v18_full_audit.json': ROOT / 'outputs/cctv_dgp_structure_v18_audit_recovery_r2/local_full_audit.json',
        'lineage/v18_review.json': ROOT / 'outputs/cctv_dgp_structure_review_v18.json',
        'lineage/v18_verified_summary.json': ROOT / 'outputs/cctv_dgp_structure_v18_verified_summary.json',
        'lineage/processing_results_v19.json': processing / 'results.json',
        'lineage/processing_protocol_v19.json': processing / 'processing_protocol_v19.json',
        'lineage/processing_audit_v19.json': processing / 'independent_processing_audit.json',
    }.items(): copy(name, source)
    for c in cp['training_cases']:
        for prefix, original in [('parity/dgp_base', 'update0'), ('parity/structure_update600', 'update600')]:
            copy(prefix + '/' + c['id'] + '.npy', returned / (original + '/' + c['id'] + '.npy'))
    refs = [r for r in broad['references'] if r['role'] == 'validation']
    cases = broad['validation_cases']
    names = {c['input'] for c in cases} | {r['target'] for r in refs} | {r['observed'] for r in refs} | set(cp['data_assets_sha256'])
    assets = {path.relative_to(OUT).as_posix(): q.sha(path) for path in OUT.rglob('*') if path.is_file()}
    p = {'format': q.FORMAT, 'design': q.DESIGN, 'assets_sha256': assets,
        'references': refs, 'cases': cases, 'preview_reference_ids': broad['validation_preview_reference_ids'],
        'training_parity_references': cp['references'], 'training_parity_cases': cp['training_cases'],
        'data_assets_sha256': {name: broad['data_assets_sha256'][name] for name in sorted(names)},
        'terminal_state_hash': q.read(returned / 'results.json')['snapshots'][-1]['state_hash'],
        'baseline_results_sha256': broad['baseline_results_sha256'],
        'training': False, 'validation_used': True, 'native_used': False, 'native_reserved_used': False,
        'teacher_used': False, 'production_promoted': False,
        'review_scope': 'Repeated-use development cohort, not independent final evaluation. Native inputs/reserved identities untouched. Pretrained corpus overlap unverified.',
        'execution_location': 'Existing forensic-dgp-thesis NVIDIA L4; no optimizer or gradient execution'}
    q.write(OUT / q.PLAN, p); pin = q.sha(OUT / q.PLAN)
    with (OUT / 'protocol.sha256').open('x', encoding='ascii', newline='\n') as stream: stream.write(pin + '\n')
    q.verify(OUT, parent, r2, mixed, baseline, pin)
    names = sorted(set(assets) | {q.PLAN, 'protocol.sha256'})
    q.require(sum((OUT / name).stat().st_size for name in names) <= 128 * 1024**2, 'Inference execution bundle exceeds128MiB')
    archive = ROOT / 'outputs/cctv-dgp-input-selection-v19-execution.tar.gz'
    with tarfile.open(archive, 'x:gz', compresslevel=5) as stream:
        for name in names:
            q.require(time.monotonic() - start <= 300, 'Preparation/archive exceeds300s')
            stream.add(OUT / name, arcname=name, recursive=False)
    digest = q.sha(archive)
    with Path(str(archive) + '.sha256').open('x', encoding='ascii', newline='\n') as stream:
        stream.write(digest + '  ' + archive.name + '\n')
    launcher = ROOT / 'outputs/launch_cctv_dgp_input_selection_v19.py'
    q.require(not launcher.exists(), 'Preserve external launcher'); shutil.copyfile(OUT / 'scripts/launch_cctv_dgp_input_selection_v19.py', launcher)
    q.write(ROOT / 'outputs/cctv_dgp_input_selection_v19_preparation.json', {'complete': True,
        'protocol_sha256': pin, 'archive_sha256': digest, 'archive_bytes': archive.stat().st_size,
        'members': len(names), 'source_assets': len(assets), 'launcher_sha256': q.sha(launcher),
        'validation_references': 104, 'validation_cases': 520, 'training_parity_cases': 50,
        'local_neural_forwards': 0, 'backward_calls': 0, 'optimizer_updates': 0, 'VM_launched': False,
        'seconds': time.monotonic() - start, 'production_promoted': False})
    print((ROOT / 'outputs/cctv_dgp_input_selection_v19_preparation.json').read_text(), flush=True)


if __name__ == '__main__': prepare()
