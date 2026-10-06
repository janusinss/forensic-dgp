"""Freeze a separate finite VM pilot after verified local zero-residual proof."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_structure_vm_v18'
SOURCE_FILES = ['cctv_dgp_structure_v18.py', 'dgp_structure_conditioner_v18.py', 'dgp_structure_objective_v18.py',
    'scripts/train_cctv_dgp_structure_v18.py', 'scripts/audit_cctv_dgp_structure_v18.py',
    'scripts/supervise_cctv_dgp_structure_v18.py', 'scripts/launch_cctv_dgp_structure_v18.py',
    'scripts/import_cctv_dgp_structure_v18.py', 'scripts/prepare_cctv_dgp_structure_v18.py',
    'scripts/verify_cctv_dgp_structure_prototype_v18.py', 'tests/test_dgp_structure_conditioner_v18.py',
    'tests/test_cctv_dgp_structure_v18.py']


def prepare():
    start = time.monotonic()
    sys.path.insert(0, str(ROOT))
    import cctv_dgp_structure_v18 as q
    parent = ROOT / 'outputs/cctv_dgp_face_code_fit_vm_v12_r2'
    r2 = ROOT / 'outputs/cctv_dgp_broader_codes_vm_v16_r2'
    mixed = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'
    baseline = ROOT / 'outputs/cctv_dgp_generalization_return_v15/outputs/generalization_v15'
    v = q.environment(parent, r2)
    legacy = v.verify(r2, parent, mixed, baseline, q.R2_PIN)
    proof = ROOT / 'outputs/cctv_dgp_structure_prototype_v18'
    v.require(v.read(proof / 'independent_saved_output_audit.json')['complete']
              and v.sha(proof / 'results.json') == q.PROTOTYPE_RESULTS
              and v.sha(ROOT / 'dgp_structure_conditioner_v18.py') == q.MODULE_PIN, 'Verified local interface proof required')
    v.require(not OUT.exists(), 'Preserve existing draft/frozen V18; no automatic regeneration')
    OUT.mkdir()
    for name in SOURCE_FILES:
        target = OUT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        ast.parse((ROOT / name).read_text(encoding='utf-8'), feature_version=(3, 10))
        shutil.copyfile(ROOT / name, target)
    copies = {
        'weights/r2_conditioner_epoch8.pth': ROOT / 'outputs/cctv_dgp_broader_codes_failure_return_v16_r2/outputs/broader_codes_v16_r2/epoch8/conditioner.pth',
        'lineage/prototype_results.json': proof / 'results.json',
        'lineage/prototype_protocol.json': proof / 'protocol.json',
        'lineage/prototype_saved_output_audit.json': proof / 'independent_saved_output_audit.json',
        'lineage/r2_full_audit.json': ROOT / 'outputs/cctv_dgp_broader_codes_v16_r2_audit_recovery_1/local_full_audit.json',
        'lineage/r2_review.json': ROOT / 'outputs/cctv_dgp_broader_codes_review_v16_r2.json',
        'lineage/v17_results.json': ROOT / 'outputs/cctv_dgp_fidelity_spotcheck_v17/results.json',
        'lineage/v17_review.json': ROOT / 'outputs/cctv_dgp_fidelity_spotcheck_v17/assistant_development_review.json',
    }
    for name, source in copies.items():
        target = OUT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    refs, cases = q.selected_cohort(legacy)
    ids = legacy['train_preview_reference_ids']
    v.write(OUT / 'schedule_v18.json', {'batches': q.schedule(refs, cases, v.SOURCES, v.PROFILES)})
    data_names = {c['input'] for c in cases} | {r['target'] for r in refs} | {r['observed'] for r in refs}
    assets = {path.relative_to(OUT).as_posix(): v.sha(path) for path in OUT.rglob('*') if path.is_file()}
    p = {'format': q.FORMAT, 'design': q.DESIGN, 'training_location': 'existing forensic-dgp-thesis Linux NVIDIA L4 VM',
        'r2_protocol_sha256': q.R2_PIN, 'r2_results_sha256': v.read(copies['lineage/r2_full_audit.json'])['results_sha256'],
        'reference_ids': ids, 'references': refs, 'training_cases': cases,
        'data_assets_sha256': {name: legacy['data_assets_sha256'][name] for name in sorted(data_names)},
        'assets_sha256': assets, 'validation_used': False, 'native_used': False, 'native_reserved_used': False,
        'checkpoint_selected': False, 'production_promoted': False,
        'source_scope': 'Two photographic source folders, not ethnicity, native CCTV or Zamboanga validation; prior/recognizer pretraining overlap unverified.',
        'pipeline': 'Frozen DGP pixel base; frozen R2 code head and declared CodeFormer feature64; our new RGB residual decoder.'}
    v.write(OUT / q.PLAN, p)
    pin = v.sha(OUT / q.PLAN)
    with (OUT / 'protocol.sha256').open('x', encoding='ascii', newline='\n') as stream: stream.write(pin + '\n')
    q.verify(OUT, parent, r2, mixed, baseline, pin)
    archive = ROOT / 'outputs/cctv-dgp-structure-v18-execution.tar.gz'
    names = sorted(set(assets) | {q.PLAN, 'protocol.sha256'})
    with tarfile.open(archive, 'x:gz', compresslevel=5) as stream:
        for name in names: stream.add(v.safe(OUT, name), arcname=name, recursive=False)
    archive_sha = v.sha(archive)
    with Path(str(archive) + '.sha256').open('x', encoding='ascii', newline='\n') as stream:
        stream.write(archive_sha + '  ' + archive.name + '\n')
    launcher = ROOT / 'outputs/launch_cctv_dgp_structure_v18.py'
    v.require(not launcher.exists(), 'Preserve existing external launcher')
    shutil.copyfile(OUT / 'scripts/launch_cctv_dgp_structure_v18.py', launcher)
    checked = 0
    with tarfile.open(archive, 'r:gz') as stream:
        v.require([item.name for item in stream.getmembers()] == names, 'Execution archive inventory differs')
        for member in stream.getmembers():
            content = stream.extractfile(member).read()
            v.require(member.isfile() and content == (OUT / member.name).read_bytes(), 'Execution transfer member differs')
            checked += 1
    receipt = {'complete': True, 'date': '2026-10-05', 'protocol_sha256': pin, 'archive_sha256': archive_sha,
        'archive_bytes': archive.stat().st_size, 'members_verified': checked, 'source_assets': len(assets),
        'launcher_sha256': v.sha(launcher), 'prototype_results_sha256': q.PROTOTYPE_RESULTS,
        'updates': 600, 'exposures': 6000, 'training_references': 10, 'training_cases': 50,
        'local_neural_calls': 0, 'backward_calls': 0, 'optimizer_updates': 0,
        'CUDA_gradient_preflight': 'pending user VM execution', 'VM_launched': False,
        'production_promoted': False, 'seconds': time.monotonic() - start}
    v.write(ROOT / 'outputs/cctv_dgp_structure_v18_preparation.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__': prepare()
