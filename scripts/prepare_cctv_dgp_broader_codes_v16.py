"""Freeze a thin verified V16 package; reuse existing parent assets without inference."""
import hashlib
from pathlib import Path
import shutil
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import cctv_dgp_broader_codes_v16 as v


def prepare():
    start = time.monotonic(); out = ROOT / 'outputs/cctv_dgp_broader_codes_vm_v16'
    v.require(not out.exists(), 'Preserve frozen/partial V16 package')
    mixed = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'
    parent = ROOT / 'outputs/cctv_dgp_face_code_fit_vm_v12_r2'
    baseline = ROOT / 'outputs/cctv_dgp_generalization_return_v15/outputs/generalization_v15'
    v.require(v.sha(mixed / 'mixed_protocol_v9.json') == v.MIXED_PIN and
              v.sha(parent / 'face_code_fit_protocol_v12.json') == v.PARENT_PIN and
              v.sha(baseline / 'results.json') == v.BASELINE_PIN, 'Frozen lineage differs')
    mp = v.read(mixed / 'mixed_protocol_v9.json'); pp = v.read(parent / 'face_code_fit_protocol_v12.json')
    br = v.read(baseline / 'results.json')
    audit = ROOT / 'outputs/cctv_dgp_generalization_return_v15/local_independent_audit.json'
    review = ROOT / 'outputs/cctv_dgp_generalization_review_v15.json'
    v.require(v.read(audit)['complete'] and v.read(audit)['results_sha256'] == v.BASELINE_PIN and
              v.read(review)['complete'] and not v.read(review)['useful_upgrade'], 'Require audited negative V15')
    tests = ROOT / 'outputs/cctv_dgp_broader_codes_v16_tests.json'; tr = v.read(tests)
    v.require(tr['complete'] and tr['backward_calls'] == tr['optimizer_updates'] == 0 and
              all(v.sha(ROOT / name) == pin for name, pin in tr['tested_sources_sha256'].items()) and
              set(tr['tested_sources_sha256']) == set(v.SOURCES_FILES), 'Tests stale or incomplete')
    native = ROOT / 'outputs/cctv_native_development_v2/frozen_subset.json'
    input_review = ROOT / 'outputs/cctv_native_development_v2/input_review.json'
    release = ROOT / 'outputs/cctv_survface_structure_audit_v1/verification.json'
    v.require(v.sha(native) == 'c063985b258b808efaa897c88593cbdbcee2848d686d3d056219f8a8e555a98e' and
              v.sha(input_review) == '41912033f5070547e120a6b384024fa929eeaf3a7ecca408c0c67c53d37d36e2' and
              v.sha(release) == 'cf9124983a815776a787eeae577bbbbd173892be2f69dc33c354626641863222',
              'Native acquisition/review lineage changed')
    out.mkdir(); assets = {}

    def copy(src, name):
        expected = v.sha(src); dst = v.safe(out, name); dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst); v.require(v.sha(dst) == expected, 'Copied source differs')
        assets[name] = expected

    for name in v.SOURCES_FILES: copy(ROOT / name, name)
    for key, path in [('boundary_tests', tests), ('negative_v15_audit', audit), ('negative_v15_review', review),
                      ('native_subset', native), ('native_input_review', input_review), ('native_release_audit', release)]:
        copy(path, 'lineage/' + key + '.json')
    data_names = {r[k] for r in mp['references'] for k in ['target', 'observed']}
    data_names.update(c['input'] for c in mp['training_cases'] + mp['validation_cases'])
    baseline_names = {'target_embeddings/' + r['id'] + '.npy' for r in mp['references'] if r['role'] == 'validation'}
    for row in br['rows']:
        baseline_names.add(row['input_metrics']['embedding'])
        for arm in ['retained_dgp_v2', 'starting_prior_none']:
            baseline_names.update(row['arms'][arm][k] for k in ['prediction', 'embedding'])
            if 'raw' in row['arms'][arm]: baseline_names.add(row['arms'][arm]['raw'])
    plan = {'format': v.FORMAT, 'date': '2026-10-04', 'design': v.DESIGN, 'assets_sha256': assets,
        'parent_protocol_sha256': v.PARENT_PIN, 'parent_assets_sha256': pp['assets_sha256'],
        'mixed_protocol_sha256': v.MIXED_PIN, 'baseline_results_sha256': v.BASELINE_PIN,
        'references': mp['references'], 'training_cases': mp['training_cases'], 'validation_cases': mp['validation_cases'],
        'data_assets_sha256': {name: mp['assets_sha256'][name] for name in sorted(data_names)},
        'baseline_assets_sha256': {name: br['artifacts_sha256'][name] for name in sorted(baseline_names)},
        'train_preview_reference_ids': v.previews([r for r in mp['references'] if r['role'] == 'train']),
        'validation_preview_reference_ids': v.previews([r for r in mp['references'] if r['role'] == 'validation']),
        'native_used': False, 'native_reserved_used': False, 'production_promoted': False, 'checkpoint_selected': False,
        'architecture': 'Our reset2422432-parameter code-only head, with retained trained DGP, declared frozen CodeFormer prior/clean teacher and ArcFace; permitted experimental component, no application adoption.',
        'initialization': 'Seeded reset head with zero code projection; no V14/V15 fitted checkpoint loaded.',
        'objective': 'Train-role clean teacher labels only; observed-token CE. No teacher labels or optimization on validation.',
        'baseline': 'Reuse independently audited V15 retained DGP and starting no-stat prior PNGs/embeddings; require new epoch0 validation output parity on all520 cases.',
        'rendering': 'Frozen no-stat w0 decoder, floor floatRGB255, copy original input outside observed support. No display sharpening/CLAHE or new output-selected processing.',
        'split_limit': 'Own exact source/target content disjointness checked. Development validation used in prior experiments; near duplicates/identity overlap beyond existing records/pretrained FFHQ exposure remain unknown. Asian is dataset provenance, not ethnicity or local CCTV validation.',
        'scope_limit': 'Photography camera proxies only. Native24/reserved32 untouched. Broader data/objective/optimizer change together; not causal attribution, native restoration or final independent review.',
        'finite_stop': 'Cache900s; fit/evaluation1200s; VM audit240s; total2400s. Early20-reference/25-update conservative projections,20GiB VRAM, nonfinite/gradient/parity/state failures stop. Stop after epoch4 when fixed training-preview CE improves less than1%. No resume or automatic selection.',
        'return_scope': 'All outputs/checkpoints/probes/teacher labels/traces/grids returned;9GB feature cache remains on VM. Local arithmetic auditor checks serialized evidence without replaying training or recognizer. Human visual review required.',
        'transport': 'Current user instruction: no configured SSH; prepare pasteable browser-SSH launch and verified transfer files. No assistant cloud launch performed.'}
    v.write(out / 'schedule_v16.json', {'steps': v.schedule(plan)})
    assets['schedule_v16.json'] = v.sha(out / 'schedule_v16.json')
    v.write(out / v.PLAN, plan); pin = v.sha(out / v.PLAN)
    (out / 'protocol.sha256').write_text(pin + '\n', encoding='ascii', newline='\n')
    v.verify(out, parent, mixed, baseline, pin)
    archive = ROOT / 'outputs/cctv-dgp-broader-codes-v16-execution.tar.gz'
    names = sorted(assets) + [v.PLAN, 'protocol.sha256']
    with tarfile.open(archive, 'x:gz', compresslevel=3) as stream:
        for name in names: stream.add(out / name, arcname=name, recursive=False)
    with tarfile.open(archive, 'r:gz') as stream:
        members = v.archive_members(stream, out, 256 * 1024**2)
        v.require({m.name for m in members} == set(names), 'Execution archive coverage differs')
        for member in members:
            v.require(member.isfile() and hashlib.sha256(stream.extractfile(member).read()).hexdigest() == v.sha(out / member.name),
                      'Archived member differs')
    archive_sha = v.sha(archive)
    Path(str(archive) + '.sha256').write_text(archive_sha + '  ' + archive.name + '\n', encoding='ascii', newline='\n')
    result = {'complete': True, 'protocol_sha256': pin, 'archive_sha256': archive_sha,
        'archive_bytes': archive.stat().st_size, 'archive_members': len(names),
        'parent_assets_verified': len(pp['assets_sha256']), 'data_assets_verified': len(data_names),
        'baseline_assets_verified': len(baseline_names), 'train_references': 781, 'validation_references': 104,
        'optimizer_updates_budget': 3128, 'exposures_budget': 31280, 'seconds': time.monotonic() - start,
        'neural_forwards': 0, 'backward_calls': 0, 'optimizer_updates': 0, 'vm_launched': False,
        'native_reserved_used': False, 'production_promoted': False, 'cuda_preflight_verified': False}
    v.write(ROOT / 'outputs/cctv_dgp_broader_codes_v16_preparation.json', result); print(result, flush=True)


if __name__ == '__main__': prepare()
