"""Independent frozen-transfer/scope audit; no neural imports or VM operations."""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs/cctv_dgp_structure_vm_v18'
PROTOCOL_SHA = 'e3a7567e9a9c53a1668464ab5c95edbf094c73e9a7c8e550635dcf8d8ef04adb'
ARCHIVE_SHA = 'ecc8a24aa77e23f80bbd7e482573ed666e443e35c7ee5633ddf4e5a6edeb2ce7'
BOOTSTRAP_SHA = 'a0aa9c6ef4e6a36783ec22bc97d981c5789550f3027a7e130d6964cd47c46e2c'


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit():
    started = time.monotonic()
    plan = read(BUNDLE / 'structure_protocol_v18.json')
    archive = ROOT / 'outputs/cctv-dgp-structure-v18-execution.tar.gz'
    assert sha(BUNDLE / 'structure_protocol_v18.json') == PROTOCOL_SHA
    assert (BUNDLE / 'protocol.sha256').read_text().strip() == PROTOCOL_SHA
    assert sha(archive) == ARCHIVE_SHA and archive.stat().st_size == 9063630
    assert Path(str(archive) + '.sha256').read_text().split() == [ARCHIVE_SHA, archive.name]
    assert sha(ROOT / 'outputs/launch_cctv_dgp_structure_v18.py') == BOOTSTRAP_SHA
    assert sha(BUNDLE / 'scripts/launch_cctv_dgp_structure_v18.py') == BOOTSTRAP_SHA
    assets = plan['assets_sha256']
    assert len(assets) == 21
    wanted = set(assets) | {'structure_protocol_v18.json', 'protocol.sha256'}
    verified = []
    with tarfile.open(archive, 'r:gz') as stream:
        members = stream.getmembers()
        assert len(members) == 23 and {m.name for m in members} == wanted
        assert len({m.name for m in members}) == 23
        assert sum(m.size for m in members) <= 64 * 1024**2
        for member in members:
            part = PurePosixPath(member.name)
            assert member.isfile() and not part.is_absolute() and '..' not in part.parts
            assert ':' not in member.name and '\\' not in member.name and str(part) == member.name
            payload = stream.extractfile(member).read()
            assert payload == (BUNDLE / member.name).read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            if member.name in assets:
                assert digest == assets[member.name]
            verified.append({'name': member.name, 'bytes': len(payload), 'sha256': digest})
    sources = [name for name in assets if name.endswith('.py')]
    assert len(sources) == 12
    for name in sources:
        assert sha(ROOT / name) == assets[name]
        ast.parse((BUNDLE / name).read_text(encoding='utf-8'), feature_version=(3, 10))

    r2_protocol = ROOT / 'outputs/cctv_dgp_broader_codes_vm_v16_r2/broader_codes_protocol_v16_r2.json'
    legacy = read(r2_protocol)
    refs, cases, ids = plan['references'], plan['training_cases'], plan['reference_ids']
    assert ids == legacy['train_preview_reference_ids'] and len(set(ids)) == 10
    assert refs == [r for r in legacy['references'] if r['id'] in ids]
    assert cases == [c for c in legacy['training_cases'] if c['reference_id'] in ids]
    assert len(refs) == 10 and len(cases) == 50 and all(r['role'] == 'train' for r in refs)
    source_groups = ['dataset/asian_faces', 'dataset/thumbnails128x128']
    profiles = ['clear', 'blur_lr24', 'lowlight_lr32', 'motion_lr48', 'compound_lr24']
    assert Counter(r['source'] for r in refs) == Counter({source: 5 for source in source_groups})
    lookup = {(c['reference_id'], c['profile']): c['id'] for c in cases}
    assert len(lookup) == 50
    by_source = {source: sorted(r['id'] for r in refs if r['source'] == source) for source in source_groups}
    batches = read(BUNDLE / 'schedule_v18.json')['batches']
    assert len(batches) == 600
    exposures = Counter()
    for index, batch in enumerate(batches):
        expected = [lookup[(by_source[source][index % 5], profile)]
                    for source in source_groups for profile in profiles]
        assert batch == expected and len(set(batch)) == 10
        exposures.update(batch)
    assert sum(exposures.values()) == 6000 and set(exposures) == {c['id'] for c in cases}
    assert all(count == 120 for count in exposures.values())
    selected_data = {c['input'] for c in cases} | {r['target'] for r in refs} | {r['observed'] for r in refs}
    assert set(plan['data_assets_sha256']) == selected_data
    for name, digest in plan['data_assets_sha256'].items():
        assert digest == legacy['data_assets_sha256'][name]
        assert sha(ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2' / name) == digest
    assert all(plan[key] is False for key in ['validation_used', 'native_used',
                                             'native_reserved_used', 'checkpoint_selected', 'production_promoted'])

    assert sha(r2_protocol) == plan['r2_protocol_sha256'] == '4f64cf2204458f8fadcab1824fc4bc293c65803e123fdebb304f7b5bd3a502a0'
    checkpoint = ROOT / 'outputs/cctv_dgp_broader_codes_failure_return_v16_r2/outputs/broader_codes_v16_r2/epoch8/conditioner.pth'
    assert sha(checkpoint) == assets['weights/r2_conditioner_epoch8.pth'] == '3ef704e70f68d633ac7624eb47f79cfada189341dfdc58bad08444595d7d6757'
    full = read(BUNDLE / 'lineage/r2_full_audit.json')
    review = read(BUNDLE / 'lineage/r2_review.json')
    assert full['complete'] and full['results_sha256'] == plan['r2_results_sha256'] == review['results_sha256']
    assert sha(BUNDLE / 'lineage/r2_full_audit.json') == review['full_audit_sha256']
    assert review['production_promoted'] is False
    proof = read(BUNDLE / 'lineage/prototype_results.json')
    proof_audit = read(BUNDLE / 'lineage/prototype_saved_output_audit.json')
    assert sha(BUNDLE / 'lineage/prototype_results.json') == proof_audit['results_sha256'] == 'b589998457496f5cbecd1e8933b3a9e5a1bc9108c14e9c6381da5d721ab98a7c'
    assert proof['complete'] and proof_audit['complete']
    assert proof['module_sha256'] == assets['dgp_structure_conditioner_v18.py'] == 'e98af67f110b4ebc276ff2394345a81bc6ca7e082b36a493ad5c3fde94593e5d'
    assert proof['frozen_before'] == proof['frozen_after'] and proof['backward_calls'] == proof['optimizer_updates'] == 0
    assert len(proof['rows']) == 2 and all(row['exact_dgp_raw_parity'] and row['exact_zero_prior_raw_parity'] for row in proof['rows'])
    for name, digest in proof['artifacts_sha256'].items():
        assert sha(ROOT / 'outputs/cctv_dgp_structure_prototype_v18' / name) == digest
    assert len(proof['artifacts_sha256']) == 8

    pairs = [(6, 24), (24, 48), (48, 96), (96, 128), (160, 128),
             (224, 96), (176, 64), (88, 32), (38, 16)]
    parameters = sum(9 * a * b + 3 * b for a, b in pairs) + (256 * 32 + 3 * 32) + (9 * 16 * 3 + 3)
    assert parameters == 684395 == plan['design']['trainable_parameters']
    expected_counts = {name: 50 + 8 for name in ['dgp', 'prior_encoder', 'prior_classifier', 'r2_head', 'prior_generator']}
    expected_counts.update(prior_RGB_tail=0, residual_head=600 + 2 + 4 * 50 + 50 + 8,
                           recognizer=10 + 50 + 2 + 600 + 4 * 50 + 50, unused_v11=0)
    module = ast.parse((BUNDLE / 'cctv_dgp_structure_v18.py').read_text())
    function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'expected_counts')
    assert ast.literal_eval(function.body[-1].value) == expected_counts
    limits = {key: plan['design'][key] for key in ['cache_cap_seconds', 'fit_cap_seconds',
                                                 'trainer_cap_seconds', 'audit_cap_seconds', 'supervisor_cap_seconds']}
    assert limits == {'cache_cap_seconds': 300, 'fit_cap_seconds': 900,
                      'trainer_cap_seconds': 1230, 'audit_cap_seconds': 300, 'supervisor_cap_seconds': 1800}
    assert plan['design']['snapshot_updates'] == [0, 50, 200, 600] and plan['design']['fit_stop_update'] == 50
    receipt = {'complete': True, 'date': datetime.now(timezone.utc).isoformat(),
        'checker_sha256': sha(Path(__file__)), 'protocol_sha256': PROTOCOL_SHA,
        'archive_sha256': ARCHIVE_SHA, 'archive_bytes': archive.stat().st_size,
        'launcher_sha256': BOOTSTRAP_SHA, 'members': verified, 'source_assets': 21,
        'Python_3_10_source_files': 12, 'selected_data_files': len(selected_data),
        'training_references': 10, 'training_cases': 50, 'schedule_updates': 600,
        'schedule_exposures': 6000, 'exposures_per_case': 120, 'parameters_arithmetic': parameters,
        'declared_success_neural_counts_independently_derived': expected_counts,
        'timing_limits': limits, 'prototype_saved_artifacts_verified': 8,
        'original_R2_protocol_checkpoint_unchanged': True,
        'neural_calls': 0, 'backward_calls': 0, 'optimizer_updates': 0, 'VM_launched': False,
        'CUDA_gradient_preflight': 'pending user VM execution',
        'trained_quality': 'pending VM return, audit and visual review',
        'production_promoted': False, 'seconds': time.monotonic() - started}
    target = ROOT / 'outputs/cctv_dgp_structure_v18_transfer_audit.json'
    with target.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    print(json.dumps({key: receipt[key] for key in ['complete', 'protocol_sha256', 'archive_bytes',
        'source_assets', 'selected_data_files', 'schedule_updates', 'schedule_exposures', 'parameters_arithmetic', 'seconds']}, indent=2))


if __name__ == '__main__':
    audit()
