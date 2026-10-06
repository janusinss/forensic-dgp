"""Separately freeze V16 r2 after audited zero-update timing failure; no inference/training."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/cctv_dgp_broader_codes_vm_v16'
WORK = ROOT / 'pilots/cctv_dgp_broader_codes_v16_r2'
OUT = ROOT / 'outputs/cctv_dgp_broader_codes_vm_v16_r2'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def replace_once(text, old, new):
    require(text.count(old) == 1, 'Original source anchor differs: ' + old[:100])
    return text.replace(old, new, 1)


def predecessor():
    p = read(BASE / 'broader_codes_protocol_v16.json')
    receipt = ROOT / 'outputs/cctv_dgp_broader_codes_failure_return_v16/local_cache_failure_audit.json'
    audit = read(receipt)
    require(audit['complete'] and not audit['success'] and audit['optimizer_updates'] == audit['backward_calls'] == 0
            and audit['protocol_sha256'] == sha(BASE / 'broader_codes_protocol_v16.json') and
            audit['failure_archive_sha256'] == sha(ROOT / 'outputs/cctv-dgp-broader-codes-v16-failure.tar.gz'),
            'Require independently audited immutable zero-update V16 failure')
    for name, expected in p['assets_sha256'].items():
        require(sha(BASE / name) == expected, 'Preserve changed original asset: ' + name)
    return p, receipt


def generate():
    p, _ = predecessor()
    require(not WORK.exists(), 'Preserve existing source drafts; edit them explicitly instead of regenerating')
    WORK.mkdir(parents=True)
    tree = ast.parse((BASE / 'cctv_dgp_broader_codes_v16.py').read_text())
    files = ast.literal_eval(next(node.value for node in tree.body if isinstance(node, ast.Assign)
                                 and any(isinstance(target, ast.Name) and target.id == 'SOURCES_FILES' for target in node.targets)))
    for name in files:
        target = WORK / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(BASE / name, target)

    common = (WORK / 'cctv_dgp_broader_codes_v16.py').read_text()
    common = replace_once(common, "PLAN = 'broader_codes_protocol_v16.json'",
        "from cctv_dgp_cache_timing_v16_r2 import cache_order\n\nPLAN = 'broader_codes_protocol_v16_r2.json'")
    common = replace_once(common, "FORMAT = 'reset-broader-code-only-component-v16'",
        "FORMAT = 'reset-broader-code-only-component-v16-r2'")
    common = replace_once(common, "'cache_timing_at_reference': 20, 'timing_update': 25,",
        "'cache_timing_at_reference': 30, 'timing_update': 25,\n    'cache_projection_method': 'source-role-steady-reference-rate-startup-warmups-once-v16-r2',")
    common = replace_once(common, "    'tests/test_cctv_dgp_broader_codes_runtime_v16.py', 'face_prior_grid_v10.py',",
        "    'tests/test_cctv_dgp_broader_codes_runtime_v16.py', 'face_prior_grid_v10.py',\n    'cctv_dgp_cache_timing_v16_r2.py', 'tests/test_cctv_dgp_v16_cache_timing_r2.py',")
    common = replace_once(common, 'def verify(root, parent, mixed, baseline, pin):\n',
        'def verify(root, parent, mixed, baseline, pin):\n    root, parent, mixed, baseline = map(Path, [root, parent, mixed, baseline])\n')
    common = replace_once(common, "    validate_cohort(p); validate_schedule(p, read(root / 'schedule_v16.json')['steps'])\n",
        "    validate_cohort(p); validate_schedule(p, read(root / 'schedule_v16.json')['steps'])\n    require(read(root / 'cache_order_v16_r2.json')['reference_ids'] == [ref['id'] for ref in cache_order(p)], 'Cache order differs')\n")
    (WORK / 'cctv_dgp_broader_codes_v16.py').write_text(common, encoding='utf-8', newline='\n')

    trainer = (WORK / 'scripts/train_cctv_dgp_broader_codes_v16.py').read_text()
    trainer = replace_once(trainer, "    out = root / 'outputs/broader_codes_v16'", "    out = root / 'outputs/broader_codes_v16_r2'")
    trainer = replace_once(trainer, "    with torch.no_grad():\n        for index, ref in enumerate(p['references'], 1):\n            clock(); rid = ref['id']\n",
        "    from cctv_dgp_cache_timing_v16_r2 import cache_order, projection, METHOD\n"
        "    clock(); initialization_seconds = time.monotonic() - start\n"
        "    v.write(out / 'cache_initialization.json', {'seconds': initialization_seconds, 'initial_state': initial, 'method': METHOD})\n"
        "    reference_timings = []; warmed = set()\n"
        "    with torch.no_grad():\n"
        "        for index, ref in enumerate(cache_order(p), 1):\n"
        "            reference_start = time.monotonic()\n"
        "            clock(); rid = ref['id']\n")
    old = """            if index == d['cache_timing_at_reference']:
                elapsed = time.monotonic() - start
                # Validation uses five encoder passes/reference, conservatively overproject all remaining work.
                equivalent_batches = (781 - index) + 520
                projection = elapsed + elapsed / index * equivalent_batches * d['timing_safety_factor'] + 30
                v.write(out / 'cache_timing.json', {'references': index, 'seconds': elapsed,
                    'projected_seconds': projection, 'cap_seconds': d['cache_cap_seconds']})
                v.require(projection <= d['cache_cap_seconds'], 'Cache projection exceeds900s')
"""
    new = """            clock()
            if index <= d['cache_timing_at_reference']:
                key = (ref['source'], ref['role'])
                reference_timings.append({'id': rid, 'source': ref['source'], 'role': ref['role'],
                    'cases': len(own_cases), 'seconds': time.monotonic() - reference_start, 'warmup': key not in warmed})
                warmed.add(key)
            if index == d['cache_timing_at_reference']:
                timing = projection(p, reference_timings, time.monotonic() - start, initialization_seconds)
                v.write(out / 'cache_timing.json', timing)
                print({'cache_projection_seconds': timing['projected_seconds'], 'initialization_seconds': initialization_seconds,
                    'cap_seconds': timing['cap_seconds']}, flush=True)
                v.require(timing['passed'], 'Cache projection exceeds900s')
"""
    trainer = replace_once(trainer, old, new)
    trainer = replace_once(trainer, "    bind('cache_manifest.json'); bind('cache_timing.json'); bind('execution.json')",
        "    bind('cache_manifest.json'); bind('cache_timing.json'); bind('cache_initialization.json'); bind('execution.json')")
    trainer = trainer.replace('V16 cache references', 'V16 r2 cache references').replace('V16 update ', 'V16 r2 update ')
    (WORK / 'scripts/train_cctv_dgp_broader_codes_v16.py').write_text(trainer, encoding='utf-8', newline='\n')

    supervisor = (WORK / 'scripts/supervise_cctv_dgp_broader_codes_v16.py').read_text()
    supervisor = supervisor.replace('outputs/broader_codes_v16', 'outputs/broader_codes_v16_r2')
    supervisor = supervisor.replace('cctv-dgp-broader-codes-v16-results', 'cctv-dgp-broader-codes-v16-r2-results')
    supervisor = supervisor.replace('cctv-dgp-broader-codes-v16-failure', 'cctv-dgp-broader-codes-v16-r2-failure')
    supervisor = replace_once(supervisor, "            'supervisor_launch.json', 'supervisor_execution.json', 'supervisor_failure.json']",
        "            'supervisor_launch.json', 'bootstrap_preflight.json', 'supervisor_execution.json', 'supervisor_failure.json']")
    (WORK / 'scripts/supervise_cctv_dgp_broader_codes_v16.py').write_text(supervisor, encoding='utf-8', newline='\n')

    auditor = (WORK / 'scripts/audit_cctv_dgp_broader_codes_v16.py').read_text()
    auditor = replace_once(auditor, "    p = v.verify(root, parent, mixed, baseline, pin)\n    failure = v.read(supervisor_failure)",
        "    p = v.verify(root, parent, mixed, baseline, pin)\n    if (out / 'cache_timing.json').is_file():\n        from cctv_dgp_cache_timing_v16_r2 import verify_cache_timing\n        verify_cache_timing(p, v.read(out / 'cache_timing.json'))\n    failure = v.read(supervisor_failure)")
    auditor = replace_once(auditor, "    start = time.monotonic(); p = v.verify(root, parent, mixed, baseline, pin); d = p['design']\n",
        "    start = time.monotonic(); p = v.verify(root, parent, mixed, baseline, pin); d = p['design']\n"
        "    from cctv_dgp_cache_timing_v16_r2 import verify_cache_timing\n"
        "    timing = verify_cache_timing(p, v.read(out / 'cache_timing.json'))\n"
        "    v.require(timing['passed'] and v.read(out / 'cache_initialization.json')['seconds'] == timing['initialization_seconds'], 'Cache timing evidence differs')\n")
    auditor = replace_once(auditor, "['neural_execution_receipt.json', 'execution.json',\n",
        "['neural_execution_receipt.json', 'execution.json', 'cache_initialization.json',\n")
    (WORK / 'scripts/audit_cctv_dgp_broader_codes_v16.py').write_text(auditor, encoding='utf-8', newline='\n')

    importer = (WORK / 'scripts/import_cctv_dgp_broader_codes_v16.py').read_text()
    importer = replace_once(importer, 'import argparse\n', 'import argparse\nimport importlib.util\n')
    importer = replace_once(importer, 'ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))\n',
        "ROOT = Path(__file__).resolve().parents[1]\nBUNDLE = ROOT / 'outputs/cctv_dgp_broader_codes_vm_v16_r2'\nsys.path.insert(0, str(BUNDLE))\n")
    importer = replace_once(importer, "    bundle = ROOT / 'outputs/cctv_dgp_broader_codes_vm_v16'; pin = v.sha(bundle / v.PLAN)",
        "    bundle = BUNDLE; pin = v.sha(bundle / v.PLAN)")
    importer = importer.replace('outputs/broader_codes_v16', 'outputs/broader_codes_v16_r2')
    importer = replace_once(importer, '        from scripts.audit_cctv_dgp_broader_codes_v16 import audit_failure\n',
        "        spec = importlib.util.spec_from_file_location('r2_failure_audit', BUNDLE / 'scripts/audit_cctv_dgp_broader_codes_v16.py')\n"
        "        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)\n"
        "        audit_failure = module.audit_failure\n")
    importer = importer.replace("str(ROOT / 'scripts/audit_cctv_dgp_broader_codes_v16.py')", "str(BUNDLE / 'scripts/audit_cctv_dgp_broader_codes_v16.py')")
    (WORK / 'scripts/import_cctv_dgp_broader_codes_v16.py').write_text(importer, encoding='utf-8', newline='\n')
    shutil.copyfile(WORK / 'scripts/import_cctv_dgp_broader_codes_v16.py', ROOT / 'scripts/import_cctv_dgp_broader_codes_v16_r2.py')
    for source, name in [(ROOT / 'cctv_dgp_cache_timing_v16_r2.py', 'cctv_dgp_cache_timing_v16_r2.py'),
                         (ROOT / 'tests/test_cctv_dgp_v16_cache_timing_r2.py', 'tests/test_cctv_dgp_v16_cache_timing_r2.py'),
                         (Path(__file__), 'scripts/prepare_cctv_dgp_broader_codes_v16.py'),
                         (ROOT / 'scripts/launch_cctv_dgp_broader_codes_v16_r2.py', 'scripts/launch_cctv_dgp_broader_codes_v16.py')]:
        shutil.copyfile(source, WORK / name)
    print('Generated separate V16 r2 source drafts; no inference, backward, training or VM launch.')


def prepare():
    started = time.monotonic(); previous, failure_receipt = predecessor()
    require(WORK.is_dir() and not OUT.exists(), 'Require reviewed source drafts and preserve any frozen r2')
    sys.path.insert(0, str(WORK))
    spec = importlib.util.spec_from_file_location('v16_r2_contract', WORK / 'cctv_dgp_broader_codes_v16.py')
    v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
    test_path = ROOT / 'outputs/cctv_dgp_v16_r2_boundary_tests.json'
    tested = read(test_path)
    require(tested['complete'] and tested['neural_forwards'] == tested['backward_calls'] == tested['optimizer_updates'] == 0
            and tested['tested_sources_sha256'] == {name: sha(WORK / name) for name in v.SOURCES_FILES}, 'Tests stale or incomplete')
    OUT.mkdir(); assets = {}

    def copy_file(source, name):
        target = OUT / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target); assets[name] = sha(target)
        require(assets[name] == sha(source), 'Source copy differs')

    for name in v.SOURCES_FILES:
        copy_file(WORK / name, name)
    for name in previous['assets_sha256']:
        if name.startswith('lineage/'):
            copy_file(test_path if name == 'lineage/boundary_tests.json' else BASE / name, name)
    copy_file(failure_receipt, 'lineage/zero_update_v16_failure_audit.json')
    copy_file(BASE / 'schedule_v16.json', 'schedule_v16.json')
    plan = copy.deepcopy(previous)
    plan.update({'format': v.FORMAT, 'date': '2026-10-05', 'design': v.DESIGN, 'assets_sha256': assets,
        'revision': 'V16 r2: corrected source/role timing with startup/warmups counted once; unchanged learning/data/limits',
        'predecessor_protocol_sha256': sha(BASE / 'broader_codes_protocol_v16.json'),
        'predecessor_failure_archive_sha256': sha(ROOT / 'outputs/cctv-dgp-broader-codes-v16-failure.tar.gz'),
        'cache_order_scope': '10 training and5 validation references per source first; first reference per source/role is warmup, cached once and excluded from rate. All885 references cached once, teacher labels only for781 train roles. Original optimizer schedule unchanged.',
        'finite_stop': 'Same cache900s, fit/evaluation1200s, audit240s, supervisor2400s,20GiB VRAM and12GiB free disk; stratified30-reference steady-rate timing and25-update fit projection, epoch4 CE improvement>=1%. No resume/selection/promotion.'})
    from cctv_dgp_cache_timing_v16_r2 import cache_order
    write(OUT / 'cache_order_v16_r2.json', {'reference_ids': [ref['id'] for ref in cache_order(plan)]})
    assets['cache_order_v16_r2.json'] = sha(OUT / 'cache_order_v16_r2.json')
    v.validate_cohort(plan); v.validate_schedule(plan, read(OUT / 'schedule_v16.json')['steps'])
    write(OUT / v.PLAN, plan)
    pin = sha(OUT / v.PLAN)
    (OUT / 'protocol.sha256').write_text(pin + '\n', encoding='ascii', newline='\n')
    v.verify(OUT, ROOT / 'outputs/cctv_dgp_face_code_fit_vm_v12_r2', ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2',
             ROOT / 'outputs/cctv_dgp_generalization_return_v15/outputs/generalization_v15', pin)
    archive = ROOT / 'outputs/cctv-dgp-broader-codes-v16-r2-execution.tar.gz'
    names = sorted(assets) + [v.PLAN, 'protocol.sha256']
    with tarfile.open(archive, 'x:gz', compresslevel=3) as stream:
        for name in names:
            stream.add(OUT / name, arcname=name, recursive=False)
    with tarfile.open(archive, 'r:gz') as stream:
        members = v.archive_members(stream, OUT, 256 * 1024**2)
        require({member.name for member in members} == set(names), 'Archive coverage differs')
        for member in members:
            require(hashlib.sha256(stream.extractfile(member).read()).hexdigest() == assets.get(member.name, sha(OUT / member.name)),
                    'Archive member differs')
    archive_sha = sha(archive)
    Path(str(archive) + '.sha256').write_text(archive_sha + '  ' + archive.name + '\n', encoding='ascii', newline='\n')
    bootstrap = ROOT / 'outputs/launch_cctv_dgp_broader_codes_v16_r2.py'
    with bootstrap.open('xb') as stream:
        stream.write((OUT / 'scripts/launch_cctv_dgp_broader_codes_v16.py').read_bytes())
    require(sha(bootstrap) == assets['scripts/launch_cctv_dgp_broader_codes_v16.py'], 'Bootstrap copy differs')
    result = {'complete': True, 'protocol_sha256': pin, 'archive_sha256': archive_sha,
              'archive_bytes': archive.stat().st_size, 'members_independently_checked': len(names), 'bootstrap_sha256': sha(bootstrap),
              'parent_assets_verified': len(plan['parent_assets_sha256']), 'data_assets_verified': len(plan['data_assets_sha256']),
              'baseline_assets_verified': len(plan['baseline_assets_sha256']), 'optimizer_updates_budget': 3128,
              'original_schedule_unchanged': sha(OUT / 'schedule_v16.json') == sha(BASE / 'schedule_v16.json'),
              'original_failure_preserved': True, 'seconds': time.monotonic() - started,
              'neural_forwards': 0, 'backward_calls': 0, 'optimizer_updates': 0, 'VM_launched': False,
              'CUDA_preflight_verified': False}
    write(ROOT / 'outputs/cctv_dgp_broader_codes_v16_r2_preparation.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate-only', action='store_true')
    args = parser.parse_args()
    generate() if args.generate_only else prepare()
