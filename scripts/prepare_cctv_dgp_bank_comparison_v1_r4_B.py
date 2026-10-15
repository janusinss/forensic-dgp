"""Seal a hashing-only B4 revision and a small verified asset-reuse transfer."""
import ast
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
PRIOR_NAME = 'cctv_dgp_bank_comparison_v1_r3_B_vm'
PRIOR_STEM = 'cctv-dgp-bank-comparison-v1-r3-B'
NAME = 'cctv_dgp_bank_comparison_v1_r4_B_vm'
STEM = 'cctv-dgp-bank-comparison-v1-r4-B'
PIN = 'f34b7b54cd322d26a1a2159cffa28aabb1258fcb484914fd781d70e3e7bc0c35'
TRANSFER_NAME = 'cctv_dgp_bank_comparison_B4_transfer'
TRANSFER_STEM = STEM + '-code-transfer'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def replace(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def archive(root, name, path):
    with tarfile.open(path, 'x:gz', compresslevel=1) as tar:
        for file in sorted(root.rglob('*')):
            if file.is_file():
                tar.add(file, arcname=name + '/' + file.relative_to(root).as_posix(), recursive=False)
    digest = sha(path)
    with Path(str(path) + '.sha256').open('x', encoding='ascii', newline='\n') as stream:
        stream.write(digest + '  ' + path.name + '\n')
    return digest


def main():
    analysis = OUT / 'cctv_dgp_bank_comparison_v1_r3_B_analysis'
    returned = OUT / 'cctv_dgp_bank_comparison_v1_r3_B_return'
    parent = OUT / PRIOR_NAME
    assert sha(parent / 'protocol.json') == sha(returned / 'protocol.json') == PIN
    imp = read(analysis / 'import.json'); exe = read(analysis / 'failure_execution_audit.json')
    audit = read(analysis / 'independent_audit.json'); regression = read(analysis / 'hash_compat_loader_regression.json')
    assert all(record['complete'] for record in [imp, exe, audit, regression])
    assert exe['actual_optimizer_updates'] == audit['optimizer_updates_in_VM'] == 0
    assert exe['missing_hash_API_reproduced'] and regression['cases_checked'] == 5
    helper = ROOT / 'scripts/cctv_dgp_bank_pinned_initializer_v2.py'
    assert sha(helper) == regression['loader_sha256']
    dest = OUT / NAME; full = OUT / (STEM + '-execution.tar.gz'); patch = OUT / TRANSFER_NAME
    small = OUT / (TRANSFER_STEM + '.tar.gz')
    assert not any(path.exists() for path in [dest, full, patch, small])
    p = read(parent / 'protocol.json')
    dest.mkdir()
    for name, expected in p['assets_sha256'].items():
        assert sha(parent / name) == sha(returned / name) == expected
        if name == 'cctv_dgp_bank_pinned_initializer_v1.py':
            continue
        target = dest / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent / name, target)
        assert sha(target) == expected
    for name in ['cctv_dgp_bank_comparison_v1_contract.py', 'cctv_dgp_bank_comparison_v1_model.py']:
        target = dest / name
        text = replace(target.read_text(encoding='utf-8'), repr(PRIOR_NAME), repr(NAME))
        if name.endswith('_contract.py'):
            text = replace(text, repr(PRIOR_STEM), repr(STEM))
            text = replace(text, repr('own256-conditioned-frozen-generative-bank-comparison-v1-r3-B-pinned-initializer'),
                           repr('own256-conditioned-frozen-generative-bank-comparison-v1-r4-B-Python310-hash'))
            text = replace(text, "    assert p['initializer_preflight_before_baseline'] is True",
                "    assert p['initializer_preflight_before_baseline'] is True\n"
                "    assert p['retained_B3_protocol_sha256'] == " + repr(PIN) + "\n"
                "    assert p['loader_hash_policy'] == 'streaming-sha256-Python310-v1'\n"
                "    assert p['loader_hash_runtime_check_in_transfer_CLI'] is True")
            text = replace(text, "'export_manifest.json', 'export.log', 'export_exit_code.txt'}",
                "'export_manifest.json', 'export.log', 'export_exit_code.txt', 'packet_install_receipt.json'}")
        target.write_text(text, encoding='utf-8', newline='\n')
    shutil.copyfile(helper, dest / helper.name)
    installer = ROOT / 'scripts/install_cctv_dgp_bank_B4_from_B3_v1.py'
    shutil.copyfile(installer, dest / 'scripts/install_packet_from_B3.py')
    worker = dest / 'scripts/cctv_dgp_bank_comparison_v1_vm.py'
    text = replace(worker.read_text(encoding='utf-8'),
                   'from cctv_dgp_bank_pinned_initializer_v1 import load_pinned_initializer',
                   'from cctv_dgp_bank_pinned_initializer_v2 import load_pinned_initializer')
    text = replace(text, '    if args.verify_transfer:\n        print(',
        "    if args.verify_transfer:\n"
        "        from cctv_dgp_bank_pinned_initializer_v2 import sha as loader_sha\n"
        "        initializer_name = 'initializers/initial_B.pth'\n"
        "        assert loader_sha(root / initializer_name) == p['assets_sha256'][initializer_name]\n"
        "        print(")
    text = replace(text, "'native_DEV_cases': 24, 'optimizer_updates': 0, 'neural_or_gradient_calls': 0}))",
        "'native_DEV_cases': 24, 'optimizer_updates': 0, 'neural_or_gradient_calls': 0,\n"
        "            'loader_hash_runtime_checked': True, 'python': platform.python_version()}))")
    worker.write_text(text, encoding='utf-8', newline='\n')
    provenance = dest / 'provenance'; provenance.mkdir(exist_ok=True)
    for source, name in [(parent / 'protocol.json', 'B3_protocol.json'),
        (returned / 'outputs/bank_comparison_v1/failure.json', 'B3_failure.json'),
        (returned / 'outputs/bank_comparison_v1/initializer_load_B_0.json', 'B3_initializer_load_failure.json'),
        (analysis / 'import.json', 'B3_return_import.json'),
        (analysis / 'failure_execution_audit.json', 'B3_failure_execution_audit.json'),
        (analysis / 'hash_compat_loader_regression.json', 'Python310_hash_compat_regression.json')]:
        shutil.copyfile(source, provenance / name)
    p = copy.deepcopy(p)
    p.update(format='own256-conditioned-frozen-generative-bank-comparison-v1-r4-B-Python310-hash',
        UTC=datetime.now(timezone.utc).isoformat(), retained_B3_protocol_sha256=PIN,
        loader_hash_policy='streaming-sha256-Python310-v1',
        loader_hash_runtime_check_in_transfer_CLI=True)
    p['assets_sha256'] = {path.relative_to(dest).as_posix(): sha(path)
                          for path in sorted(dest.rglob('*')) if path.is_file()}
    write(dest / 'protocol.json', p); pin = sha(dest / 'protocol.json')
    spec = importlib.util.spec_from_file_location('bank_B4_contract', dest / 'cctv_dgp_bank_comparison_v1_contract.py')
    contract = importlib.util.module_from_spec(spec); spec.loader.exec_module(contract)
    contract.verify(dest, pin)
    for name in p['assets_sha256']:
        if name.endswith('.py'):
            ast.parse((dest / name).read_text(encoding='utf-8-sig'), filename=name, feature_version=(3, 10))
    full_hash = archive(dest, NAME, full)
    patch.mkdir(); shutil.copyfile(dest / 'protocol.json', patch / 'protocol.json')
    shutil.copyfile(parent / 'protocol.json', patch / 'source_protocol.json')
    shutil.copyfile(installer, patch / 'install_packet.py')
    old = read(parent / 'protocol.json')
    overlay = {name: expected for name, expected in p['assets_sha256'].items()
               if old['assets_sha256'].get(name) != expected}
    for name in overlay:
        target = patch / 'overlay' / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(dest / name, target)
    transfer = dict(complete=True, source_name=PRIOR_NAME, target_name=NAME,
        protocol_sha256=pin, source_protocol_sha256=PIN,
        overlay_assets=overlay, installer_sha256=sha(installer), copy_cap_seconds=300,
        minimum_free_after_install_bytes=16*1024**3,
        source_or_research_files_deleted=0, model_gradient_or_training_calls=0)
    write(patch / 'transfer.json', transfer)
    small_hash = archive(patch, TRANSFER_NAME, small)
    record = dict(complete=True, route='B', name=NAME, stem=STEM, protocol_sha256=pin,
        archive_sha256=full_hash, archive_bytes=full.stat().st_size,
        packet_assets=len(p['assets_sha256']), prior_B3_protocol_sha256=PIN,
        transfer_name=TRANSFER_NAME, transfer_stem=TRANSFER_STEM,
        transfer_archive_sha256=small_hash, transfer_archive_bytes=small.stat().st_size,
        transfer_manifest_sha256=sha(patch / 'transfer.json'), overlay_assets=len(overlay),
        reused_assets=len(p['assets_sha256'])-len(overlay),
        retained_B3_failure=True, scientific_recipe_and_caps_unchanged=True,
        initializer_asset_bytes_unchanged=True,
        hash_compat_loader_regression_sha256=sha(analysis / 'hash_compat_loader_regression.json'),
        manual_VM_run_pending=True, current_VM_cleanup_pending=True,
        neural_gradient_or_training_calls=0, model_qualified=False, app_promotion=False, goal_complete=False)
    write(OUT / 'cctv_dgp_bank_comparison_v1_r4_B_preparation.json', record)
    print(record, flush=True)


if __name__ == '__main__':
    main()
