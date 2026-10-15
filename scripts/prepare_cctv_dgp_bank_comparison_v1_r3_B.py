"""Initializer-load repair in a new immutable B root; science and caps stay fixed."""
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
PRIOR_NAME = 'cctv_dgp_bank_comparison_v1_r2_B_vm'
PRIOR_STEM = 'cctv-dgp-bank-comparison-v1-r2-B'
NAME = 'cctv_dgp_bank_comparison_v1_r3_B_vm'
STEM = 'cctv-dgp-bank-comparison-v1-r3-B'
PIN = '68197fd20bef1eae52c4c232164a027e095d567d4bfb37aea79c4f908853ea42'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False)+'\n')


def replace(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def main():
    analysis = OUT / 'cctv_dgp_bank_comparison_v1_r2_B_analysis'
    returned = OUT / 'cctv_dgp_bank_comparison_v1_r2_B_return'
    parent = OUT / PRIOR_NAME
    assert sha(parent / 'protocol.json') == sha(returned / 'protocol.json') == PIN
    imp = read(analysis / 'import.json'); exe = read(analysis / 'failure_execution_audit.json')
    regression = read(analysis / 'pinned_loader_regression.json')
    assert imp['complete'] and exe['complete'] and regression['complete']
    assert exe['actual_optimizer_updates'] == 0 and exe['failure_message'] == 'Fresh initializer differs'
    assert regression['cases_checked'] == 5 and regression['every_loaded_tensor_bit_exact']
    helper = ROOT / 'scripts/cctv_dgp_bank_pinned_initializer_v1.py'
    assert sha(helper) == regression['loader_sha256']
    dest = OUT / NAME; archive = OUT / (STEM+'-execution.tar.gz')
    assert not dest.exists() and not archive.exists()
    p = read(parent / 'protocol.json')
    dest.mkdir()
    for name, expected in p['assets_sha256'].items():
        assert sha(parent / name) == expected
        target = dest / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent / name, target); assert sha(target) == expected
    for name in ['cctv_dgp_bank_comparison_v1_contract.py', 'cctv_dgp_bank_comparison_v1_model.py']:
        target = dest / name
        text = replace(target.read_text(encoding='utf-8'), repr(PRIOR_NAME), repr(NAME))
        if name.endswith('_contract.py'):
            text = replace(text, repr(PRIOR_STEM), repr(STEM))
            text = replace(text,
                "assert p['format'] == 'own256-conditioned-frozen-generative-bank-comparison-v1-r2-split-route'",
                "assert p['format'] == 'own256-conditioned-frozen-generative-bank-comparison-v1-r3-B-pinned-initializer'\n"
                "    assert p['initializer_loading_policy'] == 'hash-pinned-strict-state-load-v1'\n"
                "    assert p['initializer_preflight_before_baseline'] is True\n"
                "    assert p['retained_B2_protocol_sha256'] == "+repr(PIN))
        target.write_text(text, encoding='utf-8', newline='\n')
    shutil.copyfile(helper, dest / 'cctv_dgp_bank_pinned_initializer_v1.py')
    worker = dest / 'scripts/cctv_dgp_bank_comparison_v1_vm.py'
    text = worker.read_text(encoding='utf-8')
    text = replace(text, '        def create(label, learning=False):',
        '        initializer_records = []\n\n        def create(label, learning=False):')
    before = """            initial = torch.load(root / f'initializers/initial_{label}.pth', map_location='cuda', weights_only=True)
            assert state_hash(model) == state_hash(initial), 'Fresh initializer differs'
            model.load_state_dict(initial, strict=True)
            del initial"""
    after = """            from cctv_dgp_bank_pinned_initializer_v1 import load_pinned_initializer
            loading = {'torch': torch.__version__, 'loaded_on': 'cuda', 'learning_enabled': False}
            name = 'initializers/initial_' + label + '.pth'
            try:
                load_pinned_initializer(model, root / name, p['assets_sha256'][name], label, loading)
            except BaseException as exc:
                loading.update(failure_type=type(exc).__name__, failure_message=str(exc))
                write(out / ('initializer_load_' + label + '_' + str(len(initializer_records)) + '.json'), loading)
                raise
            assert state_hash(model) == loading['authoritative_state_sha256']
            initializer_records.append(loading)
            write(out / ('initializer_load_' + label + '_' + str(len(initializer_records)-1) + '.json'), loading)"""
    text = replace(text, before, after)
    text = replace(text, "        baseline = metrics('baseline', original.net)",
        "        # Resolve the previous zero-update load failure before the expensive baseline.\n"
        "        progress.update({'stage': 'pinned_initializer_before_baseline', 'route': 'B', 'route_updates': 0})\n"
        "        candidate = create('B', learning=False)\n"
        "        assert counts['B'] == counts['gradient_queries'] == counts['optimizer_updates'] == 0\n"
        "        del candidate; candidate = None; torch.cuda.empty_cache()\n"
        "        clock()\n"
        "        progress.update({'stage': 'baseline_snapshot', 'route': None})\n"
        "        baseline = metrics('baseline', original.net)")
    worker.write_text(text, encoding='utf-8', newline='\n')
    evidence = [parent / 'protocol.json', returned / 'outputs/bank_comparison_v1/failure.json',
        analysis / 'import.json', analysis / 'failure_execution_audit.json',
        analysis / 'CPU_initializer_inspection.json', analysis / 'pinned_loader_regression.json']
    for i, source in enumerate(evidence):
        target = dest / 'provenance/retained_B2_initializer_stop' / (str(i)+'_'+source.name)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
    p = copy.deepcopy(p)
    p.update(format='own256-conditioned-frozen-generative-bank-comparison-v1-r3-B-pinned-initializer',
        UTC=datetime.now(timezone.utc).isoformat(), initializer_loading_policy='hash-pinned-strict-state-load-v1',
        initializer_preflight_before_baseline=True, retained_B2_protocol_sha256=PIN,
        constructor_tensor_differences_recorded=True,
        only_new_style_conditioner_constructor_differences_permitted=True,
        CPU_VM_constructor_specific_cause_unproven=True)
    p['assets_sha256'] = {f.relative_to(dest).as_posix(): sha(f) for f in sorted(dest.rglob('*')) if f.is_file()}
    write(dest / 'protocol.json', p); pin = sha(dest / 'protocol.json')
    spec = importlib.util.spec_from_file_location('bank_B3_contract', dest / 'cctv_dgp_bank_comparison_v1_contract.py')
    contract = importlib.util.module_from_spec(spec); spec.loader.exec_module(contract); contract.verify(dest, pin)
    for name in p['assets_sha256']:
        if name.endswith('.py'):
            ast.parse((dest / name).read_text(encoding='utf-8-sig'), filename=name, feature_version=(3, 10))
    with archive.open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w:gz', compresslevel=1) as tar:
            for path in sorted(dest.rglob('*')):
                if path.is_file():
                    tar.add(path, arcname=NAME+'/'+path.relative_to(dest).as_posix(), recursive=False)
    digest = sha(archive)
    Path(str(archive)+'.sha256').write_text(digest+'  '+archive.name+'\n', encoding='ascii', newline='\n')
    record = dict(complete=True, route='B', name=NAME, stem=STEM, protocol_sha256=pin,
        archive_sha256=digest, archive_bytes=archive.stat().st_size, packet_assets=len(p['assets_sha256']),
        prior_B2_protocol_sha256=PIN, retained_B2_failure=True, scientific_recipe_and_caps_unchanged=True,
        initializer_asset_bytes_unchanged=True, constructor_mismatch_specific_VM_cause_unproven=True,
        loader_CPU_regression_sha256=sha(analysis / 'pinned_loader_regression.json'),
        manual_VM_run_pending=True, current_VM_storage_unverified=True,
        neural_gradient_or_training_calls=0, model_qualified=False, app_promotion=False, goal_complete=False)
    write(OUT / 'cctv_dgp_bank_comparison_v1_r3_B_preparation.json', record)
    print(record, flush=True)


if __name__ == '__main__':
    main()
