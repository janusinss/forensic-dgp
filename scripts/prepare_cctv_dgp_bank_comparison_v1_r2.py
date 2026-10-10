"""Resource-only revision: two independent manual roots, no neural calls.

Retains the original stop. Models, objective, data, schedule and quality gates
are unchanged. Each packet fits one route; returns must be audited separately.
"""
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
PARENT = OUT / 'cctv_dgp_bank_comparison_v1_vm'
RETURN = OUT / 'cctv_dgp_bank_comparison_v1_return'
ANALYSIS = OUT / 'cctv_dgp_bank_comparison_v1_analysis'
PARENT_PIN = '0e4f9645abc99f6d980f3b20240b196f65f3e130472795faa2f964df1a5a3b3e'
GIB = 1024**3
SNAPSHOT_BOUND = 3*GIB + 64*1024**2
STATE_BOUND = 512*1024**2
ABLATION_BOUND = 128*1024**2


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, record):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False)+'\n')


def replace_once(source, before, after):
    assert source.count(before) == 1, before
    return source.replace(before, after)


def revised_sources(route, name, stem):
    originals = {n: (PARENT/n).read_text(encoding='utf-8') for n in [
        'cctv_dgp_bank_comparison_v1_contract.py', 'cctv_dgp_bank_comparison_v1_model.py',
        'scripts/cctv_dgp_bank_comparison_v1_vm.py']}
    contract = replace_once(originals['cctv_dgp_bank_comparison_v1_contract.py'],
        "NAME = 'cctv_dgp_bank_comparison_v1_vm'", 'NAME = '+repr(name))
    contract = replace_once(contract, "STEM = 'cctv-dgp-bank-comparison-v1'", 'STEM = '+repr(stem))
    contract = replace_once(contract, "'maximum_backwards': 500, 'maximum_optimizer_updates': 100",
        "'maximum_backwards': 250, 'maximum_optimizer_updates': 50")
    contract = replace_once(contract, "'recognizer_forward_calls': 3100", "'recognizer_forward_calls': 2000")
    contract = replace_once(contract, "'maximum_gradient_queries': 6", "'maximum_gradient_queries': "+('2' if route=='A' else '4'))
    contract = replace_once(contract, "'prior_fixture_forward_calls': 1", "'prior_fixture_forward_calls': "+('0' if route=='A' else '1'))
    contract = replace_once(contract, "'B_forward_calls': 1300" if route=='A' else "'A_forward_calls': 1200",
        "'B_forward_calls': 0" if route=='A' else "'A_forward_calls': 0")
    contract = replace_once(contract, "assert p['format'] == 'own256-conditioned-frozen-generative-bank-comparison-v1'",
        "assert p['format'] == 'own256-conditioned-frozen-generative-bank-comparison-v1-r2-split-route'\n"
        "    assert p['selected_routes'] == ["+repr(route)+"]\n"
        "    assert p['parent_protocol_sha256'] == "+repr(PARENT_PIN)+"\n"
        "    assert p['resource_layout'] == {'full_candidate_snapshot_bound_bytes': "+str(SNAPSHOT_BOUND)+
        ", 'state_and_fixture_reserve_bytes': "+str(STATE_BOUND)+", 'B_ablation_bound_bytes': "+str(ABLATION_BOUND)+
        ", 'export_metadata_reserve_bytes': 33554432, 'baseline_growth_allowance': 1.05}\n"
        "    assert p['baseline_storage_receipt']['baseline_bytes'] == 2123116593\n"
        "    assert p['comparison_complete_in_this_packet'] is False\n"
        "    assert p['separate_return_audit_required_before_next_route'] is True")
    model = replace_once(originals['cctv_dgp_bank_comparison_v1_model.py'],
        "NAME = 'cctv_dgp_bank_comparison_v1_vm'", 'NAME = '+repr(name))
    worker = originals['scripts/cctv_dgp_bank_comparison_v1_vm.py']
    old = """        projected = int(baseline_bytes * 3.5 * 1.35 + 512 * 1024**2)
        write(out / 'storage_projection.json', {'baseline_bytes': baseline_bytes, 'snapshot_factor': 3.5,
            'compression_uncertainty_factor': 1.35, 'checkpoint_reserve_bytes': 512 * 1024**2,
            'projected_return_bytes': projected, 'cap_bytes': BUDGETS['return_uncompressed_bytes'],
            'export_reserve_bytes': projected, 'optimizer_updates': 0})
        assert projected <= BUDGETS['return_uncompressed_bytes'], 'Measured output projection stop before optimizer'
        assert shutil.disk_usage(root).free >= 2*projected-baseline_bytes+BUDGETS['disk_reserve_bytes'], 'Output+export reserve stop before optimizer'"""
    new = """        layout = p['resource_layout']
        ablation = layout['B_ablation_bound_bytes'] if p['selected_routes'] == ['B'] else 0
        projected = baseline_bytes + layout['full_candidate_snapshot_bound_bytes'] + layout['state_and_fixture_reserve_bytes'] + ablation
        packet_bytes = sum((root/n).stat().st_size for n in p['assets_sha256'])
        export_reserve = projected + packet_bytes + layout['export_metadata_reserve_bytes']
        write(out / 'storage_projection.json', {'baseline_bytes': baseline_bytes,
            'candidate_lossless_upper_bound_bytes': layout['full_candidate_snapshot_bound_bytes'],
            'B_ablation_upper_bound_bytes': ablation, 'state_and_fixture_reserve_bytes': layout['state_and_fixture_reserve_bytes'],
            'projected_return_bytes': projected, 'cap_bytes': BUDGETS['return_uncompressed_bytes'],
            'packet_bytes': packet_bytes, 'export_reserve_bytes': export_reserve, 'optimizer_updates': 0,
            'prior_failure_preserved': True, 'scientific_recipe_and_gates_unchanged': True})
        assert projected <= BUDGETS['return_uncompressed_bytes'], 'Measured output projection stop before optimizer'
        assert shutil.disk_usage(root).free >= projected-baseline_bytes+export_reserve+BUDGETS['disk_reserve_bytes'], 'Output+packet+export reserve stop before optimizer'"""
    worker = replace_once(worker, old, new)
    assert worker.count("for route in ['A', 'B']:") == 2
    worker = worker.replace("for route in ['A', 'B']:", "for route in p['selected_routes']:")
    worker = replace_once(worker, "for label in ['A', 'B']}", "for label in p['selected_routes']}")
    worker = replace_once(worker, "        total_projection += timing['B']['snapshot_projection_seconds'] * (44/805)",
        "        if 'B' in timing:\n            total_projection += timing['B']['snapshot_projection_seconds'] * (44/805)")
    worker = replace_once(worker, "all_pass = len(completed) == 2 and all(r['capacity_pass'] for r in completed)",
        "all_pass = len(completed) == 1 and all(r['capacity_pass'] for r in completed)")
    worker = replace_once(worker, "progress['stage'] = 'finite_comparison_complete'", "progress['stage'] = 'finite_route_complete'")
    worker = replace_once(worker, "'all_routes_capacity_pass': all_pass, 'automatic_follow_on': False",
        "'all_selected_routes_capacity_pass': all_pass, 'comparison_complete': False, 'automatic_follow_on': False")
    # Check output+export capacity before spending a full baseline snapshot.
    worker = replace_once(worker, "    # A GPU task, rather than the user's own tmux shell, is a competing workload.",
        "    layout = p['resource_layout']\n"
        "    historical = int(p['baseline_storage_receipt']['baseline_bytes'] * layout['baseline_growth_allowance'])\n"
        "    forecast = historical + layout['full_candidate_snapshot_bound_bytes'] + layout['state_and_fixture_reserve_bytes']\n"
        "    forecast += layout['B_ablation_bound_bytes'] if p['selected_routes'] == ['B'] else 0\n"
        "    assert forecast <= BUDGETS['return_uncompressed_bytes'], 'Historical lossless output bound; no neural calls'\n"
        "    packet = sum((root/n).stat().st_size for n in p['assets_sha256'])\n"
        "    assert shutil.disk_usage(root).free >= 2*forecast+packet+layout['export_metadata_reserve_bytes']+BUDGETS['disk_reserve_bytes'], 'Historical output+export reserve; no neural calls'\n"
        "    # A GPU task, rather than the user's own tmux shell, is a competing workload.")
    return {'cctv_dgp_bank_comparison_v1_contract.py': contract,
        'cctv_dgp_bank_comparison_v1_model.py': model, 'scripts/cctv_dgp_bank_comparison_v1_vm.py': worker}


def main():
    assert sha(PARENT/'protocol.json') == PARENT_PIN
    original = read(PARENT/'protocol.json')
    audit = read(ANALYSIS/'independent_audit.json')
    assert audit['complete'] and audit['optimizer_updates_in_VM'] == 0
    assert audit['saved_paired_raw_PNG_cases_checked'] == 3905 and audit['native_compositions_checked'] == 24
    failure = read(RETURN/'outputs/bank_comparison_v1/failure.json')
    storage = read(RETURN/'outputs/bank_comparison_v1/storage_projection.json')
    assert failure['message'] == 'Measured output projection stop before optimizer'
    assert storage['baseline_bytes'] == 2123116593 and storage['projected_return_bytes'] == 10568596813
    assert failure['progress']['counts']['optimizer_updates'] == 0
    for n, expected in original['assets_sha256'].items():
        assert sha(PARENT/n) == expected
    records = []
    for route in ['A', 'B']:
        name = 'cctv_dgp_bank_comparison_v1_r2_'+route+'_vm'
        stem = 'cctv-dgp-bank-comparison-v1-r2-'+route
        dest = OUT/name; archive = OUT/(stem+'-execution.tar.gz')
        assert not dest.exists() and not archive.exists()
        dest.mkdir(); modified = revised_sources(route, name, stem); bindings = {}
        for n, expected in original['assets_sha256'].items():
            f = dest/n; f.parent.mkdir(parents=True, exist_ok=True)
            if n in modified:
                f.write_text(modified[n], encoding='utf-8', newline='\n')
            else:
                shutil.copyfile(PARENT/n, f); assert sha(f) == expected
            bindings[n] = sha(f)
        evidence = [PARENT/'protocol.json', RETURN/'outputs/bank_comparison_v1/failure.json',
            RETURN/'outputs/bank_comparison_v1/storage_projection.json', ANALYSIS/'import.json', ANALYSIS/'independent_audit.json']
        for i, src in enumerate(evidence):
            n = 'provenance/retained_bank_v1_stop/'+str(i)+'_'+src.name
            f = dest/n; f.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, f); bindings[n] = sha(f)
        p = dict(original)
        p.update(format='own256-conditioned-frozen-generative-bank-comparison-v1-r2-split-route',
            UTC=datetime.now(timezone.utc).isoformat(), selected_routes=[route], parent_protocol_sha256=PARENT_PIN,
            resource_layout=dict(full_candidate_snapshot_bound_bytes=SNAPSHOT_BOUND, state_and_fixture_reserve_bytes=STATE_BOUND,
                B_ablation_bound_bytes=ABLATION_BOUND, export_metadata_reserve_bytes=32*1024**2, baseline_growth_allowance=1.05),
            baseline_storage_receipt=storage, separate_return_audit_required_before_next_route=True,
            comparison_complete_in_this_packet=False, assets_sha256=bindings)
        spec = importlib.util.spec_from_file_location('frozen_contract_'+route, dest/'cctv_dgp_bank_comparison_v1_contract.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        p['budgets'] = module.BUDGETS
        write(dest/'protocol.json', p); pin = sha(dest/'protocol.json'); module.verify(dest, pin)
        for n in bindings:
            if n.endswith('.py'):
                ast.parse((dest/n).read_text(encoding='utf-8-sig'), feature_version=(3,10))
        with archive.open('xb') as stream:
            with tarfile.open(fileobj=stream, mode='w:gz', compresslevel=1) as tar:
                for f in sorted(dest.rglob('*')):
                    if f.is_file(): tar.add(f, arcname=name+'/'+f.relative_to(dest).as_posix(), recursive=False)
        digest = sha(archive)
        Path(str(archive)+'.sha256').write_text(digest+'  '+archive.name+'\n', encoding='ascii', newline='\n')
        bound = storage['baseline_bytes']+SNAPSHOT_BOUND+STATE_BOUND+(ABLATION_BOUND if route=='B' else 0)
        records.append(dict(route=route, name=name, stem=stem, protocol_sha256=pin, archive_sha256=digest,
            archive_bytes=archive.stat().st_size, prepared_packet_assets=len(bindings), output_bound_bytes=bound,
            historical_growth_output_bound_bytes=int(storage['baseline_bytes']*1.05)+bound-storage['baseline_bytes'],
            current_VM_storage_not_verified=True, manual_training_pending=True, quality_qualified=False))
        print(records[-1], flush=True)
    write(OUT/'cctv_dgp_bank_comparison_v1_r2_preparation.json', dict(complete=True, routes=records,
        resource_revision_only=True, original_failure_retained=True, local_neural_gradient_or_optimizer_calls=0,
        model_qualified=False, comparison_execution_pending=True, goal_complete=False))


if __name__ == '__main__':
    main()
