"""Independent source/data/archive/negative-launch audit; no neural calls."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs'
DEST = OUT/'cctv_dgp_bank_path_ablation_v1_vm'
PARENT = OUT/'cctv_dgp_bank_photometric_guard_v1_return'
ANALYSIS = OUT/'cctv_dgp_bank_photometric_guard_v1_analysis'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    receipt = OUT/'cctv_dgp_bank_path_ablation_v1_transfer_audit.json'
    assert not receipt.exists()
    prep = read(OUT/'cctv_dgp_bank_path_ablation_v1_preparation.json')
    assert prep['complete'] and prep['manual_VM_diagnostic_pending']
    p = read(DEST/'protocol.json')
    assert sha(DEST/'protocol.json') == prep['protocol_sha256']
    assert p['parent_protocol_sha256'] == sha(PARENT/'protocol.json')
    old = read(PARENT/'protocol.json')
    assert p['case_ids'] == old['preview_case_ids'] and len(p['case_ids']) == 100
    selected = [i for i in range(0, len(old['cases']), 5) if old['cases'][i]['id'] in p['case_ids']]
    assert len(selected) == 20
    assert all({c['id'] for c in old['cases'][i:i+5]} <= set(p['case_ids']) for i in selected)
    assert len({old['cases'][i]['source_person_or_reference'] for i in selected}) == 20
    assert len({old['cases'][i]['source'] for i in p['parity_begins']}) == 2
    assert p['native_case_ids'] == [c['id'] for c in old['native_development']] and len(p['native_case_ids']) == 24
    assert p['preserved_scientific_thresholds'] == old['scientific_thresholds']
    assert p['optimizer_updates'] == p['gradient_queries'] == p['backwards'] == p['full_epochs'] == 0
    assert p['paired_variant_compositions'] == 7*100 and p['native_variant_compositions'] == 7*24
    assert p['model_forward_cap'] == 7*(20+24)+4 == 312
    assert p['recognizer_forward_cap'] == 7*20 == 140
    assert p['minimum_free_bytes'] >= 2*p['output_byte_cap']+p['disk_reserve_bytes']+1024**2
    assert not p['app_promotion'] and not p['failed_run_resume_allowed'] and not p['automatic_follow_on']
    for root, items in [(DEST, p['assets_sha256']), (PARENT, p['parent_required_sha256'])]:
        for name, expected in items.items():
            path = root/name
            assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root)
            assert sha(path) == expected, name
    for name, expected in p['evidence_sha256'].items():
        assert read(ROOT/name)['complete'] and sha(ROOT/name) == expected
    worker = DEST/'scripts/cctv_dgp_bank_path_ablation_v1_vm.py'
    source = worker.read_text(encoding='utf-8')
    assert 'file_digest' not in source, 'Retain Python310-compatible VM hashing'
    tree = ast.parse(source)
    functions = {n.name:n for n in tree.body if isinstance(n, ast.FunctionDef)}
    run = functions['run']
    assert isinstance(run.body[0], ast.Expr) and isinstance(run.body[0].value, ast.Call)
    assert run.body[0].value.func.id == 'require_vm'
    forbidden = {'backward', 'step', 'enable_vm_learning', 'Adam', 'AdamW', 'SGD', 'grad'}
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in forbidden
                   for n in ast.walk(tree))
    assert not any(isinstance(n, ast.ImportFrom) and 'scripts' in (n.module or '') for n in ast.walk(tree))
    assert not any(isinstance(n, ast.Import) and any('optim' in alias.name for alias in n.names) for n in ast.walk(tree))
    assert 'torch.set_grad_enabled(False)' in source
    assert "candidate.load_state_dict(stored['model'], strict=True)" in source
    assert 'torch.equal(zeroed, disabled)' in source and 'np.array_equal(raw, previous)' in source
    assert 'state_hash(candidate) == initial_state' in source
    shell = DEST/'scripts/run_paths.sh'
    text = shell.read_text(encoding='utf-8')
    assert b'\r' not in shell.read_bytes()
    assert '1230 python' in text and '330 python' in text and '--kill-after=30' in text
    assert 'Immutable used root' in text and '--verify-transfer' in text
    bash = Path('C:/Program Files/Git/bin/bash.exe')
    assert bash.is_file(), 'Git Bash required for readonly shell syntax check'
    syntax = subprocess.run([str(bash), '-n', str(shell)], capture_output=True, text=True, timeout=30)
    assert syntax.returncode == 0, syntax.stderr
    archive = OUT/(prep['stem']+'-execution.tar.gz')
    assert sha(archive) == prep['archive_sha256']
    checksum = Path(str(archive)+'.sha256').read_bytes()
    assert checksum == (prep['archive_sha256']+'  '+archive.name+'\n').encode('ascii')
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        expected = {f'{prep["name"]}/{name}': value for name,value in
                    {**p['assets_sha256'], 'protocol.json':prep['protocol_sha256']}.items()}
        assert len(members) == len(expected) == 3
        assert len({m.name for m in members}) == 3
        for member in members:
            assert member.isfile() and member.name in expected
            assert member.size == (DEST/Path(member.name).relative_to(prep['name'])).stat().st_size
            stream = tar.extractfile(member)
            assert hashlib.sha256(stream.read()).hexdigest() == expected[member.name]
    # Execute only the pre-neural rejection on Windows against the real packet.
    assert not (DEST/'outputs').exists()
    negative = subprocess.run([sys.executable, '-B', str(worker), '--root', str(DEST),
        '--parent-root', str(PARENT), '--protocol-sha', prep['protocol_sha256'], '--run'],
        capture_output=True, text=True, timeout=120)
    assert negative.returncode != 0 and "platform.system() == 'Linux'" in negative.stderr
    assert not (DEST/'outputs').exists()
    evidence = dict(returncode=negative.returncode, stdout=negative.stdout, stderr=negative.stderr,
                    before_model_imports_and_outputs=True)
    with (OUT/'cctv_dgp_bank_path_ablation_v1_windows_rejection.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(evidence, indent=2)+'\n')
    for name, expected in read(ANALYSIS/'local_protected_before.json')['files_sha256'].items():
        assert sha(ROOT/name) == expected
    record = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), checker_sha256=sha(__file__),
        preparation_sha256=sha(OUT/'cctv_dgp_bank_path_ablation_v1_preparation.json'),
        protocol_sha256=prep['protocol_sha256'], archive_sha256=sha(archive), archive_members_checked=3,
        packet_assets_checked=len(p['assets_sha256']), parent_assets_checked=len(p['parent_required_sha256']),
        data_only_validation=True, Windows_rejection_before_neural_or_output_work=True,
        Bash_readonly_syntax_pass=True, Python310_hash_compatibility_checked=True,
        input_and_native_identity_lists_unchanged=True, original_thresholds_unchanged=True,
        sealed_model_and_parent_stop_unchanged=True, local_neural_calls=0,
        local_gradient_queries=0, local_backwards=0, local_optimizer_updates=0,
        manual_VM_inference_pending=True, actual_CUDA_parity_timing_and_results_unrun=True,
        independent_final_review_pending=True, model_qualified=False, goal_complete=False)
    with receipt.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print(dict(complete=True, archive_members=3, local_neural_calls=0, manual_VM_diagnostic_pending=True))


if __name__ == '__main__':
    main()
