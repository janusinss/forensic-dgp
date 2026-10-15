"""Independent hash/namespace/array/static checks; no neural construction."""
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

from cctv_dgp_spatial_readout_v1_contract import NAME, STEM, CHECKPOINT, read, write, sha, verify_transfer, validate_protocol
from audit_cctv_dgp_spatial_readout_v1_return import safe_member, independent_gate

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/'outputs'
REVIEW = OUT/'cctv_dgp_reconstruction_design_review_v2'
PARENT = OUT/'cctv_dgp_bank_photometric_guard_v1_return'
PACKET = OUT/NAME


def main():
    start = time.monotonic()
    prepared = read(REVIEW/'preparation.json')
    pin = prepared['protocol_sha256']
    p, old = verify_transfer(PACKET, PARENT, pin)
    assert sha(OUT/'cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth') == CHECKPOINT
    archive = OUT/(STEM+'-execution.tar.gz')
    assert sha(archive) == prepared['archive_sha256']
    sidecar = (OUT/(archive.name+'.sha256')).read_bytes()
    assert b'\r' not in sidecar and not sidecar.startswith(b'\xef\xbb\xbf')
    assert sidecar.decode('utf-8') == prepared['archive_sha256']+'  '+archive.name+'\n'
    with tarfile.open(archive, 'r:gz') as stream:
        members = stream.getmembers()
        for member in members:
            safe_member(member)
        names = [m.name for m in members]
        expected = {NAME+'/'+n: digest for n, digest in p['assets_sha256'].items()} | {NAME+'/protocol.json': pin}
        assert len(names) == len(set(names)) and set(names) == set(expected)
        for member in members:
            data = stream.extractfile(member).read()
            assert hashlib.sha256(data).hexdigest() == expected[member.name]
            assert data == (PACKET/member.name.removeprefix(NAME+'/')).read_bytes()
    near = read(OUT/'cctv_dgp_nearby_gradient_learning_v1_r1_return/protocol.json')
    assert p['reference_blocks'] == near['reference_blocks']
    flat = [c for b in p['reference_blocks'] for c in b['case_ids']]
    assert p['case_ids'] == old['preview_case_ids']
    assert len(flat) == len(set(flat)) == 100 and set(flat) == set(p['case_ids'])
    assert len({b['reference'] for b in p['reference_blocks']}) == 20
    assert sha(PACKET/'readouts.npz') == near['assets_sha256']['readouts.npz']
    import numpy as np
    from PIL import Image
    targets = read(PACKET/'evidence/target_bindings.json')['targets']
    refs = {r['id']: r for r in old['references']}
    dimensions = [(16, 512, 128), (32, 512, 128), (64, 256, 128), (128, 128, 64), (256, 64, 32)]
    expanded_count = 0
    with np.load(PACKET/'readouts.npz', allow_pickle=False) as arrays:
        for size, inp, dest in dimensions:
            w, bias = arrays[f'f{size}_weight'], arrays[f'f{size}_bias']
            assert w.shape == (dest, inp, 1, 1) and bias.shape == (dest,)
            assert w.dtype == bias.dtype == np.float32 and np.isfinite(w).all() and np.isfinite(bias).all()
            assert np.linalg.matrix_rank(w[:, :, 0, 0].astype(np.float64)) == dest
            expanded_count += dest*inp*9+dest
    assert expanded_count+609219 == 2176419
    for row in targets:
        ref = refs[row['reference']]
        with Image.open(PARENT/ref['target']) as image:
            actual = hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()
        assert actual == row['canonical_target_RGB_sha256'] == old['canonical_target_RGB_sha256'][ref['id']]
        assert row['evaluation_source_kind_field_present'] == ('evaluation_source_kind' in ref)
        assert row['evaluation_source_kind'] == ref.get('evaluation_source_kind')
    syntax_files = []
    for path in PACKET.rglob('*.py'):
        tree = ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 10))
        syntax_files.append(path.relative_to(PACKET).as_posix())
        if path.name == 'cctv_dgp_spatial_readout_v1_model.py':
            assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) and any(a.name.startswith('torch') for a in n.names) for n in tree.body)
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_model')
            assert ast.unparse(function.body[0]) == 'require_manual(root)'
        if path.name == 'cctv_dgp_spatial_readout_v1_vm.py':
            train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
            load = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_dependencies')
            assert ast.unparse(train.body[0]) == 'machine_guard(root, idle=True)'
            assert ast.unparse(load.body[0]) == 'machine_guard(root)'
            source = path.read_text(encoding='utf-8')
            assert 'original.predict_float' not in source and 'original.net(values' not in source
            assert 'torch.optim.Adam(shadow' in source and 'model.zero_grad(set_to_none=True)' in source
            assert "for update, index in enumerate(p['schedule'], 1):" in source
    mutations = []
    changes = [dict(updates_per_route=51), dict(conditioning_trainable=True), dict(observation_trainable=True),
               dict(prior_trainable=True), dict(B_readout_kernel=1), dict(app_promotion=True),
               dict(final_identity_pixels=1), dict(native_pixels_used=1), dict(full_corpus_epochs=5),
               dict(distance_fraction_maximum=.002), dict(failed_recipe_resume_allowed=True)]
    for change in changes:
        modified = copy.deepcopy(p)
        modified.update(change)
        try:
            validate_protocol(modified)
        except AssertionError:
            mutations.append(dict(change=change, rejected=True))
        else:
            raise AssertionError('Altered protocol was not rejected: '+str(change))
    tests = subprocess.run([sys.executable, '-B', str(ROOT/'scripts/test_cctv_dgp_spatial_readout_v1.py')],
        cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert tests.returncode == 0 and 'Ran 14 tests' in tests.stderr
    test_path = REVIEW/('final_boundary_tests_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.txt')
    with test_path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(tests.stdout+tests.stderr)
    worker = PACKET/'scripts/cctv_dgp_spatial_readout_v1_vm.py'
    command = [sys.executable, '-B', str(worker), '--root', str(PACKET), '--parent-root', str(PARENT), '--protocol-sha', pin]
    checked = subprocess.run(command+['--verify-transfer'], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert checked.returncode == 0, checked.stderr
    check = json.loads(checked.stdout)
    assert check['complete'] and check['neural_calls'] == check['optimizer_updates'] == check['gradient_queries'] == 0
    rejected = subprocess.run(command+['--train'], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert rejected.returncode != 0 and 'Human manual Linux L4 tmux execution only' in rejected.stderr
    assert not (PACKET/'outputs').exists()
    bash_path = REVIEW/'bash_syntax_host.json'
    bash = read(bash_path)
    assert bash['complete'] and bash['exit_code'] == 0 and bash['read_only_syntax_check']
    assert bash['script_sha256'] == sha(PACKET/'scripts/run_spatial_readout.sh')
    assert bash['command'] == [r'C:\Program Files\Git\bin\bash.exe', '-n', str((PACKET/'scripts/run_spatial_readout.sh').resolve())]
    record = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), protocol_sha256=pin,
        archive_sha256=prepared['archive_sha256'], packet_members=len(members), parent_assets=check['parent_assets'],
        targets_checked=20, canonical_cases_bound=100, pointwise_readout_rank_checks=5,
        B_elements=2176419, Python310_AST_files=syntax_files, actual_hash_only_CLI=check,
        Windows_training_denial=rejected.stderr, protocol_mutations=mutations, meaningful_CPU_boundary_tests=14,
        LF_sidecar_verified=True, read_only_Bash_syntax_passed=True,
        Bash_syntax_bound_host_receipt=str(bash_path), boundary_test_log=str(test_path),
        source_hashes={path.relative_to(ROOT).as_posix(): sha(path) for path in
            [ROOT/'scripts/audit_cctv_dgp_spatial_readout_v1_preparation.py', ROOT/'scripts/test_cctv_dgp_spatial_readout_v1.py',
             REVIEW/'preparation.json', REVIEW/'preparation_attempt1.json', REVIEW/'preparation_audit_attempt1.json',
             REVIEW/'preparation_audit_attempt2.json', bash_path, test_path]},
        neural_construction_or_forward_calls=0, gradients_or_optimizer_updates=0, VM_calls=0,
        fresh_CUDA_parity_and_nonzero_gradients_PENDING=True, finite_training_and_image_usefulness_PENDING=True,
        model_qualified=False, packet_status='VERIFIED_PREPARED_UNRUN', seconds=time.monotonic()-start)
    write(REVIEW/'independent_preparation_audit.json', record)
    print(json.dumps({k: record[k] for k in ['complete', 'packet_members', 'parent_assets', 'meaningful_CPU_boundary_tests', 'packet_status', 'model_qualified']}))


if __name__ == '__main__':
    main()
