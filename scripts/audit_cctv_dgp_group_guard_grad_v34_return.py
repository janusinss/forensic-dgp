"""Prospective independent V34 audit: saved arrays and frozen CPU replays only."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
NAME = 'cctv_dgp_group_guard_grad_v34_vm'
STEM = 'cctv-dgp-group-guard-grad-v34'
BUNDLE = ROOT / 'outputs' / NAME
OUT = ROOT / 'outputs/cctv_dgp_group_guard_grad_v34_return'
PREFIX = 'cctv_dgp_group_guard_grad_v34_return/'
V33 = ROOT / 'outputs/cctv_dgp_loss_cone_probe_v33_return'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream: stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


def group_indices(cases):
    names = sorted({'all', 'clear', 'degraded'} | {c['source'] + '/' + k for c in cases for k in ['all', 'clear', 'degraded', c['profile']]})
    result = {name: [i for i,c in enumerate(cases) if name in {'all', 'clear' if c['profile'] == 'clear' else 'degraded',
        c['source'] + '/all', c['source'] + ('/clear' if c['profile'] == 'clear' else '/degraded'), c['source'] + '/' + c['profile']}]
        for name in names}
    assert len(result) == 17 and all(result.values())
    return result


def verify_basis(p):
    for name, digest in p['assets_sha256'].items(): assert sha(BUNDLE / name) == digest, name
    for name, digest in p['local_basis_sha256'].items(): assert sha(ROOT / name) == digest, name
    old = ROOT / 'outputs/cctv_dgp_v32_loss_gradient_v1_vm'
    for name,digest in p['basis_VM_sha256'].items(): assert sha(old / name) == digest, name
    for name,digest in p['V33_original_readback_sha256'].items(): assert sha(V33 / name) == digest, name
    auditor = module('pinned_V33_return_checker_for_baseline_only', ROOT / 'scripts/audit_cctv_dgp_loss_cone_probe_v33_return_r2.py')
    prior_p = read(ROOT / 'outputs/cctv_dgp_loss_cone_probe_v33_vm/protocol.json')
    basis = auditor.verify_basis(prior_p)
    assert basis['cohorts'] == p['cohorts'] and basis['parameter_layout'] == p['parameter_layout']
    prior_audit = read(ROOT / 'outputs/cctv_dgp_loss_cone_probe_v33_independent_audit_r2.json')
    assert prior_audit['complete'] and prior_audit['finite_probe_complete'] and not prior_audit['training_capacity_pass']
    return auditor, prior_p, basis


def safe_members(members, p):
    allowed = set(p['assets_sha256']) | {'protocol.json', 'export_manifest.json', 'diagnostic.log', 'diagnostic_exit_code.txt',
        'supervisor_receipt.json', 'outputs/results.json', 'outputs/failure.json'}
    for cohort in p['cohorts']:
        start = 'outputs/' + cohort['name'] + '/'; allowed.add(start + 'receipt.json')
        for index in range(10): allowed.add(start + f'batch{index}_guard_gradients.npy')
        for case in cohort['cases']: allowed.update(start + case['id'] + suffix for suffix in ['.npy', '.png'])
    seen, result, size = set(), [], 0
    for member in members:
        assert member.isfile() and not member.issym() and not member.islnk()
        assert member.name.startswith(PREFIX) and not any(c in member.name for c in ['\\', ':', '\x00'])
        name = member.name[len(PREFIX):]; parts = PurePosixPath(name).parts
        assert parts and not PurePosixPath(name).is_absolute() and all(v not in ['.', '..'] for v in parts)
        assert name in allowed and name.lower() not in seen
        assert 0 <= member.size <= 64 * 1024 ** 2
        size += member.size; seen.add(name.lower()); result.append((member, name))
        assert size <= p['budgets']['export_uncompressed_bytes'] and len(seen) <= 500
    return result, size


def import_return(p, pin, digest, size):
    archive = ROOT / 'outputs' / (STEM + '-results.tar.gz')
    assert archive.stat().st_size == size and sha(archive) == digest
    assert Path(str(archive) + '.sha256').read_text().strip().split() == [digest, archive.name]
    exported = read(ROOT / 'outputs' / (STEM + '-export.json'))
    assert exported['complete'] and exported['archive_sha256'] == digest and exported['bytes'] == size
    assert exported['training_success_not_implied'] and exported['optimizer_updates'] == 0
    assert not OUT.exists(), 'Preserve any prior or partial return audit'
    with tarfile.open(archive, 'r:gz') as tar:
        items, total = safe_members(tar.getmembers(), p); OUT.mkdir()
        for item,name in items:
            destination = (OUT / name).resolve(); assert destination.is_relative_to(OUT); destination.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(item) as source, destination.open('xb') as stream:
                for block in iter(lambda: source.read(1024 ** 2), b''): stream.write(block)
    hashes = {name:sha(OUT / name) for _,name in items}; assert hashes['protocol.json'] == pin
    assert all(hashes[name] == digest for name,digest in p['assets_sha256'].items())
    manifest = read(OUT / 'export_manifest.json')
    assert manifest['complete'] and manifest['protocol_sha256'] == pin
    assert manifest['files_sha256'] == {n:h for n,h in hashes.items() if n != 'export_manifest.json'}
    receipt = {'complete': True, 'archive_sha256': digest, 'archive_bytes': size, 'files_sha256': hashes,
        'members': len(items), 'uncompressed_bytes': total, 'returned_code_executed': False}
    write(ROOT / 'outputs/cctv_dgp_group_guard_grad_v34_return_import.json', receipt)
    return exported, receipt


def audit(digest, size):
    started = time.monotonic(); p = read(BUNDLE / 'protocol.json'); pin = sha(BUNDLE / 'protocol.json')
    old_auditor, old_p, basis = verify_basis(p); exported, imported = import_return(p, pin, digest, size)
    success, failure = OUT / 'outputs/results.json', OUT / 'outputs/failure.json'
    assert success.exists() != failure.exists(); result = read(success if success.exists() else failure)
    assert result['protocol_sha256'] == pin
    assert result['optimizer_updates'] == result['parameter_updates'] == result['epochs'] == result['backwards'] == 0
    assert not result['new_checkpoint_created'] and not result['app_promotion'] and not result['goal_complete']
    assert exported['run_results_present'] == success.exists() and exported['failure_present'] == failure.exists()
    supervisor = read(OUT / 'supervisor_receipt.json'); code = int((OUT / 'diagnostic_exit_code.txt').read_text())
    assert supervisor['protocol_sha256'] == pin and supervisor['diagnostic_exit_code'] == code and (code == 0) == success.exists()
    assert supervisor['cap_seconds'] == 630 and supervisor['kill_grace_seconds'] == 30
    assert supervisor['within_external_bound'] == (supervisor['seconds'] <= 660)
    summaries, replay = [], None
    if success.exists():
        import numpy as np
        from PIL import Image
        from skimage.metrics import structural_similarity
        assert result['gradient_queries'] == 300 and result['seconds'] <= 600
        assert result['peak_allocated_VRAM_bytes'] <= 20 * 1024 ** 3
        assert {k:result[k] for k in p['forward_call_limits']} == p['forward_call_limits']
        assert result['original_DGP_state'] == basis['original_DGP_state'] and result['recognizer_state'] == basis['recognizer_state']
        mixed = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'
        for cohort in p['cohorts']:
            label = cohort['name']; folder = OUT / 'outputs' / label; receipt = read(folder / 'receipt.json')
            expected_receipt = next(r['receipt_sha256'] for r in result['receipts'] if r['cohort'] == label)
            assert sha(folder / 'receipt.json') == expected_receipt and receipt['complete'] and receipt['state'] == 0
            assert receipt['candidate_state'] == basis['original_DGP_state'] and receipt['guard_metrics'] == p['guard_metrics']
            assert receipt['original_batch_context_preserved'] and receipt['nonhinged_derivatives_not_new_training_losses']
            cases = cohort['cases']; assert [r['id'] for r in receipt['rows']] == [c['id'] for c in cases]
            index_sets = group_indices(cases); vectors = {key:np.zeros((3,978243), np.float64) for key in index_sets}
            values = np.zeros((50,3), np.float64); queries = 0; worst_raw = worst_value = 0.
            for batch_index,batch in enumerate(receipt['batches']):
                assert time.monotonic() - started < 1200
                path = folder / f'batch{batch_index}_guard_gradients.npy'
                g = np.load(path,allow_pickle=False); assert g.dtype == np.float32 and g.shape == (15,978243) and np.isfinite(g).all()
                assert sha(path) == batch['gradient_sha256'] and batch['batch'] == batch_index
                assert np.allclose(np.linalg.norm(g.astype(np.float64),axis=1),batch['gradient_norms'],rtol=2e-10,atol=1e-11)
                assert batch['rows'] == receipt['rows'][batch_index*5:batch_index*5+5]
                for slot,row in enumerate(batch['rows']):
                    case_index = batch_index*5+slot; case = cases[case_index]; cid = case['id']
                    raw = np.load(folder / (cid + '.npy'),allow_pickle=False)
                    cached = np.load(V33 / f'outputs/state0_{label}/before/{cid}.npy',allow_pickle=False)
                    assert raw.dtype == np.float32 and raw.shape == (256,256,3) and np.isfinite(raw).all() and raw.min() >= 0 and raw.max() <= 1
                    error = float(np.abs(raw-cached).max()); assert error <= p['same_VM_raw_tolerance']; worst_raw = max(worst_raw,error)
                    with Image.open(mixed / case['target']) as im: target = np.asarray(im.convert('RGB')).astype(np.float32) / np.float32(255)
                    with Image.open(mixed / case['input']) as im: camera = np.asarray(im.convert('RGB')).copy()
                    with Image.open(mixed / case['observed']) as im: mask = np.asarray(im).copy() > 0
                    with Image.open(folder / (cid+'.png')) as im: png = np.asarray(im.convert('RGB')).copy()
                    assert np.array_equal(png,np.where(mask[...,None],np.floor(raw*np.float32(255)),camera).astype(np.uint8))
                    assert np.array_equal(png[~mask],camera[~mask])
                    _,ssmap = structural_similarity(target,raw,data_range=1,channel_axis=-1,win_size=7,full=True)
                    interior = old_auditor.module('pinned_erode_for_raw_metrics',ROOT/'scripts/cctv_dgp_loss_cone_probe_v33_metrics.py').erode(mask,3)
                    truth = np.load(V33 / f'outputs/state0_{label}/before/{cid}_target_embedding.npy',allow_pickle=False)
                    vector = np.load(V33 / f'outputs/state0_{label}/before/{cid}_raw_embedding.npy',allow_pickle=False)
                    fresh = [float(np.square((raw-target)[mask]).astype(np.float64).mean()),
                        1-float(ssmap[interior].astype(np.float64).mean()),1-float(vector@truth)]
                    values[case_index] = row['raw_guard_values']; e = float(np.abs(values[case_index]-fresh).max())
                    assert e <= p['CPU_guard_value_tolerance']; worst_value = max(worst_value,e)
                    for key,indices in index_sets.items():
                        if case_index in indices: vectors[key] += g[slot*3:slot*3+3].astype(np.float64) / len(indices)
                    queries += 3
            assert queries == 150 and len(receipt['batches']) == 10
            matrix = np.concatenate([vectors[key] for key in index_sets]); assert matrix.shape == (51,978243) and np.isfinite(matrix).all()
            theta = np.load(V33 / f'outputs/state0_{label}/theta_before.npy',allow_pickle=False).astype(np.float64)
            after = np.load(V33 / f'outputs/state0_{label}/restoration/theta_after.npy',allow_pickle=False).astype(np.float64)
            predicted = -(matrix@(theta-after))
            labels = [{'group':key,'metric':metric} for key in index_sets for metric in p['guard_metrics']]
            summary = {'cohort':label,'gradient_queries':150,'guard_derivatives':51,'parameter_values_per_gradient':978243,
                'maximum_raw_parity_error':worst_raw,'maximum_raw_guard_value_error':worst_value,
                'labels':labels,'group_guard_values':{key:values[ids].mean(0).tolist() for key,ids in index_sets.items()},
                'group_gradient_norms':np.linalg.norm(matrix,axis=1).tolist(), 'group_gradient_gram':(matrix@matrix.T).tolist(),
                'predicted_changes_under_original_V33_restoration_proposal':predicted.tolist(),
                'gradient_replay_performed_locally':False,'linear_predictions_not_finite_output_guarantees':True}
            summaries.append(summary); del matrix,vectors,g
        # Replay the previously pinned, unchanged model path; new raw outputs are
        # separately checked against all100 original V33 outputs above.
        replay = old_auditor.CPU_replay(old_p,basis,started)
    else:
        assert 0 <= result['gradient_queries'] <= 300 and not result['resume_permitted']
    assert time.monotonic()-started < 1200
    checked = {'complete':True,'protocol_sha256':pin,'checker_sha256':sha(Path(__file__)),
        'archive_sha256':digest,'archive_bytes':size,'members_verified':imported['members'],
        'diagnostic_complete':success.exists(),'failure_retained':failure.exists(),'gradient_summaries':summaries,
        'pinned_baseline_CPU_replay':replay,'local_gradient_calls':0,'local_optimizer_updates':0,
        'training_capacity_pass':False,'retained_checkpoints_modified':False,'app_promotion':False,
        'goal_complete':False,'seconds':time.monotonic()-started}
    write(ROOT/'outputs/cctv_dgp_group_guard_grad_v34_independent_audit.json',checked)
    print(json.dumps({'complete':True,'diagnostic_complete':success.exists(),'members_verified':imported['members'],'seconds':checked['seconds']}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--expected-sha',required=True);parser.add_argument('--expected-bytes',type=int,required=True)
    a=parser.parse_args();audit(a.expected_sha,a.expected_bytes)
