"""Frozen prospective checker. Never imports returned code or runs local gradients."""
import argparse
from pathlib import Path
import sys
import tarfile
import time
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'outputs/cctv_dgp_original_feature_probe_v1_vm'
OUT = ROOT/'outputs/cctv_dgp_original_feature_probe_v1_vm_return'
sys.path.insert(0, str(BUNDLE))
from cctv_dgp_original_feature_probe_v1_contract import (
    NAME, STEM, TERMS, SCOPES, FRACTIONS, BUDGETS, STATE, read, write, sha, verified_assets)
from frozen_capacity_contract import exported_pixel_metrics, detail_metric, feature_support, capacity
from frozen_raw_metrics import pixel_metrics, detail_float, deliver, mean_only, review_groups


def pixels(path, mode='RGB'):
    with Image.open(path) as im: return np.asarray(im.convert(mode)).copy()


def exact_structure(saved, fresh, atol=1e-10):
    if isinstance(fresh, dict):
        assert saved.keys() == fresh.keys()
        for k, v in fresh.items(): exact_structure(saved[k], v, atol)
    elif isinstance(fresh, list):
        assert len(saved) == len(fresh)
        for a, b in zip(saved, fresh): exact_structure(a, b, atol)
    elif isinstance(fresh, (float, np.floating)): assert abs(saved-fresh) <= atol
    else: assert saved == fresh


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--expected-sha', required=True)
    a = parser.parse_args(); started = time.monotonic(); pin = sha(BUNDLE/'protocol.json')
    p = verified_assets(BUNDLE, pin)
    for n, d in p['local_sources_sha256'].items(): assert sha(ROOT/n) == d, n
    archive_path = ROOT/'outputs'/(STEM+'-results.tar.gz')
    assert sha(archive_path) == a.expected_sha
    assert Path(str(archive_path)+'.sha256').read_text().split() == [a.expected_sha, archive_path.name]
    export = read(ROOT/'outputs'/(STEM+'-export.json'))
    assert export['complete'] and export['archive_sha256'] == a.expected_sha and export['bytes'] == archive_path.stat().st_size
    assert not OUT.exists(), 'Preserve imported evidence and audits; no overwrite'
    with tarfile.open(archive_path, 'r:gz') as tar:
        members = tar.getmembers(); seen = set(); total = 0
        for m in members:
            assert m.isfile() and not m.issym() and not m.islnk()
            parts = m.name.split('/'); assert parts[0] == NAME+'_return'
            assert all(part not in ['', '.', '..'] for part in parts) and '\\' not in m.name and ':' not in m.name
            n = '/'.join(parts[1:]); assert n not in seen and n not in p['assets_sha256']; seen.add(n)
            assert n.startswith('outputs/') or n in ['protocol.json', 'export_manifest.json', 'supervisor_receipt.json', 'diagnostic.log', 'diagnostic_exit_code.txt']
            total += m.size; assert total <= BUDGETS['return_uncompressed_bytes']+32*1024**2
        assert len(seen) <= 5000
        assert {'protocol.json', 'export_manifest.json', 'supervisor_receipt.json', 'diagnostic.log', 'diagnostic_exit_code.txt'} <= seen
        OUT.mkdir()
        for m in members:
            path = OUT/'/'.join(m.name.split('/')[1:]); path.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(m) as source, path.open('xb') as target:
                import shutil
                shutil.copyfileobj(source, target)
    assert sha(OUT/'protocol.json') == pin
    manifest = read(OUT/'export_manifest.json')
    assert manifest['complete'] and manifest['protocol_sha256'] == pin
    assert set(manifest['files_sha256']) == seen-{'export_manifest.json'}
    for n, d in manifest['files_sha256'].items(): assert sha(OUT/n) == d
    success = (OUT/'outputs/results.json').is_file()
    assert success != (OUT/'outputs/failure.json').is_file()
    result = read(OUT/('outputs/results.json' if success else 'outputs/failure.json'))
    assert result['protocol_sha256'] == pin
    assert result['optimizer_updates'] == result['epochs'] == result['committed_trajectory_updates'] == result['backwards'] == 0
    assert result['optimizer_constructed'] is False and result['new_trained_checkpoint'] is False
    outer = read(OUT/'supervisor_receipt.json')
    assert outer['protocol_sha256'] == pin and outer['cap_seconds'] == BUDGETS['external_seconds'] and outer['kill_grace_seconds'] == 30
    assert int((OUT/'diagnostic_exit_code.txt').read_text()) == outer['worker_exit_code']
    if not success:
        write(ROOT/'outputs/cctv_dgp_original_feature_probe_v1_failure_import.json', {
            'complete': True, 'archive_sha256': a.expected_sha, 'members_verified': len(members),
            'protocol_sha256': pin, 'failure': result, 'partial_neural_evidence_not_audited': True,
            'model_qualification': False, 'goal_complete': False})
        print({'complete': True, 'failure_imported': True, 'partial_neural_evidence_not_audited': True}); return
    assert outer['worker_exit_code'] == 0 and result['complete'] and result['gradient_queries'] == 30
    assert result['seconds'] <= BUDGETS['worker_seconds'] and result['trial_seconds'] <= BUDGETS['trial_seconds']
    assert result['peak_allocated_VRAM_bytes'] <= BUDGETS['peak_vram_bytes']
    assert result['states_before'] == result['states_after'] and result['states_before']['original'] == STATE
    assert result['states_before']['candidate'] == STATE and result['states_before']['recognizer'] == p['recognizer_state']
    assert result['forward_counts'] == {'original': 20, 'candidate': 210, 'recognizer': 230}
    folder = OUT/'outputs'; gradients = np.load(folder/'aggregate_gradients.npz', allow_pickle=False)
    assert set(gradients.files) == set(TERMS)
    gradient_receipt = read(folder/'gradient_receipt.json'); assert gradient_receipt['complete']
    assert len(gradient_receipt['rows']) == 30 and gradient_receipt['seconds'] <= BUDGETS['gradient_seconds']
    aggregate = {k: np.zeros(1996035, np.float64) for k in TERMS}
    for b in range(10):
        with np.load(folder/f'gradients_batch{b}.npz', allow_pickle=False) as arrays:
            for k in TERMS:
                v = arrays[k]; assert v.dtype == np.float32 and v.shape == (1996035,) and np.isfinite(v).all()
                aggregate[k] += v.astype(np.float64)/10
                row = next(r for r in gradient_receipt['rows'] if r['batch'] == b and r['term'] == k)
                assert [r['name'] for r in row['parameters']] == [r['name'] for r in p['parameter_layout']]
                for desc, report in zip(p['parameter_layout'], row['parameters']):
                    part = v[desc['start']:desc['end']].astype(np.float64)
                    assert abs(float(np.linalg.norm(part))-report['L2']) <= 1e-12
                    assert np.count_nonzero(part) == report['nonzero_values']
                    if not report['graph_connected']: assert not np.count_nonzero(part)
    for k in TERMS: assert np.array_equal(aggregate[k], gradients[k])
    initial = np.load(folder/'initial_parameters.npy', allow_pickle=False)
    assert initial.dtype == np.float32 and initial.shape == (1996035,) and np.isfinite(initial).all()
    labels = ['baseline']+[s+'_'+format(f, '.0e').replace('-', 'm') for s in SCOPES for f in FRACTIONS]
    assert result['trials_completed'] == labels[1:] and len(result['trial_summaries']) == 9
    rebuilt = {}; cases_by_id = {c['id']: c for c in p['cases']}
    for label in labels:
        saved = read(folder/label/'metrics.json'); assert saved['complete'] and saved['variant'] == label
        assert [r['id'] for r in saved['rows']] == [c['id'] for c in p['cases']]
        rows = []
        for c, report in zip(p['cases'], saved['rows']):
            assert time.monotonic()-started <= 1500, 'Local audit deadline'
            cid = c['id']; camera = pixels(BUNDLE/c['input']); target = pixels(BUNDLE/c['target']); mask = pixels(BUNDLE/c['observed'], 'L') > 0
            feature = feature_support(mask, c['landmarks5_canvas_xy'])
            raw = np.load(folder/label/(cid+'.npy'), allow_pickle=False)
            base = np.load(folder/'baseline'/(cid+'.npy'), allow_pickle=False)
            png = pixels(folder/label/(cid+'.png')); mean_png = pixels(folder/label/(cid+'_mean_only.png'))
            assert raw.dtype == np.float32 and raw.shape == (256, 256, 3) and np.isfinite(raw).all()
            assert np.array_equal(raw[~mask], camera[~mask].astype(np.float32)/np.float32(255))
            assert np.array_equal(deliver(raw, camera, mask), png)
            mraw, mpng, shift = mean_only(raw, base, camera, mask); assert np.array_equal(mpng, mean_png)
            vector = np.load(folder/label/(cid+'_embeddings.npy'), allow_pickle=False)
            assert vector.dtype == np.float32 and vector.shape == (3, 512) and np.isfinite(vector).all()
            assert np.allclose(np.sum(vector.astype(np.float64)**2, axis=1), 1, rtol=0, atol=1e-5)
            if label != 'baseline': assert np.array_equal(vector[2], np.load(folder/'baseline'/(cid+'_embeddings.npy'), allow_pickle=False)[2])
            rm = pixel_metrics(raw, target, mask); pm = exported_pixel_metrics(png, target, mask)
            rm.update({'ArcFace_observed_fixed': float(vector[0]@vector[2]), 'landmark_high_frequency_MSE': detail_float(raw, target, feature),
                'constant_mean_shift_only_MSE': pixel_metrics(mraw, target, mask)['MSE']})
            pm.update({'ArcFace_observed_fixed': float(vector[1]@vector[2]), 'landmark_high_frequency_MSE': detail_metric(png, target, feature),
                'constant_mean_shift_only_MSE': exported_pixel_metrics(mpng, target, mask)['MSE']})
            exact_structure(report, {'id': cid, 'source': c['source'], 'profile': c['profile'], 'raw': rm, 'png': pm, 'postclip_mean_RGB_shift': shift.tolist()})
            rows.append({**report, 'raw': rm, 'png': pm})
        groups = {co['name']: {s: review_groups([r for r in rows if r['id'] in co['case_ids']], s) for s in ['raw', 'png']} for co in p['cohorts']}
        # Stored rows must reproduce groups exactly; fresh pixel recomputation uses arithmetic-only tolerance.
        stored_groups = {co['name']: {s: review_groups([r for r in saved['rows'] if r['id'] in co['case_ids']], s) for s in ['raw', 'png']} for co in p['cohorts']}
        assert saved['groups'] == stored_groups; exact_structure(saved['groups'], groups)
        rebuilt[label] = groups
        if label != 'baseline':
            summary = next(r for r in result['trial_summaries'] if r['variant'] == label)
            allowed = np.zeros(1996035, bool)
            for desc in p['parameter_layout']:
                if summary['scope'] == 'joint' or desc['partition'] == summary['scope']: allowed[desc['start']:desc['end']] = True
            direction = np.where(allowed, aggregate[TERMS[0]], 0.)
            norm = float(np.linalg.norm(direction)); weight_norm = float(np.linalg.norm(initial.astype(np.float64)[allowed]))
            expected = (initial.astype(np.float64)-direction*(summary['relative_fraction']*weight_norm/norm)).astype(np.float32)
            actual = np.load(folder/(label+'_parameters.npy'), allow_pickle=False); assert np.array_equal(expected, actual)
            delta = actual.astype(np.float64)-initial.astype(np.float64)
            assert abs(float(np.linalg.norm(delta))-summary['actual_displacement_L2']) <= 1e-12
            for k in TERMS: assert abs(float(aggregate[k]@delta)-summary['gradient_dot_actual_displacement'][k]) <= 1e-12
            comparisons = {co['name']: {s: capacity(rebuilt['baseline'][co['name']][s], groups[co['name']][s], .01) for s in ['raw', 'png']} for co in p['cohorts']}
            exact_structure(summary['comparisons'], comparisons)
    # Fresh saved-vector CPU inference only: no candidate.assign_trial or gradient API.
    import torch
    from cctv_dgp_pilot import FixedObservedIdentity, grid112, state_hash
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_original_feature_probe_v1_candidate import OriginalFeatureProbe
    torch.set_num_threads(4)
    original, _ = load_frozen_dgp_restorer(BUNDLE/'weights/dgp_v2.pth', expected_sha256=p['original_checkpoint_sha256'], device='cpu')
    model = OriginalFeatureProbe(original.net, p['parameter_layout']); identity = FixedObservedIdentity(BUNDLE/'weights/w600k_r50.onnx', 'cpu')
    initial_cpu = model.vector(); assert np.array_equal(initial_cpu, initial)
    frozen_before = (state_hash(original.net), state_hash(identity))
    refs = {r['id']: r for r in p['references']}
    # First reference from each source in each TRAIN cohort:20 cases per variant,200 CPU replays.
    replay_ids = []
    for co in p['cohorts']:
        for source in sorted({c['source'] for c in p['cases']}):
            choices = [cid for cid in co['case_ids'] if cases_by_id[cid]['source'] == source]
            replay_ids.extend(choices[:5])
    worst_raw = 0.; worst_png = 0; worst_vector = 0.
    with torch.no_grad():
        for label in labels:
            values = initial if label == 'baseline' else np.load(folder/(label+'_parameters.npy'), allow_pickle=False)
            for desc, (_, value) in zip(p['parameter_layout'], model.selected):
                value.copy_(torch.from_numpy(values[desc['start']:desc['end']].reshape(desc['shape'])))
            assert state_hash(model.net) == read(folder/label/'metrics.json')['candidate_state']
            for cid in replay_ids:
                assert time.monotonic()-started <= 1500
                c = cases_by_id[cid]; camera = pixels(BUNDLE/c['input']); mask = pixels(BUNDLE/c['observed'], 'L') > 0; target = pixels(BUNDLE/c['target'])
                rgb = lambda v: torch.from_numpy(v.astype(np.float32)/np.float32(255)).permute(2, 0, 1)[None]
                x = rgb(camera); support = torch.from_numpy(mask.astype(np.float32))[None, None]
                base = original(x).detach().clone(); result_cpu = model(x, support, base)[0].permute(1, 2, 0).numpy().copy()
                raw = np.load(folder/label/(cid+'.npy'), allow_pickle=False); png = pixels(folder/label/(cid+'.png'))
                worst_raw = max(worst_raw, float(np.abs(result_cpu-raw).max()))
                worst_png = max(worst_png, int(np.abs(deliver(result_cpu, camera, mask).astype(np.int16)-png.astype(np.int16)).max()))
                grid = torch.from_numpy(grid112(refs[c['source_person_or_reference']]['matrix112']))[None]
                raw_tensor = torch.from_numpy(raw).permute(2, 0, 1)[None]
                vectors = identity.embedding(torch.cat([raw_tensor, rgb(png), rgb(target)]), support.repeat(3, 1, 1, 1), grid.repeat(3, 1, 1, 1)).numpy().copy()
                stored = np.load(folder/label/(cid+'_embeddings.npy'), allow_pickle=False)
                worst_vector = max(worst_vector, float(np.abs(vectors-stored).max()))
            if label != 'baseline':
                for desc, (_, value) in zip(p['parameter_layout'], model.selected): value.copy_(torch.from_numpy(initial[desc['start']:desc['end']].reshape(desc['shape'])))
        assert state_hash(model.net) == state_hash(original.net) == STATE
    tol = p['CPU_replay_tolerances']
    assert worst_raw <= tol['raw_max_abs'] and worst_png <= tol['PNG_byte_max'] and worst_vector <= tol['embedding_max_abs']
    assert frozen_before == (state_hash(original.net), state_hash(identity))
    verified_assets(BUNDLE, pin)
    write(ROOT/'outputs/cctv_dgp_original_feature_probe_v1_independent_audit.json', {
        'complete': True, 'archive_sha256': a.expected_sha, 'protocol_sha256': pin,
        'checker_sha256': sha(Path(__file__)), 'members_verified': len(members), 'all1000_raw_and_PNG_metrics_recomputed': True,
        'all30_gradient_vectors_and_nine_displacements_arithmetic_verified': True,
        'gradient_autograd_replay_local': False, 'CPU_replay_cases': len(labels)*len(replay_ids),
        'maximum_raw_error': worst_raw, 'maximum_PNG_byte_error': worst_png, 'maximum_embedding_error': worst_vector,
        'seconds': time.monotonic()-started, 'local_gradient_queries': 0, 'local_optimizer_updates': 0,
        'visual_review_pending': True, 'full_TRAIN_capacity_not_tested': True, 'model_qualification': False, 'goal_complete': False})
    print({'complete': True, 'diagnostic_evidence_audited': True, 'model_qualification': False})


if __name__ == '__main__': main()
