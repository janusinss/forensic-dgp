"""Prepare and bind a new manual-only diagnostic; no VM connection or autograd."""
import ast
import copy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tarfile
import time
from cctv_dgp_group_conflicts_v1_contract import (
    NAME, STEM, TERMS, FRACTIONS, BUDGETS, PARITY, definitions, sha, read, write, validate)

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'outputs/cctv_dgp_original_feature_probe_v1_vm'
BALANCE = ROOT/'outputs/cctv_dgp_original_loss_balance_v1_vm'
RETURN = ROOT/'outputs/cctv_dgp_original_loss_balance_v1_vm_return'
OUT = ROOT/'outputs'/NAME
PREP = ROOT/'outputs/cctv_dgp_group_conflicts_v1_preparation'


def text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream: stream.write(value)


def once(source, old, new):
    assert source.count(old) == 1, old[:150]
    return source.replace(old, new)


def worker_source():
    source = (OLD/'scripts/cctv_dgp_original_feature_probe_v1_vm.py').read_text()
    source = source.replace('original_feature_probe_v1', 'group_conflicts_v1')
    source = once(source, '        torch.cuda.reset_peak_memory_stats()', '''        torch.cuda.reset_peak_memory_stats()
        from cctv_dgp_group_conflicts_v1_losses import pixel_and_structure_losses
        from cctv_dgp_group_conflicts_v1_math import common_direction''')
    source = once(source, 'initial = candidate.vector(); np.save', "initial = candidate.vector(); assert np.array_equal(initial, np.load(root/'evidence/initial_parameters.npy', allow_pickle=False)); np.save")
    source = once(source, "        remaining_estimate = 9*baseline['seconds']*1.25", "        remaining_estimate = 3*baseline['seconds']*1.25")
    source = once(source, "'remaining_variants': 9", "'remaining_variants': 3")
    source = once(source, 'projected_bytes = 10*baseline_bytes*1.25 + 400*1024**2',
                  'projected_bytes = 4*baseline_bytes*1.25 + 650*1024**2')
    begin = source.index('        candidate.enable_diagnostic_gradients(root)')
    end = source.index('        trial_seconds = time.monotonic()-trials_start', begin)
    replacement = '''        # Forward parity first; no autograd or candidate forward is used here.
        parity_start = time.monotonic(); parity_rows = []
        limits = p['loss_surrogate_parity']; maximum = {'MSE': 0., 'SSIM': 0., 'structure': 0.}
        with torch.no_grad():
            for begin in range(0, 100, 5):
                clock(); ids = list(range(begin, begin+5)); b = batch(ids)
                raw = torch.where(b['mask'].bool(), b['base'], b['x'])
                values = pixel_and_structure_losses(raw, b['target'], b['mask'], b['feature'])
                for slot, idx in enumerate(ids):
                    stored = baseline['rows'][idx]['raw']
                    errors = {'MSE': abs(float(values['MSE'][slot])-stored['MSE']),
                        'SSIM': abs(1.-float(values['SSIM_loss'][slot])-stored['SSIM']),
                        'structure': abs(float(values['landmark_structure'][slot])-stored['landmark_high_frequency_MSE'])}
                    parity_rows.append({'id': items[idx]['case']['id'], 'errors': errors})
                    for key in maximum: maximum[key] = max(maximum[key], errors[key])
        passed = (maximum['MSE'] <= limits['MSE_abs'] and maximum['SSIM'] <= limits['SSIM_abs'] and
                  maximum['structure'] <= limits['landmark_structure_abs'])
        write(out/'loss_surrogate_parity.json', {'complete': True, 'passed': passed,
            'cases': 100, 'maximum_errors': maximum, 'rows': parity_rows,
            'seconds': time.monotonic()-parity_start, 'gradient_queries': 0})
        assert passed, 'Loss-surrogate forward parity failed; stop before derivatives'
        candidate.enable_diagnostic_gradients(root)
        parameters = [v for _, v in candidate.selected]
        aggregate = np.zeros((32, 1996035), np.float64); gradient_rows = []
        gradient_start = time.monotonic()
        lookup = {(r['source'], r['profile'], r['metric']): k for k, r in enumerate(p['group_losses'])}
        import hashlib
        for begin in range(0, 50, 5):
            clock(); assert time.monotonic()-gradient_start <= BUDGETS['gradient_seconds']
            ids = [by_id[cid] for cid in p['cohorts'][0]['case_ids'][begin:begin+5]]
            b = batch(ids); result = candidate(b['x'], b['mask'], b['base'])
            assert torch.equal(result, torch.where(b['mask'].bool(), b['base'], b['x']))
            values = pixel_and_structure_losses(result, b['target'], b['mask'], b['feature'])
            vector = identity.embedding(result, b['mask'], b['grid'])
            values['ArcFace_loss'] = 1.-(vector*b['truth']).sum(1)
            source_name = items[ids[0]]['case']['source']; queries = []
            for slot, idx in enumerate(ids):
                case = items[idx]['case']; assert case['source'] == source_name
                for metric in ['MSE', 'SSIM_loss', 'ArcFace_loss']:
                    queries.append((lookup[(source_name, case['profile'], metric)], values[metric][slot]))
            queries.append((lookup[(source_name, 'degraded', 'landmark_structure')], values['landmark_structure'][1:].mean()))
            assert len(queries) == 16
            for slot, (group_index, value) in enumerate(queries):
                clock(); assert time.monotonic()-gradient_start <= BUDGETS['gradient_seconds']
                gradients = torch.autograd.grad(value, parameters, retain_graph=slot < 15,
                                                allow_unused=True, materialize_grads=False)
                progress['gradient_queries'] += 1; parameter_rows = []; full = np.zeros(1996035, np.float32)
                for desc, grad in zip(p['parameter_layout'], gradients):
                    values_cpu = np.zeros(desc['elements'], np.float32) if grad is None else grad.detach().reshape(-1).cpu().numpy().copy()
                    assert values_cpu.dtype == np.float32 and np.isfinite(values_cpu).all()
                    full[desc['start']:desc['end']] = values_cpu
                    parameter_rows.append({'name': desc['name'], 'graph_connected': grad is not None,
                        'L2': float(np.linalg.norm(values_cpu.astype(np.float64))),
                        'nonzero_values': int(np.count_nonzero(values_cpu))})
                aggregate[group_index] += full.astype(np.float64)/5.
                gradient_rows.append({'batch': begin//5, 'group_index': group_index,
                    'group': p['group_losses'][group_index], 'value': float(value.detach()),
                    'case_ids': [items[i]['case']['id'] for i in ids],
                    'query_vector_sha256': hashlib.sha256(full.tobytes()).hexdigest(),
                    'query_L2': float(np.linalg.norm(full.astype(np.float64))), 'parameters': parameter_rows})
            del gradients, queries, values, result, vector
            frozen(True)
            print({'gradient_batch': begin//5+1, 'of': 10, 'gradient_queries': progress['gradient_queries']}, flush=True)
        candidate.net.requires_grad_(False); frozen(True)
        gradient_seconds = time.monotonic()-gradient_start
        assert gradient_seconds <= BUDGETS['gradient_seconds'] and progress['gradient_queries'] == 160
        assert np.isfinite(aggregate).all()
        np.savez_compressed(out/'group_gradients.npz', gradients=aggregate)
        write(out/'gradient_receipt.json', {'complete': True, 'rows': gradient_rows,
            'groups': p['group_losses'], 'seconds': gradient_seconds, 'gradient_queries': 160,
            'aggregate_vectors': 32, 'aggregation': 'five equal reference contributions per group',
            'individual_query_vectors_exported': False, 'local_autograd_replay_permitted': False,
            'optimizer_updates': 0, 'none_and_numeric_zero_separate': True})
        with np.load(root/'evidence/previous_directions.npz', allow_pickle=False) as saved:
            old_direction = saved['balanced_2'].copy()
        assert old_direction.dtype == np.float64 and old_direction.shape == (1996035,)
        norms = np.linalg.norm(aggregate, axis=1)
        old_unit = old_direction/np.linalg.norm(old_direction)
        previous_cosines = -(aggregate@old_unit)/np.maximum(norms, 1e-300)
        write(out/'previous_direction_group_predictions.json', {'complete': True,
            'direction': 'previous balanced_2', 'groups': p['group_losses'],
            'descent_cosines': previous_cosines.tolist(),
            'raw_directional_derivatives': (aggregate@old_direction).tolist(),
            'negative_cosine_group_indices': np.flatnonzero(previous_cosines < 0).tolist(),
            'sampled_first_order_only': True})
        solver_start = time.monotonic()
        direction, certificate = common_direction(aggregate, p['minimum_normalized_descent_cosine'])
        assert time.monotonic()-solver_start <= BUDGETS['solver_seconds']; clock()
        write(out/'direction_certificate.json', certificate)
        if direction is not None: np.save(out/'common_direction.npy', direction, allow_pickle=False)
        decoder = np.zeros(1996035, bool)
        for desc in p['parameter_layout']:
            if desc['partition'] == 'decoder_control': decoder[desc['start']:desc['end']] = True
        weight_norm = p['initial_decoder_weight_L2']
        assert abs(float(np.linalg.norm(initial.astype(np.float64)[decoder]))-weight_norm) <= 1e-10
        trials_start = time.monotonic(); trial_summaries = []
        if direction is not None:
            for fraction in FRACTIONS:
                clock(); assert time.monotonic()-trials_start <= BUDGETS['trial_seconds']
                label = 'common_'+format(fraction, '.0e').replace('-', 'm')
                trial = (initial.astype(np.float64)+fraction*weight_norm*direction).astype(np.float32)
                assert np.isfinite(trial).all(); actual = trial.astype(np.float64)-initial.astype(np.float64)
                candidate.assign_trial(trial); np.save(out/(label+'_parameters.npy'), trial, allow_pickle=False)
                receipt = evaluate(label)
                comparisons = {co['name']: {stage: capacity(baseline['groups'][co['name']][stage], receipt['groups'][co['name']][stage], .01)
                    for stage in ['raw', 'png']} for co in p['cohorts']}
                trial_summaries.append({'variant': label, 'scope': 'common', 'relative_fraction': fraction,
                    'actual_displacement_L2': float(np.linalg.norm(actual)), 'selected_weight_L2': weight_norm,
                    'scale_weight_partition': 'decoder_control',
                    'relative_displacement_actual': float(np.linalg.norm(actual))/weight_norm,
                    'gradient_dot_actual_displacement': (aggregate@actual).tolist(),
                    'all_actual_group_derivatives_negative': bool(np.all(aggregate@actual < 0)),
                    'candidate_state': receipt['candidate_state'], 'comparisons': comparisons,
                    'model_qualification': False})
                progress['trials_completed'].append(label)
                candidate.assign_trial(initial); frozen(True)
                print({'trial': label, 'of': 3, 'optimizer_updates': 0}, flush=True)
        else:
            print({'common_direction_found': False, 'finite_trials': 0, 'optimizer_updates': 0}, flush=True)
'''
    source = source[:begin]+replacement+source[end:]
    source = once(source, "assert counts == {'original': 20, 'candidate': 210, 'recognizer': 230}",
                  "assert counts == {'original': 20, 'candidate': 30+20*len(trial_summaries), 'recognizer': 50+20*len(trial_summaries)}")
    source = once(source, "'raw_and_PNG_cases': 1000", "'raw_and_PNG_cases': 100*(1+len(trial_summaries)), 'common_direction_found': direction is not None")
    return source


def guide(pin, digest, size):
    return f'''# Group conflicts before more epochs: manual VM commands

**Prepared and verified, not run.** This measures source/profile loss conflicts
before training. There are **160 gradient queries, zero optimizer updates and
zero epochs**. If all 32 sampled losses have a certified common descent direction,
three independent reset trials test it against the unchanged preservation gates.
Otherwise the diagnostic records the conflicts and exports without trials.
Neither outcome qualifies a model or starts follow-on training.

Estimated diagnostic **8–20 minutes**, export **1–5 minutes**; the broader loss
queries have not been timed on the L4 yet. Enforced worker1500s/external1530s,
cache120s, gradients480s, solver120s, trials300s, export300s/external330s,
30s kill grace,20GiB allocated VRAM and1.5GiB uncompressed return. Require
**4 GiB free after installation**, with512MiB protected reserve and a storage
projection before derivatives. No cleanup is bundled.

Archive: {size:,} bytes. Protocol SHA256: `{pin}`.
Execution SHA256: `{digest}`.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{STEM}-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{STEM}-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c {STEM}-execution.tar.gz.sha256 &&
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
test ! -e ~/forensic-dgp/{NAME} &&
tar -xzf {STEM}-execution.tar.gz -C ~/forensic-dgp &&
df -h ~/forensic-dgp
```

3. Open tmux:

```bash
tmux new-session -A -s dgp_group_conflicts_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/{NAME} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_group_conflicts_v1_vm.py --root . --protocol-sha {pin} --verify-transfer &&
bash scripts/run_group_conflicts.sh {pin}
```

Detach with Ctrl+B, release, D; reattach with
`tmux attach-session -t dgp_group_conflicts_v1`. Use a fresh directory. The worker
refuses competing GPU processes, a different VM, prior outputs, resume and
automatic promotion. Every timing/storage/integrity failure is retained.

5. Download from **Windows Google Cloud SDK Shell**, one remote source per command:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-export.json" "."
```

All three return files require independent audit and visual review. The 100
photographic TRAIN cases do not establish native CCTV, DEV or final performance.
Read CCTV_DGP_GROUP_CONFLICTS_V1_DESIGN.md for surrogate and derivative-audit
limits. The DGP-led app, all five milestones and seven covering families remain
active/incomplete. Changing VMs requires updating and reverifying the host guard
before a new packet; do not edit this immutable packet on the VM.
'''


def main():
    start = time.monotonic(); assert not OUT.exists() and not PREP.exists()
    parent = read(BALANCE/'protocol.json')
    for name, digest in parent['assets_sha256'].items(): assert sha(BALANCE/name) == digest
    for name, digest in parent['local_sources_sha256'].items(): assert sha(ROOT/name) == digest
    audit_path = ROOT/'outputs/cctv_dgp_original_loss_balance_v1_independent_audit.json'
    visual_path = ROOT/'outputs/cctv_dgp_original_loss_balance_v1_return_review_v1/visual_review.json'
    audit, visual = read(audit_path), read(visual_path)
    assert audit['complete'] and audit['all36_stored_row_comparisons_and_failure_decisions_exact']
    assert visual['complete'] and visual['unique_model_outputs_reviewed'] == 1000
    assert read(ROOT/'outputs/cctv_dgp_group_conflicts_v1_loss_parity_r2/forward_parity.json')['passed']
    worker = worker_source()
    candidate = (OLD/'cctv_dgp_original_feature_probe_v1_candidate.py').read_text().replace('original_feature_probe_v1', 'group_conflicts_v1')
    for code in [worker, candidate]: ast.parse(code, feature_version=(3, 10))
    text(ROOT/'scripts/cctv_dgp_group_conflicts_v1_candidate.py', candidate)
    text(ROOT/'scripts/cctv_dgp_group_conflicts_v1_vm.py', worker)
    # The prospective return checker must already exist and be syntactically valid.
    ast.parse((ROOT/'scripts/audit_cctv_dgp_group_conflicts_v1_return.py').read_text(), feature_version=(3, 10))
    OUT.mkdir(); PREP.mkdir(); mapping = {}
    for name in sorted(parent['assets_sha256']):
        if name.startswith('evidence/') or 'original_loss_balance_v1_' in name or name == 'scripts/run_loss_balance.sh': continue
        destination = OUT/name; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(BALANCE/name, destination); mapping[name] = (BALANCE/name).relative_to(ROOT).as_posix()
    evidence = {'initial_parameters.npy': BALANCE/'evidence/initial_parameters.npy',
        'previous_directions.npz': BALANCE/'evidence/canonical_directions.npz',
        'parent_protocol.json': BALANCE/'protocol.json', 'parent_results.json': RETURN/'outputs/results.json',
        'parent_independent_audit.json': audit_path, 'parent_visual_review.json': visual_path,
        'forward_parity_r2.json': ROOT/'outputs/cctv_dgp_group_conflicts_v1_loss_parity_r2/forward_parity.json',
        'forward_parity_failure.json': ROOT/'outputs/cctv_dgp_group_conflicts_v1_loss_parity/failure_receipt.json',
        'forward_parity_failed_receipt.json': ROOT/'outputs/cctv_dgp_group_conflicts_v1_loss_parity/forward_parity.json',
        'forward_parity_source_binding.json': ROOT/'outputs/cctv_dgp_group_conflicts_v1_loss_parity_r2/source_binding.json',
        'failed_losses.py.txt': ROOT/'outputs/cctv_dgp_group_conflicts_v1_loss_parity/before/losses.py'}
    for name, origin in evidence.items():
        destination = OUT/'evidence'/name; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, destination); mapping['evidence/'+name] = origin.relative_to(ROOT).as_posix()
    for part in ['candidate', 'contract', 'math', 'losses', 'vm']:
        filename = 'cctv_dgp_group_conflicts_v1_'+part+'.py'
        destination = OUT/('scripts' if part == 'vm' else '')/filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/'scripts'/filename, destination)
        mapping[destination.relative_to(OUT).as_posix()] = 'scripts/'+filename
    shell = (OLD/'scripts/run_feature_probe.sh').read_text().replace('original_feature_probe_v1', 'group_conflicts_v1')
    shell = shell.replace('original-feature diagnostic', 'group-conflicts diagnostic').replace('1230s', '1530s')
    text(OUT/'scripts/run_group_conflicts.sh', shell)
    p = copy.deepcopy(parent); by_id = {c['id']: c for c in p['cases']}
    for key in ['measured_balance_ratio', 'canonical_direction_arithmetic_atol', 'weight_norm_arithmetic_atol', 'saved_parent_gradient_queries']: p.pop(key, None)
    p.update({'format': 'own-DGP-source-profile-gradient-conflicts-v1', 'UTC': datetime.now(timezone.utc).isoformat(),
        'parent_protocol_sha256': sha(BALANCE/'protocol.json'), 'parent_archive_sha256': audit['archive_sha256'],
        'parent_audit_sha256': sha(audit_path), 'parent_result_sha256': sha(RETURN/'outputs/results.json'),
        'hypothesis': 'Mean objectives hide source/profile conflicts; measure gradients for every source/profile preservation metric before more epochs.',
        'architecture_direction': 'Unchanged isolated current DGP, same connected tensors and fixed centering; measure distinct group derivatives.',
        'terms': TERMS, 'scopes': ['common'], 'group_losses': definitions([by_id[c] for c in p['cohorts'][0]['case_ids']]),
        'gradient_queries': 160, 'maximum_trial_variants': 3, 'loss_surrogate_parity': PARITY,
        'budgets': BUDGETS, 'minimum_normalized_descent_cosine': 1e-7,
        'direction_reconstruction_arithmetic_atol': 1e-12,
        'relative_displacement_fractions': FRACTIONS, 'scale_weight_partition': 'decoder_control',
        'model_qualification': False, 'automatic_follow_on': False, 'app_promotion': False, 'goal_complete': False,
        'proposal_rule': 'Bounded minimum-norm simplex combination of32 unit group gradients; require every descent cosine>=1e-7; conditional3 reset trials scaled by original decoder L2; float32 once.',
        'raw_storage': 'All32 float64 aggregate vectors,160 query metadata rows,100 baseline raw/PNG/embedding cases and up to300 trial cases; individual query vectors omitted explicitly.',
        'assets_sha256': {q.relative_to(OUT).as_posix(): sha(q) for q in sorted(OUT.rglob('*')) if q.is_file()},
        'copied_source_mapping': mapping})
    local_names = ['CCTV_DGP_ORIGINAL_LOSS_BALANCE_V1_RESULTS.md', 'CCTV_DGP_GROUP_CONFLICTS_V1_DESIGN.md',
        'scripts/prepare_cctv_dgp_group_conflicts_v1.py', 'scripts/verify_cctv_dgp_group_conflicts_v1_packet.py',
        'scripts/audit_cctv_dgp_group_conflicts_v1_return.py', 'scripts/cctv_dgp_group_conflicts_v1_contract.py',
        'scripts/cctv_dgp_group_conflicts_v1_math.py', 'scripts/cctv_dgp_group_conflicts_v1_losses.py',
        'scripts/close_cctv_dgp_original_loss_balance_v1_review.py',
        'outputs/cctv_dgp_original_loss_balance_v1_independent_audit.json',
        'outputs/cctv_dgp_original_loss_balance_v1_return_review_v1/failure_attribution.json',
        'outputs/cctv_dgp_original_loss_balance_v1_return_review_v1/visual_review.json',
        'outputs/cctv_dgp_original_loss_balance_v1_return_review_v1/gallery_independent_audit.json',
        'outputs/cctv_dgp_group_conflicts_v1_loss_parity/failure_receipt.json',
        'outputs/cctv_dgp_group_conflicts_v1_loss_parity_r2/forward_parity.json']
    local_names.append('outputs/cctv_dgp_group_conflicts_v1_loss_parity_r2/source_binding.json')
    p['local_sources_sha256'] = {**parent['local_sources_sha256'], **{name: sha(ROOT/name) for name in local_names}}
    validate(p); write(OUT/'protocol.json', p); pin = sha(OUT/'protocol.json')
    for q in OUT.rglob('*.py'): ast.parse(q.read_text(encoding='utf-8'), feature_version=(3, 10))
    archive = ROOT/'outputs'/(STEM+'-execution.tar.gz'); assert not archive.exists()
    with tarfile.open(archive, 'w:gz', compresslevel=1) as tar:
        for q in sorted(OUT.rglob('*')):
            if not q.is_file(): continue
            info = tar.gettarinfo(str(q), arcname=NAME+'/'+q.relative_to(OUT).as_posix())
            info.uid = info.gid = 0; info.uname = info.gname = ''; info.mtime = 0; info.mode = 0o644
            with q.open('rb') as stream: tar.addfile(info, stream)
    digest = sha(archive); text(Path(str(archive)+'.sha256'), digest+'  '+archive.name+'\n')
    text(ROOT/'CCTV_DGP_GROUP_CONFLICTS_V1_VM.md', guide(pin, digest, archive.stat().st_size))
    write(PREP/'prepared.json', {'complete': True, 'protocol_sha256': pin, 'archive_sha256': digest,
        'archive_bytes': archive.stat().st_size, 'assets': len(p['assets_sha256']), 'cases': 100,
        'optimizer_updates': 0, 'neural_calls': 0, 'VM_connections': 0, 'VM_diagnostic_launched': False,
        'independent_packet_audit_pending': True, 'seconds': time.monotonic()-start})
    print({'complete': True, 'protocol_sha256': pin, 'archive_sha256': digest, 'bytes': archive.stat().st_size})


if __name__ == '__main__': main()
