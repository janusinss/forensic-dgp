"""Derive distinct sources without mutating frozen diagnostic sources."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def once(source, before, after):
    assert source.count(before) == 1, before[:100]
    return source.replace(before, after)


def main():
    old = ROOT / 'scripts/cctv_dgp_group_conflicts_v1_vm.py'
    worker = old.read_text().replace('group_conflicts_v1', 'finite_guard_v1')
    worker = once(worker, '"""Manual L4 connectivity and finite trial outputs; no optimizer or trained save."""',
        '"""Manual L4 finite mechanics study. Accepted changes count as training."""')
    worker = once(worker, 'NAME, STEM, BUDGETS, TERMS, SCOPES, FRACTIONS, STATE, read, write, sha, verified_assets)',
        'NAME, STEM, BUDGETS, FRACTIONS, STATE, read, write, sha, verified_assets)')
    worker = once(worker, "'optimizer_constructed': False, 'new_trained_checkpoint': False}",
        "'optimizer_constructed': False, 'new_trained_checkpoint': False, 'training_parameter_updates': 0}")
    worker = once(worker, "'Need4GiB free after install'", "'Need7GiB free after install; stop before neural calls'")
    # Loss/math implementations remain the earlier immutable, verified implementations.
    worker = worker.replace('from cctv_dgp_finite_guard_v1_losses import', 'from cctv_dgp_group_conflicts_v1_losses import')
    worker = worker.replace('from cctv_dgp_finite_guard_v1_math import', 'from cctv_dgp_group_conflicts_v1_math import')
    worker = once(worker, '        def evaluate(label):', "        def evaluate(label, previous_label='baseline'):")
    worker = once(worker, "                        mraw, mpng, shift = mean_only(a, base, item['camera'], item['mask8'])", """                        mraw, mpng, shift = mean_only(a, base, item['camera'], item['mask8'])
                        prior = base if previous_label == 'baseline' else np.load(out/previous_label/(cid+'.npy'), allow_pickle=False)
                        praw, ppng, pshift = mean_only(a, prior, item['camera'], item['mask8'])""")
    worker = once(worker, "                        Image.fromarray(mpng).save(folder/(cid+'_mean_only.png'))", """                        Image.fromarray(mpng).save(folder/(cid+'_mean_only.png'))
                        Image.fromarray(ppng).save(folder/(cid+'_previous_mean_only.png'))""")
    worker = once(worker, "'png': pm, 'postclip_mean_RGB_shift': shift.tolist()})", """'png': pm, 'postclip_mean_RGB_shift': shift.tolist(),
                            'previous_postclip_mean_RGB_shift': pshift.tolist(),
                            'previous_constant_mean_shift_only_MSE': {
                                'raw': pixel_metrics(praw, item['target8'], item['mask8'])['MSE'],
                                'png': exported_pixel_metrics(ppng, item['target8'], item['mask8'])['MSE']}})""")
    worker = once(worker, "            receipt = {'complete': True, 'variant': label, 'rows': rows, 'groups': groups,", """            previous_rows = [{**r, **{stage: {**r[stage],
                'constant_mean_shift_only_MSE': r['previous_constant_mean_shift_only_MSE'][stage]}
                for stage in ['raw', 'png']}} for r in rows]
            previous_groups = {co['name']: {stage: review_groups(
                [r for r in previous_rows if r['id'] in co['case_ids']], stage)
                for stage in ['raw', 'png']} for co in p['cohorts']}
            receipt = {'complete': True, 'variant': label, 'rows': rows, 'groups': groups,
                'previous_reference_variant': previous_label, 'previous_anchor_groups': previous_groups,""")
    worker = once(worker, "        remaining_estimate = 3*baseline['seconds']*1.25", "        remaining_estimate = 3*baseline['seconds']*1.25")
    worker = worker.replace("BUDGETS['trial_seconds']", "BUDGETS['trial_seconds_per_state']")
    worker = once(worker, '        projected_bytes = 4*baseline_bytes*1.25 + 650*1024**2', """        projected_bytes = 10*baseline_bytes*1.25 + p['projected_gradient_and_snapshot_bytes']""")
    begin = worker.index('        candidate.enable_diagnostic_gradients(root)')
    end = worker.index('    except BaseException as exc:', begin)
    worker = worker[:begin] + """        from cctv_dgp_finite_guard_v1_training import fit
        decoder_mask = np.zeros(1996035, bool)
        for desc in p['parameter_layout']:
            if desc['partition'] == 'decoder_control': decoder_mask[desc['start']:desc['end']] = True
        assert abs(float(np.linalg.norm(initial.astype(np.float64)[decoder_mask])) - p['initial_decoder_weight_L2']) <= 1e-10
        fit(root, p, pin, original, candidate, identity, items, by_id, batch, evaluate, baseline, initial,
            progress, counts, frozen, clock, start, out, pixel_and_structure_losses,
            common_direction, state_hash, before, states, provenance)
""" + worker[end:]
    worker = once(worker, "    result = {'complete': True, 'archive_sha256': digest, 'bytes': destination.stat().st_size,", """    run_receipt = read(root/('outputs/results.json' if (root/'outputs/results.json').is_file() else 'outputs/failure.json'))
    result = {'complete': True, 'archive_sha256': digest, 'bytes': destination.stat().st_size,""")
    worker = once(worker, "        'optimizer_updates': 0, 'run_results_present':", "        'optimizer_updates': run_receipt['optimizer_updates'], 'accepted_changes_are_actual_training': True, 'run_results_present':")
    worker = worker.replace('Finite diagnostic deadline', 'Finite mechanics deadline').replace('External diagnostic stop', 'External mechanics stop')
    candidate = (ROOT/'scripts/cctv_dgp_group_conflicts_v1_candidate.py').read_text().replace('group_conflicts_v1', 'finite_guard_v1')
    for name, source in [('cctv_dgp_finite_guard_v1_vm.py', worker), ('cctv_dgp_finite_guard_v1_candidate.py', candidate)]:
        ast.parse(source, feature_version=(3, 10))
        with (ROOT/'scripts'/name).open('x', encoding='utf-8', newline='\n') as stream: stream.write(source)
    print({'complete': True, 'new_sources': 2, 'old_sources_mutated': 0, 'neural_calls': 0})


if __name__ == '__main__': main()
