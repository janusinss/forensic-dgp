"""Explain the matched A/B stop using audited saved metrics/states only.

No VM connection, model forward, differentiation, optimizer or gate change.
The bank-disabled subset is descriptive, not a full-cohort bank ablation.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / 'outputs/cctv_dgp_bank_comparison_v1_r3_A_return'
B = ROOT / 'outputs/cctv_dgp_bank_comparison_v1_r4_B_return'
ANALYSIS = ROOT / 'outputs/cctv_dgp_bank_comparison_v1_r4_B_analysis'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def aggregate(rows, stage):
    assert rows
    return {key: float(np.mean([row[stage][key] for row in rows])) for key in
            ['MSE', 'SSIM', 'ArcFace_observed_fixed', 'landmark_high_frequency_MSE']}


def comparison(old, new):
    return dict(MSE_change=new['MSE']-old['MSE'],
        SSIM_change=new['SSIM']-old['SSIM'],
        observed_recognizer_similarity_change=new['ArcFace_observed_fixed']-old['ArcFace_observed_fixed'],
        relative_structure_gain=1-new['landmark_high_frequency_MSE']/old['landmark_high_frequency_MSE'],
        recognizer_similarity_not_identity_recovery_proof=True)


def main():
    start = time.monotonic()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    receipt = ANALYSIS / 'saved_learning_diagnostic.json'
    assert not receipt.exists()
    p = read(B / 'protocol.json')
    manifest = read(B / 'export_manifest.json')['files']
    audit = read(ANALYSIS / 'independent_audit.json')
    matched = read(ANALYSIS / 'matched_AB_comparison.json')
    execution = read(ANALYSIS / 'execution_audit.json')
    assert all(row['complete'] for row in [audit, matched, execution])
    assert audit['optimizer_updates_in_VM'] == execution['actual_optimizer_updates'] == 50
    assert matched['optimizer_updates_each'] == 50 and matched['full_epochs_each'] == 0
    bindings = {}

    def bind(path):
        bindings[path.relative_to(ROOT).as_posix()] = sha(path)
        if path.is_relative_to(B):
            assert manifest[path.relative_to(B).as_posix()] == bindings[path.relative_to(ROOT).as_posix()]

    for name in ['independent_audit.json', 'matched_AB_comparison.json', 'execution_audit.json']:
        bind(ANALYSIS / name)
    bind(B / 'protocol.json')
    a_out, b_out = [root / 'outputs/bank_comparison_v1' for root in [A, B]]
    metrics = {}
    for label, folder in [('baseline', b_out/'baseline'), ('A', a_out/'A50'),
                          ('B', b_out/'B50'), ('bank_disabled', b_out/'B50_bank_disabled')]:
        file = folder / 'metrics.json'; bind(file)
        metrics[label] = read(file)
    tables = {}
    for stage in ['raw', 'png']:
        baseline = metrics['baseline']['groups'][stage]
        tables[stage] = {group: {route: comparison(baseline[group], metrics[route]['groups'][stage][group])
                        for route in ['A', 'B']} for group in baseline}

    # Compare the same cases for the bank-on/off diagnostic. The ablation has
    # only the prospectively selected 100 paired previews, not all 3,905 cases.
    ids = [row['id'] for row in metrics['bank_disabled']['rows']]
    assert len(ids) == len(set(ids)) == 100 and set(ids) == set(p['preview_case_ids'])
    maps = {name: {row['id']: row for row in data['rows']} for name, data in metrics.items()}
    sample = {}
    for stage in ['raw', 'png']:
        sample[stage] = {}
        groups = {'all': ids, 'clear': [id_ for id_ in ids if maps['B'][id_]['profile']=='clear'],
                  'degraded': [id_ for id_ in ids if maps['B'][id_]['profile']!='clear']}
        for group, selected in groups.items():
            on = aggregate([maps['B'][id_] for id_ in selected], stage)
            off = aggregate([maps['bank_disabled'][id_] for id_ in selected], stage)
            original = aggregate([maps['baseline'][id_] for id_ in selected], stage)
            sample[stage][group] = dict(cases=len(selected), bank_on=on, bank_disabled=off,
                bank_on_vs_disabled=comparison(off, on),
                bank_on_vs_baseline=comparison(original, on),
                disabled_vs_baseline=comparison(original, off))

    # Distinguish exposed TRAIN references from other references in the same
    # TRAIN pool. Neither partition becomes held-out DEV or a clean CCTV test.
    exposed = {p['cases'][begin]['source_person_or_reference'] for step in p['schedule'] for begin in step}
    assert len(exposed) == 250
    partitions = {}
    case_ids = {case['id']: case for case in p['cases']}
    for partition, flag in [('exposed_TRAIN', True), ('unexposed_in_same_TRAIN_pool', False)]:
        selected = [id_ for id_, case in case_ids.items()
                    if (case['source_person_or_reference'] in exposed)==flag and case['profile']!='clear']
        partitions[partition] = dict(cases=len(selected), references=len(selected)//4,
            evidence_role='paired photographic TRAIN capacity', independent_validation=False,
            stages={stage: {route: comparison(
                aggregate([maps['baseline'][id_] for id_ in selected], stage),
                aggregate([maps[route][id_] for id_ in selected], stage)) for route in ['A','B']}
                for stage in ['raw','png']})

    state_path = b_out / 'state_B50/full_state.pth'
    initial_path = B / 'initializers/initial_B.pth'
    conditioning_path = b_out / 'B_conditioning_after_update1.json'
    for file in [state_path, initial_path, conditioning_path]:
        bind(file)
    state = torch.load(state_path, map_location='cpu', weights_only=True)
    initial = torch.load(initial_path, map_location='cpu', weights_only=True)
    conditioning = read(conditioning_path)
    names = conditioning['names']
    assert len(names) == len(set(names)) == p['routes']['B']['tensors'] == 37
    assert conditioning['optimizer_updates'] == 1 and conditioning['all_nonzero']
    assert all(np.isfinite(norm) and norm>0 for norm in conditioning['summed_norms'])
    assert set(state['model']) == set(initial)
    moved, frozen = [], 0
    for name, before in initial.items():
        after = state['model'][name]
        if name not in names:
            assert torch.equal(before, after), name
            frozen += 1
            continue
        assert before.shape == after.shape and before.dtype == after.dtype
        assert torch.isfinite(before).all() and torch.isfinite(after).all()
        delta = after.double()-before.double()
        old_norm, change = float(before.double().norm()), float(delta.norm())
        part = 'current_reconstruction' if name.startswith('current.') else name.split('.')[0]
        moved.append(dict(name=name, partition=part, elements=before.numel(),
            initial_L2_norm=old_norm, update50_change_L2_norm=change,
            relative_L2_change=change/old_norm if old_norm else None,
            initially_zero=old_norm==0, bitwise_changed=not torch.equal(before, after)))
    history_path = b_out / 'history_B.json'; bind(history_path)
    history = read(history_path)
    assert len(history['rows']) == 50
    first = float(np.mean([row['objective'] for row in history['rows'][:10]]))
    last = float(np.mean([row['objective'] for row in history['rows'][-10:]]))
    initializer = []
    for file in sorted(b_out.glob('initializer_load_B_*.json')):
        bind(file); row = read(file)
        assert row['strict_load_complete'] and row['every_loaded_tensor_exact'] and row['frozen_tensors_exact']
        initializer.append(dict(name=file.name, constructor_differences=row['constructor_differences'],
            loaded_state_sha256=row['loaded_state_sha256'], every_loaded_tensor_exact=True))
    assert initializer
    for name, expected in bindings.items():
        assert sha(ROOT/name) == expected
    result = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), checker_sha256=sha(__file__),
        protocol_sha256=sha(B/'protocol.json'), evidence_files_sha256=bindings,
        comparison='50 matched updates; 250 distinct references; 1250 profile exposures; zero complete epochs each',
        all_cohort_source_profile_changes=tables,
        bank_disabled_subset=dict(paired_cases=100, references=20,
            prospectively_selected=True, descriptive_only=True, full_3905_ablation_claimed=False,
            native_unpaired_cases=24, native_reference_accuracy_claimed=False, stages=sample),
        exposure_partitions=partitions, B_parameter_changes=moved,
        reference_resolution_provenance=dict(
            target_kind_counts=dict(Counter(row.get('evaluation_source_kind','original_source_reference')
                                           for row in p['references'])),
            observed_source_size_counts=dict(Counter(str(row['native_size']) for row in p['references'])),
            HQ_source_hash_bindings=sum(bool(row.get('hq_source_sha256')) for row in p['references']),
            prepared_target_size=[256,256],
            observed_native_size_is_not_supervised_target_resolution=True,
            additional_data_gap_not_established_by_this_diagnostic=True),
        changed_tensors_per_partition=dict(Counter(row['partition'] for row in moved if row['bitwise_changed'])),
        exact_frozen_state_tensors_checked=frozen, first_update_all_37_gradient_norms_nonzero=True,
        initializer_loading=initializer,
        history=dict(first10_mean_objective=first,last10_mean_objective=last,
            same_reference_batches=False, plateau_or_convergence_not_proven=True),
        original_quality_decisions=audit['metric_and_gate_decisions'],
        no_quality_gate_redefinition=True, no_failed_state_resume=True,
        final_identities_used=False, local_model_forwards=0, local_gradients=0, local_optimizer_updates=0,
        model_qualified=False, app_promotion=False, goal_complete=False, seconds=time.monotonic()-start)
    with receipt.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(dict(complete=True,bank_ablation_cases=100,changed_trainable_tensors=sum(row['bitwise_changed'] for row in moved),
        frozen_state_tensors_checked=frozen,local_forwards_gradients_updates=0),flush=True)


if __name__ == '__main__':
    main()
