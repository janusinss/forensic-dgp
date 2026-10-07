"""Interpret independently checked saved gradients; no neural or autograd calls."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_vm'
RETURN = ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_return'
AUDIT = ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_independent_audit.json'
OUT = ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_analysis'
REPORT = ROOT / 'CCTV_DGP_FEATURE_FUSION_GRADIENT_V1_RESULTS.md'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    assert not OUT.exists() and not REPORT.exists()
    a, p = read(AUDIT), read(PACKET / 'protocol.json')
    assert a['complete'] and a['diagnostic_complete'] and not a['VM_failure_retained']
    assert a['checker_sha256'] == sha(ROOT / 'scripts/audit_cctv_dgp_feature_fusion_gradient_v1_return.py')
    assert a['protocol_sha256'] == sha(PACKET / 'protocol.json')
    assert a['optimizer_updates'] == a['local_gradient_calls'] == a['local_optimizer_updates'] == 0
    assert a['CPU_replay']['cases_at_both_states'] == 200
    result = read(RETURN / 'outputs/results.json')
    assert result['component_gradient_calls'] == 280 and result['DGP_and_recognizer_unmodified']
    assert not result['new_checkpoint_created'] and not result['optimizer_constructed']
    rows = a['cohort_gradient_analysis']
    assert [(r['state'], r['cohort']) for r in rows] == [(s, c) for s in [0, 50] for c in ['exposed', 'unexposed']]
    selections = set(p['fusion_parameter_names']) | set(p['decoder_parameter_names'])
    assert len(selections) == 23 and p['selected_parameters'] == 978243
    table, disconnected = [], []
    for row in rows:
        assert set(row['selected23_improvement_norms']) == selections
        missing = [name for name, norm in row['selected23_improvement_norms'].items() if norm <= 0]
        disconnected.extend({'state': row['state'], 'cohort': row['cohort'], 'name': n} for n in missing)
        fusion, decoder = [row['partition_analysis'][key] for key in ['feature_fusion', 'decoder']]
        assert fusion['parameters'] == 479616 and decoder['parameters'] == 498627
        table.append({'state': row['state'], 'cohort': row['cohort'],
                      'fusion_improvement_norm': fusion['improvement_norm'],
                      'decoder_improvement_norm': decoder['improvement_norm'],
                      'fusion_to_decoder_improvement_norm_ratio': fusion['improvement_norm'] / decoder['improvement_norm'],
                      'fusion_component_directional_derivatives': dict(zip(p['terms'], fusion['component_directional_derivatives'])),
                      'decoder_component_directional_derivatives': dict(zip(p['terms'], decoder['component_directional_derivatives'])),
                      'fusion_preservation_norm': fusion['preservation_norm'],
                      'decoder_preservation_norm': decoder['preservation_norm'],
                      'fusion_improvement_preservation_cosine': fusion['improvement_preservation_cosine'],
                      'decoder_improvement_preservation_cosine': decoder['improvement_preservation_cosine'],
                      'all23_connected': not missing})
    early_path = ROOT / 'outputs/cctv_dgp_profile_batches_v31_return/outputs/early_structure_stop.json'
    early = read(early_path)
    assert early['minimum'] == .01 and not early['pass']
    initial_rows = [r for r in table if r['state'] == 0]
    initial_descent = all(all(r['fusion_component_directional_derivatives'][k] < 0 for k in p['terms'][:3]) for r in initial_rows)
    eligible = not disconnected and initial_descent
    bindings = [Path(__file__), AUDIT, PACKET / 'protocol.json', RETURN / 'outputs/results.json',
                ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_return_import.json', early_path]
    bindings.extend(RETURN / f'outputs/state{s}_{c}/receipt.json' for s in [0, 50] for c in ['exposed', 'unexposed'])
    analysis = {'complete': True, 'source_bindings_sha256': {f.relative_to(ROOT).as_posix(): sha(f) for f in bindings},
                'rows': table, 'disconnected_improvement_tensors': disconnected,
                'all23_improvement_gradients_nonzero_in_both_cohorts_at_both_states': not disconnected,
                'original_fusion_negative_objective_direction_decreases_all_three_improvement_terms_in_both_cohorts': initial_descent,
                'new_finite_fusion_partition_pilot_justified_for_preparation': eligible,
                'interpretation': 'A first-order connection and descent observation supports testing this previously frozen original feature-fusion path. It does not predict AdamW steps or useful restored faces.',
                'preservation_policy': 'Keep all seven loss terms and all early/final structure, clear-control, source and preservation gates; stopped-state penalties respond to drift.',
                'next_design': 'Fresh original DGP copy; add the measured 11 fusion tensors to the 12 active decoder tensors; keep backbone, head4 and all evaluation buffers frozen. Same V31 data, schedule, objective, normalizers, optimizer and gates.',
                'V31_failed_one_percent_gate_retained': early,
                'TRAIN_evidence_only': True, 'source_labels_are_not_ethnicity': True,
                'native_or_reserved_used': False, 'new_training_recipe_created': False,
                'neural_calls': 0, 'gradient_calls': 0, 'optimizer_updates': 0,
                'app_promotion': False, 'useful_output_qualification': False, 'goal_complete': False}
    OUT.mkdir()
    with (OUT / 'analysis.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(analysis, stream, indent=2, allow_nan=False)
    lines = ['# Original DGP feature-fusion diagnostic: audited findings', '',
             f"The human VM diagnostic completed {result['component_gradient_calls']} component-gradient queries in {result['seconds']:.3f} seconds, with zero optimizer updates, backwards or epochs. No new checkpoint was created.", '',
             f"The independent local checker verifies all {a['members_verified']} returned files and {sum(r['values_checked'] for r in rows):,} saved gradient values. It replays 200 outputs with CPU inference only: raw maximum error {a['CPU_replay']['raw_maximum_error']:.9g}, PNG maximum difference {a['CPU_replay']['PNG_maximum_byte_error']} byte, vector maximum error {a['CPU_replay']['vector_maximum_error']:.9g}. Returned code is not executed and no local gradients are computed.", '',
             '| State | TRAIN cohort | Fusion improvement norm | Decoder improvement norm | Fusion/decoder ratio | All 23 connected |',
             '| --- | --- | ---: | ---: | ---: | --- |']
    for row in table:
        lines.append(f"| {row['state']} | {row['cohort']} | {row['fusion_improvement_norm']:.6g} | {row['decoder_improvement_norm']:.6g} | {row['fusion_to_decoder_improvement_norm_ratio']:.3f} | {row['all23_connected']} |")
    lines.extend(['',
                  'The five lateral and three top-down convolutions comprise 11 original feature-fusion tensors with 479,616 parameters. Their values stayed fixed in V31. The comparison decoder has 12 active tensors with 498,627 parameters. All are connected to improvement losses in both measured cohorts at both states.' if not disconnected else 'Disconnected improvement tensors remain a reason to reject selecting the complete partition for training.', '',
                  'At the original state, a negative total-objective direction restricted to fusion decreases all three improvement terms in both sampled cohorts.' if initial_descent else 'Original-state fusion directional derivatives do not support a uniform three-term descent claim.',
                  'These are Euclidean first-order derivative observations on matched TRAIN subsets. Gradient norms depend on parameterization; larger norms do not prove more capacity, explain a unique cause or predict finite AdamW behavior. The stopped-state preservation terms remain necessary; they must not be removed to force a structure gain.', '',
                  f"V31 remains closed at {early['relative_feature_error_gain'] * 100:.6f}% against the unchanged 1% early requirement. Original checkpoints, stopped weights, source hashes, splits and every failed gate remain retained. No stopped run is resumed.", '',
                  'The next justified design tests one parameter-partition change on a fresh original DGP copy: enable the 11 measured fusion tensors alongside the 12 active decoder tensors. Keep backbone, inactive head4, evaluation buffers, paired clear/degraded batches, full 3,905-case TRAIN corpus, seven losses, initial normalizers, AdamW settings and all gates. Release a finite manually launched L4 packet only after independent source/data/transfer verification.' if eligible else 'No new training packet is justified by these measurements. Retain the findings for an architecture review.', '',
                  'This diagnostic qualifies no restoration or completion output. Cohorts are synthetic photographic TRAIN data, not native CCTV or independent evaluation. Source labels do not establish ethnicity. Reserved-final pixels remain unopened; useful native development, all covering families and independent final review are still outstanding. The app primary checkpoint is unchanged. Goal active/incomplete.', ''])
    with REPORT.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('\n'.join(lines))
    print(json.dumps({'complete': True, 'all23_connected': not disconnected, 'initial_fusion_three_term_descent': initial_descent,
                      'new_partition_pilot_preparation_justified': eligible, 'rows': table}, indent=2))


if __name__ == '__main__':
    main()
