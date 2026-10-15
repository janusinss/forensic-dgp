"""Prepare a separate RAW32 optimizer comparison; actual training stays manual."""
import argparse
import ast
import copy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tarfile

from import_cctv_dgp_conditioning_learning_v1_r4_return import ROOT, OUT, RETURN, ANALYSIS, PARENT, PIN, read, sha, write

NAME = 'cctv_dgp_raw_conditioning_learning_v1_vm'
STEM = 'cctv-dgp-raw-conditioning-learning-v1'


def text_file(path, text):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)


def replacements(text, rows):
    original = text
    for before, after in rows:
        assert text.count(before) == 1, before
        text = text.replace(before, after)
    restored = text
    for before, after in reversed(rows):
        assert restored.count(after) == 1, after
        restored = restored.replace(after, before)
    assert restored == original
    return text


def sources():
    p = read(RETURN / 'protocol.json')
    audit = read(ANALYSIS / 'failure_audit.json')
    visual = read(ANALYSIS / 'visual_review.json')
    prediction = read(ANALYSIS / 'gradient_prediction_review.json')
    publication = read(ANALYSIS / 'publication/publication_audit.json')
    human = read(ANALYSIS / 'diagnostic_decision.json')
    assert all(v['complete'] for v in [audit, visual, prediction, publication, human])
    assert sha(RETURN / 'protocol.json') == PIN
    assert audit['accepted_trials_verified'] == [] and human['human_answer'] == 'apply best approach'
    assert visual['all20_sheets_and600_cells_directly_viewed']
    assert all(sha(RETURN / name) == digest for name, digest in p['assets_sha256'].items())
    assert all(sha(ROOT / n) == h for n, h in read(ANALYSIS / 'local_protected_before.json')['files_sha256'].items())
    destination = OUT / NAME
    assert not destination.exists(), 'Preserve existing prepared/used recipes'
    destination.mkdir()
    namespace = [('conditioning_learning_v1_r4', 'raw_conditioning_learning_v1'),
                 ('cctv-dgp-conditioning-learning-v1-r4', STEM)]
    worker_changes = [
        ("assert p['maximum_gradient_queries']==1280 and p['model_forward_cap']==180 and p['recognizer_forward_cap']==360",
         "assert p['maximum_gradient_queries']==640 and p['model_forward_cap']==180 and p['recognizer_forward_cap']==320\n"
         "    assert p['RAW_groups_per_round']==32 and p['PNG_groups_per_round']==0\n"
         "    assert p['optimizer_constraints']=='32 RAW groups; PNG measured by actual unchanged acceptance gates'"),
        ('from conditioning_losses import delivered_surrogates,pixel_and_structure_losses as png_losses',
         'from conditioning_losses import delivered_surrogates'),
        ("method='Full-cohort constrained gradient descent with actual raw/PNG backtracking'",
         "method='RAW32 full-cohort constrained descent with actual unchanged raw/PNG backtracking'"),
        ("all_groups=[dict(g,key=stage+'/'+g['key'],metric_stage=stage) for stage in ['RAW','PNG'] for g in raw_groups]",
         "all_groups=[dict(g,key='RAW/'+g['key'],metric_stage='RAW') for g in raw_groups]"),
        ("                png=np.stack([deliver(a,c,m) for a,c,m in zip(raw,x8,mask8)])\n"
         "                png_tensor=torch.from_numpy(png).permute(0,3,1,2).contiguous().cuda()\n"
         "                rgb,feature=delivered_surrogates(pred,png_tensor)\n"
         "                stages=[('RAW',pred,raw_losses(pred,target,mask,support)),('PNG',rgb,png_losses(rgb,target,mask,support,feature))]",
         "                stages=[('RAW',pred,raw_losses(pred,target,mask,support))]"),
        ("retain_graph=not(stage=='PNG' and query==15)", "retain_graph=query!=15"),
        ('del pred,rgb,feature,stages,terms,actual,x,target,mask,support,grids,png_tensor',
         'del pred,stages,terms,actual,x,target,mask,support,grids'),
    ]
    record = []
    for name, digest in p['assets_sha256'].items():
        source = RETURN / name
        target_name = name.replace('conditioning_learning_v1_r4', 'raw_conditioning_learning_v1')
        target = destination / target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith('cctv_dgp_conditioning_learning_v1_r4_vm.py'):
            original = source.read_text(encoding='utf-8')
            text = replacements(original, worker_changes)
            for old, new in namespace:
                text = text.replace(old, new)
            ast.parse(text, feature_version=(3, 10))
            text_file(target, text)
            text_file(ROOT / 'scripts/cctv_dgp_raw_conditioning_learning_v1_vm.py', text)
        elif name == 'conditioning_contract.py':
            original = source.read_text(encoding='utf-8')
            start, end = original.index('def solve_direction('), original.index('def bounded_rate(')
            solver = original[start:end]
            for before, after in [
                ('len(vectors) == len(records) == 64', 'len(vectors) == len(records) == 32'),
                ('np.zeros((64,64))', 'np.zeros((32,32))'), ('rank_bound = 64*', 'rank_bound = 32*'),
                ('Retain all64 loss dimensions', 'Retain all32 RAW loss dimensions'),
                ('np.zeros(64)', 'np.zeros(32)'),
                ('all64_loss_dimensions_retained', 'all32_RAW_loss_dimensions_retained'),
            ]:
                assert before in solver
                solver = solver.replace(before, after)
            assert solver.count('np.float64') == original[start:end].count('np.float64')
            text = original[:start] + solver + original[end:]
            assert text[end - (len(original[start:end]) - len(solver)):] == original[end:]
            ast.parse(text, feature_version=(3, 10))
            text_file(target, text)
            text_file(ROOT / 'scripts/cctv_dgp_raw_conditioning_learning_v1_contract.py', text)
        elif name in ['conditioning_model.py', 'scripts/run_learning.sh']:
            text = source.read_text(encoding='utf-8')
            for old, new in namespace:
                text = text.replace(old, new)
            text_file(target, text)
        else:
            with source.open('rb') as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst)
            assert sha(target) == digest
        record.append(dict(source=name, source_sha256=digest, target=target_name, target_sha256=sha(target)))
    for filename in ['gradient_prediction_review.json', 'diagnostic_decision.json']:
        with (ANALYSIS / filename).open('rb') as src, (destination / filename).open('xb') as dst:
            shutil.copyfileobj(src, dst)
    write(OUT / 'cctv_dgp_raw_conditioning_learning_v1_source_preparation.json', dict(
        complete=True, UTC=datetime.now(timezone.utc).isoformat(), preparer_sha256=sha(__file__),
        old_protocol_sha256=PIN, sources=record, changes=worker_changes,
        original_VM_worker_contract_models_masks_and_research_unchanged=True,
        new_optimizer_uses32_RAW_groups_and_actual_RAW_PNG_quality_gates_unchanged=True,
        model_and_initialization_and_cohort_and_scientific_thresholds_unchanged=True,
        no_neural_model_or_gradient_or_actual_direction_or_weight_proposals_generated_locally=True,
        protocol_and_transfer_audits_not_yet_complete=True, manual_training_unrun=True))
    print(dict(source_preparation=True, RAW_optimizer_groups=32, actual_PNG_gates_retained=True,
               manual_training_unrun=True))


def seal():
    destination = OUT / NAME
    p = read(RETURN / 'protocol.json')
    source = read(OUT / 'cctv_dgp_raw_conditioning_learning_v1_source_preparation_r1.json')
    green = read(OUT / 'cctv_dgp_raw_conditioning_learning_v1_solver_green.json')
    precision = read(OUT / 'cctv_dgp_raw_conditioning_learning_v1_precision_green.json')
    assert source['complete'] and green['complete'] and precision['complete']
    assert green['contract_sha256'] == sha(destination / 'conditioning_contract.py')
    assert precision['contract_sha256'] == sha(destination / 'conditioning_contract.py')
    assert all(sha(destination / r['target']) == r['target_sha256'] for r in source['sources'])
    assert not (destination / 'protocol.json').exists()
    assert all(sha(PARENT / n) == h for n, h in p['parent_required_sha256'].items())
    q = copy.deepcopy(p)
    q.update(format='finite-RAW32-conditioning-learning-v1', name=NAME, UTC=datetime.now(timezone.utc).isoformat(),
        maximum_gradient_queries=640, recognizer_forward_cap=320, PNG_groups_per_round=0,
        refresh_all64_gradients_after_every_accepted_update=False,
        refresh_all32_RAW_gradients_after_every_accepted_update=True,
        optimizer_constraints='32 RAW groups; PNG measured by actual unchanged acceptance gates',
        PNG_surrogate_constraints_not_used_for_update_direction=True,
        canonical_exact_PNG_forward_with_explicit_approximate_ST_backward=True,
        PNG_surrogate_used_for_optimization=False,
        exact_PNG_delivery_and_all17_scientific_acceptance_groups_retained=True,
        optimizer='Stateless RAW32 constrained full-cohort descent with actual RAW/PNG checks; accepted changes are training',
        repair_scope='Changed optimizer constraint set from RAW+approximatePNG64 to RAW32; actual RAW/PNG gates unchanged',
        evidence_is_activation_and_saved_RAW_subspace_arithmetic_not_trained_output_quality=False,
        evidence_is_R4_rejected_finite_outputs_and_192_saved_gradient_prediction_comparisons=True,
        comparison_parent_protocol_sha256=PIN,
        comparison_parent_return_archive_sha256=read(ANALYSIS / 'import.json')['archive_sha256'],
        comparison_rejected_trials_retained=['r00_trial00', 'r00_trial01', 'r00_trial02'],
        comparison_preview_exposures_and_evaluation_opportunities_unchanged=True,
        runtime_difference_and_new_RAW_gradient_exposures_reported_separately=True,
        reviewed_prediction_sha256=sha(ANALYSIS / 'gradient_prediction_review.json'),
        human_diagnostic_decision_sha256=sha(ANALYSIS / 'diagnostic_decision.json'),
        assets_sha256={f.relative_to(destination).as_posix(): sha(f) for f in destination.rglob('*') if f.is_file()})
    unchanged = ['reference_blocks', 'case_ids', 'raw_gradient_groups', 'conditioning_names', 'conditioning_shapes',
        'all37_learning_names', 'initial_model_state', 'preserved_scientific_thresholds',
        'maximum_accepted_updates', 'maximum_trials_per_update', 'maximum_run_relative_parameter_distance',
        'maximum_rate', 'backtrack_factors', 'parent_protocol_sha256', 'parent_required_sha256',
        'current_decoder_and_all_readouts_and_original_prior_recognizer_buffers_frozen',
        'actual_raw_and_PNG_scientific_preservation_gate_before_every_accepted_update',
        'final_finite_checkpoint_requires_1percent_globally_and_each_source_RAW_and_PNG',
        'full_TRAIN_final_structure_requirement_remains_10percent', 'original_checkpoint_sha256',
        'worker_seconds', 'measurement_seconds', 'gradient_round_seconds', 'external_seconds',
        'minimum_free_bytes', 'disk_reserve_bytes', 'output_byte_cap', 'peak_VRAM_bytes', 'peak_RSS_bytes']
    assert all(q[k] == p[k] for k in unchanged)
    write(destination / 'protocol.json', q)
    pin = sha(destination / 'protocol.json')
    archive = OUT / (STEM + '-execution.tar.gz')
    with tarfile.open(archive, 'x:gz', compresslevel=1) as tar:
        for path in sorted(f for f in destination.rglob('*') if f.is_file()):
            tar.add(path, arcname=NAME + '/' + path.relative_to(destination).as_posix(), recursive=False)
    text_file(Path(str(archive) + '.sha256'), sha(archive) + '  ' + archive.name + '\n')
    assert all(sha(RETURN / n) == h for n, h in p['assets_sha256'].items())
    assert all(sha(ROOT / n) == h for n, h in read(ANALYSIS / 'local_protected_before.json')['files_sha256'].items())
    write(OUT / 'cctv_dgp_raw_conditioning_learning_v1_preparation.json', dict(
        complete=True, UTC=datetime.now(timezone.utc).isoformat(), preparer_sha256=sha(__file__), name=NAME,
        protocol_sha256=pin, archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
        packet_assets=len(q['assets_sha256']), parent_file_bindings=len(q['parent_required_sha256']),
        inherited_scientific_and_state_resource_fields_checked=unchanged,
        source_preparation_sha256=sha(OUT / 'cctv_dgp_raw_conditioning_learning_v1_source_preparation_r1.json'),
        solver_fixture_sha256=sha(OUT / 'cctv_dgp_raw_conditioning_learning_v1_solver_green.json'),
        precision_fixture_sha256=sha(OUT / 'cctv_dgp_raw_conditioning_learning_v1_precision_green.json'),
        distinct_optimizer_comparison_not_a_failed_recipe_resume=True,
        PNG_quality_gates_preserved=True, manual_pilot_unrun=True,
        local_neural_gradient_optimizer_or_parameter_displacement_calls=0,
        prospective_transfer_and_return_audits_required=True, model_qualified=False, goal_complete=False))
    print(dict(complete=True, protocol_sha256=pin, archive_bytes=archive.stat().st_size, manual_unrun=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seal', action='store_true')
    args = parser.parse_args()
    seal() if args.seal else sources()
