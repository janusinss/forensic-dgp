"""Prepare prospective RAW32 return auditors; keep every R4 checker intact."""
import ast

from import_cctv_dgp_conditioning_learning_v1_r4_return import ROOT, OUT, PIN, read, sha, write


def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def save(path, text):
    ast.parse(text, feature_version=(3, 10))
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)


def main():
    prep = read(OUT / 'cctv_dgp_raw_conditioning_learning_v1_preparation.json')
    assert prep['complete'] and prep['manual_pilot_unrun']
    packet = OUT / prep['name']
    assert sha(packet / 'protocol.json') == prep['protocol_sha256']
    records = []
    def record(source, target):
        records.append(dict(source=source.name, source_sha256=sha(source),
                            target=target.name, target_sha256=sha(target)))
    source = ROOT / 'scripts/import_cctv_dgp_conditioning_learning_v1_r4_return.py'
    text = source.read_text(encoding='utf-8').replace('conditioning_learning_v1_r4', 'raw_conditioning_learning_v1')
    text = text.replace('cctv-dgp-conditioning-learning-v1-r4', 'cctv-dgp-raw-conditioning-learning-v1')
    text = replace_once(text, PIN, prep['protocol_sha256'])
    target = ROOT / 'scripts/import_cctv_dgp_raw_conditioning_learning_v1_return.py'
    save(target, text); record(source, target)
    source = ROOT / 'scripts/conditioning_decision_receipt_check_v1.py'
    text = source.read_text(encoding='utf-8').replace('conditioning_learning_v1_r4', 'raw_conditioning_learning_v1')
    target = ROOT / 'scripts/raw_conditioning_decision_receipt_check_v1.py'
    save(target, text); record(source, target)
    source = ROOT / 'scripts/audit_cctv_dgp_conditioning_learning_v1_r4_return_r1.py'
    text = source.read_text(encoding='utf-8').replace('conditioning_learning_v1_r4', 'raw_conditioning_learning_v1')
    text = text.replace('cctv-dgp-conditioning-learning-v1-r4', 'cctv-dgp-raw-conditioning-learning-v1')
    changes = [
        ("keys=[stage+'/'+g['key'] for stage in ['RAW','PNG'] for g in p['raw_gradient_groups']]",
         "keys=['RAW/'+g['key'] for g in p['raw_gradient_groups']]"),
        ("expected_groups=[dict(g,key=stage+'/'+g['key'],metric_stage=stage) for stage in ['RAW','PNG'] for g in p['raw_gradient_groups']]",
         "expected_groups=[dict(g,key='RAW/'+g['key'],metric_stage='RAW') for g in p['raw_gradient_groups']]"),
        ('gram=np.zeros((64,64));dots=np.zeros(64)', 'gram=np.zeros((32,32));dots=np.zeros(32)'),
        ("cert['all64_loss_dimensions_retained']", "cert['all32_RAW_loss_dimensions_retained']"),
        ('gradient_groups=64', 'gradient_groups=32'),
        ('recognizer=40+40*len(gradient_audits)+40*trial_count', 'recognizer=40+20*len(gradient_audits)+40*trial_count'),
        ('gradient_queries=640*len(gradient_audits)', 'gradient_queries=320*len(gradient_audits)'),
        ('from conditioning_decision_receipt_check_v1 import validate_receipt',
         'from raw_conditioning_decision_receipt_check_v1 import validate_receipt'),
        ("sha(ROOT/'scripts/conditioning_decision_receipt_check_v1.py')", "sha(ROOT/'scripts/raw_conditioning_decision_receipt_check_v1.py')"),
        ("        original_checker_sha256='30f8785e8c519624760d6c41ffcf79afeee2a643b0409e5f9b876c4eea780e6e',\n", ''),
        ("        original_checker_stop_sha256=sha(ANALYSIS/'original_checker_stop.json'),\n", ''),
        ("alignment['metric_stage'] in ['RAW','PNG']", "alignment['metric_stage']=='RAW'"),
    ]
    for before, after in changes:
        text = replace_once(text, before, after)
    target = ROOT / 'scripts/audit_cctv_dgp_raw_conditioning_learning_v1_return.py'
    save(target, text); record(source, target)
    source = ROOT / 'scripts/audit_cctv_dgp_conditioning_learning_v1_r4_preparation.py'
    text = source.read_text(encoding='utf-8').replace('conditioning_learning_v1_r4', 'raw_conditioning_learning_v1')
    text = text.replace('cctv-dgp-conditioning-learning-v1-r4', 'cctv-dgp-raw-conditioning-learning-v1')
    text = text.replace("records=[dict(value=.01,metric='feature_MSE' if i in [31,63] else 'MSE') for i in range(64)]",
                        "records=[dict(value=.01,metric='feature_MSE' if i in [30,31] else 'MSE') for i in range(32)]")
    text = text.replace('np.eye(64)', 'np.eye(32)').replace('np.zeros(64,np.float32)', 'np.zeros(32,np.float32)')
    text = text.replace("certificate['all64_loss_dimensions_retained']", "certificate['all32_RAW_loss_dimensions_retained']")
    text = text.replace('declared_synthetic64_constraint_solver_and_cumulative_distance_fixture_passed',
                        'declared_synthetic32_RAW_constraint_solver_and_cumulative_distance_fixture_passed')
    text = replace_once(text,
        '    from audit_cctv_dgp_raw_conditioning_learning_v1_loss_forward_repair import audit as repair_audit\n    repair_proof=repair_audit(packet,p)\n',
        '    from audit_cctv_dgp_raw_conditioning_learning_v1_design import audit as design_audit\n    repair_proof=design_audit(packet,p)\n')
    text = text.replace('loss_forward_and_unchanged_recipe_audit=repair_proof', 'RAW_optimizer_and_unchanged_quality_gate_design_audit=repair_proof')
    marker = '    records=[dict(value=.01,metric='
    index = text.index(marker)
    additional = """    r4=OUT/'cctv_dgp_conditioning_learning_v1_r4_return/outputs/conditioning_learning_v1_r4'
    current_base=read(r4/'initial/snapshot.json')['groups']
    r4_rejected=[]
    for label in ['r00_trial00','r00_trial01','r00_trial02']:
        after=read(r4/label/'snapshot.json')['groups']
        checked=decision(current_base,current_base,after)
        assert checked==read(r4/label/'decision.json')['decision']
        assert not checked['accept_step'] and not checked['one_percent_checkpoint_pass']
        r4_rejected.append(label)
"""
    text = text[:index] + additional + text[index:]
    text = replace_once(text, 'actual_prior_rejected_joint_proposals_still_rejected=previous_rejected,',
                        'actual_prior_rejected_joint_proposals_still_rejected=previous_rejected,\n'
                        '        all_three_actual_R4_trial_decisions_replayed_and_still_rejected=r4_rejected,')
    target = ROOT / 'scripts/audit_cctv_dgp_raw_conditioning_learning_v1_preparation.py'
    save(target, text); record(source, target)
    write(OUT / 'cctv_dgp_raw_conditioning_learning_v1_checker_preparation.json', dict(
        complete=True, preparer_sha256=sha(__file__), protocol_sha256=prep['protocol_sha256'], prospective_checkers=records,
        previous_R4_importer_original_and_repaired_auditors_retained=True,
        future_derivative_group_and_actual_forward_counts_are_RAW32=True,
        actual_RAW_and_PNG_scores_and_gate_decisions_remain_independently_checked=True,
        three_returned_R4_rejections_and_four_older_rejections_remain_binding=True,
        no_local_models_or_training=True, future_full_return_audit_not_run=True))
    print(dict(complete=True, prospective_checkers=4, RAW32_derivatives=True, actual_PNG_checks_retained=True))


if __name__ == '__main__':
    main()
