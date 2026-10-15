"""Prepare immutable R4 import/audit namespaces; retain R3 checkers."""
import ast
from import_cctv_dgp_conditioning_learning_v1_r3_return import ROOT, OUT, PIN, read, sha, write


def main():
    prep = read(OUT / 'cctv_dgp_conditioning_learning_v1_r4_preparation.json')
    assert prep['complete']
    sources = ['import_cctv_dgp_conditioning_learning_v1_r3_return.py',
               'audit_cctv_dgp_conditioning_learning_v1_r3_return.py',
               'cctv_dgp_conditioning_learning_v1_r3_contract.py']
    records = []
    for name in sources:
        source = ROOT / 'scripts' / name
        original = source.read_text(encoding='utf-8')
        text = original.replace('conditioning_learning_v1_r3', 'conditioning_learning_v1_r4')
        text = text.replace('cctv-dgp-conditioning-learning-v1-r3', 'cctv-dgp-conditioning-learning-v1-r4')
        if name.startswith('import_'):
            assert text.count(PIN) == 1
            text = text.replace(PIN, prep['protocol_sha256'])
        if name.startswith('audit_'):
            marker = "    # Every complete candidate snapshot gets a sheet for every reference,"
            assert text.count(marker) == 1
            diagnostic = """    if (out/'loss_alignment_failure.json').exists():
        alignment=read(out/'loss_alignment_failure.json')
        assert failure and failure['message']=='Scientific/surrogate loss alignment differs'
        assert alignment['metric_stage'] in ['RAW','PNG'] and 0<=alignment['query']<16
        assert abs(alignment['actual_forward_value']-alignment['expected_scientific_value'])==alignment['absolute_difference']
        assert alignment['absolute_difference']>alignment['alignment_bound']
        block=next(b for b in p['reference_blocks'] if b['begin']==alignment['begin'])
        assert block['reference']==alignment['reference'] and alignment['snapshot'] in snapshots
        expected_stage=alignment['metric_stage'].lower();query=alignment['query']
        rows={r['id']:r for r in snapshots[alignment['snapshot']]['snapshot']['rows']}
        if query<15:
            score=rows[block['case_ids'][query//3]][expected_stage]
            expected=[score['MSE'],1-score['SSIM'],1-score['ArcFace_observed_fixed']][query%3]
        else:expected=float(np.mean([rows[cid][expected_stage]['landmark_high_frequency_MSE'] for cid in block['case_ids'][1:]]))
        assert alignment['expected_scientific_value']==expected
        assert alignment['raw_helper_sha256']==sha(RETURN/'conditioning_raw_losses.py')
        assert alignment['PNG_helper_sha256']==sha(RETURN/'conditioning_losses.py')
        assert alignment['candidate_state']==snapshots[alignment['snapshot']]['snapshot']['model_state']
        write(ANALYSIS/'alignment_scalar_audit.json',dict(complete=True,reported_alignment=alignment,
            scientific_snapshot_value_independently_checked=True,actual_neural_loss_not_reexecuted_locally=True))
"""
            text = text.replace(marker, diagnostic + marker)
        ast.parse(text, feature_version=(3, 10))
        target = ROOT / 'scripts' / name.replace('conditioning_learning_v1_r3', 'conditioning_learning_v1_r4')
        with target.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(text)
        records.append(dict(source=name, source_sha256=sha(source), target=target.name, target_sha256=sha(target)))
    # Retain the comprehensive pre-R3 audit; supplement it for this new loss
    # boundary rather than weakening its baseline, solver or gate checks.
    source = ROOT / 'scripts/audit_cctv_dgp_conditioning_learning_v1_r2_preparation.py'
    text = source.read_text(encoding='utf-8').replace('conditioning_learning_v1_r2', 'conditioning_learning_v1_r4')
    text = text.replace('cctv-dgp-conditioning-learning-v1-r2', 'cctv-dgp-conditioning-learning-v1-r4')
    marker = "    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']"
    assert text.count(marker) == 1
    text = text.replace(marker, "    from audit_cctv_dgp_conditioning_learning_v1_r4_loss_forward_repair import audit as repair_audit\n    repair_proof=repair_audit(packet,p)\n" + marker)
    marker = '        LF_ASCII_checksum_exact=True,parent_assets_verified=5852'
    assert text.count(marker) == 1
    text = text.replace(marker, '        loss_forward_and_unchanged_recipe_audit=repair_proof,\n' + marker)
    target = ROOT / 'scripts/audit_cctv_dgp_conditioning_learning_v1_r4_preparation.py'
    ast.parse(text, feature_version=(3, 10))
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)
    records.append(dict(source=source.name, source_sha256=sha(source), target=target.name, target_sha256=sha(target)))
    write(OUT / 'cctv_dgp_conditioning_learning_v1_r4_checker_preparation.json', dict(
        complete=True, preparer_sha256=sha(__file__), protocol_sha256=prep['protocol_sha256'],
        prospective_checkers=records, previous_checkers_retained=True,
        optional_returned_loss_scalar_audit_added=True, no_local_models_or_training=True,
        future_full_return_audit_not_run=True))
    print(dict(complete=True, prospective_checkers=4, manual_training_unrun=True))


if __name__ == '__main__':
    main()
