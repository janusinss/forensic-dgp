"""Check actual retained gradient paths and reject a mismatched loss query."""
from import_cctv_dgp_nearby_gradient_learning_v1_return import OUT, ROOT, read, sha, write
from audit_cctv_dgp_nearby_gradient_learning_v1_return import retained_initial_gradient, expected_query


def main():
    p=read(OUT/'cctv_dgp_raw_conditioning_learning_v1_r1_vm/protocol.json')
    for index,row in enumerate(p['raw_gradient_groups']):
        record,vector=retained_initial_gradient(index,'RAW/'+row['key'])
        assert record['key']=='RAW/'+row['key'] and vector.shape==(2580864,)
        assert all(c['query']==expected_query(row) for c in record['contributions'])
    try:retained_initial_gradient(0,'RAW/wrong/source/group')
    except AssertionError:rejected=True
    else:raise AssertionError('Mismatched gradient group accepted')
    write(OUT/'cctv_dgp_nearby_gradient_learning_v1_source_paths_r1.json',dict(complete=True,
        checker_sha256=sha(__file__),return_auditor_sha256=sha(ROOT/'scripts/audit_cctv_dgp_nearby_gradient_learning_v1_return.py'),
        actual_retained_gradient_files=32,mismatched_loss_key_rejected=rejected,
        source_profile_metric_query_mapping_exact=True,local_neural_or_training_calls=0))
    print(dict(complete=True,actual_gradient_paths_checked=32,query_mapping_checked=True,neural_calls=0))


if __name__=='__main__':main()
