"""Copy new receipt-only packet namespaces; retain all prior return checkers."""
import ast

from import_cctv_dgp_conditioning_learning_v1_r4_return import ROOT, OUT, read, sha, write


def main():
    old = read(OUT / 'cctv_dgp_raw_conditioning_learning_v1_preparation.json')
    new = read(OUT / 'cctv_dgp_raw_conditioning_learning_v1_r1_preparation.json')
    assert old['complete'] and new['complete']
    files = ['import_cctv_dgp_raw_conditioning_learning_v1_return.py',
             'audit_cctv_dgp_raw_conditioning_learning_v1_return.py',
             'cctv_dgp_raw_conditioning_learning_v1_contract.py',
             'raw_conditioning_decision_receipt_check_v1.py',
             'audit_cctv_dgp_raw_conditioning_learning_v1_preparation.py']
    rows = []
    for name in files:
        source = ROOT / 'scripts' / name
        text = source.read_text(encoding='utf-8')
        text = text.replace('raw_conditioning_learning_v1', 'raw_conditioning_learning_v1_r1')
        text = text.replace('cctv-dgp-raw-conditioning-learning-v1', 'cctv-dgp-raw-conditioning-learning-v1-r1')
        text = text.replace('raw_conditioning_decision_receipt_check_v1', 'raw_conditioning_decision_receipt_check_v1_r1')
        if name.startswith('import_'):
            assert text.count(old['protocol_sha256']) == 1
            text = text.replace(old['protocol_sha256'], new['protocol_sha256'])
        if name.endswith('_preparation.py'):
            marker = "    verified=ast.literal_eval(transfer.stdout.strip())\n"
            assert text.count(marker) == 1
            text = text.replace(marker, marker +
                "    assert verified['maximum_gradient_queries']==p['maximum_gradient_queries']==640\n"
                "    assert verified['maximum_accepted_updates']==p['maximum_accepted_updates']==2\n")
        target_name = name.replace('raw_conditioning_learning_v1', 'raw_conditioning_learning_v1_r1')
        target_name = target_name.replace('raw_conditioning_decision_receipt_check_v1', 'raw_conditioning_decision_receipt_check_v1_r1')
        target = ROOT / 'scripts' / target_name
        ast.parse(text, feature_version=(3, 10))
        with target.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(text)
        rows.append(dict(source=name, source_sha256=sha(source), target=target.name, target_sha256=sha(target)))
    write(OUT / 'cctv_dgp_raw_conditioning_learning_v1_r1_checker_preparation.json', dict(
        complete=True, preparer_sha256=sha(__file__), protocol_sha256=new['protocol_sha256'], checkers=rows,
        old_checkers_retained=True, exact_protocol_and_transfer_summary_count_agreement_required=True,
        future_full_return_audit_not_run=True, no_local_models_or_training=True))
    print(dict(complete=True, prospective_checkers=5, transfer_summary_counts_bound_to_protocol=True))


if __name__ == '__main__':
    main()
