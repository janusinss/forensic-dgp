"""Preserve R2 checkers and bind equivalent prospective checkers to fresh R3."""
import ast
from import_cctv_dgp_conditioning_learning_v1_r2_return import ROOT, OUT, PIN, read, sha, write


def main():
    prep = read(OUT / 'cctv_dgp_conditioning_learning_v1_r3_preparation.json')
    assert prep['complete']
    names = [
        'import_cctv_dgp_conditioning_learning_v1_r2_return.py',
        'audit_cctv_dgp_conditioning_learning_v1_r2_return.py',
        'audit_cctv_dgp_conditioning_learning_v1_r2_preparation.py',
        'cctv_dgp_conditioning_learning_v1_r2_contract.py']
    records = []
    for name in names:
        source = ROOT / 'scripts' / name
        original = source.read_text(encoding='utf-8')
        text = original.replace('conditioning_learning_v1_r2', 'conditioning_learning_v1_r3')
        text = text.replace('cctv-dgp-conditioning-learning-v1-r2', 'cctv-dgp-conditioning-learning-v1-r3')
        if name.startswith('import_'):
            assert text.count(PIN) == 1
            text = text.replace(PIN, prep['protocol_sha256'])
        if name.startswith('audit_') and name.endswith('_preparation.py'):
            # Supplement the retained audit with the new actual utility regression.
            marker = "    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']"
            assert text.count(marker) == 1
            text = text.replace(marker, "    from audit_cctv_dgp_conditioning_learning_v1_r3_startup_repair import audit as startup_audit\n    startup_proof=startup_audit(packet,p)\n" + marker)
            marker2 = '        LF_ASCII_checksum_exact=True,parent_assets_verified=5852'
            assert text.count(marker2) == 1
            text = text.replace(marker2, '        startup_shape_and_exact_unchanged_recipe_audit=startup_proof,\n' + marker2)
        ast.parse(text, feature_version=(3, 10))
        target = ROOT / 'scripts' / name.replace('conditioning_learning_v1_r2', 'conditioning_learning_v1_r3')
        with target.open('x', encoding='utf-8', newline='\n') as f:
            f.write(text)
        records.append(dict(source=name, source_sha256=sha(source), target=target.name, target_sha256=sha(target)))
    write(OUT / 'cctv_dgp_conditioning_learning_v1_r3_checker_preparation.json', dict(
        complete=True, preparer_sha256=sha(__file__), protocol_sha256=prep['protocol_sha256'],
        original_checkers_retained=True, prospective_checkers=records, no_model_calls=True,
        future_full_return_audit_not_run=True))
    print(dict(complete=True, prospective_checkers=len(records), manual_training_unrun=True))


if __name__ == '__main__':
    main()
