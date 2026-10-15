"""Close the actually viewed A-return sheets after the independent audit finishes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_bank_comparison_v1_r3_A_analysis'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    receipt = OUT / 'visual_review.json'
    assert not receipt.exists()
    audit = read(OUT / 'independent_audit.json')
    execution = read(OUT / 'execution_audit.json')
    observations = read(OUT / 'visual_observations.json')
    manifest = read(OUT / 'visual_review/sheet_manifest.json')
    previews = read(OUT / 'preview_integrity_audit.json')
    imported = read(OUT / 'import.json')
    assert all(r['complete'] for r in [audit, execution, observations, manifest, previews, imported])
    assert audit['saved_paired_raw_PNG_cases_checked'] == 7810
    assert audit['native_compositions_checked'] == 48
    assert audit['optimizer_updates_in_VM'] == execution['actual_optimizer_updates'] == 50
    assert audit['local_optimizer_updates'] == audit['local_gradient_queries'] == audit['local_backwards'] == 0
    assert not audit['model_qualified'] and not observations['restoration_candidate_qualified']
    assert audit['manifest_sha256'] == imported['export_manifest_sha256']
    assert len(audit['limited_CPU_replays']) == 2
    assert len(audit['strict_full_state_checks']) == 1
    assert all(r['strict_model_state'] and r['frozen_tensors_unchanged'] for r in audit['strict_full_state_checks'])
    assert sha(OUT / 'visual_review/sheet_manifest.json') == observations['sheet_manifest_sha256']
    assert len(manifest['sheets']) == len(observations['sheets']) == observations['viewed_sheets'] == 26
    assert not observations['independent_final_review']
    actual = {r['name']: r['sha256'] for r in manifest['sheets']}
    assert set(actual) == {r['name'] for r in observations['sheets']}
    for row in observations['sheets']:
        assert row['viewed'] and row['detail'] == 'original' and row['observation']
        assert row['sha256'] == actual[row['name']] == sha(OUT / 'visual_review' / row['name'])
    bindings = {name: sha(OUT / name) for name in [
        'import.json', 'route_import.json', 'independent_audit.json', 'execution_audit.json',
        'preview_integrity_audit.json', 'visual_observations.json', 'visual_review/sheet_manifest.json']}
    record = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=audit['protocol_sha256'], evidence_sha256=bindings,
        viewed_sheets=26, paired_sheets=20, native_sheets=6,
        paired_TRAIN_preview_cases=100, native_DEV_unpaired_cases=24,
        reviewer=observations['reviewer'], independent_final_review=False,
        full_metric_audit_complete=True, observations_retained_without_rewriting=True,
        missing_B_tiles=observations['missing_tiles'], B_not_run=True,
        raw_and_PNG_quality_rejection_retained=True, restoration_candidate_qualified=False,
        app_promotion=False, all_seven_completion_families_pending=True,
        final_evaluation_identities_used=False, goal_complete=False)
    with receipt.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(dict(complete=True, viewed_sheets=26, full_metric_audit_complete=True,
               model_qualified=False, independent_final_review=False))


if __name__ == '__main__':
    main()
