"""Close all 26 actually displayed B-failure sheets; no candidate quality claim."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_bank_comparison_v1_r2_B_analysis'
RETURN = ROOT / 'outputs/cctv_dgp_bank_comparison_v1_r2_B_return'
A_RETURN = ROOT / 'outputs/cctv_dgp_bank_comparison_v1_r3_A_return'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    receipt = OUT / 'visual_review.json'; assert not receipt.exists()
    audit = read(OUT / 'independent_audit.json')
    execution = read(OUT / 'failure_execution_audit.json')
    manifest = read(OUT / 'visual_review/sheet_manifest.json')
    assert audit['complete'] and execution['complete'] and manifest['complete']
    assert audit['saved_paired_raw_PNG_cases_checked'] == 3905 and audit['native_compositions_checked'] == 24
    assert audit['optimizer_updates_in_VM'] == execution['actual_optimizer_updates'] == 0
    assert len(manifest['sheets']) == 26 and len(manifest['missing_tiles']) == 372
    assert audit['strict_full_state_checks'] == audit['limited_CPU_replays'] == []
    observations = [
        'Ordinary glasses, hair and clear facial texture soften; all degraded central features remain blurred. No A/B candidates in this return.',
        'Clear skin and wrinkles soften. Blur, lowlight and compound eyes/mouth stay indistinct; no B result exists.',
        'Clear hair and eyes lose detail; degraded features remain diffuse and dark. Candidate columns are absent.',
        'Clear hair and teeth soften; degraded smile and eyes are smeared. No trained B output exists.',
        'Smile and fringe soften; degraded teeth/eyes remain unresolved. Candidate columns are missing.',
        'Clear eyes and mouth are softened. Dark and compound features remain indistinct; B is untested.',
        'Clear teeth and eye edges soften; blur/compound smile detail remains diffuse. No B or bank-disabled output.',
        'Ordinary hair and clear eye definition soften; degraded face stays diffuse. No candidate reconstruction.',
        'Clear eyes, hair and skin lose fine detail; dark/compound faces remain broad shapes. Candidate outputs absent.',
        'Ordinary glasses and clear teeth soften. Degraded eye and mouth detail remains limited; no B output.',
        'Mustache, skin and clear eyes soften; dark/compound features remain blurred. No B quality test.',
        'Clear mouth and eyes lose some detail; degraded mouth becomes broad. No candidate outputs.',
        'Clear eye and lip contours soften; blur and lowlight faces remain diffuse. B columns absent.',
        'Clear turned face is softened; degraded mouth and eyes remain smudged. No new restoration output.',
        'Visible hand is retained with smoothing. Eye/smile detail stays diffuse on degraded inputs; no completion result.',
        'Clear smile and teeth soften; compound central features remain unresolved. No trained B comparison.',
        'Hand/hair remain visible; clear teeth soften and degraded face remains heavily blurred. No removal or B result.',
        'Overlapping hair is retained; clear texture softens and degraded eyes/mouth stay diffuse. No completion test.',
        'Head covering remains; visible clear facial texture softens. Degraded eye/lip detail is limited; no hidden-region estimate.',
        'Clear smile texture softens; scarf/hair remain. Degraded central features stay diffuse and candidate columns are absent.',
        'DEV01/02: early dark faces smear further; later crops show softer eye/mouth edges than input. Unpaired baseline only.',
        'DEV03/04: early faces are diffuse; later detail is softened. Ordinary glasses retained; no B native output.',
        'DEV05/06: ordinary glasses remain, dark early DEV06 face is heavily diffuse; later crops lose input detail. Baseline only.',
        'DEV07/08: turned/early faces and glasses soften; later hair and face contours lose definition. No B output or reference metric.',
        'DEV09/10: reflection and glasses retained, but early face definition smears. Later eye/brow detail softens; unpaired baseline only.',
        'DEV11/12: early face detail is diffuse; later ordinary glasses, mouth and hair soften. No candidate reconstruction.'
    ]
    assert len(observations) == len(manifest['sheets'])
    sheets = []
    for row, note in zip(manifest['sheets'], observations):
        assert sha(OUT / 'visual_review' / row['name']) == row['sha256']
        sheets.append(dict(**row, viewed=True, detail='original', observation=note))
    a = read(A_RETURN / 'export_manifest.json')['files']
    b = read(RETURN / 'export_manifest.json')['files']
    names = [n for n in b if n.startswith('outputs/bank_comparison_v1/baseline/') and Path(n).suffix in {'.npz', '.npy', '.png'}]
    assert len(names) == 929
    assert all(a[n] == b[n] == sha(A_RETURN / n) == sha(RETURN / n) for n in names)
    pa, pb = [read(p / 'protocol.json') for p in [A_RETURN, RETURN]]
    fields = ['cases', 'references', 'schedule', 'native_development', 'preview_case_ids',
        'normalizers', 'active_anchor_normalizers', 'scientific_thresholds', 'optimizer', 'reserved_final_pixels']
    assert all(pa[n] == pb[n] for n in fields)
    bindings = {n: sha(OUT / n) for n in ['import.json', 'route_import.json', 'independent_audit.json',
        'failure_execution_audit.json', 'visual_review/sheet_manifest.json']}
    record = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), checker_sha256=sha(__file__),
        protocol_sha256=audit['protocol_sha256'], evidence_sha256=bindings,
        viewed_sheets=26, paired_sheets=20, native_sheets=6, paired_TRAIN_previews=100,
        native_DEV_unpaired_cases=24, sheets=sheets, missing_candidate_tiles=372,
        absent_A_reason='A outputs live in the separately audited A return',
        absent_B_reason='B stopped at initializer before its first forward/gradient/update',
        baseline_rgb_files_exactly_equal_to_A=929, matched_unchanged_input_fields=fields,
        reviewer='implementing assistant; all26 sheets displayed through view_image at original resolution',
        independent_final_review=False, B_quality_tested=False, B_candidate_outputs=0,
        local_neural_gradient_or_training_calls=0, restoration_candidate_qualified=False,
        app_promotion=False, all_seven_completion_families_pending=True,
        reserved_final_identities_used=False, goal_complete=False)
    with receipt.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print(dict(complete=True, viewed_sheets=26, baseline_files_byte_identical=929, B_quality_tested=False))


if __name__ == '__main__':
    main()
