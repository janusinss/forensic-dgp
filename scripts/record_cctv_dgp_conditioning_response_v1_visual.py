"""Bind direct original-resolution review of all 20 returned visual sheets."""
from datetime import datetime, timezone
from pathlib import Path

from import_cctv_dgp_conditioning_response_v1_return import ANALYSIS, ROOT, PIN, read, sha, write


NOTES = {
    'b036': 'Clear transparent frames and overall outline remain visible; clear-image skin detail is softened. Blur and compound rows still lose eyes, mouth and frame definition across the signed changes.',
    'b051': 'The sharp target has distinct eye, brow, wrinkle and hair detail. Initial and signed outputs soften those details; severely degraded rows remain indistinct. No convincing new detail appears.',
    'b076': 'Visible eye makeup, mouth edges and hair strands are sharper in the target. Signed outputs closely resemble initial; degraded eyes and mouth remain broad soft regions.',
    'b085': 'Clear brows, tooth separation and hair detail are softened by the baseline. Blur and compound rows do not gain convincing eye or tooth definition in the signed comparisons.',
    'b092': 'The clear target retains eye creases, teeth and ordinary hair detail. Degraded outputs remain soft; neither sign visibly resolves facial details beyond initial.',
    'b146': 'Eye outlines and tooth edges are distinct in the target. The degraded rows remain blurred and signed changes do not visibly restore that definition.',
    'b169': 'Clear teeth and hair strands are softened in the restoration columns. Broad face appearance is similar across changes; degraded eye and tooth details stay unresolved.',
    'b357': 'The visible eye, lip line and ordinary hair remain broadly represented. Severe degraded rows lose fine contours. Signed changes show no convincing clarity gain over initial.',
    'b384': 'Target eyebrows, stubble and lip edges have definition missing from degraded outputs. All signed output columns retain the same broad softness.',
    'b386': 'Clear glasses are still visible rather than removed; clear-image facial texture is softened. Degraded frame, eye and mouth details remain soft across signed changes.',
    'b391': 'The source target is itself somewhat soft. Eyebrow and facial-hair boundaries remain softened in degraded outputs; signed changes supply no convincing extra detail.',
    'b392': 'The clear target is already low in fine detail. Blurred eye, nose and open-mouth boundaries stay broad in every signed comparison; no visible improvement over initial is established.',
    'b393': 'The source has limited fine detail even in its clear row. Signed changes do not visibly improve the degraded eye, nose or lip definition.',
    'b394': 'The clear eye and open-mouth outline are visible in the target; degraded rows smear them. Signed changes remain close to initial with no convincing detail recovery.',
    'b395': 'The hand beside the cheek is part of the observed input, not a completion mask in this test. Brow, smile and fine facial lines remain soft; no useful new structure is evident.',
    'b480': 'Tooth separation and eye creases are stronger in the target. Degraded restoration columns remain blurred, with no convincing added definition from the signed changes.',
    'b615': 'The forehead hand and narrow observed crop are retained. Tooth, lip and eye definition remain blurred on degraded rows; the signed response does not visibly sharpen them.',
    'b651': 'Ordinary hair across the forehead, lips and face outline remain broadly present. Fine skin and hair detail stays softened, and severe rows remain indistinct across changes.',
    'b701': 'The observed head covering is preserved in this restoration-only comparison. Degraded eye, nose and lip contours remain soft; signed changes add no convincing facial detail.',
    'b745': 'The source target already limits fine texture. Eye and tooth definition stays blurred on degraded rows; no signed change visibly improves the full face beyond initial.',
}


def main():
    audited = read(ANALYSIS / 'independent_audit.json')
    assert audited['complete'] and audited['protocol_sha256'] == PIN
    sheets = audited['all_planned_visual_sheets']
    assert len(sheets) == 20 and {Path(r['file']).stem for r in sheets} == set(NOTES)
    reviewed = []
    for row in sheets:
        path = ROOT / row['file']
        assert sha(path) == row['sha256'] and row['cells'] == 50
        reviewed.append(dict(file=row['file'],sha256=row['sha256'],cells=50,
                             direct_original_resolution_review=True, image_size=[2560,1400],
                             all_five_rows_and_ten_columns_reviewed=True,
                             visible_useful_gain_over_initial_established=False,notes=NOTES[path.stem]))
    write(ANALYSIS / 'visual_review.json', dict(
        complete=True,UTC=datetime.now(timezone.utc).isoformat(), recorder_sha256=sha(__file__),
        independent_audit_sha256=sha(ANALYSIS/'independent_audit.json'),protocol_sha256=PIN,
        reviewed_sheets=reviewed,all20_sheets_and1000_cells_directly_viewed=True,
        input_target_initial_six_signed_and_rollback_columns_reviewed=True,
        evidence_scope='100 already exposed paired photographic TRAIN cases; no native CCTV or final evaluation',
        restored_fine_detail_or_hidden_identity_not_proved=True, visible_gain_qualified=False,
        separate_completion_families_not_evaluated=True,model_qualified=False,
        more_epochs_qualified=False,goal_complete=False,
    ))
    print(dict(complete=True,reviewed_sheets=20,reviewed_cells=1000,visible_gain_qualified=False))


if __name__ == '__main__':
    main()
