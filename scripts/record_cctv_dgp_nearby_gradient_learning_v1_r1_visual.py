"""Record direct original-resolution inspection of every returned comparison sheet."""
from datetime import datetime,timezone
from pathlib import Path
from PIL import Image

from import_cctv_dgp_nearby_gradient_learning_v1_r1_return import ROOT,ANALYSIS,PIN,read,sha,write

NOTES={
    '036':'Transparent glasses remain visible on the clear row, but eye, mouth and skin definition is softened. All degraded columns stay blurred; the three proposals do not visibly resolve frames or facial detail beyond initial.',
    '051':'The clear target has eye creases, forehead lines and hair detail lost in the initial restoration. Trial outputs retain the same broad softness on degraded rows; useful new definition is not established.',
    '076':'Eye makeup, brow edges, lips and ordinary hair strands are defined in the target. Initial and trial outputs soften them; blur and compound rows remain indistinct across all three proposals.',
    '085':'Eye borders, separated teeth and hair are clearer in the target. The restoration columns soften these details; the proposals do not supply convincing tooth or eye definition in degraded rows.',
    '092':'The clear target retains eye creases, tooth boundaries and ordinary hair. All proposal columns closely resemble initial with soft degraded eyes and mouth; no useful whole-face gain is established.',
    '146':'Clear eye contours and tooth separation are stronger in the target. Degraded restoration columns remain soft; the trial variations do not visibly resolve those contours.',
    '169':'The visible smile, individual tooth edges and hair remain softened in restoration. Broad appearance is similar across columns, while degraded eye and tooth detail remains unresolved.',
    '357':'Visible eye, lips and face outline are present, with ordinary hair preserved. Degraded fine contours stay blurred across initial and proposals; no convincing added structure is visible.',
    '384':'Target brows, stubble and lip borders have finer definition. Initial and trial outputs smooth those details, especially in degraded rows, with no convincing trial improvement.',
    '386':'Clear transparent glasses are retained rather than removed. Frame, eye, mouth and skin detail remain softened, and all degraded proposal columns stay close to initial.',
    '391':'This target already has limited fine detail. Facial-hair and eye boundaries are further softened under degradation; none of the three proposals provides convincing extra definition.',
    '392':'The target is itself somewhat soft. Degraded eye, nose and open-mouth contours remain broad in all output columns; the proposals do not visibly sharpen the full face.',
    '393':'The clear source offers limited fine texture. Eye and lip contours remain soft under degradation and closely resemble initial throughout the proposals.',
    '394':'The target shows visible eye and open-mouth contours. Degraded rows smear them; the trial columns do not convincingly restore extra structure.',
    '395':'The hand beside the cheek is retained observed content, not a completion mask. Brows, smile and facial lines remain soft in degraded trials; no hidden-region completion is tested.',
    '480':'Tooth separation and eye creases are clearer in the target. Every degraded restoration column remains blurred, without convincing trial gain over initial.',
    '615':'The narrow observed crop and hand at the forehead remain present. Lip, tooth and eye definition stays blurred under degradation; the three proposals do not visibly improve it.',
    '651':'Ordinary hair across the forehead, lip contours and face outline are retained broadly. Fine detail is smoothed and severe degraded rows stay indistinct across proposals.',
    '701':'The observed head covering remains present. Degraded eye, nose and lip definition stays soft; the trials do not add convincing facial detail beyond initial.',
    '745':'The source target already limits fine texture. Eye and tooth boundaries remain blurred in degraded outputs, with no convincing added structure from the proposals.',
}


def main():
    audited=read(ANALYSIS/'failure_audit.json');assert audited['complete'] and audited['protocol_sha256']==PIN
    sheets=audited['visual_sheets'];assert len(sheets)==20
    assert {Path(r['file']).stem.removeprefix('reference_') for r in sheets}==set(NOTES)
    expected=['input','paired TRAIN target','initial','plus02','minus02','trial00','trial01','trial02','rollback']
    reviewed=[]
    for row in sheets:
        path=ANALYSIS/row['file'];assert sha(path)==row['sha256'] and row['columns']==expected and len(row['case_ids'])==5
        with Image.open(path) as image:assert image.size==(2304,1400)
        reviewed.append(dict(file=row['file'],sha256=row['sha256'],image_size=[2304,1400],cells=45,
            direct_original_resolution_review=True,all_five_rows_and_nine_columns_reviewed=True,
            useful_visible_gain_over_initial_established=False,notes=NOTES[path.stem.removeprefix('reference_')]))
    write(ANALYSIS/'visual_review.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),recorder_sha256=sha(__file__),
        independent_failure_audit_sha256=sha(ANALYSIS/'failure_audit.json'),protocol_sha256=PIN,reviewed_sheets=reviewed,
        all20_sheets_and900_cells_directly_viewed=True,input_target_three_sampling_three_proposal_rollback_columns_reviewed=True,
        no_compelling_useful_whole_face_gain_established=True,evidence_scope='100 exposed paired photographic TRAIN cases; no native CCTV or final identities',
        transparent_glasses_and_observed_content_reviewed=True,separate_automatic_and_assisted_completion_families_not_evaluated=True,
        unique_root_cause_or_untested_future_quality_not_inferred=True,visible_gain_qualified=False,model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,sheets=20,cells=900,visible_gain_qualified=False))


if __name__=='__main__':main()
