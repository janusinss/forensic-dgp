"""Record the actual ten-sheet review after the separate saved-output audit."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_v29_paired_development_v1'
NOTES = {
    'va_asian_00209': ('Clear face keeps broad eyes, nose, mouth and outline with smoothing.', 'Degraded eyes become darker and narrower; nose and lip boundaries change. Sharper coarse edges do not establish preserved expression or gaze.'),
    'va_asian_00222': ('Clear grayscale face keeps wrinkles and open smile broadly present, with smoothing.', 'Blur, low-light and motion outputs have stronger nose/mouth boundaries, but eye apertures and smile/teeth remain uncertain. Compound output stays diffuse.'),
    'va_asian_00302': ('Clear smile and eye/nose arrangement remain broadly present.', 'Blur and low-light outputs narrow the eyes and close or reshape the visible smile. Motion is closer in broad arrangement; compound remains diffuse.'),
    'va_asian_00563': ('Clear eyes, nose and closed mouth remain broadly readable; the black support boundary is preserved.', 'Degraded eyes narrow into dark slits and the mouth changes shape. Coarse contrast increases while the paired visible expression is not retained reliably.'),
    'va_asian_00700': ('Clear eyes, nose, mouth and outline remain broadly present.', 'Degraded outputs narrow the left-visible eye and alter mouth boundaries. Motion retains more coarse layout; severe profiles do not recover the paired eye apertures.'),
    'va_ffhq_00383': ('Clear face, hair ornament, eyes and mouth survive, with smoothing.', 'Degraded outputs show sharper coarse eyes/nose/mouth than original DGP, while eyes narrow and lip/expression shape changes. Fine visible detail remains uncertain.'),
    'va_ffhq_01093': ('Clear face keeps broad eyes, nostrils, teeth, smile and outline.', 'Degraded eyes and mouth get stronger edges, but eyelid openness, smile and teeth differ from the paired face. Compound remains soft.'),
    'va_ffhq_08025': ('Clear makeup, headwear and broad facial arrangement remain present.', 'Degraded outputs strengthen coarse eye/nose/mouth edges, while eye makeup, lip colour/boundary and expression differ from the paired target. Compound stays diffuse.'),
    'va_ffhq_09056': ('Clear red glasses and eye/nose/mouth layout remain, with smoothing.', 'Degraded glasses become incomplete or diffuse and the eyes and nose shift in appearance. Sharper edges do not restore the visible glasses/eye structure reliably.'),
    'va_ffhq_10401': ('Clear eye/nose/lip arrangement, hair and outline remain, with smoothing.', 'Blur, low-light and motion outputs have clearer coarse boundaries than original DGP but change eyelid and mouth appearance. Compound stays diffuse.'),
}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    r = json.loads((OUT / 'results.json').read_text())
    a = json.loads((OUT / 'saved_output_audit.json').read_text())
    assert a['complete'] and a['cases_recomputed'] == 520
    assert len(r['diagnostic_preservation_failures']) == 21
    reviewed = [row for row in r['rows'] if row['reference_id'] in NOTES]
    assert len(reviewed) == 50 and len(r['sheets']) == 10
    value = {
        'complete': True, 'date': '2026-10-07', 'reviewer': 'Codex primary assistant',
        'method': 'Actually viewed all ten original-detail1072x1516 sheets, every50 preview face and200 unchanged256x256 cells. Quantitative audit covers all520, not a visual claim about all520.',
        'preview_faces_reviewed': 50, 'quantitatively_audited_cases': 520,
        'rows': [{'id': row['id'], 'reference_id': row['reference_id'],
                  'regions_reviewed': ['eyes', 'nose', 'mouth', 'face outline', 'visible appearance'],
                  'note': NOTES[row['reference_id']][row['profile'] != 'clear']}
                 for row in reviewed],
        'results_sha256': sha(OUT / 'results.json'),
        'saved_output_audit_sha256': sha(OUT / 'saved_output_audit.json'),
        'sheets_sha256': {s['file']: sha(OUT / s['file']) for s in r['sheets']},
        'recorder_sha256': sha(Path(__file__)),
        'conclusion': 'Sharper coarse edges coexist with eye, mouth and expression changes. V29 TRAIN capacity does not generalize acceptably:21 fixed group/metric regressions,17 of them ArcFace, with worse landmark structure in the asian_faces source. Native24 also lacks convincing useful added clarity. Retain final800 as development evidence; no app promotion.',
        'structure_gain_fraction': r['degraded_structure_gain_fraction'],
        'preservation_failures_retained': 21,
        'synthetic_paired_evidence': True, 'not_ethnicity_or_identification_accuracy': True,
        'input_usability_not_reclassified_by_output': True,
        'reserved_final_used': False, 'independent_final_review': False,
        'app_promotion': False, 'restoration_qualified': False, 'goal_complete': False,
    }
    with (OUT / 'visual_review.json').open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'complete': True, 'preview_faces_reviewed': 50, 'all520_audited': True, 'app_promotion': False}))


if __name__ == '__main__':
    main()
