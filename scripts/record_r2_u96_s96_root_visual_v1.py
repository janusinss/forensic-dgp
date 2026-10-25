"""Record root's completed coded visual inspection; no inference."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/codeformer_r2_u96_s96_objective_v1'

def pin(p):
    b=p.read_bytes()
    return {'path':p.relative_to(ROOT).as_posix(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

def main():
    manifest=json.loads((OUT/'visuals/manifest.json').read_bytes())
    paired=OUT/'paired_visual_readback_v1/root_coded_paired_visual_review_v1.json'
    prior=json.loads(paired.read_bytes())
    assert prior['root_personal_review_complete_for_paired36']
    for current,reviewed in zip(manifest['sheets'][:9],prior['sheets'],strict=True):
        actual=pin(ROOT/current['file']['path'])
        assert actual==current['file']
        assert actual['sha256']==reviewed['sha256'] and actual['bytes']==reviewed['bytes']
    for row in manifest['sheets'][9:]:
        assert pin(ROOT/row['file']['path'])==row['file']
    record={'complete':True,'reviewer':'root author; not independent human study',
        'review_mode':'Coded A/B/C before reading key; full paired scores seen before remaining native sheets',
        'paired_review_receipt':pin(paired),'paired_final_sheets_byte_identical_to_reviewed_previews':True,
        'sheets':manifest['sheets'],'personally_reviewed_sheets':17,'input_rows':68,'model_PNG_cells':204,
        'paired_findings':prior['observations'],
        'native_coded_findings':[
            'Sheets10-11: A/C are very close; B slightly changes tone/contrast. Shared generated eyes, mouth and hair; no large structural advantage. Clear glasses remain soft in darker frames.',
            'Sheets12-13: Generated eye detail, glasses geometry and smooth skin remain shared. B has a subtle tone difference, not a clearly stronger reconstruction. A/C visually near-identical.',
            'Sheets14-15: Clear-glasses rows retain glare and soft eye detail across all arms. Brighter face rendering is shared; B is modestly darker in several rows. No convincing native structural winner.',
            'Sheets16-17: Coarse QMUL probes produce plausible but weakly supported eyes/mouth or tiny compressed facial features on a broad face. One side-view row retains a poorly formed central feature. All three share these defects. Input-only uncertainty remains in force.',
            'Native images have no aligned target. These observations cannot establish recovered identity or objective native target accuracy.'
        ],
        'paired_visual_structural_winner_established':False,'native_quality_winner_established':False,
        'independent_blinded_study':False,'model_qualified':False,'main_app_default_changed':False,
        'new_neural_training_VM_calls':0}
    output=OUT/'root_visual_review_v1.json'
    with output.open('x',encoding='utf-8') as f:json.dump(record,f,indent=2);f.write('\n')
    print(json.dumps(pin(output)))

if __name__=='__main__':main()
