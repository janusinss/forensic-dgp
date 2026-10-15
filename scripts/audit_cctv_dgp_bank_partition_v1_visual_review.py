"""Independent source-pixel and actual-observation closure for all26 sheets."""
from datetime import datetime, timezone
import numpy as np
from PIL import Image
from review_cctv_dgp_bank_partition_v1 import ROOT, RETURN, PARENT, ANALYSIS, DEST, read, sha, write


def main():
    receipt=ANALYSIS/'visual_review.json'; assert not receipt.exists()
    manifest=DEST/'sheet_manifest.json'; plan=read(manifest); p=read(RETURN/'protocol.json')
    assert read(ANALYSIS/'independent_audit.json')['complete']
    assert plan['paired_sheets']==20 and plan['native_sheets']==6 and plan['cells']==720
    assert len(plan['sheets'])==26
    observations=ANALYSIS/'visual_observations'
    assert {f.name for f in observations.iterdir()}=={s['name']+'.json' for s in plan['sheets']}
    bindings={}; cells=0; paired_ids=[]; native_ids=[]
    for sheet in plan['sheets']:
        path=DEST/sheet['name']; assert sha(path)==sheet['sha256']
        observation=observations/(sheet['name']+'.json'); row=read(observation)
        assert row['complete'] and row['sheet_sha256']==sha(path) and row['sheet_manifest_sha256']==sha(manifest)
        assert row['displayed_through']=='view_image at original resolution' and not row['independent_final_review'] and len(row['notes'])>=40
        bindings[observation.relative_to(ROOT).as_posix()]=sha(observation)
        paired=sheet['role']=='paired photographic TRAIN'; columns,rows=(6,5) if paired else (5,4)
        assert len(sheet['cells'])==columns*rows
        assert {(c['row'],c['column']) for c in sheet['cells']}=={(r,c) for r in range(rows) for c in range(columns)}
        with Image.open(path) as im:
            assert im.size==(columns*256,rows*278); pixels=np.asarray(im.convert('RGB')).copy()
        for c in sheet['cells']:
            source=ROOT/c['path']; assert sha(source)==c['sha256']
            assert c['x']==c['column']*256 and c['y']==c['row']*278+22
            with Image.open(source) as im:
                assert im.size==(256,256); expected=np.asarray(im.convert('RGB')).copy()
            assert np.array_equal(pixels[c['y']:c['y']+256,c['x']:c['x']+256],expected); cells+=1
        (paired_ids if paired else native_ids).extend(sheet['case_ids'])
    assert len(paired_ids)==len(set(paired_ids))==100 and set(paired_ids)==set(p['case_ids'])
    assert native_ids==p['native_case_ids']
    protected=read(ROOT/'outputs/cctv_dgp_bank_path_ablation_v1_analysis/local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==expected for n,expected in protected.items())
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        independent_audit_sha256=sha(ANALYSIS/'independent_audit.json'),sheet_manifest_sha256=sha(manifest),
        viewed_sheets=26,sheet_cells_independently_verified=cells,observation_sha256=bindings,
        implementing_assistant_review=True,independent_final_review=False,
        fixed_weight_cross_composition_not_trained_architecture_comparison=True,
        native_cases_unpaired=24,input_only_usability_labels_not_changed=True,
        final_identities_unused=True,local_gradient_or_optimizer_calls=0,
        model_qualified=False,app_promotion=False,all_seven_completion_families_pending=True,goal_complete=False))
    print(dict(complete=True,viewed_sheets=26,exact_cells=cells,independent_final_review=False))


if __name__=='__main__': main()
