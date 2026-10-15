"""Recount delivered pixel changes from saved PNGs, without processing images."""
from datetime import datetime, timezone
import numpy as np
from PIL import Image
from import_cctv_dgp_conditioning_response_v1_return import ANALYSIS, PIN, RETURN, read, sha, write


def main():
    audit=read(ANALYSIS/'independent_audit.json')
    assert audit['complete'] and audit['protocol_sha256']==PIN and audit['actual_raw_PNG_state_protected_pixel_and_saved_embedding_replays']
    p=read(RETURN/'protocol.json');out=RETURN/'outputs/conditioning_response_v1';points=[]
    for step in p['step_plan']:
        rows=[]
        for cid in p['case_ids']:
            a_path=out/'initial/png'/(cid+'.png');b_path=out/step['label']/'png'/(cid+'.png')
            with Image.open(a_path) as a,Image.open(b_path) as b:
                assert a.mode==b.mode=='RGB' and a.size==b.size==(256,256)
                delta=np.abs(np.asarray(b).astype(np.int16)-np.asarray(a).astype(np.int16))
            rows.append(dict(id=cid,initial_png_sha256=sha(a_path),signed_png_sha256=sha(b_path),
                             changed_channels=int(np.count_nonzero(delta)),changed_pixels=int(np.count_nonzero(delta.any(2))),
                             maximum_byte_difference=int(delta.max())))
        points.append(dict(label=step['label'],cases=100,cases_with_changed_pixels=sum(r['changed_pixels']>0 for r in rows),
                           total_changed_RGB_channels=sum(r['changed_channels'] for r in rows),
                           maximum_byte_difference=max(r['maximum_byte_difference'] for r in rows),case_records=rows))
    write(ANALYSIS/'PNG_change_review.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
        checker_sha256=sha(__file__),independent_audit_sha256=sha(ANALYSIS/'independent_audit.json'),protocol_sha256=PIN,
        signed_point_recounts=points,saved_pixels_only=True,no_resaving_or_processing=True,
        scientific_export_and_quality_gates_unchanged=True,local_neural_or_training_calls=0,
        change_counts_do_not_establish_visible_or_identity_gain=True,model_qualified=False,goal_complete=False))
    print(dict(complete=True,points=[{k:v for k,v in r.items() if k!='case_records'} for r in points]))


if __name__=='__main__':main()
