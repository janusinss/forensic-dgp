"""Audit frozen reference-gate evidence without additional detector/restorer forwards."""
from pathlib import Path
import sys
import collections

import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from run_cctv_native_comparison import read,write,sha
from cctv_camera_stress import reference_canvas
from qualify_cctv_dgp_pilot_pool import admit,EXCEPTIONS
from qualify_cctv_dgp_pilot_pool_v2 import support_reasons


def main():
    old=ROOT/"outputs/cctv_dgp_reference_gate_v1";out=ROOT/"outputs/cctv_dgp_reference_gate_v2"
    a,b=read(old/"results.json"),read(out/"results.json")
    p,q=read(old/"frozen_protocol.json"),read(out/"frozen_protocol.json")
    if a["protocol_sha256"]!=sha(old/"frozen_protocol.json") or b["protocol_sha256"]!=sha(out/"frozen_protocol.json") or q["parent_results_sha256"]!=sha(old/"results.json"):
        raise ValueError("Reference-gate bindings differ")
    for protocol in (p,q):
        for name,pin in protocol["assets_sha256"].items():
            if sha(ROOT/name)!=pin:
                raise ValueError("Input-gate asset changed")
    if len(a["rows"])!=1150 or len(b["rows"])!=1150 or len({r["sha256"] for r in b["rows"]})!=1150:
        raise ValueError("Deduplicated cohort membership differs")
    from insightface.utils.face_align import estimate_norm
    groups={};counts={}
    for row,updated in zip(a["rows"],b["rows"]):
        if row["source_file"]!=updated["source_file"] or row["sha256"]!=updated["sha256"] or sha(ROOT/row["source_file"])!=row["sha256"]:
            raise ValueError("Source identity/bytes differ")
        rgb,mask,bounds=reference_canvas(ROOT/row["source_file"])
        if bounds!=row["bounds"]:
            raise ValueError("Observed bounds differ")
        points=np.asarray(row["landmarks5"],dtype=np.float32) if row["landmarks5"] is not None else None
        matrix=np.asarray(row["matrix112"],dtype=np.float64) if row["matrix112"] is not None else None
        if matrix is not None:
            np.testing.assert_allclose(matrix,estimate_norm(points[0],image_size=112),rtol=0,atol=1e-12)
        reasons,measures=admit(points,row["bboxes"],mask,matrix)
        if EXCEPTIONS.get(row["source_file"]):
            reasons.append("reviewed_input_exception")
        if reasons!=row["reasons"] or row["admitted"]!=(not reasons) or measures!=row["geometry_diagnostics"]:
            raise ValueError("Original geometric admission fails reconstruction")
        added,support=support_reasons(row,mask)
        new=[r for r in reasons if r!="reference_crop_padding"]+added
        if updated["reasons"]!=new or updated["admitted"]!=(not new) or updated["observed_feature_support"]!=support:
            raise ValueError("V2 observed-feature support fails reconstruction")
        key=row["role"]+"/"+row["source"]
        group=groups.setdefault(key,{"candidates":0,"admitted":0,"reasons":collections.Counter()})
        group["candidates"]+=1;group["admitted"]+=int(updated["admitted"]);group["reasons"].update(new)
    if groups!=b["groups"]:
        raise ValueError("Reference source-group counts differ")
    for base,data in ((old,a),(out,b)):
        for name,pin in data["artifacts_sha256"].items():
            if sha(base/name)!=pin:
                raise ValueError("Reference preview bytes differ")
    if a["reference_detection_requests"]!=1150 or a["optimizer_updates"] or b["optimizer_updates"] or b["model_forwards"]:
        raise ValueError("Input-only/finite inference evidence differs")
    report={"complete":True,"v1_results_sha256":sha(old/"results.json"),"v2_results_sha256":sha(out/"results.json"),
            "source_files_and_saved_matrices_checked":1150,"groups":groups,"model_forwards":0,"optimizer_updates":0,
            "training_ready":False,"limits":"Reconstructs recorded input-geometric support, not independent detector correctness or all-cohort pristine/uncovered quality."}
    write(out/"independent_audit.json",report)
    print({"complete":True,"references_checked":1150,"audit_sha256":sha(out/"independent_audit.json")})


if __name__=="__main__":
    main()
