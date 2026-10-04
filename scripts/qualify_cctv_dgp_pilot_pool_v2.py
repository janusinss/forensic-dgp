"""Replace a disproved context gate using saved input geometry; zero forwards."""
import collections
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_cctv_native_comparison import read, write, sha
from cctv_camera_stress import reference_canvas
from qualify_cctv_dgp_pilot_pool import admit, EXCEPTIONS, RULES

OLD = ROOT/"outputs/cctv_dgp_reference_gate_v1"
OUT = ROOT/"outputs/cctv_dgp_reference_gate_v2"
OLD_PIN = "7344d5913f392af5018a81b9f4793af78ecd9f43902d9fd1d89f21cb505eca5c"
SUPPORT = {"minimum_context_observed_fraction": .5,
           "canonical_feature_core_xyxy": [28,43,84,101], "minimum_feature_core_observed_fraction": .95,
           "observed_landmark_radius_256": 2, "minimum_each_landmark_patch_observed_fraction": .95}


def support_reasons(row, observed):
    if len(row["bboxes"]) != 1 or row["matrix112"] is None or row["landmarks5"] is None:
        return ["reference_support_unavailable"], {}
    matrix = np.asarray(row["matrix112"], dtype=np.float64)
    mask = cv2.warpAffine(observed.astype(np.uint8), matrix, (112,112), flags=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    x0,y0,x1,y1 = SUPPORT["canonical_feature_core_xyxy"]
    full, core = float(mask.mean()), float(mask[y0:y1,x0:x1].mean())
    radii = SUPPORT["observed_landmark_radius_256"]
    patches = []
    padded = np.pad(observed.astype(np.uint8), radii, constant_values=0)
    for x,y in np.round(np.asarray(row["landmarks5"])[0]).astype(int):
        patches.append(float(padded[y:y+2*radii+1,x:x+2*radii+1].mean()) if 0<=x<256 and 0<=y<256 else 0.)
    reasons = []
    if full < SUPPORT["minimum_context_observed_fraction"]:
        reasons.append("reference_context_support")
    if core < SUPPORT["minimum_feature_core_observed_fraction"]:
        reasons.append("reference_feature_core_support")
    if min(patches) < SUPPORT["minimum_each_landmark_patch_observed_fraction"]:
        reasons.append("reference_landmark_support")
    return reasons, {"observed112_fraction": full, "feature_core_observed_fraction": core,
                     "landmark_patch_observed_fractions": patches}


def main():
    if OUT.exists():
        raise ValueError("Preserve completed/partial V2")
    if sha(OLD/"results.json") != OLD_PIN:
        raise ValueError("Input geometry source differs")
    old, old_protocol = read(OLD/"results.json"), read(OLD/"frozen_protocol.json")
    if not old["complete"] or old["protocol_sha256"] != sha(OLD/"frozen_protocol.json"):
        raise ValueError("Incomplete/unbound reference geometry")
    protocol = {"format":"dgp-cctv-reference-input-gate-v2","date":"2026-10-03",
                "parent_results_sha256":OLD_PIN,"parent_protocol_sha256":sha(OLD/"frozen_protocol.json"),
                "support_diagnostic_sha256":sha(OLD/"support_diagnostic.json"),
                "assets_sha256":{"scripts/qualify_cctv_dgp_pilot_pool_v2.py":sha(Path(__file__)),
                                  "scripts/qualify_cctv_dgp_pilot_pool.py":sha(ROOT/"scripts/qualify_cctv_dgp_pilot_pool.py"),
                                  "scripts/cctv_camera_stress.py":sha(ROOT/"scripts/cctv_camera_stress.py")},
                "inherited_geometric_rules":{k:v for k,v in RULES.items() if k!="minimum_observed_fraction_in112_crop"},
                "observed_feature_support_rules":SUPPORT,"input_review_exceptions":EXCEPTIONS,
                "change_reason":"V1 incorrectly required90% observed context in the112 recognizer crop. Three inspected tight crops have all feature-core pixels observed but only80-90% contextual coverage. Require measured core/landmark support; mask unsupported context identically in reference/generated identity crops.",
                "identity_input_policy":"Fixed target matrix, float bilinear sampling; warp the target observation mask nearest; fill all unsupported112 pixels withRGB128/255 for every arm/reference. Report ArcFace_observed_fixed as a new development metric, not equivalent to earlier unmasked metrics or identity accuracy.",
                "frozen_before_new_training":True,"model_forwards":0,"optimizer_updates":0,"training_ready":False,
                "reserved_native_cctv_used":False}
    for name,pin in old_protocol["assets_sha256"].items():
        if sha(ROOT/name)!=pin:
            raise ValueError("Parent asset differs")
    OUT.mkdir()
    write(OUT/"frozen_protocol.json",protocol)
    rows=[]
    for row in old["rows"]:
        target,observed,bounds=reference_canvas(ROOT/row["source_file"])
        if sha(ROOT/row["source_file"])!=row["sha256"] or bounds!=row["bounds"]:
            raise ValueError("Parent source/geometry differs")
        points=np.asarray(row["landmarks5"],dtype=np.float32) if row["landmarks5"] is not None else None
        matrix=np.asarray(row["matrix112"],dtype=np.float64) if row["matrix112"] is not None else None
        previous,_=admit(points,row["bboxes"],observed,matrix)
        if row["input_review_exception"]:
            previous.append("reviewed_input_exception")
        if previous!=row["reasons"]:
            raise ValueError("Parent reasons fail reconstruction")
        reasons=[r for r in previous if r!="reference_crop_padding"]
        added,support=support_reasons(row,observed)
        reasons.extend(added)
        rows.append({**row,"parent_admitted":row["admitted"],"parent_reasons":previous,
                     "admitted":not reasons,"reasons":reasons,"observed_feature_support":support})
    groups={}
    for row in rows:
        key=row["role"]+"/"+row["source"]
        group=groups.setdefault(key,{"candidates":0,"admitted":0,"reasons":collections.Counter()})
        group["candidates"]+=1; group["admitted"]+=int(row["admitted"]);group["reasons"].update(row["reasons"])
    artifacts={}
    for key in groups:
        role_rows=[r for r in rows if r["role"]+"/"+r["source"]==key]
        selected=[r for r in role_rows if r["admitted"]][:8]+[r for r in role_rows if not r["admitted"]][:8]
        sheet=Image.new("RGB",(4*164,4*198),(238,238,238));draw=ImageDraw.Draw(sheet)
        for i,row in enumerate(selected):
            target,_,_=reference_canvas(ROOT/row["source_file"])
            x,y=(i%4)*164,(i//4)*198
            draw.text((x+2,y+2),"ADMIT" if row["admitted"] else "DIAGNOSTIC",fill="black")
            draw.text((x+2,y+15),Path(row["source_file"]).name[:23],fill="black")
            sheet.paste(Image.fromarray(target).resize((160,160),Image.Resampling.BILINEAR),(x,y+34))
        name="preview_"+key.replace("/","_")+".png";sheet.save(OUT/name);artifacts[name]=sha(OUT/name)
    write(OUT/"results.json",{"complete":True,"protocol_sha256":sha(OUT/"frozen_protocol.json"),"rows":rows,
          "groups":groups,"artifacts_sha256":artifacts,"model_forwards":0,"optimizer_updates":0,
          "training_ready":False,"reserved_native_cctv_used":False,
          "limits":"Input support/pose proxies and known manual exceptions only, not all-cohort clean/uncovered/eyes-open labels. Preserve poor-reference quality warnings and original rejected gate."})
    print(json.dumps({"complete":True,"groups":groups,"results_sha256":sha(OUT/"results.json"),"model_forwards":0}))


if __name__=="__main__":
    main()
