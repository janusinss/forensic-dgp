"""Independent saved-pixel audit of broad-family inference; no model loading."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/broad_face_workflow_outputs_v1"
PROTOCOL = ROOT / "outputs/broad_covering_gallery_v1/frozen_protocol.json"
PROTOCOL_SHA = "b7a1a447482d650bb0a18b13d649f0c548c3c59acd6db202111169599c805bc9"
RESULT_SHA = "afe8cb6b2d768460082890fe8ded09dd125e19f599f54f55a496e2a4073a94d3"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array(path, mode="RGB"):
    with Image.open(path) as image:
        return np.asarray(image.convert(mode)).copy()


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    destination = OUT/"independent_verification.json"
    require(not destination.exists(), "Preserve previous audit")
    require(sha(PROTOCOL)==PROTOCOL_SHA and sha(OUT/"results.json")==RESULT_SHA, "Frozen evidence differs")
    p=json.loads(PROTOCOL.read_text(encoding="utf-8"))
    r=json.loads((OUT/"results.json").read_text(encoding="utf-8"))
    cases={case["id"]:case for case in p["cases"]}
    expected={(case_id,arm) for case_id in cases for arm in ("automatic","assisted")}
    require(r["complete"] and not r["failures"] and len(r["rows"])==32, "Incomplete run")
    require({(row["id"],row["arm"]) for row in r["rows"]}==expected,"Row set differs")
    require(r["state_before"]==r["state_after"] and r["model_states_unchanged"],"Recorded model state changed")
    for case in cases.values():
        for key in ("source","input","reference","proposal"):
            require(sha(ROOT/case[key])==case[key+"_sha256"],"Source hash differs")
    for name,digest in {**r["masks_sha256"],**r["previews_sha256"]}.items():
        require(sha(OUT/name)==digest,"Saved mask/preview changed")
    aggregates, outputs, off_changes, assisted_wins, assisted_rejections, automatic_misses = {},0,0,0,0,0
    for row in r["rows"]:
        case=cases[row["id"]]
        original=array(ROOT/case["input"])
        reference=array(ROOT/case["reference"])
        reviewed=array(ROOT/case["proposal"],"L")==255
        mask_path=(OUT if row["mask_root"]=="output" else ROOT)/row["mask"]
        raw=array(mask_path,"L")
        require(np.isin(raw,(0,255)).all() and raw.shape==original.shape[:2],"Nonbinary/mismatched mask")
        active=raw==255
        if row["output"] is None:
            require(case["expected_rejection"] and row["suppressed_by_source_review"],"Unaccounted missing output")
            if row["arm"]=="assisted":
                require(row["guard_rejected"] and "less-covered" in row["guard_error"],"Missing assisted guard")
                assisted_rejections+=1
            elif not row["guard_rejected"]:
                automatic_misses+=1
            continue
        outputs+=1
        require(sha(OUT/row["output"])==row["output_sha256"],"Output hash changed")
        output=array(OUT/row["output"])
        require(output.shape==original.shape,"Output dimensions changed")
        meta=row["metadata"]
        require(hashlib.sha256(original.tobytes()).hexdigest()==meta["original_rgb_sha256"],"Decoded input differs")
        require(hashlib.sha256(active.astype(np.uint8).tobytes()).hexdigest()==meta["removal_mask_sha256"],"Mask not used exactly")
        require(hashlib.sha256(output.tobytes()).hexdigest()==meta["output_rgb_sha256"],"Decoded output differs")
        require(meta["additional_mask_expansion"]==0 and meta["optimizer_updates"]==0,"Mask expanded or training recorded")
        changed=int(np.any(output!=original,axis=-1)[~active].sum())
        require(changed==row["changed_pixels_outside_active_mask"],"Changed-pixel count differs")
        if not meta["restoration_applied"]:
            require(changed==0,"Off route changed visible pixels")
            off_changes+=changed
        elif not case["synthetically_degraded"]:
            require(row["arm"]=="automatic" and case["id"].startswith("val_244"),"Unexpected native restoration recorded")
        visible=float((np.abs(output.astype(float)-reference.astype(float))[~reviewed]/255).mean())
        baseline=float((np.abs(original.astype(float)-reference.astype(float))[~reviewed]/255).mean())
        require(abs(visible-row["visible_mae_outside_fixed_operator_proposal"])<1e-12,"Visible metric differs")
        require(abs(baseline-row["off_input_visible_mae"])<1e-12,"Off metric differs")
        group=("degraded" if case["synthetically_degraded"] else "native")+"/"+row["arm"]
        aggregates.setdefault(group,[]).append((visible,baseline))
        if case["synthetically_degraded"] and row["arm"]=="assisted" and visible<baseline:
            assisted_wins+=1
    counts=r["model_forwards"]
    require(counts=={"detector":16,"completion":21,"restoration":15},"Recorded forward count differs")
    require(r["generation_requests"]==28 and r["optimizer_updates"]==0,"Recorded budget differs")
    require(outputs==28 and assisted_rejections==2 and automatic_misses==2,"Output/guard counts differ")
    reduced={key:{"cases":len(values),"mean_visible_mae":float(np.mean([v[0] for v in values])),
                       "mean_off_input_visible_mae":float(np.mean([v[1] for v in values]))} for key,values in aggregates.items()}
    report={"verified":True,"date":"2026-10-02","protocol_sha256":PROTOCOL_SHA,"results_sha256":RESULT_SHA,
            "auditor_sha256":sha(__file__),"saved_outputs_verified":outputs,"saved_detection_masks_verified":32,
            "zero_off_visible_changed_pixels":off_changes==0,"assisted_degraded_visible_cases_improved":assisted_wins,
            "assisted_nearly_hidden_rejections":assisted_rejections,"automatic_nearly_hidden_rejection_misses":automatic_misses,
            "aggregates":reduced,"model_state_and_call_scope":"Execution records checked; models not replayed",
            "hole_preservation_scope":"No separate pre-restoration broad intermediates were saved; post-restoration hole preservation is covered by unit checks and earlier real-output comparison, not independently reconstructed here",
            "hidden_ground_truth":None,"training_admitted":False,"optimizer_updates":0}
    destination.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"verified":True,"outputs":outputs,"assisted_degraded_visible_cases_improved":assisted_wins,
                     "automatic_rejection_misses":automatic_misses,"aggregates":reduced,"verification_sha256":sha(destination)}))


if __name__=="__main__":
    main()
