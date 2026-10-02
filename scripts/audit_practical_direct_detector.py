"""Saved-pixel recount for the returned direct-head practical comparison."""
import json
from pathlib import Path

import cv2
import numpy as np

from scripts.audit_xseg_mask_comparison import mask, require, sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT / "outputs/practical_direct_detector_v1"


def main():
    destination=OUT / "independent_verification.json"
    require(not destination.exists(),"Preserve previous audit")
    require(sha(OUT / "frozen_protocol.json")=="69c6a75e4dc1af099a3cb1a621082a870966170108e9a2d2255d614b708a8db8","Protocol changed")
    require(sha(OUT / "results.json")=="4b07e9a7765486d588f025392f7cc3d2adc84e3a731a8012b4037fa7885975b0","Result changed")
    p=json.loads((OUT / "frozen_protocol.json").read_text());r=json.loads((OUT / "results.json").read_text())
    require(r["complete"] and not r["failures"] and len(r["rows"])==36,"Incomplete comparison")
    require([c["id"] for c in p["cases"]]==[row["id"] for row in r["rows"]],"Membership/order differs")
    for path,digest in p["assets_sha256"].items():require(sha(ROOT / path)==digest,"Frozen asset changed")
    for path,digest in {**r["assets_sha256"],**r["previews_sha256"]}.items():require(sha(OUT / path)==digest,"Saved evidence changed")
    groups={};controls=[];rejection_misses=[];unexpected=[];cached=0;probabilities=0
    yy,xx=np.mgrid[:256,:256];domain=((xx-127.5)/83)**2+((yy-137)/104)**2<=1
    bands={"left_eye":(60,84,119,133),"right_eye":(137,84,196,133),"nose":(108,119,149,166),"mouth":(82,165,174,208)}
    for case,row in zip(p["cases"],r["rows"]):
        raw=mask(OUT / row["raw_mask"]);direct=mask(OUT / row["direct_mask"]);parent=mask(OUT / row["baseline_mask"])
        require(np.array_equal(parent,mask(ROOT / case["baseline_cached"])),"Baseline cache differs")
        if case["direct_raw_cached"]:
            cached+=1;require(row["probability"] is None and row["raw_mask_reused"],"Cache metadata differs")
            require(np.array_equal(raw,mask(ROOT / case["direct_raw_cached"])),"Native raw cache differs")
        else:
            probabilities+=1;probability=np.load(OUT / row["probability"],allow_pickle=False)
            require(probability.dtype==np.float32 and probability.shape==(256,256) and np.isfinite(probability).all() and probability.min()>=0 and probability.max()<=1,"Invalid probability")
            require(np.array_equal(raw,probability>=.5),"Threshold differs")
        require(np.array_equal(direct,cv2.dilate(raw.astype(np.uint8),np.ones((7,7),np.uint8))!=0),"Proposal margin differs")
        truth=mask(ROOT / case["reviewed"]);protected=mask(ROOT / case["protected"]) if case["protected"] else np.zeros((256,256),bool)
        tolerated=cv2.dilate(truth.astype(np.uint8),np.ones((7,7),np.uint8))!=0
        group=groups.setdefault(row["family"]+("/degraded" if row["synthetically_degraded"] else "/native"),{"cases":0,"baseline":{},"direct":{}});group["cases"]+=1
        for arm,observed in (("baseline",parent),("direct",direct)):
            tp=int((truth&observed).sum());union=int((truth|observed).sum());positive=int(observed.sum());total=int(truth.sum())
            metrics={"predicted_pixels":positive,"reference_pixels":total,"intersection_pixels":tp,"union_pixels":union,
                     "recall":tp/total if total else None,"precision":tp/positive if positive else None,"iou":tp/union if total and union else None,
                     "excess_pixels_outside_3px_tolerance":int((observed&~tolerated).sum()),"protected_wire_pixels_changed":int((observed&protected).sum()),
                     "empty_control_marked_fraction":float(observed.mean()) if not total else None}
            for name,value in metrics.items():require(row[arm][name]==value,"Metric differs: "+name)
            for name in ("recall","precision","iou","excess_pixels_outside_3px_tolerance","empty_control_marked_fraction"):
                if metrics[name] is not None:group[arm].setdefault(name,[]).append(metrics[name])
            f={name:float(observed[y1:y2,x1:x2].mean()) for name,(x1,y1,x2,y2) in bands.items()}
            reject=observed[domain].mean()>=.8 or (f["left_eye"]>=.85 and f["right_eye"]>=.85 and f["nose"]>=.9 and f["mouth"]>=.9) or observed.mean()>=.85
            guard=row[arm]["guard"];require(bool(reject)==guard["rejected"] and f==guard["feature_covered_fractions"] and float(observed[domain].mean())==guard["face_covered_fraction"],"Guard differs")
        if not truth.any():controls.append({"id":case["id"],"direct_pixels":int(direct.sum())})
        if case["expected_rejection"] and not row["direct"]["guard"]["rejected"]:rejection_misses.append(case["id"])
        if not case["expected_rejection"] and row["direct"]["guard"]["rejected"]:unexpected.append(case["id"])
    for group in groups.values():
        for arm in ("baseline","direct"):group[arm]={name:float(np.mean(values)) for name,values in group[arm].items()}
    require(groups==r["aggregates"],"Aggregate differs")
    require(probabilities==r["new_detector_forwards"]==26 and cached==r["cached_native_raw_masks"]==10 and r["optimizer_updates"]==r["completion_forwards"]==r["restoration_forwards"]==0,"Execution counters differ")
    require(r["state_unchanged"] and r["state_before"]==r["state_after"] and not r["promoted"],"Recorded state/status differs")
    report={"verified":True,"date":"2026-10-02","protocol_sha256":sha(OUT / "frozen_protocol.json"),"results_sha256":sha(OUT / "results.json"),
            "auditor_sha256":sha(__file__),"cases_verified":36,"binary_masks_verified":108,"probability_arrays_verified":26,
            "native_cached_masks_verified":10,"controls":controls,"nearly_hidden_rejection_misses":rejection_misses,"unexpected_rejections":unexpected,
            "execution_scope":"State fingerprints/call counts checked from records; checkpoint file hash rechecked, no model replay",
            "new_model_forwards":0,"optimizer_updates":0,"promoted":False,"historical_gate_failure_preserved":True,"hidden_ground_truth":None}
    destination.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({key:report[key] for key in ("verified","cases_verified","controls","nearly_hidden_rejection_misses","unexpected_rejections")}))


if __name__=="__main__":main()
