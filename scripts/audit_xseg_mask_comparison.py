"""Independently recount saved segmentation masks/probabilities; no model forward."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT / "outputs/xseg_mask_comparison_v2"


def sha(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def mask(path):
    with Image.open(path) as image:array=np.asarray(image.convert("L")).copy()
    if array.shape!=(256,256) or not np.isin(array,(0,255)).all():raise ValueError("Invalid saved binary mask")
    return array==255


def require(value,message):
    if not value:raise ValueError(message)


def main():
    destination=OUT / "independent_verification.json"
    require(not destination.exists(),"Preserve previous audit")
    require(sha(OUT / "frozen_protocol.json")=="9fd70d1edef468e6fb6fe951e3080290e9e63bef0fc65c5fffb99a17fadb9516","Protocol changed")
    require(sha(OUT / "results.json")=="f844ec2740352f1ec5ce15fb32b06bc280fe72e6f41fea8d6b7e51f53da518df","Results changed")
    p=json.loads((OUT / "frozen_protocol.json").read_text());r=json.loads((OUT / "results.json").read_text())
    require(r["complete"] and not r["failures"] and len(r["rows"])==36,"Incomplete comparison")
    require([c["id"] for c in p["cases"]]==[row["id"] for row in r["rows"]],"Case order differs")
    for path,digest in p["assets_sha256"].items():require(sha(ROOT / path)==digest,"Frozen asset changed")
    for path,digest in {**r["assets_sha256"],**r["previews_sha256"]}.items():require(sha(OUT / path)==digest,"Saved evidence changed")
    yy,xx=np.mgrid[:256,:256]
    domain=((xx-127.5)/83)**2+((yy-137)/104)**2<=1
    aggregates={};controls=[];false_rejections=[];correct_rejections=[];cached=0
    for case,row in zip(p["cases"],r["rows"]):
        probability=np.load(OUT / row["visible_probability"],allow_pickle=False)
        require(probability.dtype==np.float32 and probability.shape==(256,256) and np.isfinite(probability).all(),"Probability type/shape differs")
        require(probability.min()>=0 and probability.max()<=1,"Probability range differs")
        if case["probability_cached"]:
            cached+=1;previous=np.load(ROOT / case["probability_cached"],allow_pickle=False)
            require(np.array_equal(probability,np.clip(previous,0,1)),"Cached probability differs")
        raw=(probability<.5)&domain
        proposed=(cv2.dilate(raw.astype(np.uint8),np.ones((7,7),np.uint8))!=0)&domain
        require(np.array_equal(raw,mask(OUT / row["raw_mask"])) and np.array_equal(proposed,mask(OUT / row["xseg_mask"])),"Threshold/domain/margin differs")
        parent=mask(OUT / row["baseline_mask"])
        require(np.array_equal(parent,mask(ROOT / case["baseline_cached"])),"Baseline cache differs")
        truth=mask(ROOT / case["reviewed"])
        protected=mask(ROOT / case["protected"]) if case["protected"] else np.zeros((256,256),bool)
        tolerance=cv2.dilate(truth.astype(np.uint8),np.ones((7,7),np.uint8))!=0
        group=aggregates.setdefault(row["family"]+("/degraded" if row["synthetically_degraded"] else "/native"),{"cases":0,"baseline":{},"xseg":{}})
        group["cases"]+=1
        for arm,observed in (("baseline",parent),("xseg",proposed)):
            tp=int((observed&truth).sum());union=int((observed|truth).sum());positive=int(observed.sum());total=int(truth.sum())
            values={"predicted_pixels":positive,"reference_pixels":total,"intersection_pixels":tp,"union_pixels":union,
                    "recall":tp/total if total else None,"precision":tp/positive if positive else None,
                    "iou":tp/union if total and union else None,"excess_pixels_outside_3px_tolerance":int((observed&~tolerance).sum()),
                    "protected_wire_pixels_changed":int((observed&protected).sum()),
                    "empty_control_marked_fraction":float(observed.mean()) if not total else None}
            for metric,value in values.items():require(value==row[arm][metric],"Saved metric differs: "+metric)
            for metric in ("recall","precision","iou","excess_pixels_outside_3px_tolerance","empty_control_marked_fraction"):
                if values[metric] is not None:group[arm].setdefault(metric,[]).append(values[metric])
            bands={"left_eye":(60,84,119,133),"right_eye":(137,84,196,133),"nose":(108,119,149,166),"mouth":(82,165,174,208)}
            fractions={name:float(observed[y1:y2,x1:x2].mean()) for name,(x1,y1,x2,y2) in bands.items()}
            rejected=(observed[domain].mean()>=.8 or (fractions["left_eye"]>=.85 and fractions["right_eye"]>=.85 and fractions["nose"]>=.9 and fractions["mouth"]>=.9) or observed.mean()>=.85)
            guard=row[arm]["guard"]
            require(bool(rejected)==guard["rejected"] and fractions==guard["feature_covered_fractions"] and float(observed[domain].mean())==guard["face_covered_fraction"],"Visibility guard differs")
        if not truth.any():controls.append({"id":case["id"],"baseline_pixels":int(parent.sum()),"xseg_pixels":int(proposed.sum())})
        if row["xseg"]["guard"]["rejected"]:
            (correct_rejections if case["expected_rejection"] else false_rejections).append(case["id"])
    for group in aggregates.values():
        for arm in ("baseline","xseg"):group[arm]={metric:float(np.mean(values)) for metric,values in group[arm].items()}
    require(aggregates==r["aggregates"],"Aggregates differ")
    require(cached==24 and r["xseg_forwards"]==12 and r["baseline_detector_forwards"]==r["completion_forwards"]==r["restoration_forwards"]==r["optimizer_updates"]==0,"Execution record differs")
    report={"verified":True,"date":"2026-10-02","protocol_sha256":sha(OUT / "frozen_protocol.json"),
            "results_sha256":sha(OUT / "results.json"),"auditor_sha256":sha(__file__),"cases_verified":36,
            "binary_masks_verified":108,"probability_arrays_verified":36,"cached_probabilities_verified":cached,
            "controls":controls,"xseg_unexpected_rejections":false_rejections,"xseg_correct_nearly_hidden_rejections":correct_rejections,
            "numeric_repair_scope":"One cached diagnostic value clipped from1+1 float32 ULP; V1 preserved",
            "execution_scope":"New forward counts verified from execution records, not by rerunning models",
            "new_model_forwards":0,"optimizer_updates":0,"promoted":False,"hidden_ground_truth":None}
    destination.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({k:report[k] for k in ("verified","cases_verified","controls","xseg_unexpected_rejections","xseg_correct_nearly_hidden_rejections")}))


if __name__=="__main__":main()
