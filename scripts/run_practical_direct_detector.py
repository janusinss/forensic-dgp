"""Inference-only transfer diagnostic of the existing returned detector head."""
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from scripts.run_practical_lama_comparison import state_sha
from scripts.run_xseg_mask_comparison import mask_metrics, pixels, sha, write_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT / "outputs/practical_direct_detector_v1"


def previews(cases,rows):
    saved={}
    for first in range(0,len(cases),6):
        selected=cases[first:first+6];canvas=Image.new("RGB",(1024,32+281*len(selected)),"#16181c");draw=ImageDraw.Draw(canvas)
        for col,label in enumerate(("input","retained proposal","returned direct-head proposal","reviewed reference")):
            draw.text((256*col+3,8),label,fill="white")
        for index,case in enumerate(selected):
            row=rows[first+index];original=pixels(ROOT / case["input"])
            arrays=[original]
            for path in (OUT / row["baseline_mask"],OUT / row["direct_mask"],ROOT / case["reviewed"]):
                marked=original.astype(float).copy();active=pixels(path,"L")==255
                marked[active]=.55*marked[active]+.45*np.array([16,185,129]);arrays.append(marked.round().astype(np.uint8))
            yy=32+index*281;draw.text((3,yy),case["id"],fill="white")
            for col,value in enumerate(arrays):canvas.paste(Image.fromarray(value),(col*256,yy+19))
        path=OUT / "preview" / ("rows_%02d_%02d.png"%(first+1,first+len(selected)));canvas.save(path)
        saved[path.relative_to(OUT).as_posix()]=sha(path)
    return saved


def main():
    if (OUT / "execution.json").exists():raise ValueError("Preserve previous run")
    p=json.loads((OUT / "frozen_protocol.json").read_text())
    if not p["frozen_before_new_forwards"] or len(p["cases"])!=36:raise ValueError("Protocol differs")
    for path,digest in p["assets_sha256"].items():
        if sha(ROOT / path)!=digest:raise ValueError("Frozen asset differs: "+path)
    sys.path.insert(0,str(ROOT / "outputs/face_extraction_dependencies"))
    from face_occlusion_adapter import load_adapter
    torch.set_num_threads(4)
    model,state=load_adapter(ROOT / p["model"],"cpu");model.eval().requires_grad_(False)
    before=state_sha(model)
    if any(parameter.requires_grad for parameter in model.parameters()):raise ValueError("Expected frozen inference")
    counts=[0]
    def count_forward(module,args,output):counts[0]+=len(args[0])
    hook=model.network.register_forward_hook(count_forward)
    for name in ("probabilities","raw_masks","direct_masks","baseline_masks","preview"):(OUT / name).mkdir()
    write_json(OUT / "execution.json",{"protocol_sha256":sha(OUT / "frozen_protocol.json"),"budget":p["budget"],
               "model_sha256":sha(ROOT / p["model"]),"torch":str(torch.__version__),"device":"cpu","threads":4,
               "optimizer_constructed":False,"state_before":before,"source_updates":state["optimizer_updates"],
               "historical_gate_status":"Ineligible; retained unchanged"})
    started=time.monotonic();rows=[];hashes={};failures=[];cached=0
    for case in p["cases"]:
        if time.monotonic()-started>300:
            failures.append({"id":case["id"],"reason":"Wall-time budget exhausted"});break
        try:
            original=pixels(ROOT / case["input"])
            probability_path=None
            if case["direct_raw_cached"]:
                raw=(pixels(ROOT / case["direct_raw_cached"],"L")==255).astype(np.uint8);cached+=1
            else:
                sample=torch.from_numpy(original.copy()).permute(2,0,1).float()[None]/255
                with torch.inference_mode():probability=model.detect(sample).sigmoid()[0,0].cpu().numpy()
                if probability.shape!=(256,256) or not np.isfinite(probability).all():raise ValueError("Invalid covering probability")
                probability_path="probabilities/"+case["id"]+".npy";np.save(OUT / probability_path,probability,allow_pickle=False)
                hashes[probability_path]=sha(OUT / probability_path);raw=(probability>=.5).astype(np.uint8)
            proposed=cv2.dilate(raw,np.ones((7,7),np.uint8))
            parent=(pixels(ROOT / case["baseline_cached"],"L")==255).astype(np.uint8)
            paths={"raw_mask":"raw_masks/"+case["id"]+".png","direct_mask":"direct_masks/"+case["id"]+".png",
                   "baseline_mask":"baseline_masks/"+case["id"]+".png"}
            for key,array in (("raw_mask",raw),("direct_mask",proposed),("baseline_mask",parent)):
                Image.fromarray(array*255).save(OUT / paths[key]);hashes[paths[key]]=sha(OUT / paths[key])
            truth=pixels(ROOT / case["reviewed"],"L")==255
            protected=pixels(ROOT / case["protected"],"L")==255 if case["protected"] else np.zeros((256,256),bool)
            rows.append({"id":case["id"],"family":case["family"],"synthetically_degraded":case["synthetically_degraded"],
                         "pose_scope":case["pose_scope"],"expected_rejection":case["expected_rejection"],**paths,
                         "probability":probability_path,"raw_mask_reused":bool(case["direct_raw_cached"]),
                         "baseline":mask_metrics(parent,truth,protected),"direct":mask_metrics(proposed,truth,protected)})
        except Exception as exc:
            failures.append({"id":case["id"],"reason":str(exc)});break
    hook.remove();after=state_sha(model)
    if before!=after or counts[0]>26 or cached>10:raise ValueError("Frozen state/budget differs")
    aggregate={}
    for row in rows:
        group=aggregate.setdefault(row["family"]+("/degraded" if row["synthetically_degraded"] else "/native"),{"cases":0,"baseline":{},"direct":{}})
        group["cases"]+=1
        for arm in ("baseline","direct"):
            for metric in ("recall","precision","iou","excess_pixels_outside_3px_tolerance","empty_control_marked_fraction"):
                value=row[arm][metric]
                if value is not None:group[arm].setdefault(metric,[]).append(value)
    for group in aggregate.values():
        for arm in ("baseline","direct"):group[arm]={metric:float(np.mean(values)) for metric,values in group[arm].items()}
    report={"format":"dgp-practical-direct-detector-results-v1","date":"2026-10-02","complete":len(rows)==36 and not failures,
            "protocol_sha256":sha(OUT / "frozen_protocol.json"),"rows":rows,"failures":failures,"assets_sha256":hashes,
            "previews_sha256":previews(p["cases"],rows) if len(rows)==36 else {},"aggregates":aggregate,
            "new_detector_forwards":counts[0],"cached_native_raw_masks":cached,"completion_forwards":0,
            "restoration_forwards":0,"optimizer_updates":0,"elapsed_seconds_after_loading":time.monotonic()-started,
            "state_before":before,"state_after":after,"state_unchanged":before==after,
            "promoted":False,"historical_selection":"Original synthetic-retention failure unchanged",
            "visual_review_pending":True,"hidden_ground_truth":None,"training_admitted":False}
    write_json(OUT / "results.json",report)
    print(json.dumps({k:report[k] for k in ("complete","new_detector_forwards","cached_native_raw_masks","elapsed_seconds_after_loading","failures","promoted")}),flush=True)


if __name__=="__main__":main()
