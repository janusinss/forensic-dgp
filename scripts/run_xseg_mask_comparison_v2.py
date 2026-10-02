"""Complete the frozen mask comparison using verified caches; no PyTorch models."""
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from scripts.run_xseg_mask_comparison import mask_metrics, pixels, sha, write_json
from xseg_occlusion import covering_proposal, facial_domain
from xseg_occlusion_v2 import XSegVisibleFaceClipped, clip_visible_probability

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/xseg_mask_comparison_v2"


def preview(cases, rows, first):
    selected = cases[first:first+6]
    canvas = Image.new("RGB", (1280, 32+281*len(selected)), "#16181c")
    draw = ImageDraw.Draw(canvas)
    for col,label in enumerate(("input","retained proposal","XSeg visible probability","XSeg covering proposal","reviewed reference")):
        draw.text((256*col+3,8),label,fill="white")
    def marked(original,mask):
        color=original.astype(float).copy()
        color[mask]=.55*color[mask]+.45*np.array([16,185,129])
        return Image.fromarray(color.round().astype(np.uint8))
    for index,case in enumerate(selected):
        row=rows[first+index];original=pixels(ROOT / case["input"])
        parent=pixels(OUT / row["baseline_mask"],"L")==255
        candidate=pixels(OUT / row["xseg_mask"],"L")==255
        reviewed=pixels(ROOT / case["reviewed"],"L")==255
        probability=np.load(OUT / row["visible_probability"],allow_pickle=False)
        gray=Image.fromarray((probability*255).round().astype(np.uint8)).convert("RGB")
        yy=32+index*281;draw.text((3,yy),case["id"],fill="white")
        for col,value in enumerate((Image.fromarray(original),marked(original,parent),gray,marked(original,candidate),marked(original,reviewed))):
            canvas.paste(value,(256*col,yy+19))
    path=OUT / "preview" / ("rows_%02d_%02d.png" % (first+1,first+len(selected)))
    canvas.save(path)
    return path.relative_to(OUT).as_posix(),sha(path)


def main():
    if (OUT / "execution.json").exists():
        raise ValueError("Preserve previous run")
    p=json.loads((OUT / "frozen_protocol.json").read_text())
    if not p["frozen_before_first_forward"] or len(p["cases"])!=36 or p["budget"]["xseg_forwards"]!=12:
        raise ValueError("Protocol differs")
    for path,digest in p["assets_sha256"].items():
        if sha(ROOT / path)!=digest:
            raise ValueError("Frozen asset changed: "+path)
    for name in ("probabilities","raw_masks","xseg_masks","baseline_masks","preview"):
        (OUT / name).mkdir()
    model=XSegVisibleFaceClipped(ROOT / "outputs/xseg_pretrained_v1/xseg_1.onnx")
    write_json(OUT / "execution.json",{"protocol_sha256":sha(OUT / "frozen_protocol.json"),"budget":p["budget"],
               "optimizer_constructed":False,"generator_loaded":False,"restorer_loaded":False,
               "providers":model.session.get_providers(),"threads":4})
    started=time.monotonic();rows=[];failures=[];hashes={};cache_count=0
    for case in p["cases"]:
        if time.monotonic()-started>300:
            failures.append({"id":case["id"],"reason":"Wall-time budget exhausted"});break
        try:
            original=pixels(ROOT / case["input"]);truth=pixels(ROOT / case["reviewed"],"L")==255
            protected=pixels(ROOT / case["protected"],"L")==255 if case["protected"] else np.zeros((256,256),bool)
            if case["probability_cached"]:
                probability=clip_visible_probability(np.load(ROOT / case["probability_cached"],allow_pickle=False))
                cache_count+=1
            else:
                probability=model(original)
            raw,proposal=covering_proposal(probability)
            parent=(pixels(ROOT / case["baseline_cached"],"L")==255).astype(np.uint8)
            paths={"visible_probability":"probabilities/"+case["id"]+".npy","raw_mask":"raw_masks/"+case["id"]+".png",
                   "xseg_mask":"xseg_masks/"+case["id"]+".png","baseline_mask":"baseline_masks/"+case["id"]+".png"}
            np.save(OUT / paths["visible_probability"],probability,allow_pickle=False)
            for key,array in (("raw_mask",raw),("xseg_mask",proposal),("baseline_mask",parent)):
                Image.fromarray(array.astype(np.uint8)*255).save(OUT / paths[key])
            for path in paths.values():hashes[path]=sha(OUT / path)
            rows.append({"id":case["id"],"family":case["family"],"synthetically_degraded":case["synthetically_degraded"],
                         "pose_scope":case["pose_scope"],"expected_rejection":case["expected_rejection"],**paths,
                         "visible_probability_min":float(probability.min()),"visible_probability_max":float(probability.max()),
                         "visible_probability_mean_in_face_domain":float(probability[facial_domain()].mean()),
                         "probability_reused":bool(case["probability_cached"]),
                         "baseline":mask_metrics(parent,truth,protected),"xseg":mask_metrics(proposal,truth,protected)})
        except Exception as exc:
            failures.append({"id":case["id"],"reason":str(exc)});break
    if model.forward_count>12 or cache_count>24:
        raise ValueError("Inference budget exceeded")
    previews={}
    if len(rows)==36:
        for first in range(0,36,6):
            name,digest=preview(p["cases"],rows,first);previews[name]=digest
    aggregates={}
    for row in rows:
        group=aggregates.setdefault(row["family"]+("/degraded" if row["synthetically_degraded"] else "/native"),{"cases":0,"baseline":{},"xseg":{}})
        group["cases"]+=1
        for arm in ("baseline","xseg"):
            for metric in ("recall","precision","iou","excess_pixels_outside_3px_tolerance","empty_control_marked_fraction"):
                value=row[arm][metric]
                if value is not None:group[arm].setdefault(metric,[]).append(value)
    for group in aggregates.values():
        for arm in ("baseline","xseg"):group[arm]={metric:float(np.mean(values)) for metric,values in group[arm].items()}
    report={"format":"dgp-xseg-mask-results-v2","date":"2026-10-02","complete":len(rows)==36 and not failures,
            "protocol_sha256":sha(OUT / "frozen_protocol.json"),"rows":rows,"failures":failures,"assets_sha256":hashes,
            "previews_sha256":previews,"aggregates":aggregates,"xseg_forwards":model.forward_count,
            "cached_xseg_probabilities":cache_count,"baseline_detector_forwards":0,"completion_forwards":0,
            "restoration_forwards":0,"optimizer_updates":0,"elapsed_seconds_after_loading":time.monotonic()-started,
            "promoted":False,"visual_review_pending":True,"hidden_ground_truth":None,"training_admitted":False}
    write_json(OUT / "results.json",report)
    print(json.dumps({key:report[key] for key in ("complete","xseg_forwards","cached_xseg_probabilities","elapsed_seconds_after_loading","failures","promoted")}),flush=True)


if __name__=="__main__":main()
