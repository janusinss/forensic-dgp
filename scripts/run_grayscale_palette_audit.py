"""Freeze then audit a palette-only improvement on cached broad outputs.

No model forward, training, target-based routing or revised removal labels.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from face_color_policy import preserve_input_palette
from scripts.run_practical_lama_comparison import pixels, sha, write_json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"outputs/broad_face_workflow_outputs_v1"
SOURCE = ROOT/"outputs/broad_covering_gallery_v1/frozen_protocol.json"
OUT = ROOT/"outputs/grayscale_palette_comparison_v1"


def main():
    if OUT.exists():
        raise ValueError("Preserve prior palette evidence")
    if sha(BASE/"results.json") != "afe8cb6b2d768460082890fe8ded09dd125e19f599f54f55a496e2a4073a94d3":
        raise ValueError("Baseline changed")
    p=json.loads(SOURCE.read_text(encoding="utf-8"))
    result=json.loads((BASE/"results.json").read_text(encoding="utf-8"))
    cases={case["id"]:case for case in p["cases"]}
    assets={}
    for row in result["rows"]:
        if row["output"]:
            require_path=BASE/row["output"]
            if sha(require_path)!=row["output_sha256"]:
                raise ValueError("Saved baseline changed")
            assets[require_path.relative_to(ROOT).as_posix()]=sha(require_path)
            c=cases[row["id"]]
            for key in ("input","reference","proposal"):
                if sha(ROOT/c[key])!=c[key+"_sha256"]:
                    raise ValueError("Frozen source changed")
                assets[c[key]]=c[key+"_sha256"]
            mask_path=(BASE if row["mask_root"]=="output" else ROOT)/row["mask"]
            assets[mask_path.relative_to(ROOT).as_posix()]=sha(mask_path)
    OUT.mkdir()
    protocol={"format":"dgp-visible-input-palette-v1","date":"2026-10-02","frozen_before_processing":True,
              "baseline_result_sha256":sha(BASE/"results.json"),"source_protocol_sha256":sha(SOURCE),
              "policy_sha256":sha(ROOT/"face_color_policy.py"),"runner_sha256":sha(__file__),"assets_sha256":assets,
              "threshold":3,"routing":"Input-only mean visible channel range after Gaussian5 sigma1 at256, eroded9 visible support >=512",
              "composition":"Gray generated region only when restoration off; gray whole result when visible restoration applied",
              "criterion":"Remove false color from grayscale inputs; preserve every off visible pixel; color inputs unchanged; no generated-region change attributable to visible-restoration selection",
              "new_model_forwards":0,"optimizer_updates":0,"hidden_ground_truth":None}
    write_json(OUT/"frozen_protocol.json",protocol)
    rows=[]
    for row in result["rows"]:
        if not row["output"]:
            continue
        case=cases[row["id"]]
        original=pixels(ROOT/case["input"])
        reference=pixels(ROOT/case["reference"])
        baseline=pixels(BASE/row["output"])
        mask_path=(BASE if row["mask_root"]=="output" else ROOT)/row["mask"]
        mask=(pixels(mask_path,"L")>0).astype(np.uint8)
        array,signal=preserve_input_palette(original,mask,baseline,row["metadata"]["restoration_applied"])
        path=OUT/(row["arm"]+"_"+row["id"]+".png")
        Image.fromarray(array).save(path)
        outside_changes=int(np.any(array!=baseline,axis=-1)[mask==0].sum())
        if not row["metadata"]["restoration_applied"] and outside_changes:
            raise ValueError("Palette changed observed pixels with restoration off")
        if not signal["grayscale_input"] and not np.array_equal(array,baseline):
            raise ValueError("Color input changed")
        selected=pixels(ROOT/case["proposal"],"L")>0
        values=np.abs(array.astype(float)-reference.astype(float))[~selected]/255
        rows.append({"id":row["id"],"arm":row["arm"],"output":path.name,"output_sha256":sha(path),
                     "signal":signal,"restoration_applied":row["metadata"]["restoration_applied"],
                     "palette_changed_pixels_outside_active_mask":outside_changes,
                     "visible_mae_outside_fixed_operator_proposal":float(values.mean()),
                     "baseline_visible_mae":row["visible_mae_outside_fixed_operator_proposal"],
                     "synthetically_degraded":case["synthetically_degraded"],"family":case["family"]})
    chosen=[row for row in rows if row["arm"]=="assisted" and row["signal"]["grayscale_input"]]
    canvas=Image.new("RGB",(1024,30+len(chosen)*278),"#16181c")
    draw=ImageDraw.Draw(canvas)
    for col,title in enumerate(("input","reviewed area","previous output","gray-preserving output")):
        draw.text((col*256+4,8),title,fill="white")
    for index,row in enumerate(chosen):
        case=cases[row["id"]];y=30+index*278
        draw.text((4,y),row["id"],fill="white")
        original=pixels(ROOT/case["input"]);mask=pixels(ROOT/case["proposal"],"L")>0
        marked=original.astype(float);marked[mask]=.55*marked[mask]+.45*np.array([16,185,129])
        arrays=(original,marked.round().astype(np.uint8),pixels(BASE/"assisted"/(row["id"]+".png")),pixels(OUT/row["output"]))
        for col,array in enumerate(arrays):
            canvas.paste(Image.fromarray(array),(col*256,y+18))
    preview=OUT/"assisted_grayscale_preview.png";canvas.save(preview)
    report={"format":"dgp-grayscale-palette-results-v1","complete":True,"protocol_sha256":sha(OUT/"frozen_protocol.json"),
            "rows":rows,"saved_outputs":len(rows),"palette_selected":sum(row["signal"]["grayscale_input"] for row in rows),
            "off_visible_palette_changes":0,"color_inputs_unchanged":True,"new_model_forwards":0,"optimizer_updates":0,
            "preview_sha256":sha(preview),"visual_review_pending":True}
    write_json(OUT/"results.json",report)
    print(json.dumps({k:report[k] for k in ("saved_outputs","palette_selected","off_visible_palette_changes","color_inputs_unchanged","new_model_forwards","protocol_sha256")}))


if __name__=="__main__":
    main()
