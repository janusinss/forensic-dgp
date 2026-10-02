"""Bounded inference comparison on source-frozen covering proposals; no training."""
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from face_workflow import FaceWorkflow, visibility_check
from scripts.run_practical_lama_comparison import pixels, sha, state_sha, write_json

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "outputs/broad_covering_gallery_v1/frozen_protocol.json"
PROTOCOL_SHA = "b7a1a447482d650bb0a18b13d649f0c548c3c59acd6db202111169599c805bc9"
OUT = ROOT / "outputs/broad_face_workflow_outputs_v1"


def visible_error(array, reference, selected):
    values = np.abs(array.astype(float)-reference.astype(float))[~selected]/255
    return float(values.mean()) if values.size else None


def main():
    if OUT.exists() or sha(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("Preserve previous outputs and frozen protocol")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if not protocol["frozen"] or sha(ROOT / "face_workflow.py") != protocol["processing_sha256"]:
        raise ValueError("Frozen processing policy differs")
    for case in protocol["cases"]:
        for key in ("source", "input", "reference", "proposal"):
            if sha(ROOT/case[key]) != case[key+"_sha256"]:
                raise ValueError("Frozen input changed")
    OUT.mkdir()
    for name in ("automatic", "assisted", "automatic_masks", "raw_masks", "preview"):
        (OUT/name).mkdir()
    engine = FaceWorkflow(device="cpu")
    engine._runtime()
    print("Loading pinned inference models; budget: 16 detector forwards, up to 28 completion/restoration requests, 600s after loading; no optimizer.", flush=True)
    engine._detector(); engine._generator(); engine._restorer()
    models = {"detector": engine.detector, "completion": engine.generator, "restoration": engine.restorer}
    before = {key: state_sha(net) for key, net in models.items()}
    if any(net.training or any(p.requires_grad for p in net.parameters()) for net in models.values()):
        raise ValueError("Expected frozen evaluation models")
    counts = {"detector": 0, "completion": 0, "restoration": 0}
    def hook(key):
        def count(*args):
            counts[key] += 1
        return count
    hooks = [engine.detector.segmenter.register_forward_hook(hook("detector")),
             engine.generator.net.register_forward_hook(hook("completion")),
             engine.restorer.net.register_forward_hook(hook("restoration"))]
    write_json(OUT/"execution.json", {"protocol_sha256": PROTOCOL_SHA,
               "processing_sha256": protocol["processing_sha256"], "state_before": before,
               "device": "cpu", "threads": 4, "optimizer_constructed": False, "optimizer_updates": 0,
               "budget": {"detector_forwards":16, "generation_requests":28, "wall_seconds_after_loading":600},
               "completion":engine.generator_provenance,"restoration":engine.restorer_provenance})
    started, rows, failures, masks, attempted = time.monotonic(), [], [], {}, 0
    for case in protocol["cases"]:
        tick = time.monotonic()
        if tick-started > 600:
            failures.append({"id": case["id"], "reason":"Inference wall-time budget exhausted"})
            break
        try:
            original = pixels(ROOT/case["input"])
            reference = pixels(ROOT/case["reference"])
            reviewed = (pixels(ROOT/case["proposal"], "L") >= 128).astype(np.uint8)
            detected = engine.review_mask(original)
            automatic = detected["mask"]
            for name, value in (("automatic_masks",automatic), ("raw_masks",detected["raw_mask"])):
                path = OUT/name/(case["id"]+".png")
                Image.fromarray(value*255).save(path)
                masks[path.relative_to(OUT).as_posix()] = sha(path)
            for arm, mask in (("automatic",automatic), ("assisted",reviewed)):
                checked = visibility_check(mask)
                row = {"id":case["id"],"arm":arm,"family":case["family"],
                       "synthetically_degraded":case["synthetically_degraded"],
                       "mask": ("automatic_masks/"+case["id"]+".png") if arm=="automatic" else case["proposal"],
                       "mask_root": "output" if arm=="automatic" else "repository",
                       "visibility":checked,"expected_rejection":case["expected_rejection"],
                       "hidden_face_mae":None}
                if case["expected_rejection"]:
                    row["output"] = None
                    row["suppressed_by_source_review"] = True
                    row["guard_rejected"] = checked["rejected"]
                    if arm=="assisted":
                        try:
                            engine.generate(original, mask, "auto")
                        except ValueError as exc:
                            if "less-covered" not in str(exc):
                                raise
                            row["guard_error"] = str(exc)
                        else:
                            raise ValueError("Nearly hidden source was not rejected")
                    # Do not knowingly synthesize this source just because a
                    # detector misses the covering. Report that miss separately.
                elif checked["rejected"]:
                    row["output"] = None
                    row["guard_rejected"] = True
                    row["unexpected_rejection"] = True
                else:
                    attempted += 1
                    if attempted > 28:
                        raise ValueError("Generation request budget exceeded")
                    output, metadata = engine.generate(original, mask, "auto")
                    path = OUT/arm/(case["id"]+".png")
                    Image.fromarray(output).save(path)
                    row.update({"output":path.relative_to(OUT).as_posix(), "output_sha256":sha(path),
                                "metadata":metadata, "visible_mae_outside_fixed_operator_proposal":visible_error(output,reference,reviewed.astype(bool)),
                                "off_input_visible_mae":visible_error(original,reference,reviewed.astype(bool)),
                                "changed_pixels_outside_active_mask":int(np.any(output!=original,axis=-1)[mask==0].sum()),
                                "mask_source_label":"baseline proposal before user review" if arm=="automatic" else "source-reviewed operator proposal"})
                rows.append(row)
            print(case["id"]+f": {time.monotonic()-tick:.2f}s", flush=True)
        except Exception as exc:
            failures.append({"id":case["id"],"reason":str(exc)})
            print(json.dumps(failures[-1]),flush=True)
        write_json(OUT/"progress.json",{"complete":False,"rows":rows,"failures":failures,"forwards":counts.copy()})
    for handle in hooks:
        handle.remove()
    after = {key: state_sha(net) for key, net in models.items()}
    if before != after or sha(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("Inference modified model/protocol state")
    previews = {}
    if not failures:
        for degraded in (False, True):
            chosen = [case for case in protocol["cases"] if case["synthetically_degraded"]==degraded]
            canvas = Image.new("RGB",(1280,32+len(chosen)*278),"#16181c")
            draw = ImageDraw.Draw(canvas)
            for col,label in enumerate(("input","automatic area","automatic output","reviewed area","assisted output")):
                draw.text((col*256+5,8),label,fill="white")
            for index,case in enumerate(chosen):
                y=32+index*278
                draw.text((5,y),case["id"],fill="white")
                original=pixels(ROOT/case["input"])
                auto=pixels(OUT/"automatic_masks"/(case["id"]+".png"),"L")>0
                reviewed=pixels(ROOT/case["proposal"],"L")>0
                images=[original]
                for arm,mask in (("automatic",auto),("assisted",reviewed)):
                    marked=original.astype(float)
                    marked[mask]=.55*marked[mask]+.45*np.array([16,185,129])
                    images.append(marked.round().astype(np.uint8))
                    row=next(r for r in rows if r["id"]==case["id"] and r["arm"]==arm)
                    if row["output"]:
                        images.append(pixels(OUT/row["output"]))
                    else:
                        text=Image.new("RGB",(256,256),"#16181c")
                        ImageDraw.Draw(text).text((8,110),"NO OUTPUT: less-covered image\n" if row.get("guard_rejected") else "NO OUTPUT: source-review hold\nautomatic covering miss",fill="white")
                        images.append(np.asarray(text))
                for col,array in enumerate(images):
                    canvas.paste(Image.fromarray(array),(col*256,y+18))
            stem="degraded" if degraded else "native"
            path=OUT/(stem+"_preview.png");canvas.save(path);previews[path.name]=sha(path)
            for start in (0,4):
                path=OUT/"preview"/(stem+f"_{start+1:02d}_{min(start+4,len(chosen)):02d}.png")
                canvas.crop((0,32+start*278,1280,32+min(start+4,len(chosen))*278)).save(path)
                previews[path.relative_to(OUT).as_posix()]=sha(path)
    result={"format":"dgp-broad-face-workflow-results-v1","date":"2026-10-02",
            "complete":not failures and len(rows)==32,"protocol_sha256":PROTOCOL_SHA,
            "runner_sha256":sha(__file__),"processing_sha256":protocol["processing_sha256"],
            "source_count":8,"cases":16,"expected_arm_rows":32,"rows":rows,"failures":failures,
            "generation_requests":attempted,"model_forwards":counts,"optimizer_updates":0,
            "elapsed_seconds":time.monotonic()-started,"state_before":before,"state_after":after,
            "model_states_unchanged":before==after,"masks_sha256":masks,"previews_sha256":previews,
            "training_admitted":False,"automatic_and_assisted_reported_separately":True,
            "automatic_selection_note":"No detector candidate promoted; manual review is mandatory",
            "review_pending":True}
    write_json(OUT/"results.json",result)
    print(json.dumps({"complete":result["complete"],"rows":len(rows),"failures":failures,
          "forwards":counts,"elapsed_seconds":result["elapsed_seconds"],"results_sha256":sha(OUT/"results.json")}),flush=True)


if __name__ == "__main__":
    main()
