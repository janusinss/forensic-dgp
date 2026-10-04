"""Version twelve ambiguous native raster pixels; preserve source/full footprints."""
import copy
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw

from scripts.prepare_cofw_covering_proposals import sha, raster

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "outputs/cofw_covering_proposals_v2"
OUT = ROOT / "outputs/cofw_covering_proposals_v3"
PARENT_SHA = "843602ba89b90d23557af6e09ba6a0585ebffef5321def33d3076e8dc29003a6"
POINTS = {935: [(52,89),(50,90),(48,91),(66,103),(91,114),(89,115)],
          955: [(50,121),(51,121),(54,122),(44,142),(46,143)],
          1324: [(99,72)]}


def main():
    if OUT.exists() or sha(PARENT / "manifest.json") != PARENT_SHA:
        raise ValueError("Preserve versions; require reviewed V2 boundary refinement")
    data = copy.deepcopy(json.loads((PARENT / "manifest.json").read_text()))
    OUT.mkdir()
    for role in ("images","masks","valid","source_valid","full_proposals"):
        (OUT / role).mkdir()
    for row in data["records"]:
        for role, asset in row["files"].items():
            assert sha(PARENT / asset["path"]) == asset["sha256"]
            shutil.copyfile(PARENT / asset["path"],OUT / asset["path"])
        index = row["matlab_train_index"]
        uncertain = POINTS.get(index,[])
        row["native_unknown_points_xy"] = [list(p) for p in uncertain]
        if not uncertain:
            continue
        width,height = row["crop_native_size"]
        full = raster((width,height),row["native_polygons"]) & ~raster((width,height),row["native_preserved_holes"])
        band = cv2.dilate(full.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool) ^ cv2.erode(
            full.astype(np.uint8),np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=1).astype(bool)
        valid = ~band & ~raster((width,height),row["native_unknown_polygons"])
        for edge in row["native_unknown_crop_edges"]:
            if edge == "top": valid[0] = False
            elif edge == "bottom": valid[-1] = False
            elif edge == "left": valid[:,0] = False
            elif edge == "right": valid[:,-1] = False
        for x,y in uncertain:
            assert valid[y,x]
            valid[y,x] = False
        affine = np.asarray(row["affine"],np.float64)
        for role,array in (("masks",full & valid),("valid",valid)):
            output = cv2.warpAffine(array.astype(np.uint8),affine,(256,256),flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT,borderValue=0) != 0
            target = OUT / row["files"][role]["path"]
            Image.fromarray(output.astype(np.uint8)*255).save(target)
            row["files"][role]["sha256"] = sha(target)
        row["native_core_pixels"] = int((full & valid).sum())
        row["native_supervised_pixels"] = int(valid.sum())
    # Save actual V3 support previews; inherited V2 page hashes are not relabelled.
    pages = []
    for start in range(0,len(data["records"]),6):
        entries = data["records"][start:start+6]
        sheet = Image.new("RGB",(1024,40+280*len(entries)),"#16181c")
        draw = ImageDraw.Draw(sheet)
        for col,title in enumerate(("source","full covering proposal","supervised core","purple unknown/padding")):
            draw.text((col*256+3,8),title,fill="white")
        for i,row in enumerate(entries):
            arrays = {}
            for role in ("images","full_proposals","masks","valid"):
                with Image.open(OUT / row["files"][role]["path"]) as image:
                    arrays[role] = np.asarray(image).copy()
            rgb = arrays["images"]; panels = [rgb]
            for role in ("full_proposals","masks"):
                selected = arrays[role] == 255
                marked = rgb.copy()
                marked[selected] = (.55*rgb[selected]+.45*np.array([16,185,129])).round().astype(np.uint8)
                panels.append(marked)
            selected = arrays["valid"] == 0
            marked = rgb.copy()
            marked[selected] = (.45*rgb[selected]+.55*np.array([165,85,215])).round().astype(np.uint8)
            panels.append(marked)
            y = 40+i*280
            draw.text((3,y),f"{row['id']} {row['family']}; V3 uncertainty; no training/model output",fill="white")
            for col,panel in enumerate(panels): sheet.paste(Image.fromarray(panel),(col*256,y+19))
        target = OUT / f"proposal_rows_{start+1:02d}_{start+len(entries):02d}.png"
        sheet.save(target)
        pages.append({"path":target.relative_to(OUT).as_posix(),"sha256":sha(target),"ids":[r["id"] for r in entries]})
    data.update(format="dgp-cofw-covering-label-draft-v3",sheets=pages,
                parent_manifest_sha256=PARENT_SHA,support_refiner_sha256=sha(__file__),
                native_raster_disagreement_policy="Exactly12 declared native pixels from V1 audit are uncertain training-only targets; RGB/full proposals unchanged. No validation/test support changes.")
    with (OUT / "manifest.json").open("x",encoding="utf-8",newline="\n") as stream:
        stream.write(json.dumps(data,indent=2)+"\n")
    lineage = {"format":"dgp-cofw-raster-uncertainty-refinement-v3","parent_manifest_sha256":PARENT_SHA,
               "new_manifest_sha256":sha(OUT / "manifest.json"),"refiner_sha256":sha(__file__),
               "geometry_failure_evidence":"outputs/cofw_covering_data_validation_v1/verification.json",
               "geometry_failure_evidence_sha256":sha(ROOT / "outputs/cofw_covering_data_validation_v1/verification.json"),
               "native_points_xy":POINTS,"unknown_points_added":12,"source_and_full_footprints_unchanged":True,
               "validation_test_or_original_gate_changed":False,"training_admitted":False,
               "model_forwards":0,"optimizer_updates":0}
    with (OUT / "refinement.json").open("x",encoding="utf-8",newline="\n") as stream:
        stream.write(json.dumps(lineage,indent=2)+"\n")
    print(json.dumps({"sources":42,"native_uncertain_pixels_added":12,"training_admitted":False,
                      "manifest_sha256":sha(OUT / "manifest.json")}))


if __name__ == "__main__":
    main()
