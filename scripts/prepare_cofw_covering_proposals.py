"""Source-grounded COFW covering proposals; no model outputs or training."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/cofw_annotation_sources_v1"
OUT = ROOT / "outputs/cofw_covering_proposals_v1"
SOURCE_SHA = "88be95ff01d0807297549eb3e27f6e41f0a6c7e9c40aeb1497434b0edb96235d"

# Native cropped pixel coordinates, inspected at2x before any model forwarding.
# Polygons are approximate observed covering/feature overlap, not hidden anatomy.
POLYGONS = {
    1030: [[(20,40),(36,29),(52,20),(68,19),(80,28),(88,41),(85,54),(78,53),(78,60),(68,70),(51,76),(34,77),(22,67)],
           [(78,46),(87,42),(101,43),(115,48),(129,62),(140,77),(136,88),(120,92),(111,92),(112,79),(101,83),(93,80),(88,86),(85,74),(80,75),(76,61)]],
    935: [[(0,91),(20,80),(37,75),(57,68),(66,72),(67,79),(62,83),(42,94),(72,81),(84,77),(92,80),(94,85),(90,89),(61,105),(100,92),(110,88),(116,94),(113,101),(85,117),(108,107),(118,112),(119,119),(113,127),(83,149),(63,160),(25,160),(7,143),(0,115)]],
    955: [[(22,96),(43,97),(62,104),(76,109),(99,108),(99,87),(104,80),(111,78),(117,83),(120,103),(126,126),(142,139),(144,173),(105,184),(71,177),(32,159),(9,143),(7,135),(12,130),(23,134),(53,146),(41,140),(17,122),(9,120),(8,113),(13,111),(25,114),(40,119),(61,124),(38,116),(15,107),(12,102),(16,98)]],
    981: [[(22,91),(38,86),(46,72),(51,66),(57,69),(59,79),(58,83),(64,80),(66,69),(72,65),(77,67),(80,84),(85,84),(86,68),(91,66),(96,70),(100,89),(108,113),(116,111),(119,104),(125,109),(125,122),(121,145),(111,164),(52,164),(43,138),(24,120)]],
    1324: [[(30,77),(24,63),(24,56),(28,55),(39,58),(50,66),(67,72),(74,73),(90,62),(97,60),(102,64),(102,69),(86,82),(105,68),(113,72),(112,77),(97,92),(105,91),(113,84),(119,85),(120,92),(111,103),(106,117),(94,129),(84,153),(53,153),(56,121),(45,111),(33,92)]],
    975: [[(45,132),(58,132),(61,109),(69,103),(76,112),(77,135),(85,153),(84,165),(93,192),(86,220),(42,222),(44,202),(31,192),(25,180),(24,159),(24,147),(30,140)]],
    1001: [[(15,80),(20,74),(30,78),(40,85),(57,89),(65,98),(74,100),(81,95),(81,73),(86,65),(94,70),(98,83),(112,95),(119,116),(125,135),(109,146),(82,153),(44,148),(31,138),(25,118),(19,106),(17,99),(24,102),(39,111),(31,99),(21,92)]],
    875: [[(85,16),(89,7),(97,8),(106,13),(111,26),(117,49),(130,76),(132,127),(127,143),(108,153),(103,117),(98,95),(94,77),(91,53)]],
    920: [[(31,37),(44,30),(56,30),(72,32),(83,30),(83,24),(72,21),(61,20),(83,18),(105,24),(107,31),(96,37),(72,42),(59,45),(49,50),(42,59),(39,82),(42,99),(41,119),(36,141),(18,153),(4,146),(4,120),(12,87),(13,65),(22,54)]],
    868: [[(46,22),(61,23),(72,25),(82,31),(93,44),(99,59),(97,74),(92,84),(87,105),(78,123),(74,137),(67,140),(69,122),(77,111),(75,92),(67,80),(62,62),(58,46)],
          [(18,19),(30,22),(37,34),(39,51),(32,56),(28,39)]],
    1217: [[(23,31),(34,19),(51,21),(49,36),(44,53),(39,66),(30,81),(35,96),(32,109),(21,105),(14,96),(10,86),(10,55)],
           [(78,24),(82,25),(62,93),(56,106),(53,104),(59,83)]],
    1338: [[(88,27),(116,27),(138,44),(139,59),(130,73),(124,92),(111,112),(109,107),(112,93),(112,87),(116,76),(106,87),(96,83),(92,74),(94,67),(87,74),(82,68),(86,57),(77,63),(75,59),(84,47)]],
    913: [[(113,21),(126,30),(132,55),(130,84),(124,107),(121,128),(115,148),(108,154),(101,145),(101,123),(107,100),(111,76),(115,58),(112,41)]],
    866: [[(79,44),(90,44),(101,60),(113,75),(120,87),(116,98),(112,105),(107,106),(104,94),(97,85),(91,78),(88,69)],
          [(36,103),(55,94),(77,90),(104,93),(108,97),(88,98),(72,98),(51,102),(37,110),(22,118),(17,114)],
          [(63,137),(78,141),(99,144),(104,149),(87,148),(74,145),(67,143)]],
    1063: [[(45,20),(78,20),(75,34),(70,41),(62,46),(57,53),(52,58),(47,53),(43,53),(42,46),(39,51),(34,49),(36,39),(39,31)]],
    1343: [[(75,18),(91,18),(100,25),(109,38),(108,53),(106,67),(99,73),(91,62),(87,50),(85,38),(79,32)],
           [(72,88),(76,87),(81,93),(84,103),(77,103)]],
    1068: [[(24,76),(45,72),(63,68),(82,64),(111,60),(128,61),(139,65),(143,74),(142,94),(137,111),(126,132),(110,152),(83,163),(59,157),(43,141),(35,124),(28,105)]],
    1057: [[(21,66),(44,63),(63,60),(78,62),(96,66),(112,71),(124,79),(124,108),(113,127),(99,140),(79,150),(56,149),(36,136),(26,122),(20,103)]],
    978: [[(88,97),(96,91),(104,84),(118,79),(125,89),(124,111),(115,130),(97,144),(74,155),(46,166),(33,172),(26,164),(44,149),(66,138),(78,128),(86,113)]],
    1092: [[(109,142),(129,136),(139,137),(140,167),(128,178),(111,181),(82,181),(75,176),(79,165),(82,158),(91,151)],
           [(44,31),(60,31),(60,58),(55,76),(45,87),(29,94),(25,87),(36,80),(41,66),(45,47)]],
    876: [[(18,70),(43,74),(67,75),(89,72),(103,78),(103,100),(97,117),(77,130),(58,137),(40,130),(27,115),(19,96)]],
    1212: [[(22,34),(65,34),(73,39),(82,51),(80,67),(82,74),(80,111),(76,133),(68,142),(56,143),(47,135),(37,116),(28,91),(19,75)]],
    922: [[(33,91),(52,91),(75,96),(90,107),(92,119),(93,126),(106,132),(117,142),(126,150),(129,161),(117,177),(90,189),(68,195),(55,183),(42,166),(33,145),(28,118)]],
    869: [[(32,20),(56,20),(56,107),(52,114),(61,126),(65,137),(62,150),(53,160),(40,164),(30,155),(31,139),(22,138),(21,122),(28,112),(28,45)]],
    954: [[(42,143),(70,135),(106,134),(126,131),(130,148),(128,162),(120,170),(97,187),(73,190),(56,179),(45,162)]],
    901: [[(41,112),(54,112),(68,115),(76,123),(75,137),(71,151),(53,153),(33,146),(34,127)]],
    854: [[(31,23),(42,25),(54,20),(70,20),(81,23),(86,27),(88,24),(112,21),(128,24),(132,30),(128,43),(122,51),(108,54),(97,52),(90,48),(84,35),(81,39),(80,48),(74,53),(61,56),(49,52),(43,45),(40,33)]],
    917: [[(80,63),(81,47),(88,34),(99,24),(117,18),(129,22),(139,31),(141,44),(138,56),(130,65),(117,68),(102,70),(89,65),(81,71),(85,84),(79,95),(67,109),(52,111),(41,105),(38,91),(39,76),(48,66),(60,61),(71,62),(77,61)]],
    1143: [[(3,39),(13,41),(19,32),(29,30),(50,35),(66,38),(73,44),(80,43),(92,46),(112,48),(122,54),(122,67),(119,82),(113,93),(100,96),(88,92),(84,81),(79,65),(73,58),(69,66),(60,76),(46,78),(31,74),(19,64),(14,53)]],
    930: [[(26,88),(50,74),(70,58),(78,57),(90,69),(112,78),(123,96),(117,124),(106,147),(81,160),(52,157),(33,143),(23,125),(17,106)],
          [(10,98),(28,105),(29,110),(11,104),(7,99)], [(15,124),(29,135),(36,138),(34,146),(16,133)],
          [(119,83),(122,90),(114,106),(106,110),(104,104)]],
    906: [[(15,10),(108,10),(118,25),(120,41),(116,69),(102,87),(95,85),(90,76),(86,71),(79,76),(67,83),(44,78),(23,69),(12,47)]],
}
HOLES = {906: [[(16,33),(22,33),(28,35),(34,39),(39,41),(39,44),(28,43),(23,42),(17,40)],
               [(78,45),(88,47),(102,47),(104,52),(95,58),(84,53),(79,53)]]}
UNKNOWN = {876: [[(79,24),(111,28),(116,66),(104,77),(81,73)]]}
EXTRA_FAMILIES = {1343: ["hand"], 1092: ["obstructing_hair"], 922: ["hand"], 869: ["hand"]}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def raster(size, polygons):
    width, height = size
    target = np.zeros((height, width), np.uint8)
    for polygon in polygons:
        if len(polygon) < 3 or any(not (0 <= x < width and 0 <= y < height) for x, y in polygon):
            raise ValueError("Native polygon leaves observed crop")
        cv2.fillPoly(target, [np.asarray(polygon, np.int32)], 1)
    return target.astype(bool)


def main():
    if OUT.exists() or sha(SOURCE / "manifest.json") != SOURCE_SHA:
        raise ValueError("Preserve proposals and verified source selection")
    source = json.loads((SOURCE / "manifest.json").read_text())
    OUT.mkdir()
    for role in ("images", "masks", "valid", "source_valid", "full_proposals"):
        (OUT / role).mkdir()
    records = []
    for original in source["records"]:
        index = original["matlab_train_index"]
        for key in ("source", "native_crop"):
            if sha(ROOT / original[key]) != original[key + "_sha256"]:
                raise ValueError("Native source asset changed")
        with Image.open(ROOT / original["native_crop"]) as im:
            rgb = np.asarray(im.convert("RGB")).copy()
        width, height = original["crop_native_size"]
        polygons = POLYGONS.get(index, [])
        holes = HOLES.get(index, [])
        full = raster((width, height), polygons) & ~raster((width, height), holes)
        band = cv2.dilate(full.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) ^ cv2.erode(
            full.astype(np.uint8), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=1).astype(bool)
        # A truncated covering touches a source-crop edge: observed RGB stays,
        # but that edge is not declared an exact positive/negative boundary.
        edges = []
        for name, values in (("top", full[0]), ("bottom", full[-1]), ("left", full[:,0]), ("right", full[:,-1])):
            if values.any():
                edges.append(name)
                if name == "top": band[0] = True
                elif name == "bottom": band[-1] = True
                elif name == "left": band[:,0] = True
                else: band[:,-1] = True
        unknown_polygons = UNKNOWN.get(index, [])
        valid = ~band & ~raster((width, height), unknown_polygons)
        core = full & valid
        if bool(polygons) != bool(core.any()):
            raise ValueError("Covered source lost all supervised positives")
        scale = 255 / max(width - 1, height - 1)
        affine = np.array([[scale, 0, (255 - (width - 1) * scale) / 2],
                           [0, scale, (255 - (height - 1) * scale) / 2]], np.float64)
        def warp(a):
            return cv2.warpAffine(a.astype(np.uint8), affine, (256,256), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        supported = warp(np.ones(full.shape, np.uint8))
        pixels = cv2.warpAffine(rgb, affine, (256,256), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(96,96,96))
        pixels[~supported] = 96
        arrays = {"images": pixels, "masks": warp(core), "valid": warp(valid),
                  "source_valid": supported, "full_proposals": warp(full)}
        if (arrays["masks"] & ~arrays["valid"]).any() or (arrays["valid"] & ~supported).any():
            raise ValueError("Invalid support containment")
        files = {}
        for role, array in arrays.items():
            path = OUT / role / f"train_{index:04d}.png"
            Image.fromarray(array if role == "images" else array.astype(np.uint8) * 255).save(path)
            files[role] = {"path": path.relative_to(OUT).as_posix(), "sha256": sha(path)}
        family = "costume_mask" if index == 906 else original["family_proposal"]
        if index == 954: family = "other_cloth_or_object"
        records.append({**original, "family": family, "additional_families": EXTRA_FAMILIES.get(index, []),
            "kind": "covered" if polygons else "uncovered", "native_polygons": polygons,
            "native_preserved_holes": holes, "native_unknown_polygons": unknown_polygons,
            "native_uncertainty_radius": 1, "native_unknown_crop_edges": edges,
            "native_full_pixels": int(full.sum()), "native_core_pixels": int(core.sum()),
            "native_supervised_pixels": int(valid.sum()), "affine": affine.tolist(), "files": files,
            "annotation_status": "draft-requires-independent-audit-and-overlay-review",
            "annotation_quality": "Assistant approximate facial-covering overlap; +/-1native boundary/crop-edge uncertainty; no expert or hidden-face truth",
            "training_admitted": False})
    pages = []
    for start in range(0, len(records), 6):
        entries = records[start:start + 6]
        sheet = Image.new("RGB", (1024, 40 + 280 * len(entries)), "#16181c")
        draw = ImageDraw.Draw(sheet)
        for column, title in enumerate(("source", "full covering proposal", "supervised core", "purple unknown/padding")):
            draw.text((column * 256 + 3, 8), title, fill="white")
        for i, row in enumerate(entries):
            arrays = {}
            for role in ("images", "full_proposals", "masks", "valid"):
                with Image.open(OUT / row["files"][role]["path"]) as im:
                    arrays[role] = np.asarray(im).copy()
            rgb = arrays["images"]
            panels = [rgb]
            for role in ("full_proposals", "masks"):
                marked = rgb.copy()
                selected = arrays[role] == 255
                marked[selected] = (.55 * rgb[selected] + .45 * np.array([16,185,129])).round().astype(np.uint8)
                panels.append(marked)
            marked = rgb.copy()
            selected = arrays["valid"] == 0
            marked[selected] = (.45 * rgb[selected] + .55 * np.array([165,85,215])).round().astype(np.uint8)
            panels.append(marked)
            y = 40 + i * 280
            draw.text((3,y), f"{row['id']} {row['family']}; draft, source overlap review pending", fill="white")
            for column, panel in enumerate(panels):
                sheet.paste(Image.fromarray(panel), (column * 256,y + 19))
        path = OUT / f"proposal_rows_{start+1:02d}_{start+len(entries):02d}.png"
        sheet.save(path)
        pages.append({"path": path.relative_to(OUT).as_posix(), "sha256": sha(path), "ids": [r["id"] for r in entries]})
    result = {"format": "dgp-cofw-covering-label-draft-v1", "date": "2026-10-03",
              "source_selection_sha256": SOURCE_SHA, "builder_sha256": sha(__file__),
              "source_url": source["source_url"], "doi": source["doi"], "attribution": source["attribution"],
              "license_record_sha256": source["license_record_sha256"], "records": records, "sheets": pages,
              "target_semantics": "Full observed covering overlapping facial area; preserve source elsewhere; unknown boundary/crop edges excluded; no hidden facial content supervision",
              "margin_policy": "No added training removal margin. Original inference3px proposal expansion remains a separate policy.",
              "source_review_correction": {"cofw_train_0906": "Initial contact-sheet candidate incorrectly tagged hand; native source is costume mask. Historical draft untouched."},
              "model_forwards": 0, "optimizer_updates": 0, "training_admitted": False,
              "held_out_practical_gallery_used_for_training": False, "test_rgb_read": False}
    with (OUT / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"drafts": len(records), "covered": sum(r["kind"] == "covered" for r in records),
                      "clear": sum(r["kind"] == "uncovered" for r in records), "training_admitted": False,
                      "manifest_sha256": sha(OUT / "manifest.json")}))


if __name__ == "__main__":
    main()
