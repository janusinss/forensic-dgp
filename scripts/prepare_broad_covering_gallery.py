"""Source-only RealOcc extension with separate operator removal proposals.

No publisher labels, memberships or historical evidence are changed. Inspect the
draft overlay before --freeze. No model inference or training occurs here.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/broad_covering_gallery_v1"
SOURCE = ROOT / "outputs/realocc_source_v1/extracted/RealOcc"
SPECS = [
    (18, "hand_eyes", "hand", False, [(64,84),(113,82),(127,93),(150,78),(205,73),(224,107),(217,155),(191,169),(161,150),(145,138),(136,131),(125,139),(115,156),(74,176),(51,162),(41,139),(45,113)]),
    (25, "hand_mouth", "hand", False, [(109,120),(125,123),(138,124),(155,133),(173,156),(186,187),(183,233),(171,255),(59,255),(55,212),(61,181),(75,157),(91,144)]),
    (362, "hair_eye", "obstructing_hair", False, [(174,76),(164,88),(157,105),(153,132),(164,159),(178,183),(201,196),(220,166),(209,128),(199,102),(188,85)]),
    (244, "knit_scarf", "scarf", False, [(57,153),(66,136),(99,123),(132,132),(154,143),(174,166),(174,220),(163,244),(54,244),(47,205)]),
    (336, "scarf_gloves", "scarf", False, [(46,142),(66,138),(94,143),(110,152),(134,158),(157,143),(207,138),(224,151),(217,215),(193,243),(60,246),(35,197)]),
    (6, "flower_mouth", "other_object", False, [(98,145),(125,147),(142,148),(168,145),(177,163),(188,197),(171,204),(166,226),(154,244),(133,237),(108,239),(93,229),(78,214),(77,188),(80,166)]),
    (7, "leaf_eye", "other_object", False, [(92,70),(116,115),(125,136),(122,177),(117,194),(101,207),(81,215),(82,240),(91,255),(0,255),(0,223),(11,193),(12,162),(24,126),(42,105),(60,102),(70,88)]),
    (26, "nearly_hidden_hands", "visibility_rejection", True, [(90,60),(130,58),(153,62),(173,77),(194,82),(203,108),(203,192),(174,230),(77,232),(53,196),(51,162),(57,121),(74,80)]),
]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare():
    if OUT.exists():
        raise ValueError("Preserve the existing draft/protocol; choose a new version")
    integrity_path = ROOT / "outputs/realocc_source_v1/source_review/integrity.json"
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    if sha(integrity_path) != "aaf1645debcbf3d43e794fdf92def776a4ba6f314dac3182a04f679f45b7114e":
        raise ValueError("RealOcc source integrity changed")
    record_by_id = {row["id"]: row for row in integrity["records"]}
    previous = json.loads((ROOT / "outputs/practical_gallery_v2/protocol.json").read_text(encoding="utf-8"))
    previous_hashes = {case["source_sha256"] for case in previous["cases"]}
    # This is an exact-file check against the known project review cohort, not
    # a claim about all 80,000 local images, identities or model pretraining.
    reviewed = json.loads((ROOT / "dataset/detector_supported_review_v1/manifest.json").read_text(encoding="utf-8"))
    def collect_hashes(value):
        if isinstance(value, dict):
            for key, entry in value.items():
                if isinstance(entry, str) and "sha256" in key:
                    previous_hashes.add(entry)
                else:
                    collect_hashes(entry)
        elif isinstance(value, list):
            for entry in value:
                collect_hashes(entry)
    collect_hashes(reviewed)
    for folder in ("inputs", "references", "proposals"):
        (OUT / folder).mkdir(parents=True, exist_ok=False)
    cases, thumbnails = [], []
    for source_id, suffix, family, reject, points in SPECS:
        name = f"val_{source_id}"
        source = SOURCE / "image" / f"{name}.jpg"
        if sha(source) != record_by_id[name]["image_sha256"] or sha(source) in previous_hashes:
            raise ValueError("Changed or previously reviewed exact source")
        native = Image.open(source).convert("RGB").resize((256,256), Image.Resampling.LANCZOS)
        reference = OUT / "references" / f"{name}_{suffix}.png"
        native.save(reference)
        mask = np.zeros((256,256), np.uint8)
        cv2.fillPoly(mask, [np.array(points, np.int32)], 1)
        mask = cv2.dilate(mask, np.ones((5,5), np.uint8))
        proposal = OUT / "proposals" / f"{name}_{suffix}.png"
        Image.fromarray(mask*255).save(proposal)
        marked = np.asarray(native).astype(float)
        marked[mask.astype(bool)] = .55*marked[mask.astype(bool)] + .45*np.array([16,185,129])
        thumbnails.append((name+" "+suffix, native, Image.fromarray(marked.round().astype(np.uint8))))
        degraded = native.filter(ImageFilter.GaussianBlur(1.2)).resize((128,128), Image.Resampling.BILINEAR).resize((256,256), Image.Resampling.BILINEAR)
        degraded = np.asarray(degraded).astype(float)+np.random.default_rng(20261002+source_id).normal(0,3,(256,256,3))
        for variant, array in (("native", np.asarray(native)), ("degraded", np.clip(degraded,0,255).round().astype(np.uint8))):
            case_id = name+"_"+suffix+"_"+variant
            path = OUT / "inputs" / (case_id+".png")
            Image.fromarray(array).save(path)
            cases.append({"id": case_id, "source": source.relative_to(ROOT).as_posix(), "source_sha256": sha(source),
                          "publisher_split": "val", "publisher_label_not_used": True, "family": family,
                          "input": path.relative_to(ROOT).as_posix(), "input_sha256": sha(path),
                          "reference": reference.relative_to(ROOT).as_posix(), "reference_sha256": sha(reference),
                          "proposal": proposal.relative_to(ROOT).as_posix(), "proposal_sha256": sha(proposal),
                          "source_review_polygon_at256": points, "proposal_margin_at256": 2,
                          "synthetically_degraded": variant == "degraded", "expected_rejection": reject,
                          "pose": "frontal/mild turn; author-aligned crop", "training_admitted": False,
                          "exposure": "All source contact sheets inspected, then native candidate review; not an unseen holdout",
                          "hidden_ground_truth": None})
    canvas = Image.new("RGB", (1024, 1096), "#16181c")
    draw = ImageDraw.Draw(canvas)
    for index, (label, native, marked) in enumerate(thumbnails):
        x, y = (index % 2)*512, (index // 2)*274
        canvas.paste(native, (x,y+18)); canvas.paste(marked, (x+256,y+18))
        draw.text((x+4,y+2), label+" | proposed removal (green)", fill="white")
    canvas.save(OUT / "source_proposals.png")
    protocol = {"format": "dgp-broad-covering-gallery-v1", "date": "2026-10-02", "frozen": False,
                "source_count": len(SPECS), "cases": cases, "new_model_forwards": 0, "optimizer_updates": 0,
                "source_integrity_sha256": sha(integrity_path), "publisher_membership_preserved": True,
                "public_training_terms_verified": False, "training_admitted": False,
                "duplicate_scope": "Exact source file hashes against the 115-source supported review manifest and prior ten-source practical cohort; not full local corpus/identity/pretraining disjointness",
                "source_sheet_sha256": sha(OUT / "source_proposals.png"), "reader_sha256": sha(__file__),
                "review_scope": "Assistant developmental removal proposals; not publisher occlusion labels or user-verified hidden anatomy",
                "criteria": ["covering removed on facial region", "plausible facial anatomy", "visible appearance retained", "no conspicuous remnant/seam", "near-total face covering rejected before model inference"],
                "planned_arms": ["baseline_detection_plus3px_review_proposal", "source_reviewed_operator_proposal"],
                "restoration": "Input-only face_workflow.py auto policy; off/on overrides verified separately",
                "metrics": "Known-visible degraded RGB MAE outside fixed operator proposal only. No real hidden-face ground truth.",
                "scope_limit": "Seven generation sources and one rejection source do not establish broad population or automatic detector accuracy"}
    (OUT / "draft.json").write_text(json.dumps(protocol, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"source_count": len(SPECS), "cases": len(cases), "frozen": False, "model_forwards": 0}))


def freeze():
    draft = OUT / "draft.json"
    frozen = OUT / "frozen_protocol.json"
    if frozen.exists():
        raise ValueError("Protocol already frozen; preserve it")
    protocol = json.loads(draft.read_text(encoding="utf-8"))
    for case in protocol["cases"]:
        for key in ("source", "input", "reference", "proposal"):
            if sha(ROOT / case[key]) != case[key+"_sha256"]:
                raise ValueError("Draft source pixels changed")
    from face_workflow import visibility_check
    for case in protocol["cases"]:
        mask = (np.asarray(Image.open(ROOT / case["proposal"]).convert("L")) >= 128).astype(np.uint8)
        result = visibility_check(mask)
        if result["rejected"] != case["expected_rejection"]:
            raise ValueError("Source proposal/visibility check disagree: "+case["id"])
        case["pre_generation_visibility_check"] = result
    protocol["frozen"] = True
    protocol["source_proposals_visually_inspected_before_generation"] = True
    protocol["draft_sha256"] = sha(draft)
    protocol["processing_sha256"] = sha(ROOT / "face_workflow.py")
    frozen.write_text(json.dumps(protocol, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"frozen_protocol_sha256": sha(frozen), "new_model_forwards": 0, "training_admitted": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    freeze() if args.freeze else prepare()
