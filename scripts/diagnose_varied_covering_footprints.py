"""Saved-pixel diagnosis and cached assisted comparison; no model imports/forwards."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/varied_covering_footprint_diagnostic_v1"
PRACTICAL = "outputs/varied_covering_practical_review_v1/"
COMPLETION = "outputs/varied_covering_completion_review_v1/"
PALETTE = "outputs/grayscale_palette_comparison_v2/"
BROAD = "outputs/broad_face_workflow_outputs_v1/"
ARMS = ("existing91", "varied133")
CACHED_IDS = ("val_18_hand_eyes_degraded", "val_25_hand_mouth_degraded",
              "val_362_hair_eye_degraded", "val_244_knit_scarf_degraded",
              "val_336_scarf_gloves_degraded", "val_6_flower_mouth_degraded",
              "val_7_leaf_eye_degraded")
PINS = {
    "outputs/varied_covering_practical_protocol_v1/protocol.json": "15f6ca20827c4221f7e12630f7d3fab54d7eceae51235eb9401c6f529563382e",
    "outputs/varied_covering_practical_validation_v1/verification.json": "fb76dce5dbe793d536d6058893edcb736e299069dd3664ffa09baf8b16d09e99",
    COMPLETION + "frozen_protocol.json": "334d7b31ad3ff7b616af7ad06048498e4efb5fdf3f1172253d0ffc570fb56845",
    COMPLETION + "results.json": "6feceb94e9c21c87f88c1b9fbdbfa6bc6f02ebe9745b35df061f60c02df841cb",
    COMPLETION + "independent_verification.json": "5cfb067b650c1069f68749a2d58ae0882d5b76e18de18e424e59ae41af117ece",
    COMPLETION + "visual_review.json": "e08298b5f213c8546d4435c7d0c2451a99672c9a4aa0e8f3ee09a596bd430ae3",
    BROAD + "results.json": "afe8cb6b2d768460082890fe8ded09dd125e19f599f54f55a496e2a4073a94d3",
    PALETTE + "results.json": "3d6e4b90fc3b6f39c230fa06dabc783412542f664d21a9a844d552be95214242",
    PALETTE + "independent_verification.json": "c026df0df3b66eb7b1e363b007037bde4a2947a3fda639e7dc57b35d9e48241e",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def pixels(name, mask=False):
    with Image.open(ROOT / name) as image:
        result = np.asarray(image.convert("L" if mask else "RGB")).copy()
    require(result.shape == ((256, 256) if mask else (256, 256, 3)), "Changed crop geometry")
    if mask:
        require(np.isin(result, (0, 255)).all(), "Nonbinary removal area")
        return result == 255
    return result


def guard(mask):
    """Independent 256px recount of the unchanged conditional visibility geometry."""
    yy, xx = np.mgrid[:256, :256]
    face = ((xx - 127.5) / 83) ** 2 + ((yy - 137) / 104) ** 2 <= 1
    bands = {"left_eye": (60, 84, 119, 133), "right_eye": (137, 84, 196, 133),
             "nose": (108, 119, 149, 166), "mouth": (82, 165, 174, 208)}
    fractions = {k: float(mask[y1:y2, x1:x2].mean())
                 for k, (x1, y1, x2, y2) in bands.items()}
    covered = float(mask[face].mean())
    all_features = (fractions["left_eye"] >= .85 and fractions["right_eye"] >= .85
                    and fractions["nose"] >= .9 and fractions["mouth"] >= .9)
    return {"rejected": bool(covered >= .8 or all_features or mask.mean() >= .85),
            "face_covered_fraction": covered, "feature_covered_fractions": fractions}


def prepare():
    require(not OUT.exists(), "Preserve existing diagnostic evidence")
    for name, pin in PINS.items():
        require(sha(ROOT / name) == pin, "Audited reference changed: " + name)
    source = read("outputs/varied_covering_practical_protocol_v1/protocol.json")
    saved = read(PRACTICAL + "results.json")
    completed = read(COMPLETION + "results.json")
    palette = read(PALETTE + "results.json")
    broad = read(BROAD + "results.json")
    cases = source["cases"]
    require(len(cases) == 36 and len({c["id"] for c in cases}) == 36, "Keep all fixed cases")
    require(read(PALETTE + "independent_verification.json")["verified"] is True,
            "Cached assisted output verification required")
    require(saved["complete"] and completed["complete"] and broad["complete"], "Incomplete prior run")
    assets = {**PINS, **source["assets_sha256"]}
    assets["scripts/diagnose_varied_covering_footprints.py"] = sha(__file__)
    assets[PRACTICAL + "results.json"] = sha(ROOT / PRACTICAL / "results.json")
    records = {row["id"]: row for row in saved["rows"]}
    require(set(records) == {c["id"] for c in cases}, "Changed practical membership")
    for row in records.values():
        for item in row["models"].values():
            for key in ("proposal", "raw", "probability"):
                if key in item:
                    name = item[key] if item["cached"] else PRACTICAL + item[key]
                    assets[name] = sha(ROOT / name)
    assisted = []
    for name in CACHED_IDS:
        case = next(c for c in cases if c["id"] == name)
        old = next(r for r in broad["rows"] if r["id"] == name and r["arm"] == "assisted")
        cached = next(r for r in palette["rows"] if r["id"] == name and r["arm"] == "assisted")
        require(old["mask"] == case["reviewed"] and old["mask_root"] == "repository",
                "Cached assisted mask differs")
        require(old["metadata"]["original_rgb_sha256"] == hashlib.sha256(pixels(case["input"]).tobytes()).hexdigest(),
                "Cached assisted input differs")
        outputs = {"cached_assisted": PALETTE + cached["output"]}
        for arm in ARMS:
            new = next(r for r in completed["rows"] if r["id"] == name and r["arm"] == arm)
            for key in ("restoration_requested", "restoration_applied", "restoration_fidelity", "visible_restoration_blend"):
                require(new["metadata"][key] == old["metadata"][key], "Restoration policies differ")
            require(cached["restoration_applied"] == new["metadata"]["restoration_applied"], "Cached palette condition differs")
            if case["family"] != "obstructing_hair":
                require(new["metadata"]["completion"] == old["metadata"]["completion"], "Completion provenance differs")
            require(new["metadata"]["visible_restoration"] == old["metadata"]["visible_restoration"], "Restorer provenance differs")
            outputs[arm] = COMPLETION + new["output"]
        for path in outputs.values():
            assets[path] = sha(ROOT / path)
        require(assets[outputs["cached_assisted"]] == cached["output_sha256"], "Cached assisted PNG changed")
        assisted.append({"id": name, "outputs": outputs, "cached_prior_inference": True,
                         "assisted_mask": case["reviewed"], "hidden_ground_truth": None})
    protocol = {"format": "dgp-varied-covering-footprint-diagnostic-v1", "date": "2026-10-03",
        "frozen_before_saved_pixel_diagnostic": True, "cases": cases, "cached_assisted_comparisons": assisted,
        "assets_sha256": assets, "models": ["retained_app_default", "camera91_source", *ARMS],
        "threshold": .5, "proposal_margin_256_pixels": 3, "core_erosion_256_pixels": 2,
        "budget": {"proposal_recounts": 144, "probability_arrays": 72, "cached_assisted_outputs": 7,
                   "new_model_forwards": 0, "optimizer_updates": 0, "wall_seconds": 120},
        "criteria": ["Separate missing operator footprint from generated quality; no hidden facial targets",
                     "Compare saved hair confidence and predicted/reviewed near-hidden guard coverage",
                     "Reinspect seven cached assisted outputs separately from current automatic estimates"],
        "scope": "Previously inspected development/operator footprints with small margins; not exact occlusion truth",
        "training": False, "application_checkpoint_selection": False, "historical_gates_unchanged": True}
    OUT.mkdir()
    write(OUT / "frozen_protocol.json", protocol)
    write(OUT / "binding.json", {"protocol_sha256": sha(OUT / "frozen_protocol.json"), "preparation_only": True,
                                 "model_forwards": 0, "optimizer_updates": 0})
    print(json.dumps({"prepared": True, "cases": 36, "cached_assisted": 7, "new_model_forwards": 0}))


def run():
    require(not (OUT / "results.json").exists(), "Preserve completed diagnostic")
    p = read(OUT.relative_to(ROOT).as_posix() + "/frozen_protocol.json")
    require(sha(OUT / "frozen_protocol.json") == read(OUT.relative_to(ROOT).as_posix() + "/binding.json")["protocol_sha256"],
            "Frozen diagnostic changed")
    require(p["training"] is False and p["budget"]["new_model_forwards"] == 0 and p["threshold"] == .5,
            "No inference/training/threshold change permitted")
    for name, pin in p["assets_sha256"].items():
        require(sha(ROOT / name) == pin, "Frozen input changed: " + name)
    started = time.perf_counter()
    saved = {r["id"]: r for r in read(PRACTICAL + "results.json")["rows"]}
    rows, probabilities = [], 0
    for case in p["cases"]:
        require(time.perf_counter() - started < p["budget"]["wall_seconds"], "Diagnostic wall budget exceeded")
        reference = pixels(case["reviewed"], True)
        core = np.asarray(Image.fromarray(reference.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(5))) == 255
        row = {"id": case["id"], "family": case["family"], "synthetically_degraded": case["synthetically_degraded"],
               "pose_scope": case["pose_scope"], "expected_rejection": case["expected_rejection"],
               "operator_pixels": int(reference.sum()), "operator_guard": guard(reference), "models": {}}
        for arm in p["models"]:
            item = saved[case["id"]]["models"][arm]
            path = item["proposal"] if item["cached"] else PRACTICAL + item["proposal"]
            proposed = pixels(path, True)
            g = guard(proposed)
            require(all(g[k] == item["metrics"]["guard"][k] for k in g), "Saved guard reproduction differs")
            missing = reference & ~proposed
            metrics = {"selected_pixels": int(proposed.sum()), "missing_operator_pixels": int(missing.sum()),
                       "missing_fraction": float(missing.sum() / reference.sum()) if reference.any() else None,
                       "missing_core_fraction": float((core & ~proposed).sum() / core.sum()) if core.any() else None,
                       "guard": g, "saved_probability_summary": None}
            if arm in ARMS:
                probability = np.load(ROOT / PRACTICAL / item["probability"], allow_pickle=False)
                require(probability.dtype == np.float32 and probability.shape == (256, 256)
                        and np.isfinite(probability).all() and probability.min() >= 0 and probability.max() <= 1,
                        "Invalid saved probability")
                require(np.array_equal(probability >= .5, pixels(PRACTICAL + item["raw"], True)), "Saved threshold differs")
                sample = probability[reference]
                metrics["saved_probability_summary"] = {"crop_max": float(probability.max()),
                    "operator_median": float(np.median(sample)) if sample.size else None,
                    "operator_p95": float(np.quantile(sample, .95)) if sample.size else None,
                    "operator_max": float(sample.max()) if sample.size else None}
                probabilities += 1
            row["models"][arm] = metrics
        rows.append(row)
    hard = [c for c in p["cases"] if c["family"] in ("obstructing_hair", "visibility_rejection")]
    sheet = Image.new("RGB", (1024, 28 + 284 * len(hard)), "#161616")
    draw = ImageDraw.Draw(sheet)
    for col, text in enumerate(("input", "existing91", "varied133", "reviewed removal")):
        draw.text((col * 256 + 4, 8), text, fill="white")
    for index, case in enumerate(hard):
        rgb = pixels(case["input"])
        y = 28 + index * 284
        draw.text((4, y + 3), case["id"], fill="white")
        sheet.paste(Image.fromarray(rgb), (0, y + 28))
        for col, arm in enumerate((*ARMS, "reviewed"), 1):
            if arm == "reviewed": path = case["reviewed"]
            else: path = PRACTICAL + saved[case["id"]]["models"][arm]["proposal"]
            mask = pixels(path, True)
            overlay = rgb.copy()
            overlay[mask] = (.45 * rgb[mask] + .55 * np.array([255, 64, 48])).astype(np.uint8)
            sheet.paste(Image.fromarray(overlay), (col * 256, y + 28))
    sheet.save(OUT / "hair_and_visibility_preview.png")
    comparison = Image.new("RGB", (1280, 28 + 284 * len(p["cached_assisted_comparisons"])), "#161616")
    draw = ImageDraw.Draw(comparison)
    for col, text in enumerate(("input", "existing91 automatic", "varied133 automatic", "prior assisted output", "assisted removal area")):
        draw.text((col * 256 + 4, 8), text, fill="white")
    for index, entry in enumerate(p["cached_assisted_comparisons"]):
        case = next(c for c in p["cases"] if c["id"] == entry["id"])
        y = 28 + index * 284
        draw.text((4, y + 3), entry["id"], fill="white")
        rgb = pixels(case["input"])
        comparison.paste(Image.fromarray(rgb), (0, y + 28))
        for col, arm in enumerate((*ARMS, "cached_assisted"), 1):
            comparison.paste(Image.fromarray(pixels(entry["outputs"][arm])), (col * 256, y + 28))
        mask = pixels(entry["assisted_mask"], True)
        overlay = rgb.copy()
        overlay[mask] = (.45 * rgb[mask] + .55 * np.array([255, 64, 48])).astype(np.uint8)
        comparison.paste(Image.fromarray(overlay), (1024, y + 28))
    comparison.save(OUT / "cached_assisted_comparison.png")
    require(len(rows) == 36 and probabilities == 72, "Incomplete saved-pixel diagnosis")
    for name, pin in p["assets_sha256"].items():
        require(sha(ROOT / name) == pin, "Source changed during diagnostic")
    write(OUT / "results.json", {"format": p["format"], "date": "2026-10-03", "complete": True,
        "protocol_sha256": sha(OUT / "frozen_protocol.json"), "rows": rows,
        "proposal_recounts": 144, "saved_probability_arrays": probabilities,
        "cached_assisted_outputs_reused": 7, "new_model_forwards": 0, "optimizer_updates": 0,
        "seconds": time.perf_counter() - started, "training": False, "promoted": False,
        "source_bindings_verified": len(p["assets_sha256"]),
        "preview_sha256": {name: sha(OUT / name) for name in ("hair_and_visibility_preview.png", "cached_assisted_comparison.png")},
        "visual_review_pending": True,
        "limitations": ["Operator areas include declared margins; these are not exact occlusion truth or hidden face targets",
                        "Seven assisted outputs are verified prior caches, not new generator executions",
                        "No threshold/margin tuning, guard changes, checkpoint selection or original425 gate evaluation"]})
    print(json.dumps({"complete": True, "proposal_recounts": 144, "saved_probabilities": 72,
                      "cached_assisted_outputs": 7, "new_model_forwards": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    prepare() if args.prepare else run()
