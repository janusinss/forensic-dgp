"""Finite CPU inference/geometry audit on six already-reviewed native dev faces.

No reserved inputs, target metrics, training, new split or quality promotion.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cctv_dgp_pilot import state_hash
from dgp_face_restoration import sha
from dgp_face_workflow_v3 import DGPFaceWorkflow, decode_crop, make_dgp_bundle
from face_workflow import png_bytes

CAP_SECONDS = 300
RAW_TOLERANCE = 1e-5
CORE = ["dev_16to23_01", "dev_16to23_05", "dev_24to39_01",
        "dev_24to39_02", "dev_24to39_06", "dev_ge40_04"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def audit(destination):
    destination = destination.resolve()
    if ROOT not in destination.parents:
        raise ValueError("Audit output must remain in the workspace")
    destination.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    subset_root = ROOT / "outputs/cctv_native_development_v2"
    native_root = ROOT / "outputs/cctv_dgp_native_pilot_review_v1"
    subset = read(subset_root / "frozen_subset.json")
    reviews = read(subset_root / "input_review.json")
    cached = read(native_root / "results.json")
    if cached["core_coarse_frontal_ids"] != CORE or not reviews["reviewed_before_model_outputs"]:
        raise ValueError("Changed input-only development criteria")
    cases = {row["id"]: row for row in subset["cases"] if row["role"] == "development"}
    if len(cases) != 24:
        raise ValueError("Changed development inventory")
    write(destination / "plan.json", {"date": "2026-10-05", "core_ids": CORE,
          "input_review_sha256": sha(subset_root / "input_review.json"),
          "frozen_subset_sha256": sha(subset_root / "frozen_subset.json"),
          "cached_native_results_sha256": sha(native_root / "results.json"),
          "cap_seconds": CAP_SECONDS, "maximum_dgp_forwards": 8,
          "cached_cpu_raw_tolerance": RAW_TOLERANCE,
          "restoration_quality_qualification": False,
          "scope": "Integration parity, preservation and declared operator input-review routing only"})
    engine = DGPFaceWorkflow(device="cpu")
    model = engine._restorer()
    before = state_hash(model)
    ledger, rendered = [], []
    forwards = 0
    try:
        for review in reviews["rows"]:
            if time.perf_counter() - started > CAP_SECONDS:
                raise TimeoutError("300-second CPU integration cap exceeded")
            case = cases[review["id"]]
            path = subset_root / case["source_file"]
            if sha(path) != case["source_sha256"]:
                raise ValueError("Native development input hash differs")
            rgb = decode_crop(path.read_bytes())
            mask = np.zeros(rgb.shape[:2], np.uint8)
            decision = "usable" if case["id"] in CORE else (
                "out_of_scope" if "profile" in review["pose_review"] else "needs_clearer")
            if decision != "usable":
                try:
                    engine.generate(rgb, mask, "on", decision, True)
                except ValueError as exc:
                    ledger.append({"id": case["id"], "input_decision": decision,
                                   "rejected_before_forward": True, "message": str(exc)})
                    continue
                raise ValueError("Operator rejection did not stop generation")
            result = engine.generate(rgb, mask, "on", decision, True)
            forwards += 1
            off = engine.generate(rgb, mask, "off", decision, True)
            np.testing.assert_array_equal(off["output"], off["original"])
            cached_row = next(row for row in cached["rows"] if row["id"] == case["id"])
            raw_path = native_root / cached_row["candidate_stages"]["camera_identity"]
            if sha(raw_path) != cached["artifacts_sha256"][cached_row["candidate_stages"]["camera_identity"]]:
                raise ValueError("Changed cached native raw")
            with np.load(raw_path, allow_pickle=False) as archive:
                cached_raw = archive["raw_rgb"]
                delta = float(np.abs(result["raw"] - cached_raw).max())
            if delta > RAW_TOLERANCE:
                raise ValueError("App DGP differs from retained native baseline")
            for name, image in (("input", result["original"]), ("off", off["output"]), ("on", result["output"])):
                (destination / (case["id"] + "_" + name + ".png")).write_bytes(png_bytes(image))
            np.save(destination / (case["id"] + "_raw.npy"), result["raw"], allow_pickle=False)
            write(destination / (case["id"] + "_metadata.json"), result["metadata"])
            columns = [result["original"], np.floor(cached_raw * np.float32(255)).astype(np.uint8),
                       np.floor(result["raw"] * np.float32(255)).astype(np.uint8), result["output"]]
            rendered.append((case["id"], columns))
            if case["id"] == CORE[0]:
                again = engine.generate(rgb, mask, "on", decision, True)
                forwards += 1
                np.testing.assert_array_equal(result["raw"], again["raw"])
                automatic = engine.generate(rgb, mask, "auto", decision, True)
                forwards += int(automatic["metadata"]["restoration_applied"])
                if automatic["metadata"]["restoration_applied"] != automatic["metadata"]["quality_signals"]["suggest_restoration"]:
                    raise ValueError("Automatic selection differs from input-only signal")
                bundle = make_dgp_bundle(rgb, mask, result["original"], result["mask"], result["output"], result["metadata"], result["raw"])
                (destination / "sample-review.zip").write_bytes(bundle)
            ledger.append({"id": case["id"], "input_decision": decision,
                           "raw_cached_max_delta": delta, "restoration_signal": result["metadata"]["quality_signals"],
                           "off_prepared_pixels_exact": True, "output_size": [256, 256],
                           "native_metrics": None})
        after = state_hash(model)
        if before != after or forwards > 8:
            raise ValueError("Changed state or exceeded finite forward count")
        sheet = Image.new("RGB", (4 * 268, 36 + len(rendered) * 282), "#020617")
        draw = ImageDraw.Draw(sheet)
        for index, heading in enumerate(("Prepared input", "Retained raw", "App raw", "App display")):
            draw.text((index * 268 + 6, 8), heading, fill="white")
        for row, (name, columns) in enumerate(rendered):
            y = 36 + row * 282
            draw.text((6, y), name, fill="white")
            for index, image in enumerate(columns):
                sheet.paste(Image.fromarray(image), (index * 268 + 6, y + 18))
        sheet.save(destination / "native-core-six.png")
        seconds = time.perf_counter() - started
        if seconds > CAP_SECONDS:
            raise TimeoutError("300-second CPU integration cap exceeded")
        artifacts = {p.name: sha(p) for p in destination.iterdir() if p.is_file()}
        receipt = {"complete": True, "seconds": seconds, "cap_seconds": CAP_SECONDS,
                   "core_cases": 6, "operator_rejected_cases": 18, "dgp_forwards": forwards,
                   "completion_forwards": 0, "state_before": before, "state_after": after,
                   "optimizer_updates": 0, "backward_calls": 0, "reserved_evaluation_used": False,
                   "structure_qualification": "Operator review replay, not an automatic classifier",
                   "native_usefulness_qualified": False, "visual_review_pending": True,
                   "covering_family_review": "Not performed in this restoration audit",
                   "checkpoint": engine.restorer_provenance, "rows": ledger, "artifacts_sha256": artifacts}
        write(destination / "receipt.json", receipt)
        print(json.dumps({"complete": True, "seconds": seconds, "dgp_forwards": forwards,
                          "destination": str(destination), "state_unchanged": before == after}))
    except Exception as exc:
        write(destination / "failure.json", {"complete": False, "seconds": time.perf_counter() - started,
                                             "error": repr(exc), "dgp_forwards": forwards, "rows": ledger})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/dgp_app_v3_inference_audit")
    audit(parser.parse_args().output)
