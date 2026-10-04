"""Recheck subset membership and reconstruct saved inference pixels; no NN imports."""
from collections import Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/cctv_native_development_v2"
OUT = ROOT / "outputs/cctv_native_comparison_v1"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(value, reason):
    if not value:
        raise ValueError(reason)


def main():
    destination = OUT / "independent_verification.json"
    require(not destination.exists(), "Preserve previous completed audit")
    subset, policy = read(BASE / "frozen_subset.json"), read(BASE / "selection_policy.json")
    protocol, execution, results = (read(OUT / name) for name in ("frozen_protocol.json", "execution.json", "results.json"))
    require(subset["complete"] and results["complete"], "Incomplete subset or comparison")
    require(sha(BASE / "selection_policy.json") == subset["selection_policy_sha256"], "Selection policy changed")
    require(sha(ROOT / "scripts/prepare_cctv_native_gallery_v2.py") == policy["selector_sha256"], "Frozen selector changed")
    require(results["protocol_sha256"] == execution["protocol_sha256"] == sha(OUT / "frozen_protocol.json"), "Comparison protocol changed")
    for name, pin in protocol["assets_sha256"].items():
        require(sha(ROOT / name) == pin, "Frozen input/code/weight changed: " + name)
    for name, pin in results["artifacts_sha256"].items():
        require(sha(OUT / name) == pin, "Saved inference artifact changed: " + name)
    roles = {role: [row for row in subset["cases"] if row["role"] == role] for role in ("development", "reserved_evaluation")}
    require(len(roles["development"]) == 24 and len(roles["reserved_evaluation"]) == 32, "Selected count differs")
    identity_sets = [{row["global_person_id"] for row in cases} for cases in roles.values()]
    require(len(identity_sets[0]) == 24 and len(identity_sets[1]) == 32 and not identity_sets[0] & identity_sets[1], "Selected person IDs overlap/repeat")
    require(len({row["source_sha256"] for row in subset["cases"]}) == 56, "Selected bytes duplicate")
    archive_path = ROOT / "dataset/cctv_survface_raw/QMUL-SurvFace-v1.zip"
    require(sha(archive_path) == subset["source_archive_sha256"], "Acquired archive changed")
    detected_formats = Counter()
    with ZipFile(archive_path) as archive:
        for case in subset["cases"]:
            source = archive.read(case["archive_member"])
            require(hashlib.sha256(source).hexdigest() == case["source_sha256"], "Native selected member changed")
            require((BASE / case["source_file"]).read_bytes() == source, "Extracted native copy differs")
            with Image.open(BytesIO(source)) as header:
                width, height = header.size
                require([width, height] == [case["native_width"], case["native_height"]], "Native dimensions differ")
                require(header.format == case["detected_format"] in ("JPEG", "PNG"), "Source format differs")
                detected_formats[header.format] += 1
            side = min(width, height)
            group = "le15" if side <= 15 else "16to23" if side <= 23 else "24to39" if side <= 39 else "ge40"
            require(group == case["size_bin"], "Native size bin differs")
            pid = int(case["archive_member"].rsplit("/", 1)[1].split("_", 1)[0])
            require(pid == case["global_person_id"], "Global native person label differs")
            expected_prefix = "QMUL-SurvFace/training_set/" if case["role"] == "development" else "QMUL-SurvFace/Face_Identification_Test_Set/gallery/"
            require(case["archive_member"].startswith(expected_prefix), "Published release role differs")
    require(protocol["cases"] == roles["development"], "Inference did not use exactly frozen development cases")
    require({row["id"] for row in results["rows"]} == {row["id"] for row in roles["development"]} and len(results["rows"]) == 24,
            "Missing/repeated output cases")
    reconstructed = []
    for row in results["rows"]:
        case = next(case for case in roles["development"] if case["id"] == row["id"])
        with Image.open(BASE / case["source_file"]) as native:
            rgb = native.convert("RGB")
            width, height = rgb.size
            side = max(width, height)
            padded = np.full((side, side, 3), 128, dtype=np.uint8)
            y, x = (side-height)//2, (side-width)//2
            padded[y:y+height, x:x+width] = np.asarray(rgb)
        expected_input = np.asarray(Image.fromarray(padded).resize((256, 256), Image.Resampling.BILINEAR))
        with Image.open(OUT / row["outputs"]["input"]) as image:
            common = np.asarray(image.convert("RGB"))
        require(np.array_equal(common, expected_input), "Shared input pixels differ from native pad/resize")
        floats = {}
        for name, arm in (("dgp", "dgp_raw"), ("codeformer", "codeformer_raw")):
            stage = np.load(OUT / f"stages/{row['id']}_{name}.npy", allow_pickle=False)
            require(stage.shape == (256, 256, 3) and stage.dtype == np.float32 and np.isfinite(stage).all() and 0 <= stage.min() <= stage.max() <= 1,
                    "Saved raw inference float is invalid")
            with Image.open(OUT / row["outputs"][arm]) as image:
                value = np.asarray(image.convert("RGB"))
            require(np.array_equal(value, (stage*255).astype(np.uint8)), "Raw PNG does not match captured model float")
            mae = float(np.abs(stage - common/255.0).mean())
            require(abs(mae-row["input_change_mae_diagnostic_only"][name]) < 1e-12, "Input-change diagnostic differs")
            floats[name] = stage
        lab = cv2.cvtColor((floats["dgp"]*255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        ref = cv2.cvtColor(common, cv2.COLOR_RGB2LAB).astype(np.float32)
        if float(lab[:, :, 0].mean()) < float(ref[:, :, 0].mean()):
            lab[:, :, 0] = np.clip((lab[:, :, 0]-float(lab[:, :, 0].mean())) *
                (max(1e-5, float(ref[:, :, 0].std()))/max(1e-5, float(lab[:, :, 0].std()))) + float(ref[:, :, 0].mean()), 0, 255).astype(np.uint8)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        calibrated = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        display = cv2.addWeighted(calibrated, 1.35, cv2.GaussianBlur(calibrated, (0, 0), 1.2), -0.35, 0)
        with Image.open(OUT / row["outputs"]["dgp_display"]) as image:
            require(np.array_equal(display, np.asarray(image.convert("RGB"))), "Legacy display pixels differ")
        reconstructed.append({"id": row["id"], "shared_input_exact": True, "raw_pngs_exact": 2, "legacy_display_exact": True})
    require(results["model_forwards"] == {"dgp": 24, "codeformer": 24}, "Forward count differs")
    require(results["model_state_after"] == execution["model_state_before"], "Model states changed")
    require(results["optimizer_updates"] == 0 and not execution["optimizer_constructed"], "Training occurred")
    require(not protocol["reserved_evaluation_used"] and not results["reserved_evaluation_used"] and not execution["reserved_evaluation_used"], "Reserved evaluation was used")
    require(results["seconds_after_loading"] <= 300 and results["PSNR"] is None and results["SSIM"] is None, "Budget/unpaired-metric claim differs")
    require(not results["checkpoint_selected"] and not results["application_change"], "Unreviewed model selection/application change")
    report = {"verified": True, "date": "2026-10-03", "auditor_sha256": sha(Path(__file__)),
              "subset_sha256": sha(BASE / "frozen_subset.json"), "protocol_sha256": sha(OUT / "frozen_protocol.json"),
              "results_sha256": sha(OUT / "results.json"), "artifact_hashes_checked": len(results["artifacts_sha256"]),
              "native_member_copies_byte_verified": 56, "selected_formats": dict(detected_formats),
              "selected_labeled_identity_overlap": 0, "raw_float_stages_checked": 48,
              "output_pngs_reconstructed": 96, "rows": reconstructed, "model_state_fingerprints_unchanged": True,
              "reserved_inputs_rendered": False, "model_forwards_in_audit": 0, "optimizer_updates": 0,
              "limitations": "Artifact and pixel reconstruction audit, not proof of clean facial identity, country attribution or independent final usefulness review"}
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"verified": True, "native_copies": 56, "output_pngs_reconstructed": 96, "audit_sha256": sha(destination)}))


if __name__ == "__main__":
    main()
