"""Reconstruct frozen paired inputs/pixels/metrics; zero model forwards or updates."""
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cctv_camera_stress import reference_canvas, degrade, PROFILES
from run_cctv_native_comparison import sha, read, write, DGP_PIN, CF_PIN, DGP, CF
from run_cctv_paired_regression import selection, normalize, SPLIT_PIN, SOURCES, summary

OUT = ROOT/"outputs/cctv_paired_regression_v1"


def png(path):
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB")).copy()


def main():
    protocol, results = read(OUT/"frozen_protocol.json"), read(OUT/"results.json")
    execution, geometry = read(OUT/"execution.json"), read(OUT/"reference_geometry.json")
    if not results["complete"] or protocol["profiles"] != PROFILES:
        raise ValueError("Frozen profiles/completion differ")
    for name, pin in protocol["assets_sha256"].items():
        if sha(ROOT/name) != pin:
            raise ValueError("Frozen asset differs: "+name)
    for asset in protocol["external_assets"].values():
        if sha(asset["path"]) != asset["sha256"]:
            raise ValueError("External inference weight differs")
    for mapping in (protocol["prepared_artifacts_sha256"], results["artifacts_sha256"]):
        for name, pin in mapping.items():
            if sha(OUT/name) != pin:
                raise ValueError("Artifact bytes differ: "+name)
    pins = [results["protocol_sha256"], execution["protocol_sha256"]]
    if pins != [sha(OUT/"frozen_protocol.json")]*2 or results["reference_geometry_sha256"] != sha(OUT/"reference_geometry.json") or execution["input_review_sha256"] != sha(OUT/"input_review.json"):
        raise ValueError("Protocol/geometry/input review bindings differ")
    if protocol["historical_split_sha256"] != SPLIT_PIN or protocol["assets_sha256"][DGP] != DGP_PIN or protocol["assets_sha256"][CF] != CF_PIN:
        raise ValueError("Checkpoint/historical split pins differ")
    split = read(ROOT/protocol["historical_split"])
    if [(r["source"], r["source_file"]) for r in protocol["references"]] != selection(split):
        raise ValueError("Deterministic historical validation selection differs")
    if len({r["source_sha256"] for r in protocol["references"]}) != 8:
        raise ValueError("Selected references contain byte duplicates")
    if execution["model_state_before"] != results["model_state_after"] or not results["model_states_unchanged"] or execution["optimizer_constructed"] or not execution["models_eval_and_frozen"]:
        raise ValueError("Frozen model state/inference-only guard differs")
    if results["model_forwards"] != {"dgp": 40, "codeformer": 40} or results["reference_detection_requests"] != 8 or results["optimizer_updates"] != 0 or results["restoration_seconds_after_loading"] > 480 or results["reference_seconds_after_loading"] > 60:
        raise ValueError("Finite inference budget differs")
    if results["checkpoint_selected"] or results["application_change"] or results["reserved_native_cctv_used"]:
        raise ValueError("No promotion/app change/reserved CCTV use permitted")
    refs, geometries = {}, {r["reference_id"]: r for r in geometry["rows"]}
    profiles = {p["id"]: p for p in PROFILES}
    from insightface.utils.face_align import estimate_norm
    for ref in protocol["references"]:
        target, mask, bounds = reference_canvas(ROOT/ref["source_file"])
        np.testing.assert_array_equal(target, png(OUT/ref["target"]))
        np.testing.assert_array_equal(mask.astype(np.uint8)*255, np.asarray(Image.open(OUT/ref["observed"])))
        if bounds != ref["bounds"]:
            raise ValueError("Observed rectangle differs")
        item = geometries[ref["id"]]
        lm = np.asarray(item["reference_landmarks"], dtype=np.float32) if item["reference_landmarks"] is not None else None
        eligible = len(item["bboxes"]) == 1 and lm is not None and np.isfinite(lm).all() and np.linalg.norm(lm[0, 0]-lm[0, 1]) >= 8
        if item["detections"] != len(item["bboxes"]) or item["eligible"] != bool(eligible):
            raise ValueError("Reference-only eligibility differs")
        if eligible:
            np.testing.assert_allclose(item["matrix"], estimate_norm(lm[0], image_size=112), rtol=0, atol=1e-12)
            emb = np.asarray(item["reference_embedding"], dtype=np.float32)
            if emb.shape != (512,) or not np.isfinite(emb).all() or abs(np.linalg.norm(emb)-1) > 1e-5:
                raise ValueError("Invalid reference embedding")
        elif item["matrix"] is not None or item["reference_embedding"] is not None:
            raise ValueError("Ineligible reference carries identity geometry")
        refs[ref["id"]] = (target, mask, bounds, item)
    if len(results["rows"]) != 40 or [r["id"] for r in results["rows"]] != [c["id"] for c in protocol["cases"]]:
        raise ValueError("Forty ordered fixed cases required")
    png_count, float_count, embedding_count = 16, 0, 0
    for case, row in zip(protocol["cases"], results["rows"]):
        for key, value in case.items():
            if row[key] != value:
                raise ValueError("Fixed case metadata differs")
        target, mask, bounds, geom = refs[case["reference_id"]]
        inp, meta = degrade(target, bounds, profiles[case["profile"]], case["seed"])
        if meta != case["degradation"]:
            raise ValueError("Degradation metadata differs")
        np.testing.assert_array_equal(inp, png(OUT/case["input"]))
        png_count += 1
        arms = {"input": inp.astype(np.float32)/255}
        for arm in ("dgp", "codeformer"):
            value = np.load(OUT/f"stages/{case['id']}_{arm}.npy", allow_pickle=False)
            if value.dtype != np.float32 or value.shape != (256, 256, 3) or not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
                raise ValueError("Invalid saved float stage")
            np.testing.assert_array_equal((value*255).astype(np.uint8), png(OUT/row["images"][arm]))
            arms[arm] = value
            png_count += 1
            float_count += 1
        reference = target.astype(np.float32)/255
        interior = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)>0
        for arm, value in arms.items():
            difference = value-reference
            mse = np.square(difference[mask]).astype(np.float64).mean()
            _, ssmap = structural_similarity(reference, value, data_range=1, channel_axis=-1, win_size=7, full=True)
            expected = {"MAE": float(np.abs(difference[mask]).astype(np.float64).mean()),
                        "SSIM": float(ssmap[interior].astype(np.float64).mean()),
                        "PSNR": float(-10*np.log10(mse)) if mse else None, "perfect_match": mse == 0,
                        "observed_pixels": int(mask.sum()), "ssim_interior_pixels": int(interior.sum())}
            actual = row["metrics"][arm]
            for key, value in expected.items():
                if value is None or isinstance(value, (bool, int, np.bool_)):
                    if actual[key] != value:
                        raise ValueError("Metric or eligible pixel count differs: "+key)
                else:
                    np.testing.assert_allclose(actual[key], value, rtol=0, atol=1e-10)
            if geom["eligible"]:
                embedding_count += 1
                emb = np.load(OUT/actual["embedding_file"], allow_pickle=False)
                if emb.shape != (512,) or emb.dtype != np.float32 or not np.isfinite(emb).all() or abs(np.linalg.norm(emb)-1) > 1e-5:
                    raise ValueError("Invalid saved output embedding")
                ref_emb = np.asarray(geom["reference_embedding"], dtype=np.float32)
                np.testing.assert_allclose(actual["ArcFace_fixed"], float(np.clip(emb@ref_emb, -1, 1)), rtol=0, atol=1e-10)
            elif actual["ArcFace_fixed"] is not None or actual["embedding_file"] is not None:
                raise ValueError("Shared ineligible cohort differs across arms")
    if results["summary"] != summary(results["rows"]):
        raise ValueError("Per-source/profile summaries differ")
    eligible_refs = sum(g["eligible"] for g in geometry["rows"])
    if results["recognition_forwards"] != eligible_refs*16 or embedding_count != eligible_refs*15 or geometry["reference_recognition_forwards"] != eligible_refs:
        raise ValueError("Identity forward/eligible case counts differ")
    report = {"complete": True, "results_sha256": sha(OUT/"results.json"),
              "protocol_sha256": sha(OUT/"frozen_protocol.json"),
              "historical_split_sha256": SPLIT_PIN, "exact_reconstructed_pngs": png_count,
              "float_stages_checked": float_count, "saved_embeddings_cosines_reconstructed": embedding_count,
              "identity_eligible_references": eligible_refs, "model_states_unchanged": True,
              "model_forwards": 0, "optimizer_updates": 0, "reserved_native_cctv_used": False,
              "checkpoint_selected": False,
              "limitations": "Reconstructs deterministic transforms, saved-stage pixels and metrics without re-forwarding networks; cannot independently prove neural correctness or physical CCTV realism. Shared degradation helper is fingerprinted, not an independent sensor implementation. No full 76000-train byte/identity dedup audit or pretrained-overlap claim."}
    write(OUT/"independent_audit.json", report)
    print(json.dumps({**report, "audit_sha256": sha(OUT/"independent_audit.json")}))


if __name__ == "__main__":
    main()
