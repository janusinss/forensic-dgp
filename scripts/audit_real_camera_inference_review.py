"""Independent saved-probability, footprint and practical metric verification."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/real_camera_inference_review_v2"
OUT = ROOT / "outputs/real_camera_inference_validation_v1"
PROTOCOL_SHA = "024a3f7aa3775fbf45ca54c6cbfa87641893b594e5aec79b3b4f869b53b1f001"
RESULT_SHA = "1ffef0024298e88c0a91e8d222aae67b0a0f898dfe06366305dc77ce9849706c"
ARMS = ("native83", "camera83", "camera91")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def binary(path):
    with Image.open(path) as image:
        a = np.asarray(image).copy()
        require(image.mode == "L" and image.format == "PNG" and a.shape == (256, 256)
                and a.dtype == np.uint8 and np.isin(a, [0, 255]).all(), "Nonbinary footprint/reference")
    return a == 255


def expand(mask):
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(7))) != 0


def independent_metrics(mask, truth, protected):
    covered = int((mask & truth).sum()); union = int((mask | truth).sum())
    height, width = np.indices((256, 256))
    face = np.square((width - 127.5) / 83) + np.square((height - 137) / 104) <= 1
    bands = {"left_eye": (60, 84, 119, 133), "right_eye": (137, 84, 196, 133),
             "nose": (108, 119, 149, 166), "mouth": (82, 165, 174, 208)}
    features = {k: float(mask[y1:y2, x1:x2].mean()) for k, (x1, y1, x2, y2) in bands.items()}
    face_fraction = float(mask[face].mean())
    reject = face_fraction >= .8 or float(mask.mean()) >= .85 or all(features[k] >= limit for k, limit in
              (("left_eye", .85), ("right_eye", .85), ("nose", .9), ("mouth", .9)))
    return {"predicted_pixels": int(mask.sum()), "reference_pixels": int(truth.sum()),
        "intersection_pixels": covered, "union_pixels": union, "recall": covered / int(truth.sum()) if truth.any() else None,
        "precision": covered / int(mask.sum()) if mask.any() else None, "iou": covered / union if truth.any() and union else None,
        "excess_pixels_outside_3px_tolerance": int((mask & ~expand(truth)).sum()),
        "protected_wire_pixels_changed": int((mask & protected).sum()),
        "empty_control_marked_fraction": float(mask.mean()) if not truth.any() else None,
        "guard": {"rejected": bool(reject), "face_covered_fraction": face_fraction,
                  "feature_covered_fractions": features,
                  "scope": "Approximate frontal-crop geometry, conditional on the reviewed removal area"}}


def main():
    import torch
    torch.set_num_threads(4)
    require(not OUT.exists(), "Preserve completed independent audit")
    require(sha(RUN / "frozen_protocol.json") == PROTOCOL_SHA and sha(RUN / "results.json") == RESULT_SHA, "Frozen evidence differs")
    p, result = read(RUN / "frozen_protocol.json"), read(RUN / "results.json")
    require(result["complete"] is True and result["protocol_sha256"] == PROTOCOL_SHA
            and len(result["rows"]) == len(p["cases"]) == 36 and result["actual_forward_images"] == 152
            and result["generator_forwards"] == result["restorer_forwards"] == result["optimizer_updates"] == 0
            and result["promoted"] is False and result["training"] is False, "Runtime/budget declarations differ")
    for name, pin in p["assets_sha256"].items():
        require(sha(ROOT / name) == pin, "Inference asset changed")
    expected = {f"{folder}/{arm}_{case['id']}.{suffix}" for folder, suffix in
                (("raw", "png"), ("proposals", "png"), ("probabilities", "npy")) for arm in ARMS for case in p["cases"]}
    expected.update(f"preview/rows_{i + 1:02d}_{i + 6:02d}.png" for i in range(0, 36, 6))
    require(set(result["artifact_sha256"]) == expected and len(expected) == 330, "Saved artifact membership differs")
    for name, pin in result["artifact_sha256"].items():
        require(sha(RUN / name) == pin, "Saved inference output changed")
    reproduction = result["reproduction"]
    require(len(reproduction) == 44 and {(r["model"], r["case"]) for r in reproduction}
            == {(a, r["name"]) for a in p["models"] for r in p["reproduction_cases"] + [{"name": "fixture/171"}]}, "Reproduction membership differs")
    require(all(type(r["different_pixels"]) is int and 0 <= r["different_pixels"] <= 4
                and r["all_differences_threshold_ambiguous"] is True for r in reproduction), "Reproduction declaration differs")
    states = {}
    for arm, path in p["models"].items():
        payload = torch.load(ROOT / path, map_location="cpu", weights_only=True)
        h = hashlib.sha256()
        for name, value in payload["model"].items():
            h.update(name.encode()); h.update(value.contiguous().numpy().tobytes())
        logged = result["model_states"][arm]
        require(logged["before"] == logged["after"] == h.hexdigest() and logged["forward_images"] == (11 if arm == "source42" else 47), "Inference state/budget differs")
        states[arm] = h.hexdigest(); del payload
    groups, controls, rejects, deficits = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for case, row in zip(p["cases"], result["rows"]):
        require(all(row[k] == case[k] for k in ("id", "family", "synthetically_degraded", "pose_scope", "expected_rejection")), "Practical case membership differs")
        truth = binary(ROOT / case["reviewed"])
        protected = binary(ROOT / case["protected"]) if case["protected"] else np.zeros_like(truth)
        for arm in p["models"]:
            model = row["models"][arm]
            if arm == "source42":
                raw, proposal = binary(ROOT / model["raw"]), binary(ROOT / model["proposal"])
                require(model["cached"] is True, "Source42 cached status differs")
            else:
                raw, proposal = binary(RUN / model["raw"]), binary(RUN / model["proposal"])
                probability = np.load(RUN / model["probability"], allow_pickle=False)
                require(probability.dtype == np.float32 and probability.shape == (256, 256)
                        and np.isfinite(probability).all() and probability.min() >= 0 and probability.max() <= 1
                        and np.array_equal(raw, probability >= .5) and model["cached"] is False, "Probability/raw threshold differs")
            require(np.array_equal(proposal, expand(raw)), "Frozen3px proposal margin differs")
            actual = independent_metrics(proposal, truth, protected)
            require(actual == model["metrics"], "Practical metric/visibility recount differs")
            groups[f"{case['family']}/{case['synthetically_degraded']}/{arm}"].append(actual)
            if not truth.any():
                controls[arm].append({"id": case["id"], "pixels": actual["predicted_pixels"]})
            if case["expected_rejection"]:
                rejects[arm].append({"id": case["id"], "rejected": actual["guard"]["rejected"]})
            if truth.any() and not proposal.any():
                deficits[arm].append(case["id"])
    summary = {key: {"cases": len(values), "mean_recall": sum(v["recall"] for v in values) / len(values) if values[0]["recall"] is not None else None,
                    "mean_iou": sum(v["iou"] for v in values) / len(values) if values[0]["iou"] is not None else None,
                    "sum_excess_pixels_outside_3px_tolerance": sum(v["excess_pixels_outside_3px_tolerance"] for v in values)}
               for key, values in groups.items()}
    OUT.mkdir()
    record = {"format": "dgp-real-camera-inference-independent-verification-v1", "date": "2026-10-03", "complete": True,
        "auditor_sha256": sha(__file__), "protocol_sha256": PROTOCOL_SHA, "results_sha256": RESULT_SHA,
        "artifacts_verified": 330, "probabilities_verified": 108, "raw_and_proposal_masks_verified": 216,
        "practical_metric_records_recounted": 144, "checkpoint_state_sha256": states,
        "reproduction_exact_masks": sum(r["different_pixels"] == 0 for r in reproduction),
        "reproduction_threshold_ambiguous_different_pixels": sum(r["different_pixels"] for r in reproduction),
        "summary": summary, "clear_controls": controls, "near_hidden_rejections": rejects,
        "empty_reference_covering_cases": deficits, "local_model_forwards_in_this_audit": 0, "local_optimizer_updates": 0,
        "promoted": False, "limitations": ["Reproduction threshold distances are runner-checked/logged; those44 probabilities were not archived",
            "Previously inspected development cases, not unseen population or original425 qualification",
            "Only detector masks; no new generated facial estimates from these checkpoints"]}
    with (OUT / "verification.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"complete": True, "artifacts_verified": 330, "metric_records": 144,
                      "verification_sha256": sha(OUT / "verification.json")}))


if __name__ == "__main__":
    main()
