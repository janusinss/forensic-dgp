"""Recount saved practical probabilities and CPU/CUDA reproductions; no inference."""
from collections import defaultdict
import hashlib
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_varied_covering_results import ARMS, require, sha, read, write, exact, binary
from scripts.audit_real_camera_inference_review import expand, independent_metrics

PROTOCOL_DIR = ROOT / "outputs/varied_covering_practical_protocol_v1"
RUN = ROOT / "outputs/varied_covering_practical_review_v1"
OUT = ROOT / "outputs/varied_covering_practical_validation_v1"
RETURN = ROOT / "outputs/downloaded_varied_covering_v1/outputs/varied_covering_vm"
MODELS = ("retained_app_default", "camera91_source", *ARMS)


def probability(path):
    value = np.load(path, allow_pickle=False)
    require(value.dtype == np.float32 and value.shape == (256, 256) and np.isfinite(value).all()
            and value.min() >= 0 and value.max() <= 1, "Invalid archived probability")
    return value


def main():
    require(not OUT.exists(), "Preserve previous independent practical audit")
    protocol, result = read(PROTOCOL_DIR / "protocol.json"), read(RUN / "results.json")
    digest = sha(PROTOCOL_DIR / "protocol.json")
    require(digest == read(PROTOCOL_DIR / "binding.json")["protocol_sha256"] == result["protocol_sha256"], "Frozen practical binding differs")
    require(result["format"] == "dgp-varied-covering-practical-result-v1" and result["complete"] is True
            and result["training"] is False and result["promoted"] is False
            and result["original425_gates_evaluated"] is False, "Result scope differs")
    exact({key: result[key] for key in ("actual_forward_images", "generator_forwards", "restorer_forwards", "optimizer_updates")},
          {"actual_forward_images": 106, "generator_forwards": 0, "restorer_forwards": 0, "optimizer_updates": 0}, "logged forward scope")
    require(type(result["seconds_excluding_loading"]) is float and 0 < result["seconds_excluding_loading"] < 300, "Inference time outside finite cap")
    for name, pin in protocol["assets_sha256"].items(): require(sha(ROOT / name) == pin, "Frozen practical asset differs")
    reproduction_names = {row["name"] for row in protocol["reproduction_cases"]} | {"fixture/171"}
    expected = {f"{folder}/{arm}_{case['id']}.{suffix}" for folder, suffix in
                (("raw", "png"), ("proposals", "png"), ("probabilities", "npy")) for arm in ARMS for case in protocol["cases"]}
    expected.update(f"reproduction/{arm}_{name.replace('/', '_')}.{suffix}" for arm in ARMS for name in reproduction_names for suffix in ("png", "npy"))
    expected.update(f"preview/rows_{first + 1:02d}_{first + 6:02d}.png" for first in range(0, 36, 6))
    require(len(expected) == 290 and set(result["artifact_sha256"]) == expected, "Artifact coverage differs")
    for name, pin in result["artifact_sha256"].items(): require(sha(RUN / name) == pin, "Practical artifact changed")
    actual_members = {path.relative_to(RUN).as_posix() for path in RUN.rglob("*") if path.is_file()}
    require(actual_members == expected | {"execution.json", "results.json"}, "Unexpected/missing practical artifact")
    require(len(result["reproduction"]) == 34 and {(row["model"], row["case"]) for row in result["reproduction"]}
            == {(arm, name) for arm in ARMS for name in reproduction_names}, "Reproduction membership differs")
    differences = 0
    for row in result["reproduction"]:
        stem = row["model"] + "_" + row["case"].replace("/", "_")
        exact(row["probability"], "reproduction/" + stem + ".npy", "reproduction probability path")
        exact(row["raw"], "reproduction/" + stem + ".png", "reproduction mask path")
        value = probability(RUN / row["probability"])
        actual = binary(RUN / row["raw"])
        require(np.array_equal(actual, value >= .5), "Reproduction raw threshold differs")
        changed = actual != binary(RETURN / row["model"] / "final_masks" / (row["case"] + ".png"))
        exact(row["different_pixels"], int(changed.sum()), "reproduction discrepancy")
        require(int(changed.sum()) <= 4 and (not changed.any() or np.all(np.abs(value[changed] - .5) <= 1e-5)), "Nonambiguous reproduction differs")
        differences += int(changed.sum())
    import torch
    states = {}
    for arm, path in protocol["models"].items():
        payload = torch.load(ROOT / path, map_location="cpu", weights_only=True)
        digest_state = hashlib.sha256()
        for name, value in payload["model"].items():
            require(torch.isfinite(value).all(), "Nonfinite checkpoint state")
            digest_state.update(name.encode()); digest_state.update(value.contiguous().numpy().tobytes())
        state = digest_state.hexdigest()
        exact(result["model_states"][arm], {"before": state, "after": state, "forward_images": 53}, "practical frozen model")
        states[arm] = state; del payload
    execution = read(RUN / "execution.json")
    require(execution["protocol_sha256"] == sha(PROTOCOL_DIR / "protocol.json") and execution["device"] == "cpu"
            and execution["optimizer_constructed"] is False and type(execution["optimizer_updates"]) is int
            and execution["optimizer_updates"] == 0, "Execution scope differs")
    groups, controls, rejections, misses = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    require(len(result["rows"]) == len(protocol["cases"]) == 36, "Practical case count differs")
    for case, row in zip(protocol["cases"], result["rows"]):
        for key in ("id", "family", "synthetically_degraded", "expected_rejection", "pose_scope", "clear_control"):
            exact(row[key], case[key], "case membership/" + key)
        require(set(row["models"]) == set(MODELS), "Comparison model membership differs")
        truth = binary(ROOT / case["reviewed"])
        protected = binary(ROOT / case["protected"]) if case["protected"] else np.zeros_like(truth)
        exact(case["clear_control"], not bool(truth.any()), "clear control target")
        for arm, item in row["models"].items():
            if arm in ARMS:
                stem = arm + "_" + case["id"]
                for key, folder, suffix in (("raw", "raw", "png"), ("proposal", "proposals", "png"), ("probability", "probabilities", "npy")):
                    exact(item[key], folder + "/" + stem + "." + suffix, "candidate artifact path")
                value = probability(RUN / item["probability"])
                raw, mask = binary(RUN / item["raw"]), binary(RUN / item["proposal"])
                require(item["cached"] is False and np.array_equal(raw, value >= .5)
                        and np.array_equal(mask, expand(raw)), "Candidate threshold/margin differs")
            else:
                require(item["cached"] is True and item["proposal"] == case["cached"][arm], "Cached proposal binding differs")
                mask = binary(ROOT / item["proposal"])
            actual = independent_metrics(mask, truth, protected)
            exact(item["metrics"], actual, "independent practical metric")
            group = "/".join((case["family"], str(case["synthetically_degraded"]), case["pose_scope"], arm))
            groups[group].append(actual)
            if case["clear_control"]: controls[arm].append({"id": case["id"], "pixels": actual["predicted_pixels"]})
            if case["expected_rejection"]: rejections[arm].append({"id": case["id"], "rejected": actual["guard"]["rejected"]})
            if truth.any() and not mask.any(): misses[arm].append(case["id"])
    summary = {key: {"cases": len(values), "mean_recall": None if values[0]["recall"] is None else sum(value["recall"] for value in values) / len(values),
               "mean_iou": None if values[0]["iou"] is None else sum(value["iou"] for value in values) / len(values),
               "excess_pixels": sum(value["excess_pixels_outside_3px_tolerance"] for value in values),
               "protected_wire_pixels": sum(value["protected_wire_pixels_changed"] for value in values)} for key, values in groups.items()}
    OUT.mkdir()
    record = {"format": "dgp-varied-covering-practical-independent-verification-v1", "date": "2026-10-03", "complete": True,
        "auditor_sha256": sha(__file__), "protocol_sha256": sha(PROTOCOL_DIR / "protocol.json"), "results_sha256": sha(RUN / "results.json"),
        "artifacts_verified": 290, "probabilities_verified": 106, "practical_metric_records": 144,
        "reproduction_masks_verified": 34, "reproduction_different_pixels": differences,
        "model_states_verified": states, "summary": summary, "clear_controls": controls, "near_hidden_rejections": rejections,
        "empty_covering_cases": misses, "seconds_excluding_loading": result["seconds_excluding_loading"],
        "local_model_forwards_in_audit": 0, "local_optimizer_updates": 0, "promoted": False,
        "visual_review_pending": True, "face_output_review_complete": False,
        "limitations": ["Exposed previously inspected development sources; no unseen population or original425 qualification",
                       "Saved probability/mask inspection and model hashes do not independently observe live forward hooks",
                       "Only covering masks are evaluated; generated-face usefulness remains a separate comparison"]}
    write(OUT / "verification.json", record)
    print({"complete": True, "artifacts": 290, "metric_records": 144, "reproduction_different_pixels": differences,
           "verification_sha256": sha(OUT / "verification.json")})


if __name__ == "__main__": main()
