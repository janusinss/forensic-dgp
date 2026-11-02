"""Independent saved-array audit. Imports no model source or neural runtime."""
import collections
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
HISTORY = Path(r"C:\xampp\htdocs\YEAR 4\Testing")
OUT = ROOT / "outputs/completion_migan256_comparison_v1"
ASSETS = ROOT / "outputs/completion_migan256_assets_v1"
AUDIT = OUT / "independent_output_audit.json"
EXPECTED = {
    "runner": "be81c4529c245217e7ee793dd68cd524c93ee37d9cef110064d048f69dd6711e",
    "protocol": "34f5558553a64d567fae0a6ab1f8f5419c27d307b8088e9a4562d718748ad00b",
    "admission": "b63f416f64cee2dd9bd91efe8fe07d1854e8af95ff7b4233e0f7964f22509802",
    "source_review": "291d797872074b3e2f129cd0756064cb2fa18ce73c43f44008058b03e67ad47a",
    "old_protocol": "d3b35fe043c198007515f3dc0a221b7c7b369b7edd8d7b9bd45122cc550cf76f",
    "source": "1e35a181e516742ae32af935848e6a9a4382f528ea7096db6658f14cd8b9ace0",
    "checkpoint": "3cab897cfcc3fcbe224510cedad06a634ae8bf5bd7120428a32fd7224f4237f4",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def pin(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def bounded(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file(), "Unbound path: " + relative)
    return path


def pixels(path, mode="RGB"):
    with Image.open(path) as image:
        return np.array(image.convert(mode))


def equal(actual, expected, label):
    require(actual.dtype == expected.dtype and actual.shape == expected.shape, label + " dtype/shape")
    require(np.array_equal(actual, expected), label + " differs")


def saved_float(relative, expected_sha):
    path = bounded(OUT, relative)
    require(sha(path) == expected_sha, "Stage hash: " + relative)
    value = np.load(path, allow_pickle=False)
    require(value.dtype == np.float32 and value.shape == (256, 256, 3), "Float256 stage shape/type")
    require(bool(np.isfinite(value).all()), "Nonfinite stage")
    return path, value


def main():
    started = time.monotonic()
    require(not AUDIT.exists(), "Preserve completed independent audit")
    required = {
        "runner": ROOT / "scripts/run_completion_migan256_v1.py",
        "protocol": OUT / "protocol.json",
        "admission": ASSETS / "source_admission.json",
        "source_review": ROOT / "outputs/completion_migan256_source_review_v1.json",
        "old_protocol": HISTORY / "outputs/completion_mat_mirror_comparison_v1/protocol.json",
        "source": ASSETS / "migan_inference.py",
        "checkpoint": ASSETS / "migan_256_ffhq.pt",
    }
    pins_before = {name: pin(path) for name, path in required.items()}
    for name, digest in EXPECTED.items():
        require(pins_before[name]["sha256"] == digest, "Pinned dependency changed: " + name)
    protocol = read(required["protocol"])
    original_protocol = read(required["old_protocol"])
    admission = read(required["admission"])
    source_review = read(required["source_review"])
    results = read(OUT / "results.json")
    supervisor = read(OUT / "supervisor_receipt.json")
    supervisor_start = read(OUT / "supervisor_started.json")
    execution = read(OUT / "execution.json")
    progress = read(OUT / "progress.json")
    require(results["complete"] is True and supervisor["complete"] is True, "Inference or supervisor incomplete")
    require(supervisor["returncode"] == 0 and supervisor["stopped_reason"] is None and not supervisor["only_owned_child_terminated"], "Supervisor stop/failure")
    require(not (OUT / "failure.json").exists(), "Retain and inspect worker failure")
    require(admission["complete"] and admission["independent"] and admission["safe_state_dict"], "Admission false")
    require(protocol["runner_sha256"] == EXPECTED["runner"], "Runner binding")
    for receipt in (results, execution, supervisor_start):
        require(receipt["protocol_sha256"] == EXPECTED["protocol"] and receipt["admission_sha256"] == EXPECTED["admission"], "Runtime receipt binding")
    require(protocol["historical_protocol_sha256"] == EXPECTED["old_protocol"], "Historical binding")
    require(len(protocol["cases"]) == len(results["rows"]) == len(original_protocol["cases"]) == 32, "Case counts")
    require(protocol["excluded_before_generation"] == original_protocol["excluded_before_generation"] and len(protocol["excluded_before_generation"]) == 4, "Exclusions changed")
    require(protocol["budgets"] == {"requests": 32, "case_forwards": 28, "parity_forwards": 1, "maximum_forwards": 29, "empty_bypasses": 4, "case_seconds": 120, "worker_seconds": 600, "external_seconds": 630, "maximum_output_bytes": 157286400}, "Budget drift")
    require(protocol["device"] == execution["device"] == "cpu" and execution["threads"] == protocol["threads"] == 4, "Runtime device/threads")
    require(execution["grad_enabled"] is False and execution["optimizer_constructed"] is False and execution["optimizer_updates"] == 0, "Execution training flag")
    require(results["optimizer_constructed"] is False and results["optimizer_updates"] == results["gradient_calls"] == 0, "Results training flag")
    require(results["assisted_only"] and not results["automatic_quality_qualification"] and not results["assisted_quality_qualification"] and not results["goal_complete"], "Qualification flags changed")
    require(0 < results["seconds"] <= 600 and 0 < supervisor["seconds"] <= 630, "Runtime limits")
    require(supervisor["seconds"] >= results["seconds"], "Supervisor shorter than worker")
    require(progress["complete"] and progress["phase"] == "finished" and progress["finished_requests"] == 32 and progress["forward_started"] == progress["forward_finished"] == 29, "Final progress")
    require(progress["worker_pid"] == execution["pid"], "Worker PID receipts disagree")
    require(results["requests"] == 32 and results["pretrained_forwards"] == 28 and results["parity_forwards"] == 1 and results["total_forwards"] == 29 and results["empty_mask_bypasses"] == 4, "Reported neural counts")
    require(results["state_unchanged"] and results["generator_state_before"] == results["generator_state_after"] and len(results["generator_state_before"]) == 64, "State digest evidence")
    require(results["weight_sha256_before"] == results["weight_sha256_after"] == EXPECTED["checkpoint"], "Weight evidence")
    mandatory = {}
    for name, digest in protocol["assets_sha256"].items():
        path = bounded(ASSETS, name)
        require(sha(path) == digest, "Asset changed: " + name)
        mandatory[str(path)] = pin(path)
    for name, digest in source_review["historical_source_bindings_sha256"].items():
        path = bounded(HISTORY, name)
        require(sha(path) == digest, "Historical binding changed: " + name)
    for name, item in source_review["historical_app_preservation_current_pins"].items():
        require(sha(bounded(HISTORY, name)) == item["sha256"], "Current app guard changed: " + name)
    cases = []
    controls = []
    kept_bytes = 0
    expected_events = []
    expected_stages = {"parity_first_raw256.npy"}
    image_names = set()
    for index, (case, row, old, reviewed) in enumerate(zip(protocol["cases"], results["rows"], original_protocol["cases"], source_review["cases"]), 1):
        name = case["id"]
        require(name == row["id"] == old["id"] == reviewed["id"], "Case identity/order")
        require(case["family"] == row["family"] and case["condition"] == row["condition"] and row["mode"] == "assisted", "Case metadata")
        require(row["hidden_PSNR_SSIM"] is None and row["exact_hidden_identity_claim"] is False, "Hidden claims")
        for key in ("input", "reviewed", "output", "stages", "metadata"):
            path = bounded(HISTORY, case[key])
            require(case[key] == old[key] and sha(path) == case["bindings_sha256"][key] == original_protocol["sources_sha256"][case[key]], "Case historical hash: " + name + "/" + key)
        for folder, key in (("inputs", "input"), ("masks", "reviewed"), ("baseline", "output")):
            copied = bounded(OUT, folder + "/" + name + ".png")
            require(sha(copied) == case["bindings_sha256"][key], "Copied asset differs: " + name)
            mandatory[str(copied)] = pin(copied)
        image = pixels(HISTORY / case["input"])
        mask = pixels(HISTORY / case["reviewed"], "L")
        baseline = pixels(HISTORY / case["output"])
        require(image.dtype == mask.dtype == np.uint8 and image.shape == baseline.shape == (256, 256, 3) and mask.shape == (256, 256), "Image/mask geometry")
        require(set(np.unique(mask)) <= {0, 255}, "Nonbinary mask")
        removal = mask == 255
        require(int(removal.sum()) == case["mask_pixels"] == row["mask_pixels"] == reviewed["mask_pixels"], "Mask support count")
        equal(baseline[~removal], image[~removal], "Cached CodeFormer visible source")
        require(row["exact_outside_reviewed_mask"] is True, "Worker preservation flag")
        require(0 <= row["seconds"] <= 120, "Whole-case120s limit")
        require(read(OUT / ("progress_%02d.json" % index)) == row, "Immutable per-case progress")
        base_float = image.astype(np.float32) / np.float32(255)
        empty = not removal.any()
        require(row["empty_mask_bypass"] == reviewed["empty_mask_bypass"] == empty, "Bypass status")
        if empty:
            controls.append(name)
            require(row["raw256"] is None and row["raw256_sha256"] is None, "Empty control raw unexpectedly exists")
            expected_decoded = base_float.copy()
            expected_completed = base_float.copy()
            expected_final = image.copy()
        else:
            expected_events.append(name)
            raw_path, raw = saved_float(row["raw256"], row["raw256_sha256"])
            expected_stages.add(raw_path.name)
            mandatory[str(raw_path)] = pin(raw_path)
            expected_decoded = np.minimum(np.maximum(raw * np.float32(0.5) + np.float32(0.5), np.float32(0)), np.float32(1))
            expected_completed = base_float.copy()
            expected_completed[removal] = expected_decoded[removal]
            generated_bytes = np.floor(expected_decoded * np.float32(255)).astype(np.uint8)
            expected_final = image.copy()
            expected_final[removal] = generated_bytes[removal]
        for label, expected_array in (("decoded256", expected_decoded), ("completed256", expected_completed)):
            path, actual = saved_float(row[label], row[label + "_sha256"])
            equal(actual, expected_array, name + " / " + label)
            mandatory[str(path)] = pin(path)
            expected_stages.add(path.name)
        file = bounded(OUT, row["output"])
        require(sha(file) == row["output_sha256"], "Output hash")
        actual = pixels(file)
        equal(actual, expected_final, name + " exact declared raw-to-PNG")
        equal(actual[~removal], image[~removal], name + " visible source bytes")
        kept_bytes += int((~removal).sum()) * 3
        mandatory[str(file)] = pin(file)
        image_names.add(name + ".png")
        cases.append({"id": name, "file": pin(file), "family": case["family"], "condition": case["condition"], "exact_outside_reviewed_mask": True, "empty_mask_bypass": empty, "raw_to_decoded_to_completed_to_PNG_exact": True, "mask_pixels": int(removal.sum()), "seconds_reported": row["seconds"]})
    require(len(controls) == 4 and len(expected_events) == 28 and kept_bytes == 5225232, "Independent recount")
    for folder in ("inputs", "masks", "baseline", "images"):
        require({x.name for x in (OUT / folder).iterdir()} == image_names, "Unexpected/missing PNG in " + folder)
    parity = results["forward_events"]
    require(len(parity) == 29 and [e["number"] for e in parity] == list(range(1, 30)), "Forward counter sequence")
    require([e["id"] for e in parity[:28]] == expected_events and not any(e["parity"] for e in parity[:28]), "Case forward sequence")
    require(parity[-1]["id"] == protocol["parity_case_id"] == protocol["cases"][0]["id"] and parity[-1]["parity"], "Parity identity")
    for event in parity:
        require(event["finished"] and 0 <= event["seconds"] <= 120 and event["finished_unix"] >= event["started_unix"], "Incomplete/out-of-budget forward")
    require(all(a["finished_unix"] <= b["started_unix"] for a, b in zip(parity, parity[1:])), "Overlapping/reordered forward evidence")
    parity_file = bounded(OUT, "stages/parity_first_raw256.npy")
    require(results["parity_first_raw_exact"] and sha(parity_file) == results["parity_raw_sha256"], "Parity saved hash")
    equal(np.load(parity_file, allow_pickle=False), np.load(OUT / results["rows"][0]["raw256"], allow_pickle=False), "Saved parity raw replay")
    mandatory[str(parity_file)] = pin(parity_file)
    require({x.name for x in (OUT / "stages").iterdir()} == expected_stages and len(expected_stages) == 93, "Unexpected/missing stage arrays")
    exclusion_rows = []
    for exclusion in source_review["excluded_before_generation"]:
        require(exclusion["generation_permitted"] is False and exclusion["expected_candidate_neural_forwards"] == 0, "Exclusion receipt")
        require(not (OUT / "images" / (exclusion["id"] + ".png")).exists(), "Excluded input generated")
        for item in exclusion["bindings"].values():
            path = bounded(HISTORY, item["relative_to_data_root"])
            require(sha(path) == item["sha256"], "Excluded asset drift")
        exclusion_rows.append({"id": exclusion["id"], "operator_input_review": exclusion["operator_input_review"], "source_mask_pins_unchanged": True, "candidate_generation_absent": True})
    pages = []
    for page in range(8):
        path = bounded(OUT, "preview/page_%02d.png" % (page + 1))
        sheet = pixels(path)
        require(sheet.shape == (1148, 1024, 3), "Sheet dimensions")
        cells = []
        for slot, case in enumerate(protocol["cases"][page * 4:(page + 1) * 4]):
            image = pixels(OUT / "inputs" / (case["id"] + ".png"))
            removal = pixels(OUT / "masks" / (case["id"] + ".png"), "L") == 255
            overlay = image.astype(np.float32)
            overlay[removal] = overlay[removal] * np.float32(0.6) + np.array([255, 120, 20], np.float32) * np.float32(0.4)
            expected_cells = (image, np.rint(overlay).astype(np.uint8), pixels(OUT / "baseline" / (case["id"] + ".png")), pixels(OUT / "images" / (case["id"] + ".png")))
            for column, expected_cell in enumerate(expected_cells):
                y, x = 28 + slot * 280, column * 256
                equal(sheet[y:y + 256, x:x + 256], expected_cell, "Sheet page/row/column %s/%s/%s" % (page + 1, slot, column))
                cells.append({"id": case["id"], "column": column, "exact_source_or_result_pixels": True})
        pages.append({"file": pin(path), "cell_count": len(cells), "cells": cells})
        mandatory[str(path)] = pin(path)
    require({x.name for x in (OUT / "preview").iterdir()} == {"page_%02d.png" % i for i in range(1, 9)}, "Sheet inventory")
    metadata_names = ["protocol.json", "results.json", "execution.json", "supervisor_started.json", "supervisor_receipt.json", "progress.json", "worker.log"] + ["progress_%02d.json" % i for i in range(1, 33)]
    for name in metadata_names:
        path = bounded(OUT, name)
        mandatory[str(path)] = pin(path)
    output_size = sum(x.stat().st_size for x in OUT.rglob("*") if x.is_file())
    require(output_size <= protocol["budgets"]["maximum_output_bytes"], "Saved output cap")
    pins_after = {name: pin(path) for name, path in required.items()}
    require(pins_before == pins_after, "Dependency changed during audit")
    for item in mandatory.values():
        require(sha(Path(item["path"])) == item["sha256"], "Artifact changed during audit")
    require(not any(name == "torch" or name.startswith("torch.") for name in sys.modules), "Neural runtime imported")
    report = {
        "complete": True, "independent": True, "audited_UTC": datetime.now(timezone.utc).isoformat(),
        "protocol": pin(required["old_protocol"]), "runner_protocol": pin(required["protocol"]),
        "results": pin(OUT / "results.json"), "admission": pin(required["admission"]),
        "source_review": pin(required["source_review"]), "runner": pin(required["runner"]),
        "auditor": pin(__file__), "cases": 32, "outputs": cases,
        "raw_compositions_recomputed_exactly": 28, "empty_mask_exact_bypasses": 4,
        "unchanged_visible_source_bytes": kept_bytes, "excluded_inputs_unchanged": exclusion_rows,
        "rendered_pages": pages, "rendered_cells_verified": 128,
        "forward_events_audited": 29, "case_forwards_reported": 28, "parity_forwards_reported": 1,
        "saved_parity_raw_equal": True, "saved_stage_arrays_verified": 93,
        "neural_replay_performed_by_auditor": False,
        "state_receipt": {"before": results["generator_state_before"], "after": results["generator_state_after"], "reported_unchanged": True, "independently_reloaded_state": False},
        "strict_state_loading_evidence": "Pinned admitted runner requires restricted CPU tensor-only loading and strict Generator256 loading before any forward; completed result/state receipts imply these guards passed. Auditor performed no model loading.",
        "worker_seconds_reported": results["seconds"], "supervisor": pin(OUT / "supervisor_receipt.json"),
        "supervisor_returncode": 0, "supervisor_seconds_reported": supervisor["seconds"],
        "maximum_case_seconds_reported": max(row["seconds"] for row in results["rows"]),
        "all_reported_runtimes_within_frozen_bounds": True,
        "saved_output_bytes_before_audit": output_size, "maximum_output_bytes": protocol["budgets"]["maximum_output_bytes"],
        "artifact_count_verified": len(mandatory), "artifacts": list(mandatory.values()),
        "dependency_pins_before": pins_before, "dependency_pins_after": pins_after,
        "model_qualified": False, "automatic_quality_qualified": False, "assisted_quality_qualified": False,
        "visual_quality_review_performed_by_this_auditor": False,
        "training_calls": 0, "neural_calls": 0, "torch_imported": False, "optimizer_updates": 0,
        "VM_operations": 0, "app_promotion": False, "goal_complete": False,
        "interpretation": "Exact saved-array, byte-preservation and provenance audit. Runtime/state/forward counts are independently cross-checked receipts from the admitted run, not fresh neural measurements. All-case visual review remains separate.",
        "audit_seconds": time.monotonic() - started,
    }
    with AUDIT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"complete": True, "audit": pin(AUDIT), "cases": 32, "raw_compositions": 28, "empty_bypasses": 4, "cells": 128, "artifacts": len(mandatory), "neural_calls": 0, "audit_seconds": report["audit_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
