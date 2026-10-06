"""One fixed causal ablation: restore after the saved reviewed completion.

Read-only historical data; no fitting, thresholds, training or app adoption.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cctv_dgp_pilot import state_hash
from dgp_face_restoration import sha
from dgp_face_workflow_v3 import DGPFaceWorkflow, canonical_tensor
from face_workflow import png_bytes
from scripts.verify_dgp_app_coverings_v3 import pixels, visible_metrics

PARENT = ROOT / "outputs/dgp_app_covering_review_v3"
OUT = ROOT / "outputs/dgp_app_completion_context_v3"


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    if OUT.exists():
        raise ValueError("Preserve context ablation; do not rerun")
    parent = json.loads((PARENT / "results.json").read_text())
    plan = json.loads((PARENT / "plan.json").read_text())
    proof = json.loads((PARENT / "saved_output_audit.json").read_text())
    assert parent["complete"] and proof["complete"] and proof["results_sha256"] == sha(PARENT / "results.json")
    cases = {r["id"]: r for r in plan["cases"]}
    supported = [r for r in parent["rows"] if not r["assisted"]["off"]["rejected"]]
    assert len(supported) == 32
    for row in supported:
        for mode in ("off", "on"):
            path = row["assisted"][mode]["output"]
            assert sha(PARENT / path) == parent["artifacts_sha256"][path]
        case = cases[row["id"]]
        assert sha(ROOT / case["reviewed"]) == plan["sources_sha256"][case["reviewed"]]
        for key in ("input",):
            assert sha(ROOT / case[key]) == plan["sources_sha256"][case[key]]
    OUT.mkdir()
    write(OUT / "plan.json", {"date": "2026-10-05", "source_sha256": sha(Path(__file__)),
          "parent_results_sha256": sha(PARENT / "results.json"), "parent_audit_sha256": sha(PARENT / "saved_output_audit.json"),
          "cases": [r["id"] for r in supported], "cap_seconds": 180, "maximum_dgp_forwards": 32,
          "fixed_variant": "Canonical DGP on the saved rendered Off completion; original visible RGB, completed reviewed holes",
          "hypothesis": "Covering texture differs from DGP's uncovered-face training context; completion first may reduce the demonstrated visible degradation",
          "scope": "Reused development photographs; no native CCTV or independent final assessment",
          "fitting": False, "training": False, "selector_changes": False, "app_changes": False})
    started = time.monotonic()
    engine = DGPFaceWorkflow(device="cpu")
    engine._runtime()
    model = engine._restorer()
    before = state_hash(model)
    assert before == parent["state_before"]["dgp"]
    rows = []
    for row in supported:
        if time.monotonic() - started > 180:
            raise TimeoutError("180-second context ablation cap exceeded")
        case = cases[row["id"]]
        context = pixels(PARENT / row["assisted"]["off"]["output"])
        removal = pixels(ROOT / case["reviewed"]).mean(axis=-1) >= 127.5
        metadata = json.loads((PARENT / "metadata" / (row["id"] + "_on.json")).read_text())
        with torch.inference_mode():
            raw = model(canonical_tensor(context, "cpu"))[0].permute(1, 2, 0).numpy().copy()
        output = np.floor(raw * np.float32(255)).astype(np.uint8)
        if metadata["display_processing"]["colour_policy"]["applied"]:
            output = np.repeat(cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)[..., None], 3, axis=-1)
        output[removal] = context[removal]
        np.save(OUT / (row["id"] + "_raw.npy"), raw, allow_pickle=False)
        (OUT / (row["id"] + "_output.png")).write_bytes(png_bytes(output))
        record = {"id": row["id"], "family": row["family"], "condition": row["condition"],
                  "completed_pixels_retained": True, "input": str(PARENT / row["assisted"]["off"]["output"])}
        if not removal.any():
            cached_raw = np.load(PARENT / "stages" / (row["id"] + "_on.npz"), allow_pickle=False)["dgp"][0].transpose(1, 2, 0)
            np.testing.assert_array_equal(raw, cached_raw)
            record["empty_mask_raw_exact"] = True
        if case["synthetically_degraded"]:
            reference = cases[row["id"].replace("_degraded", "_native")]
            target = pixels(ROOT / reference["input"])
            record["paired_known_nonremoved"] = {"off": visible_metrics(context, target, removal),
                "v3_covered_input": visible_metrics(pixels(PARENT / row["assisted"]["on"]["output"]), target, removal),
                "completed_context": visible_metrics(output, target, removal)}
        rows.append(record)
    after = state_hash(model)
    assert before == after
    seconds = time.monotonic() - started
    assert seconds <= 180
    write(OUT / "results.json", {"complete": True, "seconds": seconds, "dgp_forwards": len(rows),
          "state_before": before, "state_after": after, "rows": rows,
          "artifacts_sha256": {p.name: sha(p) for p in OUT.iterdir() if p.is_file()},
          "completion_forwards": 0, "detector_forwards": 0, "optimizer_updates": 0, "backward_calls": 0,
          "training": False, "native_cctv_used": False, "hidden_accuracy": None, "app_adopted": False,
          "visual_review_pending": True, "independent_final_review": False})
    print(json.dumps({"complete": True, "seconds": seconds, "dgp_forwards": len(rows), "state_unchanged": True}))


if __name__ == "__main__":
    main()
