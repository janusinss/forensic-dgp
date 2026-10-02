"""Freeze numerical-output repair, retaining all V1 spatial choices and evidence."""
import json
from pathlib import Path

from scripts.prepare_xseg_mask_comparison import sha

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "outputs/xseg_mask_comparison_v1"
OUT = ROOT / "outputs/xseg_mask_comparison_v2"


def main():
    if OUT.exists():
        raise ValueError("Preserve prior comparison")
    if sha(OLD / "frozen_protocol.json") != "e49e5c75d9d99e146783661eb192ec7492822a22095619ffe68360a916be5add":
        raise ValueError("V1 protocol changed")
    p = json.loads((OLD / "frozen_protocol.json").read_text())
    r = json.loads((OLD / "results.json").read_text())
    if r["complete"] or len(r["rows"]) != 23 or r["xseg_forwards"] != 24 or r["baseline_detector_forwards"] != 20:
        raise ValueError("Expected preserved numerical interruption")
    for path, digest in p["assets_sha256"].items():
        if sha(ROOT / path) != digest:
            raise ValueError("V1 asset changed")
    for path, digest in r["assets_sha256"].items():
        if sha(OLD / path) != digest:
            raise ValueError("V1 saved output changed")
    diagnostic_root = ROOT / "outputs/xseg_pretrained_v1"
    diagnostic = json.loads((diagnostic_root / "numeric_output_diagnostic_v1.json").read_text())
    if not diagnostic["finite"] or diagnostic["max"] != 1.0000001192092896 or diagnostic["above_one"] != 1:
        raise ValueError("Numeric diagnostic differs")
    rows = {row["id"]:row for row in r["rows"]}
    assets = dict(p["assets_sha256"])
    def asset(path):
        rel = Path(path).relative_to(ROOT).as_posix()
        assets[rel] = sha(path)
        return rel
    for case in p["cases"]:
        prior = rows.get(case["id"])
        if prior:
            case["probability_cached"] = asset(OLD / prior["visible_probability"])
            if case["baseline_cached"] is None:
                case["baseline_cached"] = asset(OLD / prior["baseline_mask"])
        elif case["id"] == diagnostic["case_id"]:
            case["probability_cached"] = asset(diagnostic_root / "diagnostic_val_25_hand_mouth_degraded.npy")
        else:
            case["probability_cached"] = None
    for path in (OLD / "frozen_protocol.json", OLD / "results.json", diagnostic_root / "numeric_output_diagnostic_v1.json",
                 ROOT / "xseg_occlusion_v2.py", ROOT / "scripts/run_xseg_mask_comparison_v2.py"):
        asset(path)
    p.update({"format":"dgp-xseg-mask-comparison-v2", "policy":"xseg-frontal-ellipse-proposal-v2",
              "assets_sha256":assets, "preparer_sha256":sha(__file__),
              "revision":"Numerical repair only: clip finite visible probabilities with <=1e-6 saturation error to[0,1], consistent with the official consumer. Threshold/ellipse/margin/cases/criteria unchanged. V1 interrupted evidence is retained.",
              "cache_scope":"23 finite V1 probabilities plus1 diagnostic probability; all36 retained baseline masks cached",
              "budget":{"xseg_forwards":12,"cached_xseg_probabilities":24,"baseline_detector_forwards":0,
                        "wall_seconds_after_loading":300,"completion_forwards":0,"restoration_forwards":0,"optimizer_updates":0}})
    if sum(case["probability_cached"] is None for case in p["cases"]) != 12 or any(case["baseline_cached"] is None for case in p["cases"]):
        raise ValueError("Unexpected cache coverage")
    OUT.mkdir()
    (OUT / "frozen_protocol.json").write_text(json.dumps(p,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({"cases":36,"new_forwards":12,"cached_probabilities":24,"protocol_sha256":sha(OUT / "frozen_protocol.json")}))


if __name__ == "__main__":
    main()
