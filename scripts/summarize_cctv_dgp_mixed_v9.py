"""Describe the independently audited V9 paired results; no model execution."""
import argparse
import hashlib
import json
from pathlib import Path

PROTOCOL = "6cd9ac8d5f3bf27be773979c2f628e33f5d3cf09748452eeab91a65b05b01c70"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(return_root):
    out = return_root / "outputs/cctv_dgp_mixed_v9"
    audit_path = return_root / "local_independent_audit.json"
    audit = json.loads(audit_path.read_text())
    assert audit["complete"] and audit["protocol_sha256"] == PROTOCOL
    result_path = out / "results.json"
    result = json.loads(result_path.read_text())
    assert sha(result_path) == audit["results_sha256"]
    stages = ["baseline", "epoch2", "epoch5", "epoch10", "epoch20"]
    saved = {}
    for stage in stages:
        path = out / stage / "metrics.json"
        assert sha(path) == result["artifacts_sha256"][stage + "/metrics.json"]
        saved[stage] = json.loads(path.read_text())
    base = audit["validation_summaries"]["baseline"]
    rows = []
    for stage in stages:
        for group, row in audit["validation_summaries"][stage].items():
            start = base[group]
            failures = [metric for metric in ("MSE", "SSIM", "ArcFace_observed_fixed")
                if (row[metric] > start[metric] + 1e-12 if metric == "MSE"
                    else row[metric] < start[metric] - 1e-6)]
            rows.append({"stage": stage, "group": group, **row,
                "PSNR_gain_vs_start_db": row["PSNR"] - start["PSNR"],
                "SSIM_change_vs_start": row["SSIM"] - start["SSIM"],
                "ArcFace_change_vs_start": row["ArcFace_observed_fixed"] - start["ArcFace_observed_fixed"],
                "failed_baseline_metrics": failures})
    receipt = {"complete": True, "protocol_sha256": PROTOCOL,
        "results_sha256": sha(result_path), "independent_audit_sha256": sha(audit_path),
        "analysis_source_sha256": sha(Path(__file__)), "rows": rows,
        "input_summaries": saved["baseline"]["input_summary"],
        "selected_epoch": audit["selected_epoch"], "selected_source": audit["selected_source"],
        "native_cctv_evidence": False, "identity_accuracy": None,
        "local_model_forwards": 0, "local_backward_calls": 0, "local_optimizer_updates": 0,
        "production_promoted": False,
        "limitation": "Paired photographic proxies and fixed recognizer similarity; not native CCTV or recovered identity accuracy."}
    path = return_root / "paired_diagnosis_v9.json"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"selected_epoch": audit["selected_epoch"], "degraded": [
        {key: row[key] for key in ("stage", "PSNR", "SSIM", "ArcFace_observed_fixed", "PSNR_gain_vs_start_db", "failed_baseline_metrics")}
        for row in rows if row["group"] == "degraded"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--return-root", type=Path, required=True)
    run(parser.parse_args().return_root.resolve())
