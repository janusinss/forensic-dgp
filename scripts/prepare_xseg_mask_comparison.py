"""Freeze a visible-face segmenter comparison before its first forward."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/xseg_mask_comparison_v1"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    if OUT.exists():
        raise ValueError("Preserve prior comparison")
    native_dir = ROOT / "outputs/practical_gallery_v2"
    old = json.loads((native_dir / "protocol.json").read_text())
    feet_path = ROOT / "outputs/practical_footprints_v3/frozen_protocol.json"
    feet = json.loads(feet_path.read_text())
    footprints = {c["id"].removesuffix("_native"): c for c in feet["cases"]}
    broad_path = ROOT / "outputs/broad_covering_gallery_v1/frozen_protocol.json"
    broad = json.loads(broad_path.read_text())
    broad_result = ROOT / "outputs/broad_face_workflow_outputs_v1/results.json"
    if sha(broad_path) != "b7a1a447482d650bb0a18b13d649f0c548c3c59acd6db202111169599c805bc9" or sha(broad_result) != "afe8cb6b2d768460082890fe8ded09dd125e19f599f54f55a496e2a4073a94d3":
        raise ValueError("Broad evidence changed")
    assets, cases = {}, []
    def asset(path):
        path = Path(path)
        rel = path.relative_to(ROOT).as_posix()
        assets[rel] = sha(path)
        return rel
    for source in old["cases"]:
        base_id = source["id"].removesuffix("_native").removesuffix("_degraded")
        revised = footprints.get(base_id)
        removal = ROOT / revised["new_removal"] if revised else native_dir / source["removal_proposal"]
        protected = ROOT / revised["preserved_region"] if revised else None
        cases.append({"id": source["id"], "family": source["family"],
                      "input": asset(native_dir / source["input"]), "reviewed": asset(removal),
                      "protected": asset(protected) if protected else None,
                      "synthetically_degraded": source["synthetically_degraded"], "expected_rejection": False,
                      "pose_scope": "difficult three-quarter diagnostic" if base_id.startswith("02_") else "frontal/mild turn development",
                      "baseline_cached": None, "exposure": source["exposure"], "hidden_ground_truth": None})
    for source in broad["cases"]:
        cached = ROOT / "outputs/broad_face_workflow_outputs_v1/automatic_masks" / (source["id"] + ".png")
        cases.append({"id": source["id"], "family": source["family"], "input": asset(ROOT / source["input"]),
                      "reviewed": asset(ROOT / source["proposal"]), "protected": None,
                      "synthetically_degraded": source["synthetically_degraded"], "expected_rejection": source["expected_rejection"],
                      "pose_scope": source["pose"], "baseline_cached": asset(cached), "exposure": source["exposure"],
                      "hidden_ground_truth": None})
    for path in (native_dir / "protocol.json", feet_path, broad_path, broad_result,
                 ROOT / "outputs/xseg_pretrained_v1/acquisition.json",
                 ROOT / "outputs/xseg_pretrained_v1/xseg_1.onnx",
                 ROOT / "outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth",
                 ROOT / "face_workflow.py", ROOT / "xseg_occlusion.py",
                 ROOT / "scripts/run_xseg_mask_comparison.py"):
        asset(path)
    if len(cases) != 36 or len({c["id"] for c in cases}) != 36:
        raise ValueError("Expected 36 fixed native/degraded cases")
    protocol = {"format": "dgp-xseg-mask-comparison-v1", "date": "2026-10-02", "frozen_before_first_forward": True,
                "cases": cases, "assets_sha256": assets, "preparer_sha256": sha(__file__),
                "policy": "xseg-frontal-ellipse-proposal-v1", "xseg_input": "RGB converted to BGR float32/255 NHWC 1x256x256x3",
                "xseg_output": "Visible-face probability; no FaceFusion smoothing/remap is applied",
                "proposal": "p_visible < 0.5 within fixed frontal ellipse center(127.5,137), radii(83,104); dilate 3px; constrain to ellipse",
                "baseline": "Retained detector threshold0.5 plus3px; 16 verified cached proposals and20 new inference calls",
                "budget": {"xseg_forwards": 36, "baseline_detector_forwards": 20, "wall_seconds_after_loading": 300,
                           "completion_forwards": 0, "restoration_forwards": 0, "optimizer_updates": 0},
                "device": "CPU", "threads": 4, "training_admitted": False,
                "metrics": "Full-mask recall/precision/IoU relative to fixed operator proposals; excess outside3px tolerance; protected wire edits; empty-control marked area; conditional visibility guard",
                "criterion": "No app promotion from mask averages alone. Require preservation of clear-glasses/uncovered controls, better usable native covering proposals across requested families, no spurious rejection, and reviewed output comparison before selection.",
                "scope": "Exposed small developmental gallery, 36 cases from18 sources; manual proposals are covering references, not hidden facial truth. Two three-quarter cases are diagnostic, not required first-version pose coverage. No target-based routing or threshold tuning in this version."}
    OUT.mkdir()
    (OUT / "frozen_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"cases": len(cases), "protocol_sha256": sha(OUT / "frozen_protocol.json"), "first_forward_started": False}))


if __name__ == "__main__":
    main()
