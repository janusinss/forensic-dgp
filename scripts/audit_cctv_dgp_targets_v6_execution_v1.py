"""Bind V6 exact-byte recovery and portable supervisor to the audited return."""
import argparse
import hashlib
import json
from pathlib import Path

PROTOCOL = "0fe7d036bf8567bf0860790df553afd627ac50bd5096cfda29cc7d09d71d320c"
DRIVER = "7e8f08473f417c6e5254d9e05a5c884225e4821cd121ad8af1ca0bc34fc6dc4b"
DERIVATION = "c3179744ea799e3753ca1bfd978bcd560da16373c67340090852e628afac7d5c"
ARCHIVE = "3a12b481a4950ed551c98da6955b463f33f63ee48273c7e1e3dfe20245d13c77"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def run(workspace):
    returned = workspace / "outputs/cctv_dgp_targets_return_v6"
    evidence = workspace / "outputs/cctv_dgp_targets_v6_execution_evidence_v1"
    manifest = read(evidence / "evidence_manifest.json")
    assert manifest["complete"] and manifest["protocol_sha256"] == PROTOCOL
    assert manifest["result_archive_sha256"] == ARCHIVE
    assert sha(workspace / "outputs/cctv-dgp-targets-v6-results.tar.gz") == ARCHIVE
    for name, pin in manifest["evidence_sha256"].items():
        file = evidence / name
        assert file.resolve().is_relative_to(evidence.resolve())
        assert sha(file) == pin
    core = read(returned / "local_independent_audit.json")
    assert core["complete"] and core["protocol_sha256"] == PROTOCOL
    assert core["update_trace_checked"] == 196 and not core["production_promoted"]
    assert sha(returned / "scripts/run_cctv_dgp_targets_v6_portable_v1.py") == DRIVER
    assert sha(returned / "local_source_derivation_audit_for_v6.json") == DERIVATION
    assert sha(workspace / "outputs/cctv_dgp_targets_v6_local_preparation_audit.json") == DERIVATION
    assert sha(returned / "targets_protocol_v6.json") == PROTOCOL
    launch = read(returned / "portable_execution_v1/pilot_launch.json")
    assert launch["protocol_sha256"] == PROTOCOL and launch["driver_sha256"] == DRIVER
    assert launch["local_derivation_audit_sha256"] == DERIVATION
    assert launch["host"] == "forensic-dgp-thesis" and launch["all_original_asset_sha256_verified"] == 2709
    assert not launch["vm_camera_inputs_regenerated"] and not launch["model_recipe_and_selection_changed"]
    assert launch["vm_reduced_pixels_rebuilt_by_original_verifier"]
    completion = read(workspace / "outputs/cctv_dgp_targets_v6_execution_completion_v1.json")
    assert completion["complete"] and completion["stage"] == "pilot"
    assert completion["protocol_sha256"] == PROTOCOL and completion["driver_sha256"] == DRIVER
    assert completion["optimizer_updates"] == 196 and 0 < completion["seconds"] <= 1800
    assert not completion["model_recipe_changed"] and not completion["production_promoted"]
    recovery = read(evidence / "transport_recovery_v2.json")
    assert recovery["complete"] and recovery["protocol_sha256"] == PROTOCOL and recovery["recovered_exact_files"] == 1430
    assert not recovery["model_recipe_changed"] and not recovery["camera_inputs_regenerated"] and recovery["optimizer_updates"] == 0
    for item in recovery["preserved_wrong_files"]:
        assert sha(evidence / item["preserved_as"]) == item["sha256"]
    controls = read(evidence / "control_pixel_portability_v1.json")
    assert controls["complete"] and controls["pixel_equal"] and controls["controls_checked"] == 391 and controls["optimizer_updates"] == 0
    preflight = read(evidence / "outputs/cctv_dgp_targets_v6_preflight/preflight.json")
    assert preflight["passed"] and preflight["zero_optimizer_updates"] and preflight["autograd_calls"] == 2
    assert preflight["backward_calls"] == 0 and preflight["batch_size"] == 8 and preflight["tail_batch_size"] == 7
    return {"complete": True, "protocol_sha256": PROTOCOL, "result_archive_sha256": ARCHIVE,
            "portable_driver_sha256": DRIVER, "evidence_files_checked": len(manifest["evidence_sha256"]),
            "original_assets_vm_verification_bound": 2709, "exact_files_recovered": 1430,
            "reported_vm_control_pixels_checked": 391, "standalone_preflight_autograd_calls": 2,
            "standalone_optimizer_updates": 0, "pilot_optimizer_updates": 196,
            "supervisor_seconds": completion["seconds"], "local_model_forwards": 0,
            "local_backward_calls": 0, "local_optimizer_updates": 0, "production_promoted": False,
            "limitation": "Checks exact-byte execution receipts and pinned driver; does not replay VM camera stack, CUDA gradients, recognizer forwards or prove native usefulness"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.workspace)
    with args.receipt.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(result)
