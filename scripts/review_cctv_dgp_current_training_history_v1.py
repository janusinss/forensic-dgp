"""Read existing DGP weights and logs; perform no inference or training."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/cctv_dgp_current_training_review_v1"
PILOT = ROOT / "outputs/cctv_dgp_normfix_return_v2/outputs/cctv_dgp_pilot"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def state_digest(state):
    # This is a new review convention, not the historical pilot's state-hash format.
    h = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        header = json.dumps([key, str(tensor.dtype), list(tensor.shape)],
                            separators=(",", ":")).encode("utf-8")
        h.update(len(header).to_bytes(8, "big"))
        h.update(header)
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError("Preserve the existing review; use a new version.")
    current = ROOT / "outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth"
    selected = PILOT / "camera_identity/best.pth"
    epoch2 = PILOT / "camera_identity/epoch_2.pth"
    expected = "646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b"
    assert digest(current) == expected
    a = torch.load(current, map_location="cpu", weights_only=True)
    b = torch.load(selected, map_location="cpu", weights_only=True)
    assert set(a) == set(b) and len(a) == 622
    assert all(torch.equal(a[key], b[key]) for key in a)
    assert all(bool(torch.isfinite(value).all()) for value in a.values())
    assert digest(selected) == digest(epoch2)
    assert not any(not isinstance(value, torch.Tensor) for value in a.values())

    results = read(PILOT / "results.json")
    branch = next(b for b in results["branches"] if b["arm"]["id"] == "camera_identity")
    assert branch["selection"]["selected_epoch"] == 2
    assert branch["optimizer_updates"] == 226
    assert [x["updates"] for x in branch["epochs"]] == [113, 113]
    rows = [json.loads(line) for line in (PILOT / "updates.jsonl").read_text().splitlines()
            if line.strip()]
    assert len(rows) == results["total_optimizer_updates"] == 452
    curve = []
    groups = ["degraded", "clear", "dataset/asian_faces/blur_lr24",
              "dataset/asian_faces/motion_lr48",
              "dataset/thumbnails128x128/blur_lr24",
              "dataset/thumbnails128x128/motion_lr48"]
    for epoch, directory in [(0, "baseline"), (1, "camera_identity_epoch1"),
                             (2, "camera_identity_epoch2")]:
        metrics = read(PILOT / directory / "metrics.json")
        record = {"fine_tune_epoch": epoch, "cumulative_updates": epoch * 113,
                  "export_basis": metrics["evaluation_basis"],
                  "paired_development_groups": {g: metrics["summary"][g] for g in groups}}
        if epoch:
            batch_rows = [r for r in rows if r["arm"] == "camera_identity" and r["epoch"] == epoch]
            exposures = sum(len(r["cases"]) for r in batch_rows)
            assert len(batch_rows) == 113 and exposures == 902
            record["training"] = {
                "updates": len(batch_rows), "case_exposures": exposures,
                "weighted_mean_loss": sum(r["loss"] * len(r["cases"]) for r in batch_rows) / exposures,
                "weighted_mean_identity_loss": sum(r["identity_loss"] * len(r["cases"])
                                                   for r in batch_rows) / exposures}
        curve.append(record)

    hq_path = ROOT / "outputs/cctv_dgp_hq_source_review_v3/eligible_references.json"
    hq = read(hq_path)
    roles = Counter(r["original_role"] for r in hq["references"])
    assert roles == {"train": 391, "validation": 53}
    phase4 = ROOT / "outputs/downloaded_phase4/outputs/phase4_with_progress/metrics.jsonl"
    phase4_rows = [json.loads(line) for line in phase4.read_text().splitlines() if line.strip()]
    baseline = phase4_rows[0]["ArcFace_Sim"]
    assert [r["epoch"] for r in phase4_rows[1:]] == [27, 28, 29, 30, 31]
    assert all(r["ArcFace_Sim"] < baseline for r in phase4_rows[1:])

    bindings = [current, selected, epoch2, PILOT / "results.json", PILOT / "execution.json",
                PILOT / "updates.jsonl", hq_path, phase4,
                ROOT / "outputs/cctv_dgp_normfix_return_v2/protocol.json",
                ROOT / "dgp_face_workflow_v3.py", ROOT / "CCTV_DGP_APP_V3_INTEGRATION.md",
                ROOT / "CCTV_DGP_MIXED_V9_RESULTS.md", ROOT / "CCTV_DGP_TARGETS_RESULTS_V6.md",
                ROOT / "CCTV_DGP_POST_ACTUAL_STEP_ARCHITECTURE_REVIEW.md", Path(__file__).resolve()]
    bindings.extend(PILOT / d / "metrics.json" for d in
                    ["baseline", "camera_identity_epoch1", "camera_identity_epoch2"])
    report = {
        "version": 1, "date": "2026-10-09", "training_history_review_complete": True,
        "current_app_weight": str(current.relative_to(ROOT)).replace("\\", "/"),
        "current_app_file_sha256": expected, "selected_fine_tune_epoch": 2,
        "selected_branch_optimizer_updates": 226,
        "all_branches_optimizer_updates": 452,
        "app_selected_exact_tensor_parity": True, "equal_tensors": 622,
        "tensor_hash_convention": "length-prefixed JSON key/dtype/shape then contiguous CPU numpy bytes",
        "current_tensor_state_sha256": state_digest(a),
        "weight_file_epoch_or_optimizer_metadata": False,
        "total_ancestral_training_epochs_established": False,
        "original_training_plus_two_fine_tune_epochs": True,
        "phase4_epochs_27_through_31_are_separate_rejected_history": True,
        "curve": curve,
        "existing_audited_hq_training_references": roles["train"],
        "existing_audited_hq_validation_references": roles["validation"],
        "historical_identity_overlap_fully_established": False,
        "user_decisions": {
            "start": "current app DGP checkpoint; adjust training only where evidence justifies it",
            "native_review": "already audited native CCTV development crops; reserved final stays separate",
            "additional_references": "apply best approach; reuse audited HQ references before unnecessary acquisition",
            "credit_balance_usd_user_reported": 34,
            "per_experiment_monetary_cap_user_supplied": False,
            "migration_trigger": "user reports balance near USD 5; export and verify resumable files",
            "finite_training_timing_and_stop_rules_still_required": True,
            "deadline": "thesis submission or defense around or after 20 November 2026",
            "training_execution": "manual transfer and pasteable commands on existing L4 until replacement is identified"},
        "inference_forwards": 0, "gradient_queries": 0, "backward_calls": 0,
        "optimizer_updates_this_review": 0, "new_training_protocol_prepared": False,
        "vm_connected_this_review": False, "checkpoint_promoted": False,
        "goal_complete": False,
        "source_bindings": {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in bindings}}
    OUT.mkdir()
    (OUT / "training_history.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"complete": True, "selected_fine_tune_epoch": 2,
                      "equal_tensors": 622, "hq_training_references": 391,
                      "neural_or_training_calls": 0,
                      "report": str(OUT / "training_history.json")}))


if __name__ == "__main__":
    main()
