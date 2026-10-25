"""Publish audited status while retaining every byte of the preceding documents."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLICATION = ROOT / "outputs/codeformer_u64_role_gradients_v1_return_audit/publication_v1"
DOCUMENTS = (
    "SYSTEM_WORKFLOW_AND_GOAL.md",
    "PRACTICAL_OUTPUT_SCOPE.md",
    "PROJECT_HANDOFF.md",
    "CCTV_CODEFORMER_FINETUNING_PLAN_V1.md",
)
GUIDE = "CCTV_CODEFORMER_U64_ROLE_GRADIENTS_V1_VM.md"
PREFIX = """# 23 October 2026: original-u64 diagnostic audited; next design is source-only

[Audited return and diagnosis](CCTV_CODEFORMER_U64_ROLE_GRADIENTS_V1_RETURN_REVIEW.md).
[Next gate-aligned design](CCTV_CODEFORMER_U64_GATE_ALIGNED_V1_DESIGN.md).

The user's manual zero-update diagnostic completed. Its archive is
1,226,001,016 bytes, SHA256
`c052f2b8c09e42d6ee82c263e5a9b1777b3eeb114c6e111a82ba46b4b1862be7`;
protocol SHA256 `07d00785b4aef82c2e915a032fa2808064d0f728972f8c518694698a13f5d5e3`.
The worker exited 0. All 72 forward outputs, 192 saved gradient files and the
unchanged complete 575-tensor original-u64 state independently verify. The
operation journal contains 799 events. Saved-pixel/objective checks pass 2,466
checks; root inspected all 18 planned sheets. Final supervisor/export process
exit receipts are unavailable by export design; verified transport does not
establish those exit codes or model quality. This run created no new model.

The diagnosis supports changing the update objective before adding epochs.
Low-light gradients contribute 62.23% of the mean per-photo sum of role-objective
gradient norms. They oppose localized/global/clear objectives on 14/15/12 of
16 matched photographs. The mixed direction can improve dark images while
worsening fidelity elsewhere; the objective also conflicts with some structure
gradients. These are initializer measurements, not a unique explanation of every
earlier finite-step failure.

A separate saved-array exploration finds a direction that descends all eight
pooled fidelity/structure gradients in the existing heads. It still worsens
12 of 128 individual derivatives and proves no finite-step, PNG or guard outcome.
The next inert gate-aligned solver and bound design pass 12 analytical fixtures
and independent source review. They retain exact original trained u64 and all
original preservation requirements. A prospective test is limited to four
training updates, 540 forwards and 512 gradient queries, with actual RAW/PNG
checks after each update. These are design limits, not an implemented worker.
No runnable worker, sealed transfer packet or new launch guide is ready.
The completed gradient diagnostic and failed eight-update training must not be
rerun, resumed or extended. All actual training remains manually launched on L4.

STATUS: COMPLETED_ZERO_UPDATE_DIAGNOSTIC_AUDITED_NEXT_DESIGN_INERT.
Original trained u64 remains the retained challenger; U96 remains the experimental
app incumbent. No winner was qualified and automatic goal advancement did not
trigger. Preserve all original checkpoints, states, splits, gate failures and
broader inherited DEV failures. Reviewed Protect/Apply/bypass remains separate;
automatic blur matching and later completion development remain deferred.
Native/wild evidence is unpaired. FINAL remains unopened/execution-unfrozen;
independent reviewers are pending. Full goal ACTIVE/incomplete.

The historical prepared status below is superseded by this completed-return
status. Exact prior document bytes follow for provenance.

---

"""
GUIDE_PREFIX = """# Completed diagnostic: do not repeat the historical launch steps below

23 October 2026. **Executed and independently audited; zero updates.**
The user's returned archive and all 72 outputs/192 gradient files verify.
All 575 original-u64 model tensors remained unchanged. This diagnostic created
no new model. Its historical upload and tmux instructions below are retained
for provenance and must not be rerun or resumed.

Read the [return review](CCTV_CODEFORMER_U64_ROLE_GRADIENTS_V1_RETURN_REVIEW.md)
and [next source-only design](CCTV_CODEFORMER_U64_GATE_ALIGNED_V1_DESIGN.md).
The new design has no runnable worker, transfer package or launch guide yet.
U96 remains the experimental app incumbent; original trained u64 is retained.
Full goal ACTIVE/incomplete. Exact prior guide bytes follow.

---

"""


def pin(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected ordinary file: {path}")
    data = path.read_bytes()
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": len(data),
            "sha256": sha256(data).hexdigest()}


def main():
    if PUBLICATION.exists():
        raise ValueError("Retain existing publication; never overwrite its receipts")
    requirements = {
        "outputs/codeformer_u64_role_gradients_v1_return_transport/transport_receipt_v1.json": "complete",
        "outputs/codeformer_u64_role_gradients_v1_return_pixels/audit_v1.json": "all_checks_pass",
        "outputs/codeformer_u64_role_gradients_v1_return_geometry/audit.json": "all_192_exact_schema_canonical_description_file_bindings_pass",
        "outputs/codeformer_u64_role_gradients_v1_return_audit/saved_state_audit_v1_r1.json": "saved_state_audit_pass",
        "outputs/codeformer_u64_role_gradients_v1_return_audit/independent_state_auditor_r1_readback.json": "complete",
        "outputs/codeformer_u64_role_gradients_v1_return_audit/independent_state_visual_readback_v1.json": "complete",
        "outputs/codeformer_u64_role_gradients_v1_return_audit/root_visual_review_v1.json": "root_personally_inspected_all_planned_sheets",
        "outputs/codeformer_u64_role_gradients_v1_return_audit/return_report_readback_v1.json": "return_report_numerical_status_readback_pass",
        "outputs/codeformer_u64_gate_aligned_v1_design/root_source_tests_v1.json": "all_tests_pass",
        "outputs/codeformer_u64_gate_aligned_v1_design/independent_source_review_v1.json": "review_pass",
        "outputs/codeformer_u64_gate_aligned_v1_design/prototype_tests_receipt_v1.json": "complete",
    }
    consumed = []
    for name, success in requirements.items():
        path = ROOT / name
        data = json.loads(path.read_bytes())
        if data.get(success) is not True:
            raise ValueError(f"Missing required audit: {name}: {success}")
        if data.get("unresolved_blockers"):
            raise ValueError(f"Unresolved audit blockers: {name}")
        if data.get("model_qualified") is True or data.get("manual_packet_ready") is True:
            raise ValueError("Publication boundaries disagree with receipt")
        consumed.append(pin(path))
    for name, expected in {
        "CCTV_CODEFORMER_U64_ROLE_GRADIENTS_V1_RETURN_REVIEW.md":
            "889000dbe19f29c9b063c43baf9221fc6ea17d8e0525d30740c35f3c5c6328c4",
        "CCTV_CODEFORMER_U64_GATE_ALIGNED_V1_DESIGN.md":
            "5bd6ae7c482d8432abe4247d578144be4d8a3b47ed4633dbfaf43b25f91eb2c4",
    }.items():
        value = pin(ROOT / name)
        if value["sha256"] != expected:
            raise ValueError(f"Reviewed report changed: {name}")
        consumed.append(value)
    for name in (
        "research/codeformer_u64_gate_aligned_v1/solver.py",
        "research/codeformer_u64_gate_aligned_v1/contract.py",
        "tests/test_u64_gate_aligned_solver_v1.py",
        "scripts/prove_u64_gate_aligned_solver_v1.py",
        "outputs/codeformer_u64_gate_aligned_v1_design/prototype_contract_v1.json",
    ):
        consumed.append(pin(ROOT / name))

    pending = []
    for name in (*DOCUMENTS, GUIDE):
        path = ROOT / name
        before_pin = pin(path)
        original = path.read_bytes()
        prefix = (GUIDE_PREFIX if name == GUIDE else PREFIX).encode("utf-8")
        pending.append((path, original, before_pin, prefix))
    if pending[-1][2]["sha256"] != "9109eadc9ac45354835f3096c44bb3bd211701e2c2da86b0e8dc59ca0430e931":
        raise ValueError("Original executed guide differs from reviewed guide")

    PUBLICATION.mkdir()
    changes = []
    for path, original, before_pin, prefix in pending:
        if pin(path) != before_pin:
            raise ValueError("Concurrent document change; stop before writing")
        backup = PUBLICATION / (path.name + ".before")
        backup.write_bytes(original)
        if backup.read_bytes() != original:
            raise ValueError("Prior document backup differs")
        path.write_bytes(prefix + original)
        current = path.read_bytes()
        if current[len(prefix):] != original:
            raise ValueError("Historical bytes were not preserved")
        changes.append({"before": before_pin, "backup": pin(backup),
                        "after": pin(path), "prefix_bytes": len(prefix),
                        "prior_bytes_preserved_exactly": True})
    for value in consumed:
        if pin(ROOT / value["path"]) != value:
            raise ValueError("Consumed audit or reviewed source changed")
    receipt = {
        "complete": True, "created_UTC": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETED_ZERO_UPDATE_DIAGNOSTIC_AUDITED_NEXT_DESIGN_INERT",
        "changes": changes, "consumed_before_after_exact": consumed,
        "all_prior_document_bytes_retained": True,
        "new_worker_or_transfer_package_ready": False,
        "training_or_VM_calls": 0, "model_qualified": False,
        "app_incumbent": "experimental U96", "challenger": "retained original trained u64",
        "automatic_goal_advancement_triggered": False, "full_goal": "ACTIVE/incomplete",
        "source": pin(Path(__file__).resolve()),
    }
    destination = PUBLICATION / "publication_receipt_v1.json"
    destination.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"complete": True, "documents": len(changes),
                      "receipt": str(destination), "model_qualified": False,
                      "manual_packet_ready": False}))


if __name__ == "__main__":
    main()
