"""Versioned correction for the confirmed zero-update InstanceNorm failure.

Reuse the unchanged pilot code/data/protocol. Archive only its known zero-update
failure, apply the frozen-statistics adapter, and add a six-image VM preflight.
"""
import argparse
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import run_cctv_dgp_pilot_vm as original
from cctv_dgp_pilot import read, write, sha, state_hash, verify_bundle, PilotDataset
from cctv_dgp_frozen_norm import install_frozen_instance_norm

PARENT_PROTOCOL = "b652914335e33375f8108ba427e4b5be99fd86e811f455b307ab72ae7cf152f6"
CORRECTION_ID = "instance-norm-eval-buffer-clones-v2"
FAILED_PARTIAL = "a8fa8cc1505e280236880016e871cdcc34766b6003ed476a2a4a97f01db67cfa"


def verify_correction(root, fix_dir):
    root, fix_dir = Path(root).resolve(), Path(fix_dir).resolve()
    protocol = verify_bundle(root)
    if sha(root / "protocol.json") != PARENT_PROTOCOL:
        raise ValueError("Correction requires the unchanged original pilot protocol")
    for name, pin in protocol["assets_sha256"].items():
        if name.startswith("models/") or name in (
                "cctv_dgp_pilot.py", "scripts/run_cctv_dgp_pilot_vm.py",
                "scripts/audit_cctv_dgp_pilot_results.py"):
            if sha(ROOT / name) != pin:
                raise ValueError("Executed original code differs: " + name)
    manifest = read(fix_dir / "normfix_v2.json")
    if (manifest["correction_id"] != CORRECTION_ID or
            manifest["parent_protocol_sha256"] != PARENT_PROTOCOL or
            manifest["expected_instance_norm_layers"] != 5):
        raise ValueError("Correction manifest differs")
    for name, pin in manifest["assets_sha256"].items():
        path = (fix_dir / name).resolve()
        if not path.is_relative_to(fix_dir) or sha(path) != pin:
            raise ValueError("Changed correction asset: " + name)
    import cctv_dgp_frozen_norm
    if (sha(Path(__file__)) != manifest["assets_sha256"]["scripts/run_cctv_dgp_normfix_vm.py"] or
            sha(Path(cctv_dgp_frozen_norm.__file__)) != manifest["assets_sha256"]["cctv_dgp_frozen_norm.py"]):
        raise ValueError("Executed correction code differs from the manifest")
    return protocol, manifest


def archive_zero_update_failure(root):
    """Preserve the specific user-reported failed run; refuse arbitrary repeats."""
    root = Path(root)
    out = root / "outputs/cctv_dgp_pilot"
    archived = root / "outputs/cctv_dgp_pilot_failed_v1"
    if archived.exists() or not out.is_dir():
        raise ValueError("Expected one original failure and no existing failure archive")
    failure = read(out / "failure.json")
    if (failure.get("error") != "Matched branch did not start at identical baseline" or
            failure.get("optimizer_updates_recorded") != 0 or
            failure.get("partial_state_sha256") != FAILED_PARTIAL or
            sha(out / "partial_state.pth") != FAILED_PARTIAL or
            (out / "updates.jsonl").exists() or (out / "results.json").exists()):
        raise ValueError("Not the confirmed zero-update failure; preserve it and stop")
    for name in ("pilot.log", "environment.txt", "cuda_runtime_before.txt", "pip_check.log"):
        source = root / name
        if source.is_file():
            target = out / ("original_" + name)
            if target.exists():
                raise ValueError("Existing historical runtime record: " + target.name)
            shutil.copyfile(source, target)
    out.rename(archived)
    print("Preserved original zero-update failure: " + str(archived), flush=True)
    return archived


def configure_runner(fix_dir, manifest):
    model_class, old_preflight = original.DGPSynthesizer, original.preflight

    def corrected_model(*args, **kwargs):
        model = model_class(*args, **kwargs)
        count = install_frozen_instance_norm(model)
        if count != 5:
            raise ValueError("Expected five corrected student InstanceNorm layers")
        return model

    def corrected_preflight(root, out, protocol, model, identity, perceptual, deadline):
        teacher_before = old_preflight(root, out, protocol, model, identity, perceptual, deadline)
        before = state_hash(model)
        dataset = PilotDataset(root, protocol, protocol["validation_cases"][-6:])
        low = torch.stack([dataset[index]["low"] for index in range(6)]).cuda()
        with torch.no_grad():
            output = model(low)
        if not torch.isfinite(output).all() or state_hash(model) != before:
            raise ValueError("Corrected six-image evaluation changed student state")
        original.check_clock(deadline)
        revision = {
            "correction_id": CORRECTION_ID,
            "parent_protocol_sha256": sha(root / "protocol.json"),
            "manifest_sha256": sha(fix_dir / "normfix_v2.json"),
            "expected_instance_norm_layers": 5,
            "six_image_eval_state_unchanged": True,
            "six_image_eval_state_hash": before,
            "additional_student_forward_calls": 1,
            "additional_optimizer_updates": 0,
            "original_cuda_gradient_preflight_passed": True,
            "data_loss_seed_update_budget_unchanged": True,
        }
        write(out / "normfix_revision.json", revision)
        saved = out / "runtime_revision"
        saved.mkdir()
        for name in ["normfix_v2.json", *manifest["assets_sha256"]]:
            target = saved / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fix_dir / name, target)
        print("Norm correction passed: batch6, identical student state; training follows baseline validation", flush=True)
        return teacher_before

    original.DGPSynthesizer, original.preflight = corrected_model, corrected_preflight


def record_failure(out, error):
    if not out.exists() or (out / "failure.json").exists():
        return
    context = original.RUN_CONTEXT
    partial_sha = partial_error = None
    if "model" in context:
        try:
            partial = {key: value for key, value in context.items()
                       if key not in ("model", "optimizer", "start")}
            partial["model"] = context["model"].state_dict()
            if "optimizer" in context:
                partial["optimizer"] = context["optimizer"].state_dict()
            torch.save(partial, out / "partial_state.pth")
            partial_sha = sha(out / "partial_state.pth")
        except Exception as export_error:
            partial_error = type(export_error).__name__ + ": " + str(export_error)
    write(out / "failure.json", {
        "complete": False, "error_type": type(error).__name__, "error": str(error),
        "partial_state_sha256": partial_sha, "partial_export_error": partial_error,
        "optimizer_updates_recorded": context.get("total_updates", 0),
        "resume_permitted": False, "correction_id": CORRECTION_ID,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--fix-dir", type=Path, default=ROOT)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--archive-zero-update-failure", action="store_true")
    args = parser.parse_args()
    root, fix_dir = args.root.resolve(), args.fix_dir.resolve()
    protocol, manifest = verify_correction(root, fix_dir)
    if args.verify:
        print({"original_bundle_verified": True, "correction_verified": CORRECTION_ID,
               "references": len(protocol["references"]), "optimizer_updates": 0})
        return
    original.require_vm(root)
    if args.archive_zero_update_failure:
        archive_zero_update_failure(root)
    out = root / "outputs/cctv_dgp_pilot"
    if out.exists():
        raise ValueError("Output already exists; no automatic repeat or resume")
    configure_runner(fix_dir, manifest)
    try:
        original.run(root, out)
    except Exception as error:
        record_failure(out, error)
        raise


if __name__ == "__main__":
    main()
