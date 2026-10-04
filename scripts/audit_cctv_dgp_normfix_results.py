"""Original independent return audit plus frozen normalization-correction lineage."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from cctv_dgp_pilot import read, sha, write
from audit_cctv_dgp_pilot_results import audit, extract_return, require
from run_cctv_dgp_normfix_vm import verify_correction, CORRECTION_ID


def audit_corrected(root, out, fix_dir, verify_recognizer=False):
    _, manifest = verify_correction(root, fix_dir)
    require(sha(Path(__file__)) == manifest["assets_sha256"]["scripts/audit_cctv_dgp_normfix_results.py"],
            "Executed correction auditor differs from the manifest")
    receipt = audit(root, out, verify_recognizer)
    revision = read(out / "normfix_revision.json")
    require(revision["correction_id"] == CORRECTION_ID and
            revision["manifest_sha256"] == sha(fix_dir / "normfix_v2.json") and
            revision["parent_protocol_sha256"] == receipt["protocol_sha256"] and
            revision["expected_instance_norm_layers"] == 5 and
            revision["six_image_eval_state_unchanged"] and
            revision["additional_student_forward_calls"] == 1 and
            revision["additional_optimizer_updates"] == 0 and
            revision["original_cuda_gradient_preflight_passed"] and
            revision["data_loss_seed_update_budget_unchanged"],
            "Norm correction compatibility/lineage record differs")
    execution = read(out / "execution.json")
    require(revision["six_image_eval_state_hash"] == execution["initial_model_state"],
            "Corrected tail evaluation changed starting state")
    for name in ["normfix_v2.json", *manifest["assets_sha256"]]:
        require(sha(out / "runtime_revision" / name) == sha(fix_dir / name),
                "Returned correction source differs: " + name)
    return {**receipt, "correction_id": CORRECTION_ID,
            "correction_manifest_sha256": revision["manifest_sha256"],
            "six_image_eval_state_unchanged": True}


def save_receipt(out, receipt, receipt_path=None):
    out = Path(out).resolve()
    standard = out / "independent_audit.json"
    target = Path(receipt_path).resolve() if receipt_path else standard
    require(target == standard or not target.is_relative_to(out),
            "Save an additional audit outside the returned output inventory")
    require(not target.exists(), "Preserve the existing audit receipt; choose a new --receipt")
    write(target, receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--fix-dir", type=Path, default=ROOT)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--extract-to", type=Path)
    parser.add_argument("--verify-recognizer", action="store_true")
    parser.add_argument("--receipt", type=Path,
                        help="Separate local receipt outside output when the VM receipt already exists")
    args = parser.parse_args()
    root, fix_dir = args.root.resolve(), args.fix_dir.resolve()
    if args.archive:
        if not args.extract_to:
            parser.error("--extract-to is required with --archive")
        # Verify preparation/correction before extracting any return files.
        verify_correction(root, fix_dir)
        out = extract_return(args.archive, args.extract_to)
        require(sha(args.extract_to / "protocol.json") == sha(root / "protocol.json"),
                "Transferred parent protocol differs")
    else:
        out = args.results or root / "outputs/cctv_dgp_pilot"
    receipt = audit_corrected(root, out, fix_dir, args.verify_recognizer)
    save_receipt(out, receipt, args.receipt)
    print(receipt)
