"""Build a small additive correction package. No training or model forwards."""
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from cctv_dgp_pilot import verify_bundle, sha, write
from run_cctv_dgp_normfix_vm import PARENT_PROTOCOL, CORRECTION_ID, verify_correction

PARENT = ROOT / "outputs/cctv_dgp_vm_bundle_v1"
DEST = ROOT / "outputs/cctv_dgp_normfix_v2"
ARCHIVE = ROOT / "outputs/cctv-dgp-normfix-v2.tar.gz"
FILES = [
    "cctv_dgp_frozen_norm.py", "scripts/run_cctv_dgp_normfix_vm.py",
    "scripts/audit_cctv_dgp_normfix_results.py", "scripts/run_cctv_dgp_normfix_vm.sh",
    "tests/test_cctv_dgp_frozen_norm.py", "tests/test_cctv_dgp_normfix_preservation.py",
    "scripts/prepare_cctv_dgp_normfix_v2.py", "CCTV_DGP_NORMFIX_V2.md",
]


def main():
    if DEST.exists() or ARCHIVE.exists() or Path(str(ARCHIVE) + ".sha256").exists():
        raise ValueError("Preserve an existing correction package")
    parent = verify_bundle(PARENT)
    if sha(PARENT / "protocol.json") != PARENT_PROTOCOL:
        raise ValueError("Original prepared protocol changed")
    if set(FILES) & set(parent["assets_sha256"]):
        raise ValueError("Correction would overwrite a frozen parent asset")
    DEST.mkdir()
    assets = {}
    for name in FILES:
        target = DEST / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith(".sh"):
            text = (ROOT / name).read_text(encoding="utf-8").replace("\r\n", "\n")
            with target.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
        else:
            shutil.copyfile(ROOT / name, target)
        assets[name] = sha(target)
    write(DEST / "normfix_v2.json", {
        "format": "dgp-cctv-runtime-correction-v2", "date": "2026-10-04",
        "correction_id": CORRECTION_ID, "parent_protocol_sha256": PARENT_PROTOCOL,
        "expected_instance_norm_layers": 5,
        "policy": "Cloned stored statistics for frozen InstanceNorm evaluation; original native operator and gradient path",
        "data_loss_seed_update_budget_unchanged": True,
        "vm_evidence_basis": "User-pasted L4 / torch2.9.1+cu129 forward-only diagnostic; not an independently received return",
        "reported_vm_batch8_changed_tensors": 0, "reported_vm_batch6_changed_tensors": 10,
        "reported_vm_largest_buffer_difference": 0.00048828125,
        "corrected_cuda_backward_verified_locally": False,
        "local_optimizer_updates": 0, "application_checkpoint_promoted": False,
        "assets_sha256": assets,
    })
    verify_correction(PARENT, DEST)
    with tarfile.open(ARCHIVE, "x:gz", compresslevel=5) as archive:
        for path in sorted(DEST.rglob("*")):
            if path.is_file():
                info = tarfile.TarInfo("cctv_dgp_vm_bundle/" + path.relative_to(DEST).as_posix())
                info.size, info.mtime, info.mode = path.stat().st_size, 0, 0o644
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
    with Path(str(ARCHIVE) + ".sha256").open("x", encoding="ascii", newline="\n") as stream:
        stream.write(sha(ARCHIVE) + "  " + ARCHIVE.name + "\n")
    print({"complete": True, "archive": str(ARCHIVE), "archive_bytes": ARCHIVE.stat().st_size,
           "archive_sha256": sha(ARCHIVE), "manifest_sha256": sha(DEST / "normfix_v2.json"),
           "parent_protocol_unchanged": True, "new_asset_files": len(assets),
           "local_optimizer_updates": 0, "corrected_vm_run_pending": True})


if __name__ == "__main__":
    main()
