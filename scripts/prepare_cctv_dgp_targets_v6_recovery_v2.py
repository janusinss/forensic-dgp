"""Exact-byte data recovery for failed cross-platform transport; no model work."""
from pathlib import Path
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cctv_dgp_targets_v6 import read, sha, verify, write

FIXED = ROOT / "outputs/cctv_dgp_targets_vm_v6"
FAILED = ROOT / "outputs/cctv_dgp_targets_v6_transport_failure_v1/transport_cache_inventory_failure_v1.json"
DEST = ROOT / "outputs/cctv_dgp_targets_v6_recovery_v2"
ARCHIVE = ROOT / "outputs/cctv-dgp-targets-v6-recovery-v2.tar.gz"


def main():
    started = time.monotonic()
    if DEST.exists() or ARCHIVE.exists() or Path(str(ARCHIVE) + ".sha256").exists():
        raise ValueError("Preserve existing recovery preparation")
    p = verify(FIXED)
    inventory = read(FAILED)
    if inventory["protocol_sha256"] != sha(FIXED / "targets_protocol_v6.json") or inventory["optimizer_updates"] or inventory["training_started"]:
        raise ValueError("Recovery must precede training and bind the same recipe")
    wrong = {r["name"]: r for r in inventory["mismatched_assets"]}
    missing = set(inventory["missing_assets"])
    good = set(inventory["correct_assets"])
    if good & (missing | set(wrong)) or len(good) + len(missing) + len(wrong) != 2709 or good | missing | set(wrong) != set(p["assets_sha256"]):
        raise ValueError("VM inventory does not partition all frozen assets")
    names = missing | set(wrong)
    if any(not n.startswith(("data/inputs/", "data/validation/", "data/reduced_targets/")) for n in names):
        raise ValueError("Unexpected non-image recovery asset")
    for name in names:
        if sha(FIXED / name) != p["assets_sha256"][name]:
            raise ValueError("Original prepared file changed")
    DEST.mkdir()
    plan = {"format": "cctv-dgp-v6-exact-byte-data-recovery-v2", "protocol_sha256": sha(FIXED / "targets_protocol_v6.json"),
        "failure_inventory_sha256": sha(FAILED), "recovery_assets_sha256": {n: p["assets_sha256"][n] for n in sorted(names)},
        "mismatched_before_recovery": inventory["mismatched_assets"], "existing_correct_assets": len(good),
        "missing_assets": len(missing), "replaced_mismatch_assets": len(wrong),
        "preserve_wrong_files_before_install": True, "regenerate_inputs_on_vm": False,
        "training_recipe_changed": False, "native_reserved_used": False}
    write(DEST / "recovery_plan_v2.json", plan)
    with tarfile.open(ARCHIVE, "x:gz", compresslevel=5) as stream:
        for name in sorted(names):
            path = FIXED / name
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = path.stat().st_size, 0, 0o644
            with path.open("rb") as blob:
                stream.addfile(info, blob)
    with Path(str(ARCHIVE) + ".sha256").open("x", encoding="ascii", newline="\n") as stream:
        stream.write(sha(ARCHIVE) + "  " + ARCHIVE.name + "\n")
    receipt = {"complete": True, "archive_sha256": sha(ARCHIVE), "archive_bytes": ARCHIVE.stat().st_size,
        "recovery_plan_sha256": sha(DEST / "recovery_plan_v2.json"), "protocol_sha256": plan["protocol_sha256"],
        "recover_exact_files": len(names), "original_recipe_changed": False,
        "model_forwards": 0, "backward_calls": 0, "optimizer_updates": 0,
        "seconds": time.monotonic() - started}
    write(DEST / "preparation.json", receipt)
    print(receipt)


if __name__ == "__main__":
    main()
