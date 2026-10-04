"""Independent local materialization check; zero model forwards/backward/updates."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / "outputs/cctv_dgp_targets_v6_transport_v1/payload"
CACHE = ROOT / "outputs/cctv_dgp_vm_bundle_v1"
CHECK = ROOT / "scratch/cctv_dgp_targets_v6_transport_check_v1"
sys.path.insert(0, str(ROOT))
from cctv_dgp_targets_v6 import read, sha, verify, write


def main():
    if CHECK.exists():
        raise ValueError("Preserve previous materialization check")
    recipe = read(PAYLOAD / "targets_protocol_v6.json")
    manifest_pin = sha(PAYLOAD / "transport_manifest_v1.json")
    # Bootstrap the one cached module imported before materialization begins.
    # It remains the exact pinned historical asset, not a changed implementation.
    bootstrap = "scripts/cctv_camera_stress.py"
    if sha(CACHE / bootstrap) != recipe["assets_sha256"][bootstrap]:
        raise ValueError("Bootstrap camera implementation changed")
    shutil.copytree(PAYLOAD, CHECK)
    (CHECK / bootstrap).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CACHE / bootstrap, CHECK / bootstrap)
    subprocess.run([sys.executable, "-X", "utf8", "-u", str(CHECK / "scripts/materialize_cctv_dgp_targets_v6_transport_v1.py"),
        "--root", str(CHECK), "--cache-root", str(CACHE), "--manifest-sha256", manifest_pin], check=True, timeout=200)
    rebuilt = verify(CHECK)
    if rebuilt != recipe:
        raise ValueError("Lossless transport changed frozen recipe")
    receipt = {"complete": True, "protocol_sha256": sha(CHECK / "targets_protocol_v6.json"),
        "manifest_sha256": manifest_pin, "materialization_receipt_sha256": sha(CHECK / "transport_materialization_v1.json"),
        "all_original_assets_reproduced": len(recipe["assets_sha256"]), "recipe_byte_identical": True,
        "local_model_forwards": 0, "local_backward_calls": 0, "local_optimizer_updates": 0,
        "native_reserved_used": False, "bootstrap_asset_sha256": sha(CHECK / bootstrap)}
    write(ROOT / "outputs/cctv_dgp_targets_v6_transport_v1/local_materialization_audit.json", receipt)
    print(receipt)


if __name__ == "__main__":
    main()
