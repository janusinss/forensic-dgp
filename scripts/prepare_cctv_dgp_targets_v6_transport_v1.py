"""Prepare/verify a smaller lossless transport; the V6 protocol stays byte identical."""
import ast
from collections import Counter
from pathlib import Path
import shutil
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cctv_dgp_targets_v6 import read, sha, verify, write

FIXED = ROOT / "outputs/cctv_dgp_targets_vm_v6"
CACHE = ROOT / "outputs/cctv_dgp_vm_bundle_v1"
DEST = ROOT / "outputs/cctv_dgp_targets_v6_transport_v1"
ARCHIVE = ROOT / "outputs/cctv-dgp-targets-v6-transport-v1.tar.gz"
SCRIPT = "scripts/materialize_cctv_dgp_targets_v6_transport_v1.py"


def main():
    start = time.monotonic()
    if DEST.exists() or ARCHIVE.exists() or Path(str(ARCHIVE) + ".sha256").exists():
        raise ValueError("Preserve existing transport preparation")
    p = verify(FIXED)
    cases = {c["input"] for c in sum(p["training_epochs"].values(), []) + p["validation_cases"]}
    controls = {r["reduced_target"] for r in p["references"] if r["role"] == "train"}
    methods = {}
    for name, pin in p["assets_sha256"].items():
        existing = CACHE / name
        if existing.is_file() and sha(existing) == pin:
            methods[name] = {"method": "cache", "source": name}
        elif name in controls:
            methods[name] = {"method": "reduce"}
        elif name in cases:
            methods[name] = {"method": "camera"}
        else:
            methods[name] = {"method": "ship"}
    ast.parse((ROOT / SCRIPT).read_text(encoding="utf-8"), feature_version=(3, 10))
    DEST.mkdir()
    write(DEST / "transport_manifest_v1.json", {"format": "cctv-dgp-v6-lossless-transport-v1",
        "protocol_sha256": sha(FIXED / "targets_protocol_v6.json"), "methods": methods,
        "materializer_sha256": sha(ROOT / SCRIPT), "recipe_changed": False})
    manifest_sha = sha(DEST / "transport_manifest_v1.json")
    files = {n: FIXED / n for n, item in methods.items() if item["method"] == "ship"}
    files.update({n: FIXED / n for n in ("targets_protocol_v6.json", "targets_protocol_v6.sha256")})
    files["transport_manifest_v1.json"] = DEST / "transport_manifest_v1.json"
    files[SCRIPT] = ROOT / SCRIPT
    for name, source in files.items():
        target = DEST / "payload" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    with tarfile.open(ARCHIVE, "x:gz", compresslevel=5) as stream:
        for name in sorted(files):
            source = files[name]
            info = tarfile.TarInfo("cctv_dgp_targets_vm_v6/" + name)
            info.size, info.mtime, info.mode = source.stat().st_size, 0, 0o644
            with source.open("rb") as blob:
                stream.addfile(info, blob)
    with Path(str(ARCHIVE) + ".sha256").open("x", encoding="ascii", newline="\n") as stream:
        stream.write(sha(ARCHIVE) + "  " + ARCHIVE.name + "\n")
    receipt = {"complete": True, "protocol_sha256": sha(FIXED / "targets_protocol_v6.json"),
        "manifest_sha256": manifest_sha, "archive_sha256": sha(ARCHIVE), "archive_bytes": ARCHIVE.stat().st_size,
        "methods": dict(Counter(x["method"] for x in methods.values())),
        "original_assets": len(methods), "seconds": time.monotonic() - start,
        "model_forwards": 0, "backward_calls": 0, "optimizer_updates": 0, "training_recipe_changed": False}
    write(DEST / "preparation.json", receipt)
    print(receipt)


if __name__ == "__main__":
    main()
