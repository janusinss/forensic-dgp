"""Use exact prepared V6 bytes across camera-library versions; training/gates unchanged.

The original launcher regenerates camera proxies on the VM and is unsuitable
when its camera stack differs. This driver binds the independent local source
derivation receipt, verifies every original asset, and invokes the unchanged V6
trainer and result auditor. It never monkeypatches losses, guards or verification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cctv_dgp_targets_v6 import require_vm, sha, verify, write

PROTOCOL_SHA = "0fe7d036bf8567bf0860790df553afd627ac50bd5096cfda29cc7d09d71d320c"
LOCAL_DATA_AUDIT_SHA = "c3179744ea799e3753ca1bfd978bcd560da16373c67340090852e628afac7d5c"


def supervised(command, log, deadline):
    with log.open("x", encoding="utf-8") as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"child_pid": child.pid, "command": command, "log": str(log)}), flush=True)
        try:
            code = child.wait(timeout=max(1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGINT)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise TimeoutError("Bounded portable V6 stage exceeded deadline")
    print(log.read_text(encoding="utf-8")[-4000:], flush=True)
    if code:
        raise subprocess.CalledProcessError(code, command)


def run(preflight_only):
    require_vm(ROOT)
    started = time.monotonic()
    p = verify(ROOT)  # Original verifier, including all exact hashes and reduced pixels.
    if sha(ROOT / "targets_protocol_v6.json") != PROTOCOL_SHA:
        raise ValueError("Original V6 protocol differs")
    local = ROOT / "local_source_derivation_audit_for_v6.json"
    if sha(local) != LOCAL_DATA_AUDIT_SHA:
        raise ValueError("Independent prepared-data receipt differs")
    proof = json.loads(local.read_text())
    if not proof["complete"] or proof["protocol_sha256"] != PROTOCOL_SHA or (
        proof["input_pngs_rebuilt"], proof["reduced_targets_rebuilt"], proof["hq_sources_independently_rebuilt"]) != (1302, 391, 444):
        raise ValueError("Prepared-data independent derivation scope differs")
    import torch, torchvision
    versions = (ROOT.parent / "cctv_dgp_vm_bundle/cuda_runtime_before.txt").read_text().splitlines()
    if versions != [torch.__version__, torchvision.__version__]:
        raise ValueError("Existing CUDA runtime changed")
    (ROOT / "portable_execution_v1").mkdir(exist_ok=True)
    execution = ROOT / "portable_execution_v1"
    launch = {"protocol_sha256": PROTOCOL_SHA, "local_derivation_audit_sha256": LOCAL_DATA_AUDIT_SHA,
        "driver_sha256": sha(Path(__file__)), "host": platform.node(), "pid": os.getpid(),
        "preflight_only": preflight_only, "started_unix_seconds": time.time(),
        "all_original_asset_sha256_verified": len(p["assets_sha256"]),
        "vm_camera_inputs_regenerated": False, "vm_reduced_pixels_rebuilt_by_original_verifier": True,
        "model_recipe_and_selection_changed": False, "torch": torch.__version__, "torchvision": torchvision.__version__}
    tag = "preflight" if preflight_only else "pilot"
    write(execution / (tag + "_launch.json"), launch)
    trainer = ROOT / "scripts/run_cctv_dgp_targets_vm_v6.py"
    auditor = ROOT / "scripts/audit_cctv_dgp_targets_v6.py"
    if preflight_only:
        command = [sys.executable, "-u", str(trainer), "--preflight-only", "--output", str(ROOT / "outputs/cctv_dgp_targets_v6_preflight")]
        supervised(command, ROOT / "standalone_preflight_v6.log", started + 180)
        result = json.loads((ROOT / "outputs/cctv_dgp_targets_v6_preflight/preflight.json").read_text())
        if not result["passed"] or not result["zero_optimizer_updates"]:
            raise ValueError("Standalone CUDA preflight not passed")
    else:
        out = ROOT / "outputs/cctv_dgp_targets_v6"
        archive = ROOT / "cctv-dgp-targets-v6-results.tar.gz"
        if out.exists() or archive.exists():
            raise ValueError("Preserve existing V6 pilot; no automatic repeat/resume")
        proof = json.loads((ROOT / "outputs/cctv_dgp_targets_v6_preflight/preflight.json").read_text())
        if not proof["passed"] or not proof["zero_optimizer_updates"]:
            raise ValueError("Prior standalone CUDA preflight required")
        deadline = started + 1800
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
        with (ROOT / "environment_portable_v6.txt").open("x", encoding="utf-8") as stream:
            stream.write(freeze.stdout)
        supervised([sys.executable, "-u", str(trainer)], ROOT / "pilot_targets_v6.log", deadline)
        supervised([sys.executable, "-u", str(auditor), "--root", str(ROOT), "--results", str(out),
            "--receipt", str(out / "independent_audit_vm.json")], ROOT / "audit_targets_v6.log", deadline)
        names = ["targets_protocol_v6.json", "targets_protocol_v6.sha256", "local_source_derivation_audit_for_v6.json",
                 "environment_portable_v6.txt", "pilot_targets_v6.log", "audit_targets_v6.log", "portable_execution_v1",
                 "scripts/run_cctv_dgp_targets_v6_portable_v1.py", "outputs/cctv_dgp_targets_v6"]
        with tarfile.open(archive, "x:gz", compresslevel=5) as stream:
            for name in names:
                if time.monotonic() > deadline:
                    raise TimeoutError("Portable audit/export deadline exceeded")
                stream.add(ROOT / name, arcname=name)
        with Path(str(archive) + ".sha256").open("x", encoding="ascii", newline="\n") as stream:
            stream.write(sha(archive) + "  " + archive.name + "\n")
        result = json.loads((out / "results.json").read_text())
    receipt = {"complete": True, "stage": tag, "protocol_sha256": PROTOCOL_SHA,
        "driver_sha256": sha(Path(__file__)), "seconds": time.monotonic() - started,
        "optimizer_updates": 0 if preflight_only else result["total_optimizer_updates"],
        "model_recipe_changed": False, "production_promoted": False}
    write(execution / (tag + "_completion.json"), receipt)
    print(receipt, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    try:
        run(args.preflight_only)
    except Exception as error:
        folder = ROOT / "portable_execution_v1"
        if folder.exists():
            name = "preflight_failure.json" if args.preflight_only else "pilot_failure.json"
            if not (folder / name).exists():
                write(folder / name, {"complete": False, "error_type": type(error).__name__,
                    "error": str(error), "resume_permitted": False})
        raise
