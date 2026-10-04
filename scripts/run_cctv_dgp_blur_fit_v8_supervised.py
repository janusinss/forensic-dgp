"""Finite VM-only V8 training, separate audit and export; never automatically resume."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time


def run(root, expected_protocol_sha):
    sys.path.insert(0, str(root))
    from cctv_dgp_targets_v6 import require_vm
    require_vm(root)
    spec = importlib.util.spec_from_file_location("blur_fit_pinned", root / "scripts/cctv_dgp_blur_fit_v8.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert module.sha(root / "blur_fit_protocol_v8.json") == expected_protocol_sha
    p = module.verify(root)
    assert not (root / "supervisor_execution_v8.json").exists(), "Preserve previous execution"
    parent = root.parent / "cctv_dgp_capacity_vm_v7"
    capacity_results = parent / "outputs/cctv_dgp_capacity_v7"
    import torch, torchvision
    assert (root.parent / "cctv_dgp_vm_bundle/cuda_runtime_before.txt").read_text().splitlines() == [torch.__version__, torchvision.__version__]
    started = time.monotonic(); deadline = started + 900
    module.write(root / "supervisor_execution_v8.json", {"protocol_sha256": expected_protocol_sha,
        "source_sha256": module.sha(Path(__file__)), "auditor_sha256": module.sha(root / "scripts/audit_cctv_dgp_blur_fit_v8.py"),
        "pid": os.getpid(), "torch": torch.__version__, "torchvision": torchvision.__version__,
        "started_unix_seconds": time.time(), "trainer_cap_seconds": 600, "supervisor_cap_seconds": 900,
        "expected_updates": 1000, "production_promoted": False})

    def call(args, name):
        with (root / name).open("x", encoding="utf-8") as log:
            child = subprocess.Popen(args, cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            print(json.dumps({"child_pid": child.pid, "command": args, "log": name}), flush=True)
            try:
                code = child.wait(timeout=max(1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGINT)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
                raise TimeoutError("V8 supervisor exceeded 900 seconds")
        if code:
            raise subprocess.CalledProcessError(code, args)
        print((root / name).read_text()[-2200:], flush=True)

    try:
        call([sys.executable, "-u", str(root / "scripts/cctv_dgp_blur_fit_v8.py"), "--root", str(root)], "blur_fit_training_v8.log")
        out = root / "outputs/cctv_dgp_blur_fit_v8"
        call([sys.executable, "-u", str(root / "scripts/audit_cctv_dgp_blur_fit_v8.py"), "--root", str(root),
            "--parent-root", str(parent), "--capacity-results", str(capacity_results),
            "--expected-protocol-sha", expected_protocol_sha, "--results", str(out),
            "--receipt", str(out / "independent_audit_vm.json")], "blur_fit_audit_v8.log")
        result = module.read(out / "results.json")
        module.write(root / "training_audit_completion_v8.json", {"complete": True,
            "optimizer_updates": result["total_optimizer_updates"], "training_only": True,
            "seconds": time.monotonic() - started, "production_promoted": False})
        archive = root / "cctv-dgp-blur-fit-v8-results.tar.gz"
        with tarfile.open(archive, "x:gz", compresslevel=5) as stream:
            for name in ["blur_fit_protocol_v8.json", "blur_fit_protocol_v8.sha256", "scripts", "outputs/cctv_dgp_blur_fit_v8",
                "blur_fit_training_v8.log", "blur_fit_audit_v8.log", "supervisor_execution_v8.json", "supervisor_launch_v8.json",
                "training_audit_completion_v8.json"]:
                if time.monotonic() > deadline:
                    raise TimeoutError("V8 export exceeded its deadline")
                stream.add(root / name, arcname=name)
        Path(str(archive) + ".sha256").write_text(module.sha(archive) + "  " + archive.name + "\n", encoding="ascii")
        assert time.monotonic() <= deadline, "V8 complete execution exceeded its deadline"
        module.write(root / "supervisor_completion_v8.json", {"complete": True, "protocol_sha256": expected_protocol_sha,
            "archive_sha256": module.sha(archive), "bytes": archive.stat().st_size,
            "seconds": time.monotonic() - started, "optimizer_updates": 1000, "production_promoted": False})
        print((root / "supervisor_completion_v8.json").read_text(), flush=True)
    except Exception as error:
        if not (root / "supervisor_failure_v8.json").exists():
            module.write(root / "supervisor_failure_v8.json", {"complete": False,
                "error_type": type(error).__name__, "error": str(error), "resume_permitted": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha", required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.expected_protocol_sha)
