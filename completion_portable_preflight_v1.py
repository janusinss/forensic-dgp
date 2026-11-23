"""Read-only inventory for the current completion prototype, not an installer.

No app, Torch, model or image-processing modules are imported. The output report is
written only when --output is supplied. Exit 0 means current pinned files and
runtime metadata match; it does not mean portable installation or model quality.
--require-portable deliberately exits 2 until separately reviewed packaging exists.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import stat
import struct
import sys

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = Path(r"C:\xampp\htdocs\YEAR 4\Testing\checkpoints\codeformer_inpainting.pth")
WEIGHT_PIN = {"bytes": 370786611, "sha256": "b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797"}
ADMISSION_PIN = {"bytes": 10356, "sha256": "188e54fd998ad97e9bef84607b8e92643f555565bce5a5c64c4a7a759f2c61ab"}
DEPENDENCIES = {"torch": "2.13.0", "numpy": "2.5.2", "Pillow": "12.3.0", "opencv-python": "5.0.0.93"}
SOURCE_PINS = {'completion_codeformer_web_v1.py': {'bytes': 24612, 'sha256': 'be708101d0ded43a1ea0a1a2072c2b23fc6310d43a6f15e62ea9dded766daa6e'}, 'scripts/completion_codeformer_worker_v1.py': {'bytes': 17460, 'sha256': '9230025591e6fd4ef5cbd254e826009f938dea9408c70c859bc54d0dadb1b861'}, 'templates/completion_codeformer_v1.html': {'bytes': 22345, 'sha256': '03f0f752d7ee3d1a71615d84fd38bca2735d33a3498f7bb9d583846ec9b96eb0'}, 'pretrained_completion.py': {'bytes': 3559, 'sha256': '29f2d03d692d762d1d9b39c9747e7becf30a14399651e8c419de2a5b39cae9ce'}, 'completion.py': {'bytes': 5022, 'sha256': '7debe38b0936add26c663f053fa7ef19649a93b9a717194652911b6d0bdcda72'}, 'face_workflow.py': {'bytes': 13648, 'sha256': 'ed043f9819e4655c297f092037af156cbfa9e32506f6b15cca0c5d5dfe4cacd1'}, 'third_party/__init__.py': {'bytes': 51, 'sha256': '22387ba5f42e539104f611c22b709e7e8fee3633d0b48dfa47fd7be8c8b79c07'}, 'third_party/codeformer/__init__.py': {'bytes': 74, 'sha256': 'e02b6c974f00a7fe9d54a8a22dfdf17e59936aca65608521dfb523018dc61c66'}, 'third_party/codeformer/codeformer_arch.py': {'bytes': 11377, 'sha256': 'ee9758522173b72fc1f292df825b9a32179aebfd96142ececf9aa68f6e12e9eb'}, 'third_party/codeformer/vqgan_arch.py': {'bytes': 15855, 'sha256': 'b47033f66b4a8434e0f04d4df8a69a7ac4b6c5bdf52f5b74912371dff0a20eaf'}, 'third_party/codeformer/registry.py': {'bytes': 2267, 'sha256': 'bbc3507148101bb202e8f52f5cca75d2cc21d780def757c055d82cb8e6bd594e'}, 'third_party/codeformer/LICENSE': {'bytes': 1751, 'sha256': '4122e9944900d9e23b8622b829f4343384fbb6e0667b39e4cf474b5d88a185dd'}, 'third_party/codeformer/NOTICE.md': {'bytes': 988, 'sha256': '6e06e1c0fa1dd295b54c5169d3e8b57859b3c58e2f201788992185f044c9349a'}, 'completion_process_guard_v1.py': {'bytes': 19546, 'sha256': '1277b0fde51c35337a0391d4089b193aa837f21c7e00ecc9ee4ed7ecd5f687c0'}}

def fingerprint(path):
    path = Path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink() or getattr(before, "st_file_attributes", 0) & 0x400:
        raise ValueError("Expected an ordinary file without a reparse point")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.lstat()
    identity = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns)
    if identity(before) != identity(after):
        raise ValueError("File changed during fingerprint")
    return {"bytes": after.st_size, "sha256": digest}

def check_file(path, expected):
    try:
        observed = fingerprint(path)
        return {"path": str(Path(path).resolve()), **observed, "pass": observed == expected,
                "expected": expected}
    except (OSError, ValueError) as exc:
        return {"path": str(path), "pass": False, "error": str(exc), "expected": expected}

def source_file(root, name):
    parts = Path(name).parts
    if Path(name).is_absolute() or ".." in parts:
        raise ValueError("Source allowlist must be relative")
    path = root.joinpath(*parts)
    for ancestor in [path.parent, *path.parents]:
        if ancestor == root.parent:
            break
        if ancestor.is_symlink() or getattr(ancestor.lstat(), "st_file_attributes", 0) & 0x400:
            raise ValueError("Source closure contains a redirected directory")
        if ancestor == root:
            break
    if not path.resolve().is_relative_to(root):
        raise ValueError("Source escapes selected checkout")
    return path

def path_assignments(path, names):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    return [{"name": target.id, "line": node.lineno, "expression": ast.unparse(node.value)}
            for node in tree.body if isinstance(node, ast.Assign)
            for target in node.targets if isinstance(target, ast.Name) and target.id in names]

def inventory(root, weights):
    root = Path(root).resolve(strict=True)
    sources = []
    for name, expected in SOURCE_PINS.items():
        try:
            row = check_file(source_file(root, name), expected)
        except (OSError, ValueError) as exc:
            row = {"path": str(root/name), "pass": False, "error": str(exc), "expected": expected}
        sources.append({"relative_path": name, **row})
    admission = check_file(root/"outputs/completion_codeformer_manual_v1/source_admission.json", ADMISSION_PIN)
    checkpoint = check_file(weights, WEIGHT_PIN)
    packages = []
    for name, expected in DEPENDENCIES.items():
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        packages.append({"name": name, "expected": expected, "observed": version, "pass": version == expected})
    runtime = {"python": sys.version.split()[0], "bits": struct.calcsize("P")*8,
               "platform": sys.platform, "packages": packages,
               "pass": sys.platform == "win32" and sys.version_info[:2] == (3, 13) and struct.calcsize("P") == 8 and all(p["pass"] for p in packages),
               "wheel_flavor": "not established by distribution metadata; see retained CPU execution receipt"}
    blockers = [
        {"code": "absolute_asset_root", "source": "completion_codeformer_web_v1.py", "detail": "ASSET_ROOT/CHECKPOINT points into original Testing checkout"},
        {"code": "absolute_worker_default", "source": "scripts/completion_codeformer_worker_v1.py", "detail": "DEFAULT_CHECKPOINT points into original Testing checkout"},
        {"code": "checkout_bound_admission", "source": "outputs/completion_codeformer_manual_v1/source_admission.json", "detail": "Existing admission pins absolute source/import/asset locations; a fresh checkout needs a new reviewed admission"},
        {"code": "no_completion_installer", "detail": "No accepted fresh-clone installer or inpainting asset entry exists in IMPLEMENTATION"},
        {"code": "unmeasured_resource_envelope", "detail": "Peak RAM, installed environment size, minimum hardware and fresh-install behavior remain unmeasured"},
        {"code": "quality_acceptance_pending", "detail": "One functional CPU request is not all-family usefulness or user acceptance"},
    ]
    bindings = []
    if all(row["pass"] for row in sources):
        bindings += path_assignments(root/"completion_codeformer_web_v1.py", {"ROOT", "ASSET_ROOT", "CHECKPOINT", "OUTPUT"})
        bindings += path_assignments(root/"scripts/completion_codeformer_worker_v1.py", {"ROOT", "DEFAULT_CHECKPOINT"})
    if "torch" in sys.modules:
        raise RuntimeError("Inventory unexpectedly imported Torch")
    integrity = all(row["pass"] for row in sources) and admission["pass"] and checkpoint["pass"]
    return {"format": "completion-portable-design-preflight-v1", "created_UTC": datetime.now(timezone.utc).isoformat(),
            "complete": True, "mode": "read_only_inventory_no_install_no_neural_calls", "root": str(root),
            "source_files": sources, "source_bytes": sum(p["bytes"] for p in SOURCE_PINS.values()),
            "source_file_count": len(sources), "checkpoint": checkpoint, "source_admission": admission,
            "runtime": runtime, "integrity_pass": integrity, "current_reference_preflight_pass": integrity and runtime["pass"],
            "portable_ready": False, "blockers": blockers, "path_bindings": bindings,
            "installed_environment_bytes": None, "peak_RAM_bytes": None, "minimum_hardware": None,
            "neural_calls": 0, "training_calls": 0, "torch_imported": False, "model_qualified": False}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Read-only checkout inventory root")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Read-only checkpoint path; SHA and bytes are fixed")
    parser.add_argument("--output", type=Path, help="Write one report only; its parent must already exist, existing file is not overwritten")
    parser.add_argument("--require-portable", action="store_true", help="Exit 2 while portability blockers remain")
    args = parser.parse_args()
    try:
        report = inventory(args.root, args.weights)
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(encoded)
        print(encoded, end="")
        return 0 if report["current_reference_preflight_pass"] and not args.require_portable else 2
    except (OSError, ValueError, SyntaxError) as exc:
        print(json.dumps({"complete": False, "error": str(exc), "neural_calls": 0, "training_calls": 0, "torch_imported": "torch" in sys.modules}))
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
