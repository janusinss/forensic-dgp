"""Separate, portable completion assets; no restoration asset or setup changes.

This installer stages explicitly supplied local files. Published download URLs
may be added only after the accepted asset release exists and is verified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "completion"
MANIFEST_PATH = Path(__file__).with_name("completion_assets.json")
R1_SHA256 = "7ae2486739de0f3efa99cf721a0fcb755f8f56cf69e295412f74190d2fda37e0"
R1_SIZE = 370764381
DETECTOR_SHA256 = "ccc22d524073368bc4f5d6b6d2136c90d0e2a117083e4c75fa98d42d7bfff6cf"
DETECTOR_SIZE = 2681549


class CompletionAssetError(RuntimeError):
    """An absent or changed completion asset; restoration stays available."""


def load_manifest(path=MANIFEST_PATH):
    try:
        body = json.loads(Path(path).read_text(encoding="utf-8"))
        if (body.get("schema_version") != 1 or body.get("component") != "reviewed-r1-face-completion"
                or body.get("default_model") != "completion_r1" or body.get("model_directory") != "models/completion"):
            raise ValueError("unsupported completion manifest")
        assets = body["assets"]
        if set(assets) != {"completion_r1", "covering_detector"}:
            raise ValueError("unexpected completion asset list")
        expected = {"completion_r1": ("codeformer_completion_r1.pth", R1_SIZE, R1_SHA256),
                    "covering_detector": ("covering_detector_segmenter_v1.pth", DETECTOR_SIZE, DETECTOR_SHA256)}
        for key, asset in assets.items():
            if (asset.get("filename"), asset.get("size"), asset.get("sha256")) != expected[key]:
                raise ValueError("completion asset differs from the admitted model")
            if not re.fullmatch(r"[0-9a-f]{64}", asset.get("logical_sha256", "")):
                raise ValueError("invalid logical state fingerprint")
            if asset.get("model_qualified") is not False or asset.get("url") is not None:
                raise ValueError("unreviewed qualification or download URL")
        return body
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise CompletionAssetError(f"Invalid completion asset manifest: {exc}") from exc


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordinary_file(path):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        info = item.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 1024:
            raise CompletionAssetError("Linked/reparse model paths are not accepted.")
    if not stat.S_ISREG(path.lstat().st_mode):
        raise CompletionAssetError("An ordinary model file is required.")
    return path


def verify_asset(path, asset):
    try:
        path = ordinary_file(path)
        before = path.stat()
        if before.st_size != asset["size"]:
            return False
        digest = sha256_file(path)
        after = path.stat()
        return ((before.st_size, before.st_mtime_ns, before.st_ino)
                == (after.st_size, after.st_mtime_ns, after.st_ino)
                and digest == asset["sha256"])
    except (OSError, CompletionAssetError):
        return False


def asset_path(key, *, model_dir=MODEL_DIR, verify=True):
    body = load_manifest()
    if key not in body["assets"]:
        raise CompletionAssetError("Unknown completion asset.")
    asset = body["assets"][key]
    path = Path(model_dir) / asset["filename"]
    if verify and not verify_asset(path, asset):
        raise CompletionAssetError(f"Completion asset is missing or changed: {path.name}. Run the separate completion asset setup.")
    return path


def verify_assets(*, keys=None, model_dir=MODEL_DIR):
    keys = tuple(keys or ("completion_r1", "covering_detector"))
    return {key: asset_path(key, model_dir=model_dir) for key in keys}


def stage_local(key, source, *, model_dir=MODEL_DIR):
    asset = load_manifest()["assets"][key]
    source = ordinary_file(source)
    if not verify_asset(source, asset):
        raise CompletionAssetError(f"Supplied {key} file failed the exact size/SHA-256 check.")
    folder = Path(model_dir)
    for parent in (folder, *folder.parents):
        if parent.exists() or parent.is_symlink():
            info = parent.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 1024:
                raise CompletionAssetError("Linked/reparse model directories are not accepted.")
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / asset["filename"]
    if destination.exists() or destination.is_symlink():
        if verify_asset(destination, asset):
            return destination
        raise CompletionAssetError(f"Existing completion file was preserved because its hash differs: {destination.name}")
    partial = folder / f".{destination.name}.{uuid.uuid4().hex}.partial"
    try:
        with source.open("rb") as reader, partial.open("xb") as writer:
            shutil.copyfileobj(reader, writer, 1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        if not verify_asset(partial, asset):
            raise CompletionAssetError("Staged completion file failed verification; partial retained.")
        if os.name == "nt":
            partial.rename(destination)
        else:
            os.link(partial, destination)
            partial.unlink()
        return destination
    except Exception:
        # Retain interrupted/invalid partial evidence; never overwrite a model.
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--local-r1", type=Path)
    parser.add_argument("--local-detector", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.verify_only and (args.local_r1 or args.local_detector):
            raise CompletionAssetError("Use verify-only separately from local staging.")
        if args.local_r1:
            stage_local("completion_r1", args.local_r1)
        if args.local_detector:
            stage_local("covering_detector", args.local_detector)
        verified = verify_assets()
        print(json.dumps({"complete": True, "verified_assets": list(verified), "inference_or_training_calls": 0}))
        return 0
    except (CompletionAssetError, OSError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
