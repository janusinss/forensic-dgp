"""Standard-library offline staging and verification for pinned Auto Mode ONNX models."""
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

from .managed_paths import data_path, managed_root

try:
    from .offline_payload import CopyProgress, load_payload
except ImportError:
    CopyProgress = None

    def load_payload():
        return None


try:
    from .setup_events import check_cancel, event
except ImportError:
    def check_cancel():
        return None

    def event(*_args, **_kwargs):
        return None


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "implementation" / "onnx_assets.json"
MARKER_NAME = ".onnx-bundle.json"
VALID_SUBDIRS = ("detection", "occlusion")
CHUNK_SIZE = 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_KEY_RE = re.compile(r"^[a-z0-9_]{1,80}$")
_WINDOWS_RESERVED_RE = re.compile(r"(?i)^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)")


class OnnxInstallError(RuntimeError):
    """ONNX model staging or verification failed with existing files preserved."""


def _identity(info) -> tuple:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_nlink,
        getattr(info, "st_uid", None),
    )


def _exists(path: Path) -> bool:
    try:
        path.lstat()
        return True
    except FileNotFoundError:
        return False


def _ordinary_file(path: Path) -> Path:
    path = Path(os.path.abspath(path))
    for parent in path.parents:
        info = parent.lstat()
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISDIR(info.st_mode)
            or getattr(info, "st_file_attributes", 0) & 1024
        ):
            raise OnnxInstallError(f"ONNX path ancestor is not an ordinary directory: {parent}")
    info = path.lstat()
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or getattr(info, "st_file_attributes", 0) & 1024
        or info.st_nlink != 1
    ):
        raise OnnxInstallError(f"ONNX asset must be an ordinary unlinked file: {path}")
    return path


def _ordinary_directory(folder: Path) -> Path:
    folder = Path(os.path.abspath(folder))
    for parent in (folder, *folder.parents):
        try:
            info = parent.lstat()
        except FileNotFoundError:
            continue
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISDIR(info.st_mode)
            or getattr(info, "st_file_attributes", 0) & 1024
        ):
            raise OnnxInstallError(f"ONNX installation requires ordinary directories: {parent}")
    return folder


def _valid_filename(name: str) -> bool:
    if not isinstance(name, str) or not name or len(name) > 128:
        return False
    if "/" in name or "\\" in name or name in (".", "..") or name.endswith((".", " ")):
        return False
    if re.search(r'[\x00-\x1f<>"|?*:]', name) or _WINDOWS_RESERVED_RE.match(name):
        return False
    return Path(name).name == name


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise OnnxInstallError(f"Duplicate key in ONNX manifest: {key}")
        result[key] = value
    return result


def load_manifest(manifest_path: Path = MANIFEST_PATH) -> dict:
    try:
        checked = _ordinary_file(manifest_path)
    except (OSError, OnnxInstallError) as exc:
        raise OnnxInstallError(f"ONNX manifest is missing or invalid: {manifest_path}") from exc
    if checked.stat().st_size > 1024 * 1024:
        raise OnnxInstallError("ONNX manifest exceeds its size limit.")
    try:
        data = json.loads(checked.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise OnnxInstallError(f"ONNX manifest could not be parsed: {exc}") from exc
    if type(data) is not dict:
        raise OnnxInstallError("ONNX manifest must be a JSON object.")
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise OnnxInstallError("Unsupported ONNX manifest schema_version.")
    if data.get("component") != "auto-mode-onnx-assets":
        raise OnnxInstallError("Unsupported ONNX manifest component.")
    assets = data.get("assets")
    if type(assets) is not dict or not assets or len(assets) > 32:
        raise OnnxInstallError("ONNX manifest assets table is missing or invalid.")
    seen_targets = set()
    for key, asset in assets.items():
        if not isinstance(key, str) or not _KEY_RE.fullmatch(key):
            raise OnnxInstallError(f"Invalid ONNX asset key: {key!r}")
        if type(asset) is not dict or set(asset) != {"filename", "subdir", "size", "sha256"}:
            raise OnnxInstallError(f"Invalid ONNX asset fields for {key}.")
        if not _valid_filename(asset["filename"]):
            raise OnnxInstallError(f"Invalid ONNX asset filename for {key}.")
        if asset["subdir"] not in VALID_SUBDIRS:
            raise OnnxInstallError(f"Invalid ONNX asset subdir for {key}: {asset['subdir']!r}")
        if type(asset["size"]) is not int or asset["size"] <= 0 or asset["size"] > 2 * 1024**3:
            raise OnnxInstallError(f"Invalid ONNX asset size for {key}.")
        if not isinstance(asset["sha256"], str) or not _SHA256_RE.fullmatch(asset["sha256"]):
            raise OnnxInstallError(f"Invalid ONNX asset sha256 for {key}.")
        target_key = (asset["subdir"], asset["filename"].casefold())
        if target_key in seen_targets:
            raise OnnxInstallError(f"Duplicate ONNX asset target: {asset['subdir']}/{asset['filename']}")
        seen_targets.add(target_key)
    return data


def _admitted(path: Path, asset: dict) -> bool:
    try:
        checked = _ordinary_file(path)
        before = checked.lstat()
        if before.st_size != asset["size"]:
            return False
        digest = hashlib.sha256()
        with checked.open("rb") as stream:
            if _identity(os.fstat(stream.fileno())) != _identity(before):
                return False
            for block in iter(lambda: stream.read(CHUNK_SIZE), b""):
                check_cancel()
                digest.update(block)
            if _identity(os.fstat(stream.fileno())) != _identity(before):
                return False
        return _identity(checked.lstat()) == _identity(before) and digest.hexdigest() == asset["sha256"]
    except (OSError, OnnxInstallError):
        return False


def _copy_local(source: Path, partial: Path, asset: dict) -> None:
    if not _admitted(source, asset):
        raise OnnxInstallError(f"Supplied ONNX source failed exact size/SHA-256 checks: {source}")
    progress = CopyProgress(asset["filename"], asset["size"]) if CopyProgress is not None else None
    with Path(source).open("rb") as reader, partial.open("xb") as writer:
        remaining = asset["size"]
        while True:
            check_cancel()
            block = reader.read(min(CHUNK_SIZE, remaining + 1))
            if not block:
                break
            if len(block) > remaining:
                raise OnnxInstallError(f"ONNX source grew during copy; partial retained: {partial.name}")
            writer.write(block)
            remaining -= len(block)
            if progress is not None:
                progress.update(asset["size"] - remaining)
        writer.flush()
        os.fsync(writer.fileno())
    if not _admitted(source, asset):
        raise OnnxInstallError(f"ONNX source changed during copy; partial retained: {partial.name}")


def _publish(partial: Path, destination: Path, asset: dict) -> None:
    if not _admitted(partial, asset):
        raise OnnxInstallError(f"ONNX partial failed verification and was retained: {partial.name}")
    _ordinary_directory(destination.parent)
    if os.name == "nt":
        partial.rename(destination)
    else:
        os.link(partial, destination)
        partial.unlink()
    if not _admitted(destination, asset):
        raise OnnxInstallError(f"Published ONNX asset changed or has an alias: {destination.name}")


def _write_bundle_marker(marker_path: Path, asset_keys: list[str]) -> None:
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    _ordinary_directory(marker_path.parent)
    tmp = marker_path.with_name(f".{marker_path.name}.{uuid.uuid4().hex}.tmp")
    payload = {
        "schema_version": 1,
        "component": "auto-mode-onnx-assets",
        "assets": sorted(asset_keys),
    }
    try:
        with tmp.open("x", encoding="utf-8", newline="\n") as writer:
            json.dump(payload, writer, indent=2)
            writer.write("\n")
            writer.flush()
            os.fsync(writer.fileno())
        tmp.replace(marker_path)
    finally:
        tmp.unlink(missing_ok=True)


def install_onnx_assets(
    *,
    verify_only: bool = False,
    manifest_path: Path = MANIFEST_PATH,
    models_root: Path | None = None,
) -> dict:
    check_cancel()
    manifest = load_manifest(manifest_path)
    payload = load_payload()
    has_onnx_payload = payload is not None and any(
        group == "onnx" for group, _ in getattr(payload, "models", {})
    )
    if models_root is not None:
        marker_dir = _ordinary_directory(Path(models_root) / "detection")
    else:
        marker_dir = _ordinary_directory(data_path(ROOT, "models", "detection"))
    marker_path = marker_dir / MARKER_NAME

    targets = []
    for key, asset in sorted(manifest["assets"].items()):
        if models_root is not None:
            folder = _ordinary_directory(Path(models_root) / asset["subdir"])
        else:
            folder = _ordinary_directory(data_path(ROOT, "models", asset["subdir"]))
        targets.append((key, asset, folder, folder / asset["filename"]))

    any_asset_present = any(_exists(dest) for _, _, _, dest in targets)
    marker_present = _exists(marker_path)
    strict_presence = bool(has_onnx_payload or models_root is not None or marker_present or any_asset_present)

    planned = []
    for key, asset, folder, destination in targets:
        source_path = None
        if has_onnx_payload and not verify_only:
            source_path = payload.model_source("onnx", key, asset)
            if not _admitted(source_path, asset):
                raise OnnxInstallError(f"Offline payload ONNX model failed verification: {asset['filename']}")

        if _exists(destination):
            if _admitted(destination, asset):
                action = "existing_verified"
            else:
                info = destination.lstat()
                is_ordinary_reg = (
                    not stat.S_ISLNK(info.st_mode)
                    and stat.S_ISREG(info.st_mode)
                    and not (getattr(info, "st_file_attributes", 0) & 1024)
                    and info.st_nlink == 1
                )
                if verify_only or not has_onnx_payload or not is_ordinary_reg:
                    raise OnnxInstallError(
                        f"Existing ONNX model is mismatched, corrupted, or linked: {destination}"
                    )
                action = "verified_local_repair"
        elif verify_only:
            if strict_presence:
                raise OnnxInstallError(f"ONNX asset is missing: {asset['subdir']}/{asset['filename']}")
            action = "skipped_optional"
        elif has_onnx_payload:
            action = "verified_local_copy"
        else:
            action = "skipped_optional"
        planned.append((key, asset, folder, destination, source_path, action))

    acquired = []
    for key, asset, folder, destination, source_path, action in planned:
        item = {
            "key": key,
            "subdir": asset["subdir"],
            "filename": asset["filename"],
            "size": asset["size"],
            "sha256": asset["sha256"],
            "acquisition": action,
        }
        if action in ("existing_verified", "skipped_optional"):
            if action == "existing_verified" and not verify_only:
                event(
                    "download:" + asset["filename"],
                    "complete",
                    asset["filename"],
                    current=asset["size"],
                    total=asset["size"],
                )
            acquired.append(item)
            continue

        folder.mkdir(parents=True, exist_ok=True)
        _ordinary_directory(folder)
        if shutil.disk_usage(folder).free < asset["size"] + 4 * 1024**2:
            raise OnnxInstallError(f"Not enough free space to stage {destination.name}.")
        partial = folder / f".model-{uuid.uuid4().hex}.partial"
        backup = None
        try:
            print(f"[onnx] Copying and verifying {asset['subdir']}/{asset['filename']}...", flush=True)
            _copy_local(source_path, partial, asset)
            if action == "verified_local_repair":
                backup = folder / f".model-{uuid.uuid4().hex}.previous"
                destination.rename(backup)
            try:
                _publish(partial, destination, asset)
            except BaseException:
                if backup is not None and not _exists(destination):
                    backup.rename(destination)
                raise
        except Exception as exc:
            raise OnnxInstallError(f"{exc} Existing files and any partial are preserved: {partial}") from exc
        event(
            "download:" + asset["filename"],
            "complete",
            asset["filename"],
            current=asset["size"],
            total=asset["size"],
        )
        acquired.append(item)

    for _, asset, _, destination, _, action in planned:
        if action != "skipped_optional" and not _admitted(destination, asset):
            raise OnnxInstallError(f"ONNX asset changed before final verification: {destination.name}")

    if has_onnx_payload and not verify_only:
        _write_bundle_marker(marker_path, list(manifest["assets"].keys()))

    return {
        "schema_version": 1,
        "component": "auto-mode-onnx-assets",
        "complete": True,
        "verified_only": bool(verify_only),
        "skipped": all(item["acquisition"] == "skipped_optional" for item in acquired),
        "assets": acquired,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = install_onnx_assets(verify_only=args.verify_only)
        if result["skipped"]:
            print("[onnx] Optional ONNX models absent in unbundled checkout; skipping.", flush=True)
        else:
            print(json.dumps(result, allow_nan=False), flush=True)
        return 0
    except (OnnxInstallError, OSError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("[ERROR] ONNX installation interrupted; existing models and partial evidence are retained.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
