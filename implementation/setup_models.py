"""Download the published frozen weights and verify them before use.

Uses only the Python standard library so setup failures remain easy to diagnose.
Existing files are verified and never replaced by a download.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "implementation"
MANIFEST_PATH = Path(__file__).with_name("models.json")
RELEASE_BASE = "https://github.com/janusinss/forensic-dgp/releases/download/implementation-models-v1/"
CHUNK_SIZE = 1024 * 1024


class ModelSetupError(RuntimeError):
    """A missing, invalid, or unavailable published model asset."""


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported schema_version")
        if manifest.get("default_model") not in {"u96", "u64"}:
            raise ValueError("default_model must be u96 or u64")
        assets = manifest["assets"]
        if set(assets) != {"codeformer_r2", "codeformer_u96_adapter", "terminal_u0064_heads"}:
            raise ValueError("unexpected asset list")
        filenames = set()
        for asset in assets.values():
            name = asset["filename"]
            if not isinstance(name, str) or Path(name).name != name or "/" in name or "\\" in name:
                raise ValueError("asset filename must be a plain filename")
            if name in filenames:
                raise ValueError("duplicate asset filename")
            filenames.add(name)
            if type(asset["size"]) is not int or asset["size"] <= 0:
                raise ValueError("asset size must be a positive integer")
            if not re.fullmatch(r"[0-9a-f]{64}", asset["sha256"]):
                raise ValueError("asset sha256 must contain 64 lowercase hexadecimal characters")
            if asset["url"] != RELEASE_BASE + urllib.parse.quote(name):
                raise ValueError("asset URL must use the published GitHub release")
        return manifest
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ModelSetupError(f"Invalid model manifest {path}: {exc}") from exc


def verify_asset(path: Path, asset: dict) -> bool:
    """Check complete size and SHA-256; missing/invalid files return False."""
    if not path.is_file() or path.is_symlink():
        return False
    try:
        if path.stat().st_size != asset["size"]:
            return False
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
                digest.update(chunk)
        return digest.hexdigest() == asset["sha256"]
    except OSError:
        return False


def verify_models(manifest: dict | None = None, model_dir: Path = MODEL_DIR) -> dict[str, Path]:
    manifest = manifest or load_manifest()
    verified = {}
    for key, asset in manifest["assets"].items():
        path = model_dir / asset["filename"]
        if not verify_asset(path, asset):
            raise ModelSetupError(f"Model is missing or failed its size/SHA-256 check: {path}. Run install.bat.")
        verified[key] = path
    return verified


class _HTTPSRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise ModelSetupError("Refusing a model download redirect outside HTTPS.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _publish_without_overwrite(partial: Path, destination: Path) -> None:
    # Windows rename fails if the destination exists. POSIX hard-link creation
    # gives the same atomic no-overwrite guarantee on the shared filesystem.
    if os.name == "nt":
        partial.rename(destination)
    else:
        os.link(partial, destination)
        partial.unlink()


def download_asset(asset: dict, model_dir: Path = MODEL_DIR) -> Path:
    destination = model_dir / asset["filename"]
    if destination.exists() or destination.is_symlink():
        if verify_asset(destination, asset):
            print(f"Verified {destination.name}", flush=True)
            return destination
        raise ModelSetupError(
            f"Existing file failed its size/SHA-256 check and was preserved: {destination}. "
            "Move it aside, then run install.bat again."
        )
    model_dir.mkdir(parents=True, exist_ok=True)
    partial = model_dir / f".{destination.name}.{uuid.uuid4().hex}.partial"
    digest = hashlib.sha256()
    downloaded = 0
    next_progress = 0
    request = urllib.request.Request(asset["url"], headers={"User-Agent": "Forensic-DGP-Implementation/1"})
    opener = urllib.request.build_opener(_HTTPSRedirectHandler())
    print(f"Downloading {destination.name} ({asset['size'] / 1048576:.1f} MiB)...", flush=True)
    try:
        with opener.open(request, timeout=30) as response, partial.open("xb") as stream:
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) != asset["size"]:
                raise ModelSetupError(f"Unexpected download size for {destination.name}.")
            while chunk := response.read(CHUNK_SIZE):
                downloaded += len(chunk)
                if downloaded > asset["size"]:
                    raise ModelSetupError(f"Download exceeded the expected size for {destination.name}.")
                digest.update(chunk)
                stream.write(chunk)
                if downloaded >= next_progress:
                    print(f"  {100 * downloaded / asset['size']:.0f}%", flush=True)
                    next_progress = downloaded + 32 * CHUNK_SIZE
            stream.flush()
            os.fsync(stream.fileno())
        if downloaded != asset["size"] or digest.hexdigest() != asset["sha256"]:
            raise ModelSetupError(f"Download failed its complete size/SHA-256 check for {destination.name}.")
        _publish_without_overwrite(partial, destination)
        print(f"Verified and saved {destination.name}", flush=True)
        return destination
    except urllib.error.HTTPError as exc:
        raise ModelSetupError(
            f"Model release download failed (HTTP {exc.code}) for {destination.name}. "
            "Check internet access and that implementation-models-v1 is published."
        ) from exc
    except (OSError, ValueError, urllib.error.URLError) as exc:
        raise ModelSetupError(f"Could not download {destination.name}: {exc}") from exc
    finally:
        # Only this invocation's unique incomplete file can be removed.
        partial.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="verify all local files without network access")
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest()
        if args.verify_only:
            verify_models(manifest)
            print("All 3 model files passed size and SHA-256 verification.")
        else:
            for asset in manifest["assets"].values():
                download_asset(asset)
            verify_models(manifest)
        return 0
    except ModelSetupError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
