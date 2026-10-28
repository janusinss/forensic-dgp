"""Install the separate R1 completion/proposal assets, never restoration assets.

Normal installation uses the fixed GitHub release below. The release's existence
is checked by the actual download; this module does not assert it was published.
Explicit local files enable offline installation; existing mismatches and all
failed partial downloads/copies remain preserved for inspection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import time
import urllib.parse
import urllib.request
import uuid

from . import completion_assets as assets

RELEASE_URL = "https://github.com/janusinss/forensic-dgp/releases/tag/completion-r1-models-v1"
RELEASE_BASE = "https://github.com/janusinss/forensic-dgp/releases/download/completion-r1-models-v1/"
CHUNK_SIZE = 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 60
DOWNLOAD_TOTAL_SECONDS = 1800


class CompletionInstallError(RuntimeError):
    """Completion setup stopped with existing files and failure evidence retained."""


def _https(url):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise CompletionInstallError("Completion downloads and redirects must use HTTPS without embedded credentials.")
    return url


class HTTPSRedirectHandler(urllib.request.HTTPRedirectHandler):
    max_redirections = 5

    def redirect_request(self, request, fp, code, message, headers, newurl):
        _https(newurl)
        return super().redirect_request(request, fp, code, message, headers, newurl)


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
            info.st_nlink, getattr(info, "st_uid", None))


def _admitted(path, asset):
    """Verify one ordinary, unaliased inode and its complete bytes."""
    try:
        path = assets.ordinary_file(path)
        before = path.lstat()
        if before.st_nlink != 1 or before.st_size != asset["size"]:
            return False
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            if _identity(os.fstat(stream.fileno())) != _identity(before):
                return False
            for block in iter(lambda: stream.read(CHUNK_SIZE), b""):
                digest.update(block)
            if _identity(os.fstat(stream.fileno())) != _identity(before):
                return False
        return (_identity(path.lstat()) == _identity(before)
                and digest.hexdigest() == asset["sha256"])
    except (OSError, assets.CompletionAssetError):
        return False


def _directory(folder):
    folder = Path(folder).absolute()
    for parent in (folder, *folder.parents):
        try:
            info = parent.lstat()
        except FileNotFoundError:
            continue
        if (stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & 1024):
            raise CompletionInstallError("Completion installation needs ordinary directories without links or reparse points.")
    return folder


def _exists(path):
    try:
        path.lstat()
        return True
    except FileNotFoundError:
        return False


def _publish(partial, destination, asset):
    if not _admitted(partial, asset):
        raise CompletionInstallError(f"Completion partial failed verification and was retained: {partial.name}")
    _directory(destination.parent)
    # rename() on Windows and link()+unlink() on POSIX refuse an existing
    # destination. No source model is hard-linked, and the final inode has nlink1.
    if os.name == "nt":
        partial.rename(destination)
    else:
        os.link(partial, destination)
        partial.unlink()
    if not _admitted(destination, asset):
        raise CompletionInstallError(f"Published completion file changed or has an alias; it was retained: {destination.name}")


def _copy_local(source, partial, asset):
    if not _admitted(source, asset):
        raise CompletionInstallError(f"Supplied local completion file failed exact size/SHA-256 or ordinary-file checks: {source}")
    with Path(source).open("rb") as reader, partial.open("xb") as writer:
        remaining = asset["size"]
        while True:
            block = reader.read(min(CHUNK_SIZE, remaining + 1))
            if not block:
                break
            if len(block) > remaining:
                raise CompletionInstallError(f"Local source grew during copy; partial retained: {partial.name}")
            writer.write(block)
            remaining -= len(block)
        writer.flush()
        os.fsync(writer.fileno())
    if not _admitted(source, asset):
        raise CompletionInstallError(f"Local completion source changed during copy; partial retained: {partial.name}")


def _download(url, partial, asset, *, opener=None, clock=time.monotonic):
    _https(url)
    opener = opener or urllib.request.build_opener(HTTPSRedirectHandler())
    request = urllib.request.Request(url, headers={"User-Agent": "ForensicDGP-CompletionInstaller/1", "Accept-Encoding": "identity"})
    started = clock()
    # Open exclusively first so an unsuccessful network attempt has a named,
    # retained partial. There is no overwrite, retry, append or resume path.
    with partial.open("xb") as writer:
        with opener.open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            _https(response.geturl())
            status = getattr(response, "status", 200)
            if status != 200:
                raise CompletionInstallError(f"Completion download returned HTTP {status}.")
            encoding = response.headers.get("Content-Encoding", "identity").strip().lower()
            if encoding not in ("", "identity"):
                raise CompletionInstallError("Unexpected completion download content encoding.")
            length = response.headers.get("Content-Length")
            if length is not None:
                try:
                    if int(length) != asset["size"]:
                        raise CompletionInstallError("Completion download Content-Length differs from the admitted size.")
                except ValueError as exc:
                    raise CompletionInstallError("Invalid completion download Content-Length.") from exc
            remaining, digest = asset["size"], hashlib.sha256()
            while True:
                if clock() - started > DOWNLOAD_TOTAL_SECONDS:
                    raise CompletionInstallError("Completion download exceeded 30 minutes; partial retained. No automatic retry.")
                block = response.read(min(CHUNK_SIZE, remaining + 1))
                if clock() - started > DOWNLOAD_TOTAL_SECONDS:
                    raise CompletionInstallError("Completion download exceeded 30 minutes; partial retained. No automatic retry.")
                if not block:
                    break
                if len(block) > remaining:
                    raise CompletionInstallError("Completion download exceeded its exact size; partial retained.")
                writer.write(block)
                digest.update(block)
                remaining -= len(block)
            writer.flush()
            os.fsync(writer.fileno())
            if remaining or digest.hexdigest() != asset["sha256"]:
                raise CompletionInstallError("Completion download failed its complete size/SHA-256 check; partial retained.")


def install_assets(*, local_r1=None, local_detector=None, verify_only=False,
                   model_dir=assets.MODEL_DIR, opener=None, clock=time.monotonic):
    if verify_only and (local_r1 is not None or local_detector is not None):
        raise CompletionInstallError("Use --verify-only separately from local staging.")
    manifest = assets.load_manifest()
    folder = _directory(model_dir)
    supplied = {"completion_r1": local_r1, "covering_detector": local_detector}
    offline = any(value is not None for value in supplied.values())
    planned = []
    # Refuse every existing mismatch and every invalid supplied source before
    # downloading or writing either model.
    for key, asset in manifest["assets"].items():
        destination = folder / asset["filename"]
        if supplied[key] is not None and not _admitted(Path(supplied[key]), asset):
            raise CompletionInstallError(f"Supplied {key} file failed exact size/SHA-256 checks.")
        if _exists(destination):
            if not _admitted(destination, asset):
                raise CompletionInstallError(f"Existing completion model was preserved because it is mismatched or linked: {destination}")
            source = "existing_verified"
        elif verify_only:
            raise CompletionInstallError(f"Completion asset is missing: {destination.name}")
        elif supplied[key] is not None:
            source = "verified_local_copy"
        elif offline:
            raise CompletionInstallError(f"Offline setup also needs --local-{'r1' if key == 'completion_r1' else 'detector'} for missing {destination.name}.")
        else:
            source = "verified_https_download"
        planned.append((key, asset, destination, source))
    acquired = []
    if any(item[3] != "existing_verified" for item in planned):
        folder.mkdir(parents=True, exist_ok=True)
        _directory(folder)
    for key, asset, destination, source in planned:
        item = {"key": key, "filename": asset["filename"], "size": asset["size"],
                "sha256": asset["sha256"], "acquisition": source}
        if source == "existing_verified":
            acquired.append(item)
            continue
        if shutil.disk_usage(folder).free < asset["size"] + 4 * 1024**2:
            raise CompletionInstallError(f"Not enough free space to stage {destination.name} while preserving existing files.")
        partial = folder / f".{destination.name}.{uuid.uuid4().hex}.partial"
        try:
            if source == "verified_local_copy":
                print(f"[completion] Copying and verifying {destination.name}...", flush=True)
                _copy_local(Path(supplied[key]), partial, asset)
                item["local_source_name"] = Path(supplied[key]).name
            else:
                url = RELEASE_BASE + urllib.parse.quote(asset["filename"], safe="")
                print(f"[completion] Downloading and verifying {destination.name} ({asset['size']:,} bytes)...", flush=True)
                _download(url, partial, asset, opener=opener, clock=clock)
                item["download_url"] = url
            _publish(partial, destination, asset)
        except Exception as exc:
            raise CompletionInstallError(f"{exc} Existing models and any partial are preserved: {partial}") from exc
        acquired.append(item)
    for _, asset, destination, _ in planned:
        if not _admitted(destination, asset):
            raise CompletionInstallError(f"Completion asset changed before final verification: {destination.name}")
    result = {"schema_version": 1, "component": "completion-r1-distribution-installation",
              "complete": True, "verified_only": bool(verify_only), "assets": acquired,
              "restoration_files_modified": False, "inference_calls": 0, "training_calls": 0,
              "model_quality_qualified": False}
    if any(item["acquisition"] != "existing_verified" for item in acquired):
        result["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
        result["distribution_release"] = RELEASE_URL
        result["distribution_record"] = "Downloads verified against immutable local fingerprints; this is not a model-quality or prior-publication claim."
        result["completion_manifest_sha256"] = assets.sha256_file(assets.MANIFEST_PATH)
        receipt = folder / f"installation-{uuid.uuid4().hex}.json"
        with receipt.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        result["installation_receipt"] = receipt.name
    return result


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--local-r1", type=Path)
    parser.add_argument("--local-detector", type=Path)
    parser.add_argument("--configured-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(arguments)
    try:
        # install.bat invokes this through its bootstrap Python after the existing
        # environment installation succeeds. Use only that checkout's validated
        # environment for the ordinary download path. Verification/offline file
        # copying are standard-library-only and can be performed independently.
        if not args.verify_only and args.local_r1 is None and args.local_detector is None:
            from .environment import configured_python, check_python, run_foreground
            configured = configured_python()
            check_python(configured)
            if not args.configured_child:
                return run_foreground([str(configured), "-B", "-m", "implementation.completion_install", *arguments, "--configured-child"])
            if not Path(sys.executable).samefile(configured):
                raise CompletionInstallError("Completion setup must use the validated checkout environment.")
        result = install_assets(local_r1=args.local_r1, local_detector=args.local_detector, verify_only=args.verify_only)
        print(json.dumps(result, allow_nan=False))
        return 0
    except (CompletionInstallError, assets.CompletionAssetError, OSError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("[ERROR] Completion installation interrupted; existing models and partial evidence are retained.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
