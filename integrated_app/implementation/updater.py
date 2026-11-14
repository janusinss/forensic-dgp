"""In-place application version check and self-updater."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from urllib.parse import urlsplit
import zipfile

from .setup_models import ROOT

APP_VERSION = "0.2.12"
DEFAULT_MANIFEST_URL = "https://forensic-dgp.pages.dev/release.json"
MAX_MANIFEST_BYTES = 64 * 1024
MAX_PACKAGE_BYTES = 2 * 1024**3

_SEMVER_RE = re.compile(r"^(?:0|[1-9]\d{0,3})\.(?:0|[1-9]\d{0,3})\.(?:0|[1-9]\d{0,3})$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TRUSTED_HTTPS_HOSTS = frozenset({
    "github.com",
    "objects.githubusercontent.com",
    "release-assets.githubusercontent.com",
    "forensic-dgp.pages.dev",
})
TRUSTED_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

_DETACHED_PROCESS = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
_CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
_CREATE_BREAKAWAY_FROM_JOB = 0x01000000


class UpdateValidationError(ValueError):
    """Manifest, download URL, or package integrity validation failed."""


class UpdateUnavailableError(RuntimeError):
    """Remote update manifest or package could not be retrieved."""


def _parse_version(version: str) -> tuple[int, int, int]:
    """Validate and parse a bounded three-part semantic version string."""
    cleaned = str(version or "").strip()
    if not _SEMVER_RE.fullmatch(cleaned):
        raise UpdateValidationError(f"Invalid version format: {cleaned!r}")
    major, minor, patch = cleaned.split(".")
    return int(major), int(minor), int(patch)


def _read_version_from_file(path: Path) -> str | None:
    try:
        if not path.is_file() or path.stat().st_size > MAX_MANIFEST_BYTES:
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            version = str(data.get("version", "")).strip()
            if _SEMVER_RE.fullmatch(version):
                return version
    except (OSError, ValueError, TypeError):
        return None
    return None


def get_current_version() -> str:
    """Resolve the active application version from environment, release metadata, or fallback."""
    env_version = os.environ.get("DGP_APP_VERSION", "").strip()
    if env_version and _SEMVER_RE.fullmatch(env_version):
        return env_version

    version_file = os.environ.get("DGP_VERSION_FILE", "").strip()
    if version_file:
        parsed = _read_version_from_file(Path(version_file))
        if parsed:
            return parsed

    install_root = os.environ.get("DGP_INSTALL_ROOT", "").strip()
    if install_root:
        parsed = _read_version_from_file(Path(install_root) / "release.json")
        if parsed:
            return parsed

    parsed = _read_version_from_file(ROOT / "release.json")
    if parsed:
        return parsed

    return APP_VERSION


def _manifest_source() -> str:
    return (
        os.environ.get("DGP_UPDATE_MANIFEST_FILE")
        or os.environ.get("DGP_UPDATE_MANIFEST_file")
        or os.environ.get("DGP_UPDATE_MANIFEST_URL")
        or DEFAULT_MANIFEST_URL
    ).strip()


def _extra_allowed_hosts() -> set[str]:
    raw = os.environ.get("DGP_UPDATE_ALLOWED_HOSTS", "")
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def _is_windows_drive_path(value: str) -> bool:
    return len(value) >= 2 and value[1] == ":" and value[0].isalpha()


def _read_local_bytes(path: Path, *, max_bytes: int) -> bytes:
    if not path.is_file():
        raise UpdateUnavailableError("Update manifest file was not found.")
    size = path.stat().st_size
    if size > max_bytes:
        raise UpdateValidationError("Update manifest exceeds size limit.")
    return path.read_bytes()


def _fetch_manifest_bytes(source: str) -> tuple[bytes, bool, str]:
    if not source:
        raise UpdateValidationError("Update manifest source is empty.")
    if _is_windows_drive_path(source) or "://" not in source:
        return _read_local_bytes(Path(source), max_bytes=MAX_MANIFEST_BYTES), True, ""

    parsed = urlsplit(source)
    scheme = parsed.scheme.lower()
    if scheme == "file":
        local_path = Path(urllib.request.url2pathname(parsed.path))
        return _read_local_bytes(local_path, max_bytes=MAX_MANIFEST_BYTES), True, ""

    host = (parsed.hostname or "").lower()
    if scheme == "http":
        if host not in TRUSTED_LOOPBACK_HOSTS:
            raise UpdateValidationError("Manifest URL must use HTTPS.")
    elif scheme != "https":
        raise UpdateValidationError("Unsupported manifest URL scheme.")

    request = urllib.request.Request(
        source,
        headers={"User-Agent": f"Forensic-DGP/{APP_VERSION}", "Cache-Control": "no-cache"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                raise UpdateUnavailableError(f"Update server returned HTTP {status}.")
            data = response.read(MAX_MANIFEST_BYTES + 1)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise UpdateUnavailableError("Could not contact the update server.") from exc

    if len(data) > MAX_MANIFEST_BYTES:
        raise UpdateValidationError("Update manifest exceeds size limit.")
    return data, host in TRUSTED_LOOPBACK_HOSTS, host


def _validate_download_url(download_url: str, *, allow_local: bool, manifest_host: str) -> tuple[str, Path | None]:
    cleaned = str(download_url or "").strip()
    if not cleaned:
        raise UpdateValidationError("Update manifest is missing download URL.")

    if _is_windows_drive_path(cleaned) or "://" not in cleaned:
        if not allow_local:
            raise UpdateValidationError("Untrusted local download path.")
        return "file", Path(cleaned)

    parsed = urlsplit(cleaned)
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()

    if scheme == "file":
        if not allow_local:
            raise UpdateValidationError("Untrusted file:// download URL.")
        return "file", Path(urllib.request.url2pathname(parsed.path))

    if scheme == "http":
        if host not in TRUSTED_LOOPBACK_HOSTS:
            raise UpdateValidationError("Untrusted non-HTTPS download URL.")
        return "http", None

    if scheme == "https":
        if (
            host in TRUSTED_HTTPS_HOSTS
            or host.endswith(".githubusercontent.com")
            or (manifest_host and host == manifest_host)
            or host in _extra_allowed_hosts()
        ):
            return "https", None
        raise UpdateValidationError(f"Untrusted update download host: {host or 'unknown'}.")

    raise UpdateValidationError("Unsupported download URL scheme.")


def _fetch_manifest() -> tuple[dict, bool, str]:
    raw_bytes, allow_local, manifest_host = _fetch_manifest_bytes(_manifest_source())
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise UpdateValidationError("Update manifest is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise UpdateValidationError("Update manifest must be a JSON object.")

    latest_version = str(payload.get("version", "")).strip()
    _parse_version(latest_version)

    download_url = str(payload.get("downloadUrl") or payload.get("download_url") or "").strip()
    sha256 = str(payload.get("sha256", "")).strip().lower()
    size_bytes = payload.get("bytes")
    if size_bytes is not None:
        if type(size_bytes) is not int or size_bytes < 1 or size_bytes >= MAX_PACKAGE_BYTES:
            raise UpdateValidationError("Update manifest size metadata is invalid.")

    normalized = {
        "schema": payload.get("schema", 1),
        "version": latest_version,
        "download_url": download_url,
        "sha256": sha256,
        "bytes": size_bytes,
        "filename": payload.get("filename"),
    }
    return normalized, allow_local, manifest_host


def check_for_updates() -> dict:
    """Check whether a newer version is published in the release manifest."""
    current_version = get_current_version()
    manifest, allow_local, manifest_host = _fetch_manifest()
    latest_version = manifest["version"]

    if _parse_version(latest_version) <= _parse_version(current_version):
        return {
            "status": "up_to_date",
            "update_available": False,
            "current_version": current_version,
            "latest_version": latest_version,
            "message": "Up to date",
        }

    _validate_download_url(manifest["download_url"], allow_local=allow_local, manifest_host=manifest_host)
    if not _SHA256_RE.fullmatch(manifest["sha256"]):
        raise UpdateValidationError("Update manifest is missing a valid SHA-256 digest.")

    return {
        "status": "update_available",
        "update_available": True,
        "current_version": current_version,
        "latest_version": latest_version,
        "download_url": manifest["download_url"],
        "sha256": manifest["sha256"],
        "bytes": manifest["bytes"],
        "message": f"Update {latest_version} available",
    }


def _download_and_verify_package(
    download_url: str,
    expected_sha256: str,
    expected_bytes: int | None,
    *,
    allow_local: bool,
    manifest_host: str,
) -> Path:
    if not _SHA256_RE.fullmatch(expected_sha256):
        raise UpdateValidationError("Update manifest is missing a valid SHA-256 digest.")

    kind, local_path = _validate_download_url(download_url, allow_local=allow_local, manifest_host=manifest_host)
    url_path = urlsplit(download_url).path.lower()
    suffix = ".zip" if url_path.endswith(".zip") else ".exe" if url_path.endswith(".exe") else ".bin"

    tmp = tempfile.NamedTemporaryFile(prefix="dgp-update-", suffix=suffix, delete=False)
    tmp_path = Path(tmp.name)
    hasher = hashlib.sha256()
    total = 0

    try:
        if kind == "file":
            if local_path is None or not local_path.is_file():
                raise UpdateUnavailableError("Update package file was not found.")
            with tmp, local_path.open("rb") as src:
                while True:
                    chunk = src.read(65536)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total >= MAX_PACKAGE_BYTES:
                        raise UpdateValidationError("Update package exceeds maximum size limit.")
                    hasher.update(chunk)
                    tmp.write(chunk)
        else:
            request = urllib.request.Request(
                download_url,
                headers={"User-Agent": f"Forensic-DGP/{APP_VERSION}"},
            )
            try:
                with tmp, urllib.request.urlopen(request, timeout=60) as response:
                    status = getattr(response, "status", 200)
                    if status != 200:
                        raise UpdateUnavailableError(f"Download failed with HTTP {status}.")
                    while True:
                        chunk = response.read(65536)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total >= MAX_PACKAGE_BYTES:
                            raise UpdateValidationError("Update package exceeds maximum size limit.")
                        hasher.update(chunk)
                        tmp.write(chunk)
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                raise UpdateUnavailableError("Failed to download the update package.") from exc

        if expected_bytes is not None and total != expected_bytes:
            raise UpdateValidationError("Downloaded update package size does not match manifest.")
        if hasher.hexdigest() != expected_sha256:
            raise UpdateValidationError("Update package SHA-256 verification failed.")
        return tmp_path
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _ps_single_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _release_metadata_payload(manifest: dict) -> str:
    record = {
        "schema": manifest.get("schema", 1),
        "version": manifest["version"],
        "filename": manifest.get("filename") or f"ForensicDGP_Setup-{manifest['version']}.exe",
        "sha256": manifest["sha256"],
        "downloadUrl": manifest["download_url"],
    }
    if manifest.get("bytes") is not None:
        record["bytes"] = manifest["bytes"]
    return json.dumps(record, indent=2) + "\n"


def _write_json_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _safe_extract_zip(archive_path: Path, target_dir: Path) -> None:
    target_resolved = target_dir.resolve()
    target_resolved.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as zf:
        for member in zf.infolist():
            name = member.filename.replace("\\", "/")
            if not name or name.startswith("/") or _is_windows_drive_path(name):
                raise UpdateValidationError("Archive contains an unsafe path.")
            parts = [part for part in name.split("/") if part not in ("", ".")]
            if any(part == ".." for part in parts):
                raise UpdateValidationError("Archive contains path traversal.")
            dest = (target_resolved / Path(*parts)).resolve()
            if not dest.is_relative_to(target_resolved):
                raise UpdateValidationError("Archive target escaped installation directory.")
            if member.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as src, dest.open("wb") as out:
                while True:
                    chunk = src.read(65536)
                    if not chunk:
                        break
                    out.write(chunk)


def _launch_desktop_inplace_updater(installer_path: Path, install_root: Path, manifest: dict) -> None:
    """Spawn a detached PowerShell helper to install in-place and relaunch ForensicDGP.exe."""
    release_json = _release_metadata_payload(manifest)
    script_body = "\n".join([
        "$ErrorActionPreference = 'Stop'",
        "Start-Sleep -Milliseconds 600",
        f"$installer = {_ps_single_quote(str(installer_path))}",
        f"$installRoot = {_ps_single_quote(str(install_root))}",
        "$releaseFile = Join-Path $installRoot 'release.json'",
        "$exePath = Join-Path $installRoot 'ForensicDGP.exe'",
        f"$releaseJson = {_ps_single_quote(release_json)}",
        "try {",
        "    $proc = Start-Process -FilePath $installer -ArgumentList @('/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART','/CLOSEAPPLICATIONS',('/DIR=' + $installRoot)) -Wait -PassThru",
        "    if ($proc.ExitCode -eq 0) {",
        "        [System.IO.File]::WriteAllText($releaseFile, $releaseJson, [System.Text.UTF8Encoding]::new($false))",
        "    }",
        "} finally {",
        "    Remove-Item -LiteralPath $installer -Force -ErrorAction SilentlyContinue",
        "    if (Test-Path -LiteralPath $exePath) {",
        "        Start-Process -FilePath $exePath | Out-Null",
        "    }",
        "    Remove-Item -LiteralPath $MyInvocation.MyCommand.Path -Force -ErrorAction SilentlyContinue",
        "}",
        "",
    ])
    script_file = tempfile.NamedTemporaryFile(
        prefix="dgp-inplace-update-",
        suffix=".ps1",
        mode="w",
        encoding="utf-8",
        delete=False,
    )
    script_path = Path(script_file.name)
    try:
        script_file.write(script_body)
        script_file.close()
    except BaseException:
        script_file.close()
        script_path.unlink(missing_ok=True)
        raise

    cmd = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-WindowStyle",
        "Hidden",
        "-File",
        str(script_path),
    ]
    spawn_kwargs = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    base_flags = _DETACHED_PROCESS | _CREATE_NEW_PROCESS_GROUP
    try:
        subprocess.Popen(cmd, creationflags=base_flags | _CREATE_BREAKAWAY_FROM_JOB, **spawn_kwargs)
    except PermissionError:
        subprocess.Popen(cmd, creationflags=base_flags, **spawn_kwargs)


def schedule_process_restart(delay_seconds: float = 0.5) -> bool:
    """Schedule a clean server restart/reload after returning the HTTP response."""
    in_test_harness = "unittest" in sys.modules or "pytest" in sys.modules
    if in_test_harness and os.environ.get("DGP_ENABLE_SELF_RESTART", "") != "1":
        return True

    def _restart() -> None:
        if os.environ.get("DGP_SELF_RESTART_EXEC", "") == "1":
            os.execv(sys.executable, [sys.executable, *sys.argv])
        else:
            os.kill(os.getpid(), signal.SIGTERM)

    timer = threading.Timer(delay_seconds, _restart)
    timer.daemon = True
    timer.start()
    return True


def install_update() -> dict:
    """Download, verify, and apply the latest update in-place, then schedule restart."""
    current_version = get_current_version()
    manifest, allow_local, manifest_host = _fetch_manifest()
    latest_version = manifest["version"]

    if _parse_version(latest_version) <= _parse_version(current_version):
        raise UpdateValidationError("Application is already up to date.")

    package_path = _download_and_verify_package(
        manifest["download_url"],
        manifest["sha256"],
        manifest["bytes"],
        allow_local=allow_local,
        manifest_host=manifest_host,
    )

    install_root_str = os.environ.get("DGP_INSTALL_ROOT", "").strip()
    install_root = Path(install_root_str) if install_root_str else None
    inplace_mode = os.environ.get("DGP_INPLACE_UPDATE_MODE", "").strip().lower()
    desktop_exe = (install_root / "ForensicDGP.exe") if install_root else None

    if (
        inplace_mode != "reload"
        and install_root is not None
        and desktop_exe is not None
        and desktop_exe.is_file()
        and package_path.suffix.lower() == ".exe"
    ):
        try:
            _launch_desktop_inplace_updater(package_path, install_root, manifest)
        except BaseException:
            package_path.unlink(missing_ok=True)
            raise
        return {
            "status": "updating",
            "target_version": latest_version,
            "restart": True,
            "restart_scheduled": True,
            "mode": "desktop_installer",
            "message": f"Installing update {latest_version} and restarting...",
        }

    try:
        extract_root = os.environ.get("DGP_UPDATE_TARGET_ROOT", "").strip()
        if package_path.suffix.lower() == ".zip" and extract_root:
            _safe_extract_zip(package_path, Path(extract_root))

        version_file_str = os.environ.get("DGP_VERSION_FILE", "").strip()
        metadata_text = _release_metadata_payload(manifest)
        if version_file_str:
            _write_json_atomic(Path(version_file_str), metadata_text)
        elif install_root is not None:
            _write_json_atomic(install_root / "release.json", metadata_text)
        else:
            _write_json_atomic(ROOT / "release.json", metadata_text)
    finally:
        package_path.unlink(missing_ok=True)

    restart_scheduled = schedule_process_restart()
    return {
        "status": "updating",
        "target_version": latest_version,
        "restart": True,
        "restart_scheduled": restart_scheduled,
        "mode": "reload",
        "message": f"Installing update {latest_version} and restarting...",
    }
