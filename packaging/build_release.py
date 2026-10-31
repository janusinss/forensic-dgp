"""Build a pinned online installer from a committed source tree."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
INNO_VERSION = "7.1.1"
INNO_INSTALLER_SHA256 = "8a5a6711c3766a6e5fb74f280ba0e3551e719ce94538d901fdbd1f8f9920b3f0"


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, stdout=subprocess.PIPE).stdout


def checked_version(value: str) -> str:
    if not re.fullmatch(r"(0|[1-9]\d{0,3})\.(0|[1-9]\d{0,3})\.(0|[1-9]\d{0,3})", value):
        raise ValueError("Version must be three bounded decimal integers.")
    return value


def check_inno_version(iscc: Path) -> None:
    # Inno 7's help banner omits the engine's patch version; query it directly.
    output = subprocess.run([str(iscc), "--version"], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=True).stdout
    if output.decode("utf-8-sig").strip() != INNO_VERSION:
        raise RuntimeError("Use verified Inno Setup " + INNO_VERSION + ".")


def source_archive(commit: str) -> bytes:
    # Unspecified text files must keep their Git blob bytes on Windows. Explicit
    # *.bat eol=crlf remains declared by attributes and is verified by the backend.
    return git("-c", "core.autocrlf=false", "archive", "--format=zip",
               "--prefix=forensic-dgp-" + commit + "/", commit)


def source_manifest(commit: str, archive: bytes, tree: bytes, ancestors: bytes) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Expected exact source commit.")
    files = {}
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        header, path = entry.split(b"\t", 1)
        mode, kind, blob = header.decode("ascii").split()
        name = path.decode("utf-8")
        if mode not in ("100644", "100755") or kind != "blob" or "\\" in name:
            raise ValueError("Unsupported source entry.")
        if any(p in ("", ".", "..") or p.endswith((".", " ")) for p in name.split("/")):
            raise ValueError("Unsafe source path.")
        files[name] = blob
    history = ancestors.decode("ascii").splitlines()
    if not files or not history or history[0] != commit or any(not re.fullmatch(r"[0-9a-f]{40}", v) for v in history):
        raise ValueError("Invalid source history.")
    return dict(schema=1, commit=commit, files=files, source_zip_sha256=hashlib.sha256(archive).hexdigest(),
                source_zip_size=len(archive), ancestors=history)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iscc", type=Path, required=True)
    parser.add_argument("--version", type=checked_version, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "scratch" / "dist")
    args = parser.parse_args()
    if os.name != "nt":
        raise RuntimeError("Windows build required.")
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise RuntimeError("Commit tracked source changes before a release build.")
    commit = git("rev-parse", "HEAD").decode("ascii").strip()
    # Untracked files are never included: every payload and packaging input is read
    # from the committed tree or generated from those bytes.
    archive = source_archive(commit)
    manifest = source_manifest(commit, archive, git("ls-tree", "-rz", commit), git("rev-list", commit))
    stage = ROOT / "scratch" / "build" / commit
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "app-source.zip").write_bytes(archive)
    write_json(stage / "app-source.json", manifest)
    inputs = {
        "distribution.ps1": "implementation/distribution.ps1",
        "uninstall.ps1": "packaging/uninstall.ps1",
        "gui_setup.ps1": "packaging/gui_setup.ps1",
        "installer.iss": "packaging/installer.iss",
        "Program.cs": "packaging/launcher/Program.cs",
        "LICENSE.txt": "packaging/LICENSE.txt",
    }
    for dest, source in inputs.items():
        (stage / dest).write_bytes(git("show", commit + ":" + source))
    compiler = Path(os.environ["SystemRoot"]) / "Microsoft.NET/Framework64/v4.0.30319/csc.exe"
    subprocess.run([str(compiler), "/nologo", "/target:winexe", "/platform:x64", "/optimize+",
                    "/reference:System.Windows.Forms.dll", "/reference:System.Drawing.dll",
                    "/reference:System.Web.Extensions.dll", "/out:" + str(stage / "ForensicDGP.exe"),
                    str(stage / "Program.cs")], check=True)
    check_inno_version(args.iscc)
    build = dict(schema=1, version=args.version, commit=commit, inno_version=INNO_VERSION,
                 inno_installer_sha256=INNO_INSTALLER_SHA256, signed=False,
                 payload="online-source-bundled", files={})
    for name in ["app-source.zip", "app-source.json", "distribution.ps1", "uninstall.ps1",
                 "gui_setup.ps1", "ForensicDGP.exe"]:
        build["files"][name] = hashlib.sha256((stage / name).read_bytes()).hexdigest()
    write_json(stage / "release.json", build)
    (stage / "shell-owner.json").write_text('{"schema":1,"application":"Forensic DGP"}\n', encoding="ascii")
    (stage / "release.inc").write_text(
        '#define ReleaseManifestSHA256 "' + hashlib.sha256((stage / "release.json").read_bytes()).hexdigest() + '"\n'
        '#define UninstallSHA256 "' + build["files"]["uninstall.ps1"] + '"\n', encoding="ascii")
    args.output.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(args.iscc), "/DStageDir=" + str(stage), "/DOutputDir=" + str(args.output.resolve()),
                    "/DReleaseVersion=" + args.version, "/DReleaseCommit=" + commit,
                    str(stage / "installer.iss")], check=True)
    exe = args.output / "ForensicDGP_Setup.exe"
    build["artifact_sha256"] = hashlib.sha256(exe.read_bytes()).hexdigest()
    build["artifact_bytes"] = exe.stat().st_size
    write_json(args.output / "release.json", build)
    print(json.dumps(dict(artifact=str(exe.resolve()), **build), indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print("[ERROR] " + str(exc), file=sys.stderr)
        raise SystemExit(1)
