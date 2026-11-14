"""Build a pinned online or offline installer from a committed source tree."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
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


def artifact_record(path: Path) -> dict:
    size = path.stat().st_size
    if size < 1 or size >= 2 * 1024**3:
        raise RuntimeError("The single release EXE must be nonempty and below GitHub's 2 GiB asset limit.")
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return dict(artifact_bytes=size, artifact_sha256=digest.hexdigest())


def release_defines(build: dict, manifest_sha256: str) -> str:
    result = ('#define ReleaseManifestSHA256 "' + manifest_sha256 + '"\n' +
              '#define UninstallSHA256 "' + build['files']['uninstall.ps1'] + '"\n')
    if build['payload'] == 'offline-bundled':
        result += ('#define OfflinePayload 1\n' +
                   '#define OfflinePayloadBytes ' + str(build['payload_bytes']) + '\n' +
                   '#define OfflineExpandedBytes ' + str(build['payload_expanded_bytes']) + '\n')
    return result


def stage_offline_payload(payload: Path, stage: Path, commit: str) -> dict:
    helper = ROOT / 'packaging' / 'prepare_offline.py'
    if helper.read_bytes() != git('show', commit + ':packaging/prepare_offline.py'):
        raise RuntimeError('Offline preparer differs from committed source.')
    spec = importlib.util.spec_from_file_location('dgp_offline_preparer', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    locks = {group:git('show', commit + ':implementation/locks/windows-cp313-' + group + '.txt')
             for group in ('application','capture')}
    model_specs = []
    for group, name in (('restoration','models.json'),('completion','completion_assets.json')):
        manifest = json.loads(git('show', commit + ':implementation/' + name))
        for identity, asset in sorted(manifest['assets'].items()):
            model_specs.append(dict(group=group,id=identity,filename=asset['filename'],size=asset['size'],sha256=asset['sha256']))
    return module.stage_payload(payload,stage / 'payload',locks,model_specs=model_specs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iscc", type=Path, required=True)
    parser.add_argument("--version", type=checked_version, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "scratch" / "dist")
    parser.add_argument("--offline-payload", type=Path, help="Verified payload prepared with packaging/prepare_offline.py.")
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
        "install_location.ps1": "packaging/install_location.ps1",
        "install_state.ps1": "packaging/install_state.ps1",
        "check_update.ps1": "packaging/check_update.ps1",
        "installer.iss": "packaging/installer.iss",
        "Program.cs": "packaging/launcher/Program.cs",
        "LICENSE.txt": "packaging/LICENSE.txt",
        "app.ico": "static/branding/favicon.ico",
    }
    for dest, source in inputs.items():
        (stage / dest).write_bytes(git("show", commit + ":" + source))
    assembly = (
        'using System.Reflection;\n'
        'using System.Resources;\n'
        'using System.Runtime.InteropServices;\n'
        '[assembly: AssemblyTitle("Forensic DGP")]\n'
        '[assembly: AssemblyDescription("Forensic DGP Desktop Launcher")]\n'
        '[assembly: AssemblyCompany("Forensic DGP Research")]\n'
        '[assembly: AssemblyProduct("Forensic DGP")]\n'
        '[assembly: AssemblyCopyright("Copyright (c) 2026 Forensic DGP Research")]\n'
        '[assembly: ComVisible(false)]\n'
        '[assembly: Guid("FEE649E9-195D-4576-8F30-DAE06A5099C8")]\n'
        '[assembly: NeutralResourcesLanguage("en-US")]\n'
        f'[assembly: AssemblyVersion("{args.version}.0")]\n'
        f'[assembly: AssemblyFileVersion("{args.version}.0")]\n'
        f'[assembly: AssemblyInformationalVersion("{args.version}+{commit[:12]}")]\n'
    )
    (stage / 'AssemblyInfo.cs').write_text(assembly, encoding='utf-8')
    manifest_file = stage / 'ForensicDGP.manifest'
    manifest_file.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<assembly manifestVersion="1.0" xmlns="urn:schemas-microsoft-com:asm.v1">\n'
        '  <assemblyIdentity version="1.0.0.0" name="ForensicDGP.DesktopLauncher" type="win32"/>\n'
        '  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v2">\n'
        '    <security>\n'
        '      <requestedPrivileges xmlns="urn:schemas-microsoft-com:asm.v3">\n'
        '        <requestedExecutionLevel level="asInvoker" uiAccess="false"/>\n'
        '      </requestedPrivileges>\n'
        '    </security>\n'
        '  </trustInfo>\n'
        '  <compatibility xmlns="urn:schemas-microsoft-com:compatibility.v1">\n'
        '    <application>\n'
        '      <supportedOS Id="{8e0f7a12-bfb3-4fe8-b9a5-48fd50a15a9a}"/>\n'
        '    </application>\n'
        '  </compatibility>\n'
        '  <application xmlns="urn:schemas-microsoft-com:asm.v3">\n'
        '    <windowsSettings>\n'
        '      <dpiAware xmlns="http://schemas.microsoft.com/SMI/2005/WindowsSettings">true/pm</dpiAware>\n'
        '      <dpiAwareness xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">PerMonitorV2, PerMonitor</dpiAwareness>\n'
        '      <longPathAware xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">true</longPathAware>\n'
        '    </windowsSettings>\n'
        '  </application>\n'
        '</assembly>\n',
        encoding='utf-8',
    )
    compiler = Path(os.environ["SystemRoot"]) / "Microsoft.NET/Framework64/v4.0.30319/csc.exe"
    subprocess.run([str(compiler), "/nologo", "/target:winexe", "/platform:x64", "/optimize+",
                    "/reference:System.Windows.Forms.dll", "/reference:System.Drawing.dll",
                    "/reference:System.Web.Extensions.dll", "/win32icon:" + str(stage / "app.ico"),
                    "/win32manifest:" + str(manifest_file), "/out:" + str(stage / "ForensicDGP.exe"),
                    str(stage / "Program.cs"), str(stage / "AssemblyInfo.cs")], check=True)
    check_inno_version(args.iscc)
    build = dict(schema=1, version=args.version, commit=commit, inno_version=INNO_VERSION,
                 inno_installer_sha256=INNO_INSTALLER_SHA256, signed=False,
                 payload="offline-bundled" if args.offline_payload else "online-source-bundled", files={})
    for name in ["app-source.zip", "app-source.json", "distribution.ps1", "uninstall.ps1",
                 "gui_setup.ps1", "ForensicDGP.exe", "install_location.ps1", "install_state.ps1", "check_update.ps1"]:
        build["files"][name] = hashlib.sha256((stage / name).read_bytes()).hexdigest()
    if args.offline_payload:
        verified = stage_offline_payload(args.offline_payload, stage, commit)
        build['files']['offline-payload-manifest.json'] = verified['manifest_sha256']
        build['payload_bytes'] = verified['payload_bytes']
        build['payload_expanded_bytes'] = verified['payload_expanded_bytes']
    write_json(stage / "release.json", build)
    (stage / "shell-owner.json").write_text('{"schema":1,"application":"Forensic DGP"}\n', encoding="ascii")
    (stage / "release.inc").write_text(
        release_defines(build, hashlib.sha256((stage / "release.json").read_bytes()).hexdigest()), encoding="ascii")
    args.output.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(args.iscc), "/DStageDir=" + str(stage), "/DOutputDir=" + str(args.output.resolve()),
                    "/DReleaseVersion=" + args.version, "/DReleaseCommit=" + commit,
                    str(stage / "installer.iss")], check=True)
    exe = args.output / "ForensicDGP_Setup.exe"
    build.update(artifact_record(exe))
    write_json(args.output / "release.json", build)
    print(json.dumps(dict(artifact=str(exe.resolve()), **build), indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print("[ERROR] " + str(exc), file=sys.stderr)
        raise SystemExit(1)
