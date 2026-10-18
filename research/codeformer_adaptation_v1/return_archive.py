"""Read-only verification and exact new-directory import of a manual VM return.

This auditor never imports Torch, runs training, contacts the VM or overwrites a
previous return. A valid export is evidence transport, never model qualification.
"""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import time

from .common import require, sha256, write_json

PROTOCOL_SHA256 = "e463222441c0930a01a85ee0711a1ea31b0d80ed3487a4a5459c71e65f517d3e"
MAX_PAYLOAD_BYTES = 16 * 1024**3
MAX_MEMBERS = 20000
DEVICE = re.compile(r"^(con|prn|aux|nul|com[1-9¹²³]|lpt[1-9¹²³])(?:\.|$)", re.I)


def canonical_member(name):
    require(isinstance(name, str) and name and "\\" not in name and ":" not in name,
            "Nonportable return member name")
    p = PurePosixPath(name)
    require(bool(p.parts) and not p.is_absolute() and p.as_posix() == name and
            all(part not in ("", ".", "..") for part in p.parts), "Escaped return member")
    require(all(not DEVICE.match(part) and not part.endswith((" ", ".")) and
                all(ord(char) >= 32 and char not in '?*<>|"' for char in part) for part in p.parts),
            "Unsafe Windows member name")
    return name


def _json(payload):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "Duplicate JSON field")
            result[key] = value
        return result
    return json.loads(payload.decode("utf-8-sig"), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def inspect_and_import(archive, sidecar, export_path, packet, destination, receipt_path):
    started = time.monotonic()
    archive, sidecar, export_path, packet = map(Path, (archive, sidecar, export_path, packet))
    destination, receipt_path = Path(destination), Path(receipt_path)
    require(not destination.exists(), "Retain prior imported return; do not overwrite")
    require(not receipt_path.exists(), "Retain prior import receipt")
    journal = destination.parent / (destination.name + "_import_journal.jsonl")
    require(not journal.exists(), "Retain prior import journal")
    require(archive.is_file() and not archive.is_symlink(), "Missing regular return archive")
    before_stat = archive.stat()
    export = _json(export_path.read_bytes())
    require(export.get("complete") is True and export.get("model_qualified") is False and
            export.get("protocol_sha256") == PROTOCOL_SHA256, "Unexpected return receipt")
    digest = sha256(archive)
    expected_sidecar = (digest + "  " + archive.name + "\n").encode("ascii")
    require(sidecar.read_bytes().replace(b"\r\n", b"\n") == expected_sidecar,
            "Return checksum sidecar differs")
    require(digest == export["archive_sha256"] and before_stat.st_size == export["bytes"],
            "Incomplete/changed archive download")
    require(sha256(packet / "protocol.json") == PROTOCOL_SHA256, "Local frozen protocol differs")
    protocol = _json((packet / "protocol.json").read_bytes())
    manifest_name = canonical_member(export["manifest_member"])
    require(manifest_name.startswith(protocol["outputs"] + "/export_manifest_"), "Unexpected manifest location")
    names, folded, payload_bytes, manifest_payload = {}, set(), 0, None
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            name = canonical_member(member.name)
            require(member.isfile() and not member.issym() and not member.islnk(), "Only regular return members allowed")
            require(name not in names and name.casefold() not in folded, "Duplicate/colliding return member")
            require(member.size >= 0 and len(names) < MAX_MEMBERS, "Invalid/oversized member namespace")
            payload_bytes += member.size
            require(payload_bytes <= MAX_PAYLOAD_BYTES, "Return payload exceeds finite limit")
            names[name] = member.size
            folded.add(name.casefold())
            if name == manifest_name:
                require(member.size < 8 * 1024**2, "Manifest exceeds8MiB")
                manifest_payload = source.extractfile(member).read()
    require(manifest_payload is not None and
            hashlib.sha256(manifest_payload).hexdigest() == export["manifest_sha256"],
            "Returned manifest fingerprint differs")
    manifest = _json(manifest_payload)
    require(manifest.get("complete") is True and manifest.get("protocol_sha256") == PROTOCOL_SHA256
            and manifest.get("model_qualified") is False, "Manifest scope differs")
    for flag in ("run_results_present", "failure_present"):
        require(type(manifest.get(flag)) is bool and manifest[flag] == export.get(flag),
                "Manifest/run receipt status differs")
    expected = dict(manifest["files_sha256"])
    require(manifest_name not in expected, "Manifest unexpectedly hashes itself")
    expected[manifest_name] = export["manifest_sha256"]
    require(set(expected) == set(names), "Unlisted or missing return member")
    require(set(protocol["export_files"]) <= set(expected), "Mandatory export companion missing")
    allowed = set(protocol["export_files"]) | {"protocol.json", "trainer.log", "adaptation_v1.lock"}
    for name, file_sha in expected.items():
        canonical_member(name)
        require(name.startswith(protocol["outputs"] + "/") or name in allowed,
                "Return contains an unexpected namespace")
        require(isinstance(file_sha, str) and re.fullmatch(r"[0-9a-f]{64}", file_sha), "Malformed file hash")
        if name in protocol["files_sha256"]:
            require(file_sha == protocol["files_sha256"][name], "Returned executable/input changed")
    require(expected["protocol.json"] == PROTOCOL_SHA256, "Returned protocol changed")
    all_inputs = dict(protocol["files_sha256"])
    all_inputs.update({row["path"]: row["sha256"] for row in protocol["assets"].values()})
    retained = manifest["retained_not_exported_sha256"]
    require(retained == {name: value for name, value in all_inputs.items() if name not in expected},
            "Retained-input binding namespace differs")
    for name, file_sha in retained.items():
        require(sha256(packet / canonical_member(name)) == file_sha, "Retained original packet input changed")
    require(len(retained) == export["retained_not_exported_files"], "Retained input count differs")
    require(shutil.disk_usage(destination.parent).free >= payload_bytes + 4 * 1024**3,
            "Require complete payload plus4GiB reserve for import")
    destination.mkdir()
    imported = {}
    try:
        with journal.open("x", encoding="utf-8", newline="\n") as log, tarfile.open(archive, "r:gz") as source:
            for member in source:
                name = canonical_member(member.name)
                target = destination.joinpath(*PurePosixPath(name).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                require(target.resolve().is_relative_to(destination.resolve()) and not target.exists(),
                        "Destination exists or escapes return directory")
                file_digest = hashlib.sha256()
                stream = source.extractfile(member)
                written = 0
                with target.open("xb") as output:
                    for block in iter(lambda: stream.read(4 * 1024**2), b""):
                        output.write(block); file_digest.update(block); written += len(block)
                require(written == names[name] and file_digest.hexdigest() == expected[name],
                        "Returned payload differs: " + name)
                imported[name] = file_digest.hexdigest()
                log.write(json.dumps({"file": name, "bytes": written, "sha256": imported[name]}) + "\n")
                log.flush()
        require(imported == expected, "Imported namespace differs")
        require((archive.stat().st_size, archive.stat().st_mtime_ns) ==
                (before_stat.st_size, before_stat.st_mtime_ns), "Archive changed during import")
        out = destination / protocol["outputs"]
        require((out / "failure.json").exists() == export["failure_present"] and
                (out / "run_results.json").exists() == export["run_results_present"],
                "Run/failure presence differs")
        receipt = {"complete": True, "archive_sha256": digest, "archive_bytes": before_stat.st_size,
            "protocol_sha256": PROTOCOL_SHA256, "members": len(imported), "payload_bytes": payload_bytes,
            "retained_inputs_verified": len(retained), "all_members_hash_verified": True,
            "exact_namespace": True, "import_directory": str(destination),
            "run_results_present": export["run_results_present"], "failure_present": export["failure_present"],
            "model_qualified": False, "gradient_queries": 0, "optimizer_updates": 0,
            "VM_calls": 0, "seconds": time.monotonic() - started}
        write_json(receipt_path, receipt)
        return receipt
    except BaseException as error:
        write_json(receipt_path.with_name(receipt_path.stem + "_failure.json"),
                   {"complete": False, "error": str(error), "imported": len(imported),
                    "original_archive_and_partial_evidence_retained": True, "model_qualified": False})
        raise
