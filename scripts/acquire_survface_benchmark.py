"""Acquire the official native CCTV release; inventory only, no extraction/training."""
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import time
import urllib.request
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/cctv_survface_acquisition_v1"
TARGET = ROOT / "dataset/cctv_survface_raw/QMUL-SurvFace-v1.zip"
URL = "https://drive.usercontent.google.com/download?id=13ch6BPaexlKt8gXB_I8aX7p1G3yPm2Bl&export=download&confirm=t"
EXPECTED_BYTES = 408_164_983
MAX_SECONDS = 600


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    partial = TARGET.with_suffix(".zip.part")
    if OUT.exists() or TARGET.exists() or partial.exists():
        raise ValueError("Preserve existing acquisition or partial download; inspect it before retrying")
    OUT.mkdir(parents=True)
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    write(OUT / "request.json", {
        "date": "2026-10-03", "source": "https://qmul-survface.github.io/",
        "official_link": "https://drive.google.com/open?id=13ch6BPaexlKt8gXB_I8aX7p1G3yPm2Bl",
        "download_url": URL, "expected_content_length_from_preflight": EXPECTED_BYTES,
        "time_cap_seconds": MAX_SECONDS, "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "publisher_checksum_available": False, "purpose": "Research CCTV evaluation acquisition, not model training",
        "training": False, "model_forwards": 0, "archive_extraction": False,
    })
    digest, count, next_progress = hashlib.sha256(), 0, 32 * 1024 * 1024
    try:
        request = urllib.request.Request(URL, headers={"User-Agent": "Forensic-DGP-CCTV-Benchmark/1.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 200 or response.headers.get("Content-Type", "").split(";")[0] != "application/octet-stream":
                raise ValueError("Official endpoint did not provide the public archive")
            if response.headers.get("Content-Length") != str(EXPECTED_BYTES):
                raise ValueError("Official release size differs from observed preflight; recheck provenance")
            with partial.open("xb") as stream:
                while True:
                    if time.monotonic() - started > MAX_SECONDS:
                        raise TimeoutError("Finite public dataset transfer cap exceeded")
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    if count == 0 and not chunk.startswith(b"PK\x03\x04"):
                        raise ValueError("Download is not a ZIP archive")
                    count += len(chunk)
                    if count > EXPECTED_BYTES:
                        raise ValueError("Archive exceeds fixed size limit")
                    stream.write(chunk)
                    digest.update(chunk)
                    if count >= next_progress:
                        print(json.dumps({"received_bytes": count, "expected_bytes": EXPECTED_BYTES,
                                          "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
                        next_progress += 32 * 1024 * 1024
        if count != EXPECTED_BYTES:
            raise ValueError("Incomplete archive; partial file preserved")
        files, roots, extensions, metadata = set(), Counter(), Counter(), []
        inventory_digest, uncompressed, member_count = hashlib.sha256(), 0, 0
        with ZipFile(partial) as archive:
            if not 1 <= len(archive.infolist()) <= 600_000:
                raise ValueError("Unexpected archive member count")
            for info in archive.infolist():
                path = PurePosixPath(info.filename)
                if path.is_absolute() or ".." in path.parts or "\\" in info.filename or ":" in info.filename or info.filename in files:
                    raise ValueError("Unsafe/duplicate archive member; extraction remains disabled")
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ValueError("Linked archive member; extraction remains disabled")
                files.add(info.filename)
                if info.is_dir():
                    continue
                uncompressed += info.file_size
                member_count += 1
                if uncompressed > 4 * 1024 ** 3 or info.file_size > 128 * 1024 ** 2:
                    raise ValueError("Unexpected archive expansion/member size")
                roots["/".join(path.parts[:2])] += 1
                extensions[path.suffix.lower()] += 1
                if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    metadata.append({"path": info.filename, "bytes": info.file_size, "crc32": info.CRC})
                encoded = json.dumps([info.filename, info.file_size, info.compress_size, info.CRC], ensure_ascii=True, separators=(",", ":"))
                inventory_digest.update((encoded + "\n").encode("ascii"))
        if len(metadata) > 100:
            raise ValueError("Unexpected release metadata contents; inspect before use")
        # No supplied image is decoded and no archive code is executed here.
        if TARGET.exists():
            raise ValueError("Destination appeared during transfer; preserve both files for inspection")
        partial.rename(TARGET)
        with TARGET.with_suffix(".zip.sha256").open("x", encoding="ascii", newline="\n") as stream:
            stream.write(digest.hexdigest() + "  " + TARGET.name + "\n")
        inventory = {"complete": True, "archive_sha256": digest.hexdigest(), "archive_bytes": count,
                     "regular_members": member_count, "uncompressed_bytes": uncompressed,
                     "extensions": dict(extensions), "top_directories": dict(roots), "metadata_members": metadata,
                     "central_directory_digest_sha256": inventory_digest.hexdigest(),
                     "all_member_crc_contents_checked": False, "extracted_members": 0, "decoded_images": 0,
                     "heldout_subset_selected": False, "benchmark_protocol_frozen": False,
                     "model_forwards": 0, "optimizer_updates": 0}
        write(OUT / "zip_inventory.json", inventory)
        write(OUT / "acquisition.json", {"complete": True, "archive": str(TARGET.relative_to(ROOT)).replace("\\", "/"),
              "archive_bytes": count, "archive_sha256": digest.hexdigest(), "checksum_scope": "Observed acquisition fingerprint, not publisher-supplied",
              "inventory_sha256": hashlib.sha256((OUT / "zip_inventory.json").read_bytes()).hexdigest(),
              "seconds": time.monotonic() - started, "training": False, "model_forwards": 0,
              "restoration_quality_evaluated": False, "zamboanga_validation": False})
        print(json.dumps({"complete": True, "bytes": count, "archive_sha256": digest.hexdigest(),
                          "regular_members": member_count, "extensions": dict(extensions)}), flush=True)
    except Exception as exc:
        if not (OUT / "failure.json").exists():
            write(OUT / "failure.json", {"complete": False, "error_type": type(exc).__name__, "message": str(exc),
                  "received_bytes": count, "elapsed_seconds": time.monotonic() - started,
                  "partial_preserved": partial.exists(), "training": False})
        raise


if __name__ == "__main__":
    main()
