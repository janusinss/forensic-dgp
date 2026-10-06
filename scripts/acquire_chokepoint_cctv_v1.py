"""Acquire three author-linked CCTV files sequentially; retain source terms/checksums."""
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/cctv_chokepoint_acquisition_v1"
DATA = ROOT / "dataset/cctv_chokepoint_raw_v1"
PAGE = "https://arma.sourceforge.net/chokepoint/"
RECORD = "https://zenodo.org/api/records/815657"
FILES = [("README.txt", 1446, "dfdc23d2934fc07b63ecf391d85db9dd"),
         ("groundtruth.tar.xz", 456396, "5c9d4d38a1c614905fe48da48ecdf06c"),
         ("P1E_S1.tar.xz", 385315324, "195191772856c1a62650a1d47e897b2a")]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def open_url(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "ForensicDGP-Thesis-Research/1.0"}), timeout=25)


def main():
    if OUT.exists() or DATA.exists():
        raise ValueError("Preserve partial/completed acquisition; no automatic repeat or overwrite")
    if shutil.disk_usage(ROOT).free < 2 * 1024 ** 3:
        raise ValueError("Require 2 GiB free for the bounded acquisition and later audit")
    OUT.mkdir()
    DATA.mkdir()
    write(OUT / "request.json", {"date": "2026-10-05", "frozen_before_download": True,
          "source_page": PAGE, "author_linked_repository_record": RECORD,
          "acquirer_sha256": sha(Path(__file__)),
          "files": [{"name": n, "bytes": s, "publisher_md5": m} for n, s, m in FILES],
          "maximum_payload_bytes": sum(s for _, s, _ in FILES), "cap_seconds": 600,
          "outer_timeout_seconds": 660, "socket_timeout_seconds": 25, "parallel_downloads": 1,
          "source_choice": "QMUL larger-size extension yielded no supported inputs. COX author-linked endpoint returned404; no access/terms claimed. ChokePoint is a separately reported broader source, not an Asian/Zamboanga/ethnicity proxy.",
          "native_input_choice": "Acquire original P1E_S1 camera frames and eye/identity annotations; avoid mistaking publisher-normalized96x96 face crops for native captured face resolution",
          "split_status": "No development/evaluation split chosen until release metadata is independently audited",
          "training": False, "models_executed": False, "images_decoded": False,
          "identity_overlap": "Unknown across datasets and historical training; no cross-source independence claim"})
    started = time.monotonic()
    completed = []
    try:
        with open_url(PAGE) as response:
            page = response.read(500001)
            if len(page) > 500000 or b"non-commercial" not in page or b"license notice" not in page.lower():
                raise ValueError("Require the complete original research-license source notice")
        # Keep the original notice with source material and all later derivatives.
        (DATA / "LICENSE_SOURCE.html").write_bytes(page)
        with open_url(RECORD) as response:
            raw_record = response.read(500001)
        if len(raw_record) > 500000:
            raise ValueError("Metadata cap")
        (OUT / "publisher_record.json").write_bytes(raw_record)
        record = json.loads(raw_record)
        published = {file["key"]: file for file in record["files"]}
        for name, size, md5 in FILES:
            file = published[name]
            url = f"https://zenodo.org/api/records/815657/files/{name}/content"
            if file["size"] != size or file["checksum"] != "md5:" + md5 or file["links"]["self"] != url:
                raise ValueError("Publisher fingerprint/source endpoint differs: " + name)
            partial = DATA / (name + ".part")
            digest, total, next_progress = hashlib.md5(), 0, 64 * 1024 ** 2
            with open_url(url) as response, partial.open("xb") as stream:
                if response.status != 200:
                    raise ValueError("Require successful full download")
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    if time.monotonic() - started > 600:
                        raise TimeoutError("Finite acquisition cap; partial file retained")
                    total += len(block)
                    if total > size:
                        raise ValueError("Payload exceeds pinned size")
                    digest.update(block)
                    stream.write(block)
                    if total >= next_progress:
                        print(json.dumps({"file": name, "bytes": total, "total_bytes": size,
                                          "seconds": time.monotonic() - started}), flush=True)
                        next_progress += 64 * 1024 ** 2
            if total != size or digest.hexdigest() != md5:
                raise ValueError("Transfer size/publisher MD5 mismatch: " + name)
            target = DATA / name
            partial.rename(target)
            fingerprint = sha(target)
            (DATA / (name + ".sha256")).write_text(fingerprint + "  " + name + "\n", encoding="ascii", newline="\n")
            completed.append({"name": name, "url": url, "bytes": total, "publisher_md5": md5,
                              "md5_verified": True, "observed_sha256": fingerprint})
            print(json.dumps({"downloaded": name, "bytes": total, "seconds": time.monotonic() - started}), flush=True)
        seconds = time.monotonic() - started
        if seconds > 600:
            raise TimeoutError("Acquisition completion cap")
        write(OUT / "acquisition.json", {"complete": True, "date": "2026-10-05", "seconds": seconds,
              "request_sha256": sha(OUT / "request.json"), "publisher_record_sha256": sha(OUT / "publisher_record.json"),
              "license_source_sha256": sha(DATA / "LICENSE_SOURCE.html"), "files": completed,
              "training_calls": 0, "model_forwards": 0, "images_decoded": 0,
              "dataset_audit_complete": False, "native_restoration_qualified": False,
              "independent_final_review": False, "source_country_per_image": None,
              "ethnicity_inferred": False, "zamboanga_validation": False})
        print(json.dumps({"complete": True, "seconds": seconds, "bytes": sum(f["bytes"] for f in completed)}), flush=True)
    except Exception as error:
        write(OUT / "failure.json", {"complete": False, "type": type(error).__name__, "error": str(error),
              "seconds": time.monotonic() - started, "completed_files": completed,
              "partial_files_preserved": True, "model_forwards": 0, "training_calls": 0})
        raise


if __name__ == "__main__":
    main()
