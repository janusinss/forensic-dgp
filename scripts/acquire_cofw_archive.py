"""Bounded official COFW acquisition; never extracts or admits training data."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = "https://data.caltech.edu/api/records/bc0bf-nc666/files/COFW_color.zip/content"
EXPECTED_SIZE = 503327162
EXPECTED_MD5 = "8b21d126c4e1fb307cb463578eef0511"


def write(path, data):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(data, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT) or out.exists():
        raise ValueError("Use a new directory inside the workspace")
    out.mkdir(parents=True)
    workers = 8
    chunk = (EXPECTED_SIZE + workers - 1) // workers
    started = time.monotonic()
    protocol = {
        "format": "dgp-cofw-official-acquisition-v2", "date": "2026-10-03",
        "record_url": "https://data.caltech.edu/records/bc0bf-nc666", "url": URL,
        "expected_bytes": EXPECTED_SIZE, "publisher_md5": EXPECTED_MD5,
        "workers": workers, "maximum_seconds_per_request": 240,
        "automatic_retries": 0, "training_admitted": False,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "previous_partial_preserved": "outputs/cofw_source_research_v1/COFW_color.zip.partial",
        "rationale": "A 1 MB official GET returned HTTP206 at 374114 bytes/s; prior HEAD403 used a method different from the signed GET. Require all ranges and full publisher checksum before extraction.",
    }
    write(out / "protocol.json", protocol)

    def fetch(index):
        first, last = index * chunk, min((index + 1) * chunk, EXPECTED_SIZE) - 1
        target = out / f"range_{index:02d}.partial"
        proc = subprocess.run([
            "curl.exe", "--fail", "--location", "--silent", "--show-error",
            "--connect-timeout", "10", "--max-time", "240", "--range",
            f"{first}-{last}", "--output", str(target), "--write-out", "%{json}", URL,
        ], capture_output=True, text=True)
        raw = json.loads(proc.stdout) if proc.stdout.strip() else {}
        report = {key: raw.get(key) for key in (
            "http_code", "size_download", "speed_download", "time_total", "content_type", "errormsg")}
        size = target.stat().st_size if target.exists() else 0
        report.update(index=index, first=first, last=last, bytes=size,
                      expected_bytes=last - first + 1, curl_exit=proc.returncode)
        report["complete"] = proc.returncode == 0 and raw.get("http_code") == 206 and size == last - first + 1
        write(out / f"range_{index:02d}.json", report)
        print(json.dumps({"range": index, "complete": report["complete"], "bytes": size,
                          "seconds": round(time.monotonic() - started, 2)}), flush=True)
        return report

    records = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for task in as_completed([executor.submit(fetch, i) for i in range(workers)]):
            records.append(task.result())
    records.sort(key=lambda row: row["index"])
    result = {"complete": False, "ranges": records, "training_admitted": False,
              "source_extractions": 0, "model_forwards": 0, "optimizer_updates": 0}
    if all(row["complete"] for row in records):
        target = out / "COFW_color.zip"
        md5, digest = hashlib.md5(), hashlib.sha256()
        with target.open("xb") as combined:
            for row in records:
                with (out / f"range_{row['index']:02d}.partial").open("rb") as source:
                    while block := source.read(1024 * 1024):
                        combined.write(block)
                        md5.update(block)
                        digest.update(block)
        result.update(bytes=target.stat().st_size, md5=md5.hexdigest(), sha256=digest.hexdigest())
        if result["bytes"] == EXPECTED_SIZE and result["md5"] == EXPECTED_MD5:
            with zipfile.ZipFile(target) as archive:
                result["members"] = [{"name": row.filename, "bytes": row.file_size,
                                      "compressed_bytes": row.compress_size, "crc": row.CRC}
                                     for row in archive.infolist()]
                result["crc_error_member"] = archive.testzip()
            result["complete"] = result["crc_error_member"] is None
    result["elapsed_seconds"] = time.monotonic() - started
    write(out / "acquisition.json", result)
    print(json.dumps({k: result[k] for k in ("complete", "elapsed_seconds", "training_admitted")}), flush=True)
    if not result["complete"]:
        raise SystemExit("Acquisition incomplete; preserved partial evidence is excluded from dataset use")


if __name__ == "__main__":
    main()
