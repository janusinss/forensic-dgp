"""Extract verified COFW matrices and prepare TRAINING source inspection only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "outputs/cofw_source_acquisition_v3/COFW_color.zip"
OUT = ROOT / "outputs/cofw_source_review_v1"
EXPECTED_SHA = "bc6a79bda1bd88705082af9ebdd31188cf5841917b09534b25067e424a0130e5"
MEMBERS = {"COFW_test_color.mat": 120945708, "COFW_train_color.mat": 388659959}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, data):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(data, indent=2) + "\n")


def extract():
    if OUT.exists() or sha(ARCHIVE) != EXPECTED_SHA:
        raise ValueError("Preserve existing source review and require verified archive")
    with zipfile.ZipFile(ARCHIVE) as source:
        entries = source.infolist()
        if len(entries) != 2 or {entry.filename: entry.file_size for entry in entries} != MEMBERS:
            raise ValueError("Unexpected COFW archive members or size")
        if any(stat.S_ISLNK(entry.external_attr >> 16) or entry.flag_bits & 1 for entry in entries):
            raise ValueError("Unsafe ZIP member")
        OUT.mkdir()
        records = []
        for entry in entries:
            target = OUT / entry.filename
            with source.open(entry) as src, target.open("xb") as dest:
                shutil.copyfileobj(src, dest, length=1024 * 1024)
            if target.stat().st_size != entry.file_size:
                raise ValueError("Extraction size differs")
            records.append({"file": entry.filename, "bytes": entry.file_size,
                            "sha256": sha(target), "archive_crc": entry.CRC})
    write(OUT / "extraction.json", {
        "format": "dgp-cofw-matrix-extraction-v1", "archive_sha256": EXPECTED_SHA,
        "files": records, "reader_sha256": sha(__file__), "training_admitted": False,
        "model_forwards": 0, "optimizer_updates": 0, "test_images_inspected": 0,
    })
    print(json.dumps({"extracted_matrices": 2, "training_admitted": False}), flush=True)


def inspect():
    # The isolated reader was installed during prior COFW research. It does not
    # modify the application's package environment or construct any ML models.
    sys.path.insert(0, str(ROOT / "outputs/cofw_read_dependencies"))
    import h5py
    import numpy as np
    extracted = json.loads((OUT / "extraction.json").read_text())
    expected = {row["file"]: row["sha256"] for row in extracted["files"]}
    info = {"format": "dgp-cofw-matrix-schema-v1", "h5py_version": h5py.__version__,
            "matrices": {}, "test_rgb_read": False, "training_admitted": False}
    for name in MEMBERS:
        path = OUT / name
        if sha(path) != expected[name]:
            raise ValueError("Matrix fingerprint differs")
        with h5py.File(path, "r") as file:
            keys = {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                          "attributes": {str(k): repr(v) for k, v in value.attrs.items()}}
                    for key, value in file.items() if isinstance(value, h5py.Dataset)}
            info["matrices"][name] = keys
            # Only train arrays are inspected semantically. Test RGB references
            # remain unread; shape metadata preserves publisher membership.
            if name == "COFW_train_color.mat":
                for key in file:
                    if key != "#refs#" and file[key].dtype != h5py.ref_dtype:
                        array = np.asarray(file[key])
                        print(json.dumps({"key": key, "shape": array.shape,
                                          "first_values": array.reshape(-1)[:12].tolist()}), flush=True)
                refs = np.asarray(file["IsTr"])
                sample = np.asarray(file[refs.reshape(-1)[0]])
                print(json.dumps({"first_train_image_shape": sample.shape,
                                  "first_train_image_dtype": str(sample.dtype)}), flush=True)
    write(OUT / "schema.json", info)
    print(json.dumps(info), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    if args.extract:
        extract()
    if args.inspect:
        inspect()


if __name__ == "__main__":
    main()
