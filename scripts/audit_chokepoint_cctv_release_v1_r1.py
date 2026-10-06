"""Reader r1: preserve v1 failure; admit only three fingerprinted non-image metadata files."""
from collections import Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import tarfile
import time
import xml.etree.ElementTree as ET

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset/cctv_chokepoint_raw_v1"
ACQ = ROOT / "outputs/cctv_chokepoint_acquisition_v1"
OUT = ROOT / "outputs/cctv_chokepoint_release_audit_v1_r1"
PARENT_BINDINGS = {'scripts/audit_chokepoint_cctv_release_v1.py': '03253ffa83244ef808175c9817b9dd80f01fcde618e600930741950f8e0d57fa', 'outputs/cctv_chokepoint_release_audit_v1/request.json': '8b6c11e10420a38eb63d01fcfabac56a6a083d5154f42731c468ede214eca7eb', 'outputs/cctv_chokepoint_release_audit_v1/failure.json': '337393969cf8406a8e4784a383c9e1c1c18e8fa0d439141578f326d821a86003', 'outputs/cctv_chokepoint_release_audit_v1/reader_format_diagnostic_r1.json': '4bd6adfd437f7957193a384bbac108438b25724f9e1f28bf6013895e9e08f6f4'}
DOLPHIN_METADATA = {'P1E_S1_C1': {'member': 'P1E_S1_C1/.directory', 'bytes': 50, 'sha256': '0dacbbadb0220784f2e6cd82facf8640daa9f2ca885b7fdf381f7347f7c40fc6'}, 'P1E_S1_C2': {'member': 'P1E_S1_C2/.directory', 'bytes': 48, 'sha256': '66fd2b14c8ee1cd282819358b7c7f887170f10024d1010158b562688f690f3d3'}, 'P1E_S1_C3': {'member': 'P1E_S1_C3/.directory', 'bytes': 50, 'sha256': 'edca4d7c47a8142b53f751d49f9b19dc60a452294f7a694610fe8bbca0584d12'}}
CAMERAS = ("P1E_S1_C1", "P1E_S1_C2", "P1E_S1_C3")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def require(value, message):
    if not value:
        raise ValueError(message)


def safe_member(member):
    path = PurePosixPath(member.name)
    require(not path.is_absolute() and ".." not in path.parts and ":" not in member.name
            and "\\" not in member.name, "Unsafe container path")
    require(member.isfile() or member.isdir(), "Unexpected container member type")


def main():
    require(not OUT.exists(), "Preserve completed/partial audit; no automatic repeat")
    for path, expected in PARENT_BINDINGS.items():
        require(sha(ROOT / path) == expected, "Preserved parent reader/failure binding")
    acquisition, request = read(ACQ / "acquisition.json"), read(ACQ / "request.json")
    require(acquisition["complete"] and acquisition["request_sha256"] == sha(ACQ / "request.json"), "Completed acquisition")
    OUT.mkdir()
    (OUT / "nested").mkdir()
    write(OUT / "request.json", {"date": "2026-10-05", "frozen_before_audit": True,
          "acquisition_sha256": sha(ACQ / "acquisition.json"), "auditor_sha256": sha(Path(__file__)),
          "cap_seconds": 240, "outer_timeout_seconds": 300, "maximum_container_members": 30000,
          "maximum_uncompressed_bytes": 2 * 1024 ** 3,
          "scope": "Independent transfer/container/label/header audit. No image pixels decoded or rendered, no models or training.",
          "native_source": "Original camera sequence contains three nested camera archives; publisher face crops normalized to96x96 were not acquired",
          "restoration_evidence": "Identity and eye-location labels are not aligned clean-face references; native restoration remains unpaired",
          "split_chosen": False, "original_reserved_32_used": False,
          "reader_revision": "r1: fingerprint the three observed Dolphin .directory files as non-image metadata; never execute",
          "preserved_parent_bindings": PARENT_BINDINGS, "known_metadata": DOLPHIN_METADATA})
    started = time.monotonic()
    try:
        publisher = read(ACQ / "publisher_record.json")
        entries = {file["key"]: file for file in publisher["files"]}
        verified = []
        for item in request["files"]:
            path = DATA / item["name"]
            require(path.stat().st_size == item["bytes"] == entries[item["name"]]["size"], "File size")
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "md5").hexdigest()
            require(digest == item["publisher_md5"] and entries[item["name"]]["checksum"] == "md5:" + digest, "Publisher MD5")
            fingerprint = sha(path)
            transferred = next(file for file in acquisition["files"] if file["name"] == item["name"])
            require(fingerprint == transferred["observed_sha256"], "Acquisition SHA256")
            require((DATA / (item["name"] + ".sha256")).read_bytes() ==
                    (fingerprint + "  " + item["name"] + "\n").encode("ascii"), "LF sidecar")
            verified.append({"name": item["name"], "bytes": item["bytes"], "publisher_md5": digest, "sha256": fingerprint})
        require(sha(DATA / "LICENSE_SOURCE.html") == acquisition["license_source_sha256"], "Original license notice")
        (OUT / "LICENSE_SOURCE.html").write_bytes((DATA / "LICENSE_SOURCE.html").read_bytes())
        annotation_summary, selected_annotations, annotation_ids = [], {}, {}
        with tarfile.open(DATA / "groundtruth.tar.xz", "r:xz") as archive:
            members = archive.getmembers()
            require(len(members) <= 200, "Annotation member cap")
            for member in members:
                safe_member(member)
                if member.isdir():
                    continue
                require(member.name.endswith(".xml") and member.size <= 2 * 1024 ** 2, "Annotation type/size")
                raw = archive.extractfile(member).read()
                require(b"<!ENTITY" not in raw.upper(), "XML entity declarations are not supported")
                xml = ET.fromstring(raw)
                require(xml.tag == "dataset", "Dataset XML root")
                name, frames = PurePosixPath(member.name).stem, xml.findall("frame")
                declared_name = xml.attrib["name"]
                # Portal2 .1/.2 files represent separate recording dates, while
                # their XML roots omit that suffix. Preserve the member namespace.
                require(declared_name == name.split(".", 1)[0], "Annotation member/root relation")
                require(name not in annotation_ids, "Duplicate annotation member namespace")
                frame_numbers, person_ids, records, invalid_eyes = set(), set(), [], 0
                for frame in frames:
                    number = frame.attrib["number"]
                    require(number.isdecimal() and number not in frame_numbers, "Unique numeric frame labels")
                    frame_numbers.add(number)
                    for person in frame.findall("person"):
                        pid = person.attrib["id"]
                        require(pid.isdecimal(), "Numeric publisher person label")
                        person_ids.add(pid)
                        left, right = person.find("leftEye"), person.find("rightEye")
                        valid = left is not None and right is not None
                        eyes = []
                        if valid:
                            eyes = [[int(eye.attrib["x"]), int(eye.attrib["y"])] for eye in (left, right)]
                            valid = all(0 <= x < 800 and 0 <= y < 600 for x, y in eyes) and eyes[0] != eyes[1]
                        if not valid:
                            invalid_eyes += 1
                        records.append({"frame": number, "person_id": pid, "eyes": eyes, "valid_eyes": valid})
                annotation_ids[name] = person_ids
                annotation_summary.append({"member": member.name, "sha256": hashlib.sha256(raw).hexdigest(),
                    "sequence": name, "declared_dataset_name": declared_name,
                    "frames": len(frames), "person_rows": len(records),
                    "person_ids": sorted(person_ids), "invalid_eye_rows": invalid_eyes})
                if name in CAMERAS:
                    selected_annotations[name] = records
        require(set(selected_annotations) == set(CAMERAS), "All acquired camera annotations present")
        write(OUT / "selected_annotations.json", selected_annotations)
        nested = {}
        with tarfile.open(DATA / "P1E_S1.tar.xz", "r|xz") as archive:
            for member in archive:
                safe_member(member)
                require(member.isfile() and member.name in [name + ".tar.xz" for name in CAMERAS]
                        and member.name not in nested, "Exactly three expected nested camera archives")
                require(member.size <= 512 * 1024 ** 2, "Nested archive cap")
                target = OUT / "nested" / member.name
                with archive.extractfile(member) as source, target.open("xb") as stream:
                    for block in iter(lambda: source.read(1024 ** 2), b""):
                        require(time.monotonic() - started <= 240, "Audit wall cap")
                        stream.write(block)
                require(target.stat().st_size == member.size, "Nested archive copy size")
                nested[member.name] = {"path": target.relative_to(ROOT).as_posix(), "bytes": member.size, "sha256": sha(target)}
        require(len(nested) == 3, "Nested archive count")
        inventory, camera_summary = [], []
        total_bytes = 0
        for camera in CAMERAS:
            resolutions, formats, frame_numbers, metadata_files = Counter(), Counter(), set(), []
            path = ROOT / nested[camera + ".tar.xz"]["path"]
            with tarfile.open(path, "r|xz") as archive:
                for member in archive:
                    safe_member(member)
                    if member.isdir():
                        continue
                    require(time.monotonic() - started <= 240 and len(inventory) < 30000, "Finite audit bounds")
                    total_bytes += member.size
                    require(total_bytes <= 2 * 1024 ** 3 and member.size <= 4 * 1024 ** 2, "Uncompressed byte/member cap")
                    raw = archive.extractfile(member).read()
                    filename = PurePosixPath(member.name)
                    item = {"camera": camera, "member": member.name, "bytes": member.size,
                            "sha256": hashlib.sha256(raw).hexdigest()}
                    if filename.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        require(filename.stem.isdecimal(), "Numeric native frame filename")
                        frame_number = filename.stem.zfill(8)
                        require(frame_number not in frame_numbers, "Duplicate native frame number")
                        frame_numbers.add(frame_number)
                        with Image.open(BytesIO(raw)) as image:
                            width, height, detected = image.width, image.height, image.format
                        require((width, height) == (800, 600) and detected in ("JPEG", "PNG"), "Published camera dimensions/header")
                        resolutions[f"{width}x{height}"] += 1
                        formats[detected] += 1
                        item.update({"frame": frame_number, "width": width, "height": height, "format": detected})
                    elif filename.name == ".directory":
                        known = DOLPHIN_METADATA[camera]
                        require(member.name == known["member"] and member.size == known["bytes"]
                                and item["sha256"] == known["sha256"], "Observed Dolphin metadata fingerprint")
                        item["payload_type"] = "non-image Dolphin directory configuration; never executed"
                        metadata_files.append(member.name)
                    else:
                        require(filename.suffix == ".txt", "Unexpected native container payload")
                        metadata_files.append(member.name)
                    inventory.append(item)
            missing = sorted({row["frame"] for row in selected_annotations[camera]} - frame_numbers)
            require(not missing, "Annotated native frame missing")
            camera_summary.append({"camera": camera, "native_frames": len(frame_numbers),
                 "header_resolutions": dict(resolutions), "header_formats": dict(formats),
                 "person_ids": sorted(annotation_ids[camera]), "missing_annotated_frames": missing,
                 "metadata_files": metadata_files})
            print(json.dumps({"audited_camera": camera, "native_frames": len(frame_numbers),
                              "seconds": time.monotonic() - started}), flush=True)
        write(OUT / "frame_inventory.json", inventory)
        write(OUT / "annotation_inventory.json", annotation_summary)
        require(all(annotation_ids[camera] == annotation_ids[CAMERAS[0]] for camera in CAMERAS), "Camera person sets differ")
        seconds = time.monotonic() - started
        require(seconds <= 240, "Audit completion cap")
        write(OUT / "verification.json", {"complete": True, "date": "2026-10-05", "seconds": seconds,
              "request_sha256": sha(OUT / "request.json"), "verified_files": verified, "nested_archives": nested,
              "cameras": camera_summary, "native_frame_headers_audited": sum(c["native_frames"] for c in camera_summary),
              "annotation_sequences_audited": len(annotation_summary), "acquired_sequence_persons": len(annotation_ids[CAMERAS[0]]),
              "artifacts_sha256": {p.name: sha(p) for p in OUT.iterdir() if p.is_file()},
              "image_pixels_decoded": 0, "images_rendered": 0, "model_forwards": 0, "training_calls": 0,
              "restoration_paired_ground_truth": False, "source_country_per_image": None, "ethnicity_inferred": False,
              "release_level_source": "Australian NICTA release; capture site/country not established by these acquired metadata",
              "cross_dataset_person_overlap": "Unknown; namespace source IDs. ChokePoint is absent from QMUL's published17-source table, which is not proof of identity disjointness.",
              "reserved_qmul_32_used": False, "restoration_split_frozen": False,
              "independent_final_review": False, "restoration_qualified": False,
              "next": "Freeze source-specific development/reserved person IDs and native eye-based crop rule before input review or any model comparison"})
        print(json.dumps({"complete": True, "seconds": seconds, "persons": len(annotation_ids[CAMERAS[0]])}), flush=True)
    except Exception as error:
        write(OUT / "failure.json", {"complete": False, "error": str(error), "type": type(error).__name__,
              "seconds": time.monotonic() - started, "partial_artifacts_preserved": True,
              "model_forwards": 0, "training_calls": 0})
        raise


if __name__ == "__main__":
    main()
