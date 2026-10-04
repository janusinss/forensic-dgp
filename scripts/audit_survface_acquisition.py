"""Independently audit native CCTV release structure and copied verification images."""
from collections import Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import time
from zipfile import ZipFile

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "dataset/cctv_survface_raw/QMUL-SurvFace-v1.zip"
PIN = "2fbb0876bc4761217c6de5576905e2524b8ca50ad7905720a5b4b378a0ff8e13"
OUT = ROOT / "outputs/cctv_survface_structure_audit_v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def names(array):
    result = []
    for cell in array.reshape(-1):
        require(isinstance(cell, np.ndarray) and cell.size == 1 and cell.dtype.kind in "US",
                "Unexpected MATLAB filename-cell structure")
        name = str(cell.item())
        require(PurePosixPath(name).name == name and name.endswith(".jpg"), "Invalid release image basename")
        result.append(name)
    return result


def person(name):
    prefix = name.split("_", 1)[0]
    require(prefix.isdecimal(), "Missing global person prefix")
    return int(prefix)


def main():
    require(not OUT.exists(), "Preserve previous release-structure audit")
    require(sha(ARCHIVE) == PIN, "Public archive differs from observed acquisition")
    acquisition_path = ROOT / "outputs/cctv_survface_acquisition_v1/acquisition.json"
    acquired = json.loads(acquisition_path.read_text(encoding="utf-8"))
    require(acquired["complete"] is True and acquired["archive_sha256"] == PIN and acquired["training"] is False,
            "Acquisition receipt does not bind this archive")
    started = time.monotonic()
    train_ids, test_ids, canonical, verification = set(), set(), {}, {}
    counts, metadata_hashes = Counter(), {}
    with ZipFile(ARCHIVE) as archive:
        for info in archive.infolist():
            if not info.filename.endswith(".jpg"):
                continue
            path = PurePosixPath(info.filename)
            name = path.name
            require(path.parts[0] == "QMUL-SurvFace", "Unexpected archive root")
            if path.parts[1] == "Face_Verification_Test_Set":
                require(path.parts[2] == "verification_images" and name not in verification,
                        "Unexpected duplicate verification name")
                verification[name] = info.filename
                counts["verification_copy_images"] += 1
                continue
            require(name not in canonical, "Canonical filename collides across train/test roles")
            canonical[name] = info.filename
            if path.parts[1] == "training_set":
                pid = person(name)
                require(int(path.parts[2]) == pid, "Training folder/person prefix mismatch")
                train_ids.add(pid)
                counts["training_images"] += 1
            else:
                require(path.parts[1] == "Face_Identification_Test_Set" and path.parts[2] in ("gallery", "mated_probe", "unmated_probe"),
                        "Unknown canonical test role")
                if name.startswith("distractors_"):
                    require(path.parts[2] == "unmated_probe" and name[12:-4].isdecimal(),
                            "Unexpected unlabeled distractor filename")
                    counts["unlabeled_distractor_images"] += 1
                else:
                    test_ids.add(person(name))
                counts[path.parts[2] + "_images"] += 1
        require(not train_ids & test_ids, "Published release labeled train/test person IDs overlap")
        metadata = {}
        for info in archive.infolist():
            if info.filename.endswith((".mat", ".txt", ".m")):
                raw = archive.read(info)  # ZipFile checks the member CRC here.
                metadata_hashes[info.filename] = hashlib.sha256(raw).hexdigest()
                if info.filename.endswith(".mat"):
                    metadata[info.filename] = loadmat(BytesIO(raw))
        label_map, role_checks = {}, []
        for role, name_key, id_key, filename in (
            ("gallery", "gallery_set", "gallery_ids", "gallery_img_ID_pairs.mat"),
            ("mated_probe", "mated_probe_set", "mated_probe_ids", "mated_probe_img_ID_pairs.mat"),
        ):
            data = metadata[f"QMUL-SurvFace/Face_Identification_Test_Set/{filename}"]
            image_names, ids = names(data[name_key]), data[id_key].reshape(-1)
            require(len(image_names) == len(ids) and np.isfinite(ids).all() and np.all(ids == np.floor(ids)),
                    "Identification filename/label lengths or types differ")
            require(len(image_names) == len(set(image_names)), "Duplicate identification metadata filename")
            actual = {name for name, path in canonical.items() if f"/Face_Identification_Test_Set/{role}/" in path}
            require(set(image_names) == actual, "Identification metadata and archive role membership differ")
            role_persons = set()
            for name, label in zip(image_names, ids):
                pid, label = person(name), int(label)
                require(label > 0 and (pid not in label_map or label_map[pid] == label), "Person/recognition-label mapping differs")
                label_map[pid] = label
                role_persons.add(pid)
            role_checks.append({"role": role, "images": len(image_names), "person_ids": len(role_persons),
                                "metadata_membership_exact": True})
        require(len(set(label_map.values())) == len(label_map), "Identification labels merge distinct global persons")
        pair_checks = []
        for key, expected_match in (("positive_pairs_names", True), ("negative_pairs_names", False)):
            data = metadata[f"QMUL-SurvFace/Face_Verification_Test_Set/{key}.mat"][key]
            require(data.shape == (5320, 2), "Verification pair shape differs from downloaded README")
            first, second = names(data[:, 0]), names(data[:, 1])
            require(all(name in verification for name in first + second), "Verification pair references missing image")
            mismatches = sum((person(a) == person(b)) != expected_match for a, b in zip(first, second))
            require(mismatches == 0, "Positive/negative pair global person labels conflict")
            pair_checks.append({"type": key, "pairs": len(first), "person_label_conflicts": mismatches})
        copied_digest = hashlib.sha256()
        for name, copy in sorted(verification.items()):
            require(time.monotonic() - started < 120, "Finite duplicate/metadata audit cap exceeded")
            require(name in canonical and person(name) in test_ids, "Verification copy not in canonical test data")
            copied, original = archive.read(copy), archive.read(canonical[name])
            require(copied == original, "Verification image differs from its canonical test copy")
            copied_digest.update((name + "\0" + hashlib.sha256(copied).hexdigest() + "\n").encode("ascii"))
    observed_test = sum(counts[role + "_images"] for role in ("gallery", "mated_probe", "unmated_probe"))
    report = {
        "date": "2026-10-03", "complete": True, "archive_sha256": PIN,
        "acquisition_receipt_sha256": sha(acquisition_path), "auditor_sha256": sha(Path(__file__)),
        "jpeg_entries": len(canonical) + len(verification), "canonical_filenames": len(canonical),
        "roles": dict(counts), "training_person_ids": len(train_ids), "labeled_test_person_ids": len(test_ids),
        "labeled_train_test_person_overlap": 0, "all_test_person_labels_available": False,
        "unlabeled_distractor_identity_overlap_known": False,
        "identification_metadata": role_checks, "verification_pairs": pair_checks,
        "verification_copy_images_byte_checked": len(verification),
        "verification_copy_registry_digest": copied_digest.hexdigest(), "metadata_sha256": metadata_hashes,
        "published_website_comparison": {"training_images": {"published": 220890, "observed": counts["training_images"],
                                        "difference": counts["training_images"] - 220890},
                                        "test_images": {"published": 242617, "observed": observed_test,
                                        "difference": observed_test - 242617},
                                        "all_canonical_images": {"published": 463507, "observed": len(canonical),
                                        "difference": len(canonical) - 463507}},
        "discrepancies_preserved": True, "claims_original_challenge_protocol_reproduction": False,
        "readme_identification_id_comparison": {"gallery_ids_claimed": 5319, "observed_from_mat": len(label_map),
                                               "difference": len(label_map) - 5319},
        "image_decode_quality_audited": False, "decoded_images": 0, "extracted_members": 0,
        "all_archive_members_crc_verified": False, "source_country_per_image_available_in_release_metadata": False,
        "original_dgp_pretraining_identity_overlap_known": False, "benchmark_subset_frozen": False,
        "training": False, "model_forwards": 0, "optimizer_updates": 0,
        "seconds": time.monotonic() - started,
        "next": "Select a source-native development subset and a separate fixed evaluation subset before model inference; preserve actual V1 counts and unknown country/ethnicity mapping",
    }
    OUT.mkdir()
    with (OUT / "verification.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"complete": True, "canonical_images": len(canonical), "copied_verification_images": len(verification),
                      "training_person_ids": len(train_ids), "labeled_test_person_ids": len(test_ids),
                      "labeled_train_test_person_overlap": 0, "unlabeled_distractor_images": counts["unlabeled_distractor_images"],
                      "published_count_differences": report["published_website_comparison"],
                      "verification_sha256": sha(OUT / "verification.json")}))


if __name__ == "__main__":
    main()
