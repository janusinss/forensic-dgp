"""Read-only accepted-source verification and prospective TRAIN cohort selection.

Only the two named preparation JSON files are created, exclusively. No Torch,
neural inference, image generation, training, VM access or source mutation.
"""
from __future__ import annotations

import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import stat
import time

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "outputs/codeformer_filter_train_v2_preparation"
CATALOG = "outputs/cctv_codeformer_adaptation_v2_vm/references.json"
SUPPORT = (
    "outputs/codeformer_training_readiness_v1/data_review/hq_train_candidates_v1.json",
    "outputs/codeformer_training_readiness_v1/data_review/ffhq_sources_with_roles_v1.json",
    "outputs/codeformer_training_readiness_v1/data_review/inventory_audit_v2.json",
    "outputs/codeformer_training_readiness_v1/data_review/DATA_READINESS_REVIEW_V1.md",
    "outputs/codeformer_filter_pairs_v1_preparation/cohort_selection_v1.json",
    "outputs/codeformer_filter_pairs_v1_preparation/input_only_review_criteria_v1.json",
    "outputs/codeformer_filter_pairs_v1_preparation/root_input_only_review_v1.json",
    "outputs/cctv_codeformer_role_preserving_training_v1_r2_vm/data_bundle.json",
)
EXPECTED = {
    CATALOG: "55b892ee5d55a123ea6cb55f84f322824faaec956f59168e443ff94b93337cde",
    SUPPORT[0]: "7af5f313743361ae435629171c9e92e2a4fbcf119bdcac8221338a6fd457a493",
    SUPPORT[1]: "4b77dc04c3da7520caf2e2fca5836727d9183e539d4fdc1817e06df50ab7b73c",
    SUPPORT[2]: "ea4c71204b50cc48fee532459ec46b4117f926a6e81024562a7b601c62259e2c",
    SUPPORT[3]: "4c82f6d3c8111bb4c9a9fc90e7f1b80a6b435deea07a10f7ca9966b08b3f5a98",
    SUPPORT[4]: "f1e0fa95dd2140a70f5ac091ecf83d09f51ac80e6bc1eabd8acb9223594ced9e",
    SUPPORT[5]: "fd0ae5aa624f74775a65474957f42cf3cc69e0838febac4f1feb9d063340f903",
    SUPPORT[6]: "51bff1593cd33ad2b369a7c20d2b5282ed67521df4b62e17a53f082941fc3b50",
}
SELECT_PREFIX = "codeformer-filter-fusion-train-admission-v1|"
RECT_PREFIX = "codeformer-filter-train-v2/rectangle/"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pin(path: Path) -> dict:
    path = path.absolute()
    resolved = path.resolve(strict=True)
    assert resolved.is_relative_to(ROOT), str(path)
    s = path.lstat()
    assert stat.S_ISREG(s.st_mode) and not path.is_symlink(), str(path)
    assert not getattr(s, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    assert s.st_nlink == 1, str(path)
    data = path.read_bytes()
    return {"path": path.relative_to(ROOT).as_posix(), "absolute_path": str(path),
            "bytes": len(data), "sha256": sha(data), "regular_file": True,
            "symlink_or_reparse": False, "hardlink_count": s.st_nlink}


def arrays(rgb: np.ndarray) -> dict:
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    value = np.ascontiguousarray(rgb.astype(np.float32) / np.float32(255))
    def entry(a: np.ndarray) -> dict:
        h = sha(a.tobytes(order="C"))
        return {"shape": list(a.shape), "dtype": str(a.dtype),
                "layout": "HWC RGB C-contiguous", "pixel_sha256": h,
                "array_sha256": h}
    return {"rgb8": entry(rgb), "float32": entry(value)}


def selection(train: list[dict]) -> tuple[list[dict], list[dict]]:
    groups = collections.defaultdict(list)
    for r in train:
        groups[(r["attribution"]["license_url"], r["official_category"])].append(r)
    assert len(train) == 391 and len(groups) == 9
    extra = 64 - len(groups)
    denominator = len(train) - len(groups)
    quotas = {k: 1 + extra * (len(v) - 1) // denominator for k, v in groups.items()}
    priority = sorted(groups, key=lambda k: (-(extra * (len(groups[k]) - 1) % denominator), k))
    for key in priority[:64 - sum(quotas.values())]:
        quotas[key] += 1
    selected, strata = [], []
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda r: (
            sha((SELECT_PREFIX + r["id"] + "|" + r["source_sha256"]).encode("utf-8")), r["id"]))
        selected.extend(ordered[:quotas[key]])
        strata.append({"license_url": key[0], "official_category": key[1],
                       "population": len(groups[key]), "selected": quotas[key]})
    selected.sort(key=lambda r: r["id"])
    assert len(selected) == len({r["id"] for r in selected}) == 64
    return selected, strata


def write_exclusive(path: Path, value: dict) -> dict:
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    assert path.read_bytes() == data
    return pin(path)


def main() -> None:
    start = time.monotonic()
    PREP.mkdir(parents=True, exist_ok=True)
    audit_path, cohort_path = PREP / "source_audit_v1.json", PREP / "cohort_selection_v1.json"
    assert not audit_path.exists() and not cohort_path.exists(), "Immutable preparation outputs already exist"
    inputs = {name: pin(ROOT / name) for name in (CATALOG, *SUPPORT)}
    for name, expected in EXPECTED.items():
        assert inputs[name]["sha256"] == expected, name
    catalog = json.loads((ROOT / CATALOG).read_bytes())
    refs = catalog["references"]
    candidates = json.loads((ROOT / SUPPORT[0]).read_bytes())["candidates"]
    roles = json.loads((ROOT / SUPPORT[1]).read_bytes())["sources"]
    r2 = json.loads((ROOT / SUPPORT[7]).read_bytes())
    by_candidate, by_role = ({r["id"]: r for r in rows} for rows in (candidates, roles))
    train = [r for r in refs if r["historical_role"] == "train"]
    dev = [r for r in refs if r["historical_role"] == "validation"]
    assert len(refs) == 444 and len(train) == 391 and len(dev) == 53 and len(roles) == 510
    assert {r["id"] for r in train} == set(by_candidate)
    selected, strata = selection(train)
    selected_ids = {r["id"] for r in selected}
    assert sha(("\n".join(sorted(selected_ids)) + "\n").encode()) == "44c81522f7624c87adc1aeb75e48624f7046bbcc5b5e7da8f70003d42354a761"
    rows, chosen, counters = [], {}, collections.defaultdict(collections.Counter)
    namespaces = {role: {key: set() for key in ("source_bytes", "decoded_pixels", "photo_url")}
                  for role in ("train", "validation")}
    for count, ref in enumerate(refs, 1):
        role, rid = ref["historical_role"], ref["id"]
        assert ref["decision"] == "accept_clean_restoration" and not ref["use_for_optimization"]
        prior = by_role[rid]
        for key in ("historical_role", "decision", "source_sha256", "existing_target_sha256", "attribution"):
            assert prior[key] == ref[key], (rid, key)
        source = pin(ROOT / ref["original_source_path"])
        target = pin(ROOT / ref["original_existing_target_path"])
        assert source["sha256"] == ref["source_sha256"] and target["sha256"] == ref["existing_target_sha256"]
        if role == "train":
            for key in ("source_sha256", "existing_target_sha256", "attribution"):
                assert by_candidate[rid][key] == ref[key]
        with Image.open(ROOT / source["path"]) as im, Image.open(ROOT / target["path"]) as target_im:
            assert im.mode == target_im.mode == "RGB" and im.size == (1024, 1024) and target_im.size == (256, 256)
            source_rgb = np.asarray(im)
            target_rgb = np.asarray(target_im)
            assert np.array_equal(np.asarray(im.resize((256, 256), Image.Resampling.LANCZOS)), target_rgb)
            clean = np.asarray(im.resize((512, 512), Image.Resampling.LANCZOS))
            clear = np.asarray(im.resize((256, 256), Image.Resampling.BILINEAR))
        source.update(mode="RGB", size=[1024, 1024], **arrays(source_rgb))
        target.update(mode="RGB", size=[256, 256], **arrays(target_rgb))
        clean_proof, clear_proof = arrays(clean), arrays(clear)
        assert clean_proof["rgb8"]["pixel_sha256"] == ref["proposed_512_rgb_sha256"]
        counters[role].update(sources=1, source_bytes=source["bytes"], target_bytes=target["bytes"])
        namespaces[role]["source_bytes"].add(source["sha256"])
        namespaces[role]["decoded_pixels"].add(source["rgb8"]["pixel_sha256"])
        namespaces[role]["photo_url"].add(ref["attribution"]["photo_url"])
        row = {"reference_id": rid, "historical_role": role, "source1024": source,
               "target256": target, "clean512": clean_proof, "clear256": clear_proof,
               "target256_direct_LANCZOS_exact": True, "clean512_catalog_fingerprint_exact": True,
               "source_reviewed_at_original_resolution": ref["source_reviewed_at_original_resolution"],
               "use_for_optimization": False}
        rows.append(row)
        if rid in selected_ids:
            digest = hashlib.sha256((RECT_PREFIX + source["sha256"]).encode("ascii")).digest()
            x, y = (digest[0] % 9) * 16, (digest[1] % 9) * 16
            chosen[rid] = {**row, "split": "TRAIN", "reference_manifest_row": ref,
                           "attribution": ref["attribution"], "official_category": ref["official_category"],
                           "local_rectangle_xyxy_half_open": [x, y, x + 128, y + 128],
                           "local_rectangle_is_corruption_footprint_not_face_or_removal_mask": True,
                           "input_review_decision": "PENDING", "optimization_admission": "PENDING_PROTOCOL_AND_INPUT_REVIEW"}
        if count % 100 == 0:
            print(json.dumps({"checked": count, "of": 444, "seconds": time.monotonic() - start}), flush=True)
    overlaps = {key: sorted(namespaces["train"][key] & namespaces["validation"][key])
                for key in namespaces["train"]}
    assert all(not overlap for overlap in overlaps.values())
    assert len(namespaces["train"]["photo_url"]) == 391 and len(namespaces["validation"]["photo_url"]) == 53
    # Guard the incoming manifests against concurrent replacement during verification.
    assert all(pin(ROOT / name) == proof for name, proof in inputs.items())
    r2_ids = {r["reference_id"] for r in r2["hq"]}
    assert len(r2_ids) == 100 and len(r2_ids & selected_ids) == 22
    audit = {"format": "codeformer-filter-train-v2-fresh-source-audit-v1", "complete": True,
             "UTC": datetime.datetime.now(datetime.timezone.utc).isoformat(), "audit_source": pin(Path(__file__)),
             "inputs": inputs, "counts_bytes": {key: dict(value) for key, value in counters.items()},
             "TRAIN_DEV_overlap": overlaps, "source_rows": rows, "all444_geometry_and_target_derivations_exact": True,
             "official_prior_MD5_audit_retained_not_independently_repeated": inputs[SUPPORT[2]],
             "all_current_sources_hash_bound_to_prior_audit": True, "person_identity_disjointness_proven": False,
             "CodeFormer_pretraining_overlap": "possible/unknown", "native_CCTV_or_FINAL_pixels_opened": 0,
             "user_wild_sources_selected": 0, "new_images_written": 0, "neural_calls": 0,
             "optimizer_updates": 0, "training_authorized_by_audit": False, "seconds": time.monotonic() - start}
    audit_pin = write_exclusive(audit_path, audit)
    cohort = {"format": "codeformer-filter-train-v2-prospective-source-cohort-v1", "complete": True,
              "UTC": datetime.datetime.now(datetime.timezone.utc).isoformat(), "source_audit": audit_pin,
              "catalog": inputs[CATALOG], "supporting_audit_and_approval_manifests": [inputs[x] for x in SUPPORT],
              "catalog_counts": {"TRAIN": 391, "DEV": 53, "total": 444},
              "selected_counts": {"TRAIN": 64, "DEV": 0, "FINAL": 0, "native_CCTV": 0, "wild": 0},
              "selection_rule": {"population": "All391 historicalTRAIN accepted-clean source candidates; no DEV or excluded source",
                  "strata": ["attribution.license_url", "official_category"], "nonempty_strata": 9,
                  "reserve_per_stratum": 1, "remaining_slots": 55,
                  "remaining_weight": "population_in_stratum minus1; denominator382",
                  "allocation": "integer floor(55*(n-1)/382), then largest integer remainders; ties lexicographic(license_url,official_category)",
                  "within_stratum_order": "ascending SHA256 UTF8('codeformer-filter-fusion-train-admission-v1|'+id+'|'+source_sha256), then id",
                  "global_cohort_order": "ascending reference_id",
                  "image_model_output_or_new_quality_based_selection": False,
                  "selected_ids_newline_UTF8_trailing_LF_sha256": "44c81522f7624c87adc1aeb75e48624f7046bbcc5b5e7da8f70003d42354a761"},
              "strata": strata, "cohort": [chosen[rid] for rid in sorted(chosen)],
              "selected_source_bytes": sum(chosen[rid]["source1024"]["bytes"] for rid in chosen),
              "selected_target_bytes": sum(chosen[rid]["target256"]["bytes"] for rid in chosen),
              "overlap_with_R2_100_TRAIN_sources": 22, "new_sources_relative_to_R2_100": 42,
              "rectangle_rule": {"SHA256_ASCII_prefix": RECT_PREFIX, "suffix": "source1024.sha256 lowercasehex",
                  "x": "digestbyte0 modulo9 times16", "y": "digestbyte1 modulo9 times16",
                  "width": 128, "height": 128, "coordinate_system": "256x256 half-open XYXY",
                  "anatomical_or_detector_mask": False},
              "terms": {"per_photo": "Exact original author/title/photoURL/licenseURL retained for all64; retain change notices",
                  "release": "Retain FFHQ CC BY-NC-SA4.0 dataset metadata/documentation and noncommercial notices; recognition-use restriction",
                  "notice_source": inputs[SUPPORT[4]], "CodeFormer_license": "Retain upstream license independently with any training packet",
                  "blanket_commercial_portrait_or_public_distribution_clearance": False},
              "limits": {"bounded_first_pilot_cohort_not_complete_training_population": True,
                  "all391_population_retained": True, "source_eligibility_is_not_input_admission": True,
                  "all_optimization_still_false": True, "input_review_pending": True,
                  "person_identity_disjointness_proven": False, "official_category_does_not_override_historical_role": True,
                  "no_native_CCTV_Zamboanga_ethnicity_or_recovered_identity_claim": True,
                  "all512_inputs_must_be_input_only_reviewed_before_any_optimization_admission": True,
                  "existing_DEV_controls_preserved": True, "automatic_quality_threshold_calibrated": False,
                  "new_training_or_model": False}}
    cohort_pin = write_exclusive(cohort_path, cohort)
    print(json.dumps({"complete": True, "source_audit": audit_pin, "cohort": cohort_pin,
                      "seconds": time.monotonic() - start, "NN_calls": 0, "new_images": 0}, indent=2))


if __name__ == "__main__":
    main()
