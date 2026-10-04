"""Independently reconstruct frozen V3 camera pixels; no model imports."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/cofw_camera_pairs_v2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def local(path):
    result = (ROOT / path).resolve()
    require(result.is_relative_to(ROOT) and result.is_file(), "Missing or non-local asset")
    return result


def rgb(path):
    with Image.open(path) as image:
        require(image.mode == "RGB" and image.size == (256, 256), "Fixed native RGB required")
        return np.asarray(image).copy()


def binary(path):
    with Image.open(path) as image:
        a = np.asarray(image).copy()
        require(image.mode == "L" and a.shape == (256, 256) and np.isin(a, [0, 255]).all(), "Fixed binary mask required")
        return a == 255


def main():
    destination = OUT / "independent_verification.json"
    require(not destination.exists(), "Preserve completed pair audit")
    manifest = OUT / "manifest.json"
    require(sha(manifest) == "b6f4a800427e96a52468a0ec10eec478ec8f5875e1302b35ffd6c072dfec041c",
            "Frozen paired-input manifest differs")
    data = json.loads(manifest.read_text())
    registry_path = local(data["registry"])
    require(sha(registry_path) == data["registry_sha256"] ==
            "abab07e152941c4ae24d8aea3755fa7aaedb18965b25f6d0d1b6a7e00094aadd", "Reviewed registry differs")
    require(sha(local(data["dataset_verification"])) == data["dataset_verification_sha256"] ==
            "a403f14457d7ba850ff2dbea4236b1db88276cd9e5d4f8192a65e0a0159261cc", "Dataset audit differs")
    require(sha(ROOT / "real_camera_pairs.py") == data["camera_helper_sha256"], "Frozen camera helper changed")
    require(sha(ROOT / "scripts/prepare_cofw_camera_pairs.py") == data["builder_sha256"], "Frozen builder changed")
    registry = json.loads(registry_path.read_text())
    rows = [r for r in registry["supported_records"] if r["split"] == "train"]
    require(len(rows) == 133 and len(data["cases"]) == 266, "Fixed case budget differs")
    old_path = ROOT / "outputs/real_camera_pairs_v1/manifest.json"
    require(sha(old_path) == "872b1e0d4cb75835892d4df2b6a17fd7dafde954a076724e10426ea0e71a047c", "Previous paired cache differs")
    old = json.loads(old_path.read_text())
    checked = []
    for index, row in enumerate(rows):
        source_id = row.get("source_id", f"previous_train_{index:03d}")
        native, degraded = data["cases"][2 * index:2 * index + 2]
        require([native["case_id"], degraded["case_id"]] == [2 * index, 2 * index + 1]
                and native["condition"] == "native" and degraded["condition"] == "degraded", "Case ordering differs")
        for case in (native, degraded):
            require(case["real_train_index"] == index and case["source_id"] == source_id
                    and case["source_sha256"] == row["source_sha256"] and case["group"] == row["group"]
                    and case["split"] == row["split"] == "train" and case["kind"] == row["kind"]
                    and case["family"] == row.get("occlusion_stratum", "previous_untagged")
                    and case["data_origin"] == row["data_origin"]
                    and case["target_support_or_geometry_changed"] is False, "Case membership/support differs")
            require(sha(local(case["input"])) == case["input_sha256"], "Case input changed")
        require(native["target"] == degraded["target"] and native["camera"] is None, "Pair target/metadata differs")
        for field in ("mask", "valid", "source_valid"):
            asset = native["target"][field]
            expected = (registry_path.parent / row[field]).resolve()
            require(local(asset["path"]) == expected and sha(expected) == asset["sha256"] == row[field + "_sha256"],
                    "Original label/support asset differs")
        require(local(native["input"]) == (registry_path.parent / row["image"]).resolve()
                and native["input_sha256"] == row["image_sha256"], "Native input differs")
        if index < 91:
            for case, prior in zip((native, degraded), old["cases"][2 * index:2 * index + 2]):
                require(case["input_sha256"] == prior["input_sha256"] and case["camera"] == prior["camera"],
                        "Previous source camera pixels/parameters changed")
        original, actual = rgb(local(native["input"])), rgb(local(degraded["input"]))
        target = binary(local(native["target"]["mask"]["path"]))
        valid = binary(local(native["target"]["valid"]["path"]))
        source = binary(local(native["target"]["source_valid"]["path"]))
        require(valid.any() and not (target & ~valid).any() and not (valid & ~source).any()
                and bool(target.any()) == (row["kind"] == "covered"), "Target/support semantics differ")
        require(np.all(original[~source] == 96) and np.all(actual[~source] == 96), "True padding changed")
        yy, xx = np.nonzero(source)
        x0, x1, y0, y1 = int(xx.min()), int(xx.max()) + 1, int(yy.min()), int(yy.max()) + 1
        require(source.sum() == (x1 - x0) * (y1 - y0), "Source support is not rectangular")
        seed = int.from_bytes(hashlib.sha256(("real-camera-v1:" + row["source_sha256"]).encode()).digest()[:4], "little")
        rng = np.random.default_rng(seed)
        sigma, low = float(rng.uniform(.6, 1.4)), int(rng.choice([128, 160, 192]))
        noise, quality = float(rng.uniform(1, 6)), int(rng.integers(65, 96))
        part = original[y0:y1, x0:x1]
        height, width = part.shape[:2]
        size = (max(1, round(width * low / 256)), max(1, round(height * low / 256)))
        expected_camera = {"seed": seed, "source_roi_xyxy": [x0, y0, x1, y1], "blur_sigma": sigma,
                           "blur_kernel": 9, "low_full_side": low, "low_roi_size": list(size),
                           "noise_sigma_255": noise, "jpeg_quality": quality,
                           "padding_in_filtered_source": False, "target_or_support_changed": False, "spatial_warp": False}
        require(degraded["camera"] == expected_camera, "Camera metadata differs")
        filtered = cv2.GaussianBlur(part, (9, 9), sigma, borderType=cv2.BORDER_REFLECT_101)
        resampled = cv2.resize(cv2.resize(filtered, size, interpolation=cv2.INTER_AREA),
                               (width, height), interpolation=cv2.INTER_LINEAR)
        noisy = np.clip(np.rint(resampled.astype(float) + rng.normal(0, noise, part.shape)), 0, 255).astype(np.uint8)
        ok, blob = cv2.imencode(".jpg", cv2.cvtColor(noisy, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        require(ok, "Independent JPEG encode failed")
        reconstructed = original.copy()
        reconstructed[y0:y1, x0:x1] = cv2.cvtColor(cv2.imdecode(blob, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        require(np.array_equal(actual, reconstructed) and np.any(actual[source] != original[source]),
                "Independent camera pixel reconstruction differs")
        checked.append({"real_train_index": index, "source_id": source_id,
                        "changed_source_pixels": int(np.any(actual != original, axis=2)[source].sum()),
                        "positive_pixels": int(target.sum()), "valid_pixels": int(valid.sum()),
                        "unknown_observed_pixels": int((source & ~valid).sum())})
    require({p.name for p in (OUT / "degraded").iterdir()} == {f"{i:03d}.png" for i in range(133)}, "Cache inventory differs")
    for preview in data["previews"]:
        require(sha(OUT / preview["path"]) == preview["sha256"], "Preview changed")
        require(all(source_id in {r["source_id"] for r in rows[91:]} for source_id in preview["source_ids"]), "Preview outside new training cohort")
    require(data["native_source_count"] == data["native_cases"] == data["degraded_cases"] == 133
            and data["cofw_sources"] == 42 and data["previous_sources"] == 91
            and data["held_out_inputs_in_cache"] == data["model_forwards"] == data["optimizer_updates"] == 0
            and data["training_recipe_ready"] is False, "Cache readiness declaration differs")
    report = {"format": "dgp-cofw-camera-pairs-independent-verification-v1", "date": "2026-10-03",
              "complete": True, "manifest_sha256": sha(manifest), "registry_sha256": sha(registry_path),
              "previous_camera_manifest_sha256": sha(old_path), "auditor_sha256": sha(__file__),
              "native_cases_verified": 133, "degraded_cases_verified": 133,
              "previous_degraded_views_exact": 91, "new_cofw_degraded_views": 42,
              "degradation_pixels_independently_reconstructed": True,
              "targets_support_source_groups_exactly_preserved": True, "held_out_inputs_in_cache": 0,
              "checks": checked, "model_forwards": 0, "optimizer_updates": 0,
              "trained_quality_verified": False, "training_recipe_ready": False}
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"complete": True, "cases": 266, "verification_sha256": sha(destination)}))


if __name__ == "__main__":
    main()
