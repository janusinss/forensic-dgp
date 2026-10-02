"""Verify frozen real-input pairs, original targets and crop-safe degradation pixels."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/real_camera_pairs_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).copy()


def binary(path):
    with Image.open(path) as image:
        a = np.asarray(image)
    assert a.shape == (256, 256) and a.dtype == np.uint8 and np.isin(a, [0, 255]).all()
    return a == 255


def main():
    destination = OUT / "independent_verification.json"
    if destination.exists():
        raise ValueError("Preserve completed pair audit")
    manifest = OUT / "manifest.json"
    assert sha(manifest) == "872b1e0d4cb75835892d4df2b6a17fd7dafde954a076724e10426ea0e71a047c"
    data = json.loads(manifest.read_text())
    registry_path = ROOT / data["registry"]
    assert sha(registry_path) == data["registry_sha256"]
    registry = json.loads(registry_path.read_text())
    rows = [r for r in registry["supported_records"] if r["split"] == "train"]
    assert len(rows) == 91 and len(data["cases"]) == 182
    assert sha(ROOT / "real_camera_pairs.py") == data["camera_helper_sha256"]
    checked = []
    for i, row in enumerate(rows):
        native, degraded = data["cases"][2 * i:2 * i + 2]
        assert [native["case_id"], degraded["case_id"]] == [2 * i, 2 * i + 1]
        assert native["condition"] == "native" and degraded["condition"] == "degraded"
        assert native["real_train_index"] == degraded["real_train_index"] == i
        assert native["group"] == degraded["group"] == row["group"]
        assert native["source_sha256"] == degraded["source_sha256"] == row["source_sha256"]
        assert native["split"] == degraded["split"] == row["split"] == "train"
        assert native["target"] == degraded["target"]
        for field in ("mask", "valid", "source_valid"):
            asset = native["target"][field]
            expected_path = registry_path.parent / row[field]
            assert (ROOT / asset["path"]).resolve() == expected_path.resolve()
            assert sha(expected_path) == row[field + "_sha256"] == asset["sha256"]
        assert (ROOT / native["input"]).resolve() == (registry_path.parent / row["image"]).resolve()
        assert sha(ROOT / native["input"]) == row["image_sha256"] == native["input_sha256"]
        assert sha(ROOT / degraded["input"]) == degraded["input_sha256"]
        original, actual = rgb(ROOT / native["input"]), rgb(ROOT / degraded["input"])
        target = binary(ROOT / native["target"]["mask"]["path"])
        valid = binary(ROOT / native["target"]["valid"]["path"])
        source = binary(ROOT / native["target"]["source_valid"]["path"])
        assert not (target & ~valid).any() and not (valid & ~source).any()
        assert np.all(original[~source] == 96) and np.all(actual[~source] == 96)
        yy, xx = np.nonzero(source)
        x0, x1, y0, y1 = int(xx.min()), int(xx.max()) + 1, int(yy.min()), int(yy.max()) + 1
        assert source.sum() == (x1 - x0) * (y1 - y0)
        camera = degraded["camera"]
        seed = int.from_bytes(hashlib.sha256(("real-camera-v1:" + row["source_sha256"]).encode()).digest()[:4], "little")
        assert seed == camera["seed"] and camera["source_roi_xyxy"] == [x0, y0, x1, y1]
        rng = np.random.default_rng(seed)
        sigma = float(rng.uniform(.6, 1.4))
        low = int(rng.choice([128, 160, 192]))
        noise = float(rng.uniform(1, 6))
        quality = int(rng.integers(65, 96))
        assert (sigma, low, noise, quality) == (camera["blur_sigma"], camera["low_full_side"], camera["noise_sigma_255"], camera["jpeg_quality"])
        part = original[y0:y1, x0:x1]
        h, w = part.shape[:2]
        size = (round(w * low / 256), round(h * low / 256))
        filtered = cv2.GaussianBlur(part, (9, 9), sigma, borderType=cv2.BORDER_REFLECT_101)
        resampled = cv2.resize(cv2.resize(filtered, size, interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
        noisy = np.clip(np.rint(resampled.astype(float) + rng.normal(0, noise, part.shape)), 0, 255).astype(np.uint8)
        ok, blob = cv2.imencode(".jpg", cv2.cvtColor(noisy, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        assert ok
        reconstructed = original.copy()
        reconstructed[y0:y1, x0:x1] = cv2.cvtColor(cv2.imdecode(blob, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        assert np.array_equal(actual, reconstructed)
        assert np.any(actual[source] != original[source])
        assert native["target_support_or_geometry_changed"] is False and degraded["target_support_or_geometry_changed"] is False
        checked.append({"real_train_index": i, "changed_source_pixels": int(np.any(actual != original, axis=2)[source].sum()),
                        "positive_pixels": int(target.sum()), "valid_pixels": int(valid.sum()),
                        "unknown_observed_pixels": int((source & ~valid).sum())})
    assert sha(OUT / data["preview"]["path"]) == data["preview"]["sha256"]
    assert {p.name for p in (OUT / "degraded").iterdir()} == {f"{i:03d}.png" for i in range(91)}
    report = {"format": "dgp-real-camera-pairs-independent-verification-v1", "date": "2026-10-02",
              "complete": True, "manifest_sha256": sha(manifest), "registry_sha256": sha(registry_path),
              "auditor_sha256": sha(__file__), "native_cases_verified": 91, "degraded_cases_verified": 91,
              "degradation_pixels_independently_reconstructed": True,
              "targets_support_and_source_group_exactly_preserved": True, "held_out_inputs_in_cache": 0,
              "checks": checked, "model_forwards": 0, "optimizer_updates": 0,
              "trained_quality_verified": False, "training_recipe_ready": False}
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"complete": True, "cases": 182, "verification_sha256": sha(destination)}))


if __name__ == "__main__":
    main()
