"""Reconstruct the four output PNGs from captured stages without loading models."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/real_camera_completion_review_v1"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rgb(path):
    with Image.open(path) as image:
        assert image.mode == "RGB" and image.size == (256, 256)
        return np.asarray(image).copy()


def main():
    destination = RUN / "independent_verification.json"
    assert not destination.exists(), "Preserve previous independent verification"
    assert sha(RUN / "frozen_protocol.json") == "b8f682ffa6af62a7746026ab8458398a235de476f0e6261838819e3c8c49f825"
    assert sha(RUN / "results.json") == "122fe6f12385e367355b3bc81a6fb63b7bbbe373b8b3a79b6a6837354ccfde64"
    p, result = read(RUN / "frozen_protocol.json"), read(RUN / "results.json")
    assert result["complete"] is True and result["state_before"] == result["state_after"]
    assert result["generation_requests"] == 4 and result["forwards"] == {"completion": 4, "restoration": 4}
    assert result["detector_forwards"] == result["optimizer_updates"] == 0 and result["promoted"] is False
    for name, pin in p["assets_sha256"].items():
        assert sha(ROOT / name) == pin
    expected_rows = {(c["id"], a) for c in p["cases"] for a in p["arms"]}
    assert len(result["rows"]) == 4 and {(r["id"], r["arm"]) for r in result["rows"]} == expected_rows
    files = {"preview.png"}
    for row in result["rows"]:
        files.update([row["output"], *row["stage_arrays"].values()])
    assert set(result["artifacts_sha256"]) == files and len(files) == 13
    for name, pin in result["artifacts_sha256"].items():
        assert sha(RUN / name) == pin
    verified = []
    for row in result["rows"]:
        case = next(c for c in p["cases"] if c["id"] == row["id"])
        original, output = rgb(ROOT / case["input"]), rgb(RUN / row["output"])
        assert row["mask"] == case["masks"][row["arm"]]
        with Image.open(ROOT / row["mask"]) as image:
            assert image.mode == "L"
            mask_bytes = np.asarray(image).copy()
        assert mask_bytes.shape == (256, 256) and np.isin(mask_bytes, [0, 255]).all()
        mask = mask_bytes == 255
        completed, restored = (np.load(RUN / row["stage_arrays"][k], allow_pickle=False) for k in ("completion", "restoration"))
        assert all(a.dtype == np.float32 and a.shape == (1, 3, 256, 256) and np.isfinite(a).all() for a in (completed, restored))
        observed = original.transpose(2, 0, 1)[None].astype(np.float32) / 255
        selected = mask[None, None]
        completed = np.where(selected, completed, observed)
        blended = np.where(selected, completed, .5 * restored + .5 * completed)
        assert np.array_equal(blended[selected.repeat(3, axis=1)], completed[selected.repeat(3, axis=1)])
        quantized = (np.clip(blended[0].transpose(1, 2, 0), 0, 1) * 255).round().astype(np.uint8)
        support = cv2.erode((~mask).astype(np.uint8), np.ones((9, 9), np.uint8)) != 0
        filtered = cv2.GaussianBlur(original.astype(np.float32), (5, 5), 1)
        chroma = filtered.max(-1) - filtered.min(-1)
        signal = float(chroma[support].mean())
        grayscale = int(support.sum()) >= 512 and signal <= 4
        if grayscale:
            quantized = np.repeat(cv2.cvtColor(quantized, cv2.COLOR_RGB2GRAY)[..., None], 3, -1)
        assert np.array_equal(quantized, output), "Captured stages do not reconstruct final PNG exactly"
        m = row["metadata"]
        assert m["policy"] == "reviewed-face-workflow-v2" and m["restoration_requested"] == "auto"
        assert m["restoration_applied"] is True and m["additional_mask_expansion"] == m["optimizer_updates"] == 0
        assert m["original_rgb_sha256"] == hashlib.sha256(original.tobytes()).hexdigest()
        assert m["removal_mask_sha256"] == hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest()
        assert m["output_rgb_sha256"] == hashlib.sha256(output.tobytes()).hexdigest()
        assert m["color_policy"]["grayscale_input"] == grayscale and m["color_policy"]["mean_visible_channel_range_255"] == signal
        verified.append({"id": row["id"], "arm": row["arm"], "final_png_reconstructed_exactly": True,
                         "completed_region_unchanged_by_float_restoration": True,
                         "grayscale_projection": grayscale, "hidden_ground_truth": None})
    record = {"format": "dgp-real-camera-downstream-independent-verification-v1", "date": "2026-10-03",
              "complete": True, "auditor_sha256": sha(__file__), "files_verified": 13, "rows": verified,
              "model_forwards_in_this_audit": 0, "optimizer_updates": 0, "promoted": False,
              "limitations": ["Four estimates on two exposed development inputs; not whole-family readiness",
                              "Plausibility requires visual review; no hidden facial ground truth"]}
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"complete": True, "pngs_reconstructed": 4, "verification_sha256": sha(destination)}))


if __name__ == "__main__":
    main()
