"""Independently bind the small HQ source audit to official metadata and old roles."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def sha(path, kind="sha256"):
    h = hashlib.new(kind)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pixels(path):
    with Image.open(path) as image:
        return np.asarray(image).copy()


def audit(root, output, metadata, receipt):
    if receipt.exists():
        raise FileExistsError("Preserve the successful independent receipt")
    started = time.monotonic()
    protocol, manifest, results = read(root / "protocol.json"), read(output / "manifest.json"), read(output / "results.json")
    assert results["complete"] and results["training_ready"] is False
    assert sha(root / "protocol.json") == (root / "protocol.sha256").read_text().strip() == manifest["reference_protocol_sha256"]
    assert sha(output / "manifest.json") == results["manifest_sha256"]
    assert metadata.stat().st_size == 267793842 and sha(metadata, "md5") == "425ae20f06a4da1d4dc0f46d40ba5fd6"
    assert sha(metadata) == results["metadata_sha256"]
    expected = []
    for role, count in (("train", 10), ("validation", 6)):
        pool = sorted((x for x in protocol["references"] if x["source"] == "dataset/thumbnails128x128" and x["role"] == role), key=lambda x: int(Path(x["source_file"]).stem))
        indices = np.rint(np.linspace(0, len(pool) - 1, count)).astype(int)
        expected.extend(pool[i] for i in indices)
    assert manifest["references"] == expected and len({x["id"] for x in expected}) == 16
    assert len(results["counterparts"]) == 16
    full_metadata = read(metadata)
    selected = {ref["id"]: full_metadata[str(int(Path(ref["source_file"]).stem))] for ref in expected}
    del full_metadata
    checked, old_pixels, new_pixels, image_pixels, summaries = {}, [], [], [], []
    roles_by_photo, roles_by_pixels = {}, {}
    for ref, row in zip(expected, results["counterparts"]):
        assert row["reference_id"] == ref["id"] and row["original_role"] == ref["role"]
        original = selected[ref["id"]]
        assert row["image_spec"] == original["image"] and row["thumbnail_spec"] == original["thumbnail"]
        assert row["original_photo_spec"] == original["in_the_wild"] and row["metadata"] == original["metadata"]
        assert row["official_category"] == original["category"]
        assert ref["bounds"] == [0, 0, 256, 256]
        for name in (ref["native"], ref["target"]):
            assert sha(root / name) == protocol["assets_sha256"][name]
        low = pixels(root / ref["native"])
        assert low.shape == (128, 128, 3)
        assert hashlib.md5(low.tobytes()).hexdigest() == original["thumbnail"]["pixel_md5"] == row["native_thumbnail_pixel_md5"]
        source, target = output / row["image"], output / row["target"]
        assert source.resolve().is_relative_to(output.resolve()) and target.resolve().is_relative_to(output.resolve())
        assert source.stat().st_size == original["image"]["file_size"] and sha(source, "md5") == original["image"]["file_md5"]
        high = pixels(source)
        assert list(high.shape[:2][::-1]) == original["image"]["pixel_size"] == [1024, 1024]
        assert hashlib.md5(high.tobytes()).hexdigest() == original["image"]["pixel_md5"]
        assert sha(source) == row["image_sha256"] and sha(target) == row["target_sha256"]
        with Image.open(source) as image:
            rebuilt = np.asarray(image.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS))
        actual = pixels(target)
        assert actual.shape == (256, 256, 3) and np.array_equal(actual, rebuilt)
        old = pixels(root / ref["target"])
        old_pixels.append(old); new_pixels.append(actual); image_pixels.append(high)
        checked[row["image"]] = sha(source); checked[row["target"]] = sha(target)
        error = (old.astype(np.float64) - actual.astype(np.float64)) / 255
        quad = np.asarray(original["in_the_wild"].get("face_quad", []), dtype=np.float64)
        edges = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1) if quad.shape == (4, 2) else None
        summaries.append({"reference_id": ref["id"], "original_role": ref["role"], "official_category": original["category"], "old_vs_source256_MAE": float(np.abs(error).mean()), "old_vs_source256_MSE": float(np.square(error).mean()), "original_photo_pixel_size": original["in_the_wild"]["pixel_size"], "original_alignment_quad_min_max_edge_pixels": [float(edges.min()), float(edges.max())] if edges is not None else None})
        roles_by_photo.setdefault(original["metadata"]["photo_url"], set()).add(ref["role"])
        roles_by_pixels.setdefault(hashlib.sha256(high.tobytes()).hexdigest(), set()).add(ref["role"])
    assert {p.name for p in (output / "images1024").glob("*.png")} == {Path(x["source_file"]).stem + ".png" for x in expected}
    assert len(results["previews"]) == 4
    grid_cells = 0
    for page, info in enumerate(results["previews"]):
        path = output / info["path"]
        assert sha(path) == info["sha256"]
        grid = pixels(path)
        assert grid.shape == (1144, 768, 3)
        for local_row in range(4):
            index = page * 4 + local_row
            difference = np.abs(old_pixels[index].astype(np.int16) - new_pixels[index].astype(np.int16)).astype(np.uint8)
            for col, expected_cell in enumerate((old_pixels[index], new_pixels[index], difference)):
                actual = grid[local_row*286+27:local_row*286+283, col*256:(col+1)*256]
                assert np.array_equal(actual, expected_cell)
                grid_cells += 1
        checked[info["path"]] = sha(path)
    outcome = {"complete": True, "manifest_sha256": sha(output / "manifest.json"), "results_sha256": sha(output / "results.json"), "metadata_sha256": sha(metadata), "acquisition_source_sha256": sha(ROOT / "scripts/acquire_cctv_ffhq_counterparts_v1.py"), "audit_source_sha256": sha(Path(__file__)), "references_bound_to_official_metadata": 16, "thumbnail_pixels_checked": 16, "hq_file_and_pixel_checksums_checked": 16, "target_derivations_rebuilt": 16, "preview_cells_checked": grid_cells, "original_roles_preserved": True, "exact_image_overlap_across_sample_roles": any(len(v)>1 for v in roles_by_pixels.values()), "same_source_photo_across_sample_roles": any(len(v)>1 for v in roles_by_photo.values()), "full_identity_overlap_established": False, "source_comparison_summaries": summaries, "artifacts_checked_sha256": checked, "elapsed_seconds": time.monotonic()-started, "model_forwards": 0, "backward_calls": 0, "optimizer_updates": 0, "native_reserved_used": False, "model_improvement_established": False, "training_ready": False}
    with receipt.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(outcome, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({k:outcome[k] for k in ("complete", "references_bound_to_official_metadata", "target_derivations_rebuilt", "preview_cells_checked", "exact_image_overlap_across_sample_roles", "same_source_photo_across_sample_roles", "elapsed_seconds", "model_forwards", "optimizer_updates", "training_ready")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/cctv_dgp_vm_bundle_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/cctv_dgp_hq_counterparts_v1")
    parser.add_argument("--metadata", type=Path, default=ROOT / "outputs/cctv_dgp_hq_metadata_v1/ffhq-dataset-v2.json")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    audit(args.root, args.output, args.metadata, args.receipt or args.output / "local_independent_audit.json")
