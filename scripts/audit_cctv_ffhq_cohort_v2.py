"""Audit all acquired HQ sources/targets/preview cells without model inference."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PLAN_SHA = "ad65e35c70104b72a38dfe3cdac1f1027029445313f9d5530dbb18af7cde9501"
RUBRIC_SHA = "1e0aa441917280168a87f73509e1846b15ada15ee4d8b0e62ec35bf7cc6bad3e"


def sha(path, kind="sha256"):
    h = hashlib.new(kind)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pixels(path):
    with Image.open(path) as image:
        return np.asarray(image).copy()


def run():
    started = time.monotonic()
    out = ROOT / "outputs/cctv_dgp_hq_cohort_v2"
    plan_dir = ROOT / "outputs/cctv_dgp_hq_cohort_plan_v2"
    root = ROOT / "outputs/cctv_dgp_vm_bundle_v1"
    receipt = out / "local_independent_audit.json"
    if receipt.exists():
        raise FileExistsError("Preserve the successful independent receipt")
    assert sha(plan_dir / "manifest.json") == PLAN_SHA
    assert sha(plan_dir / "source_review_rubric.json") == RUBRIC_SHA
    plan, result, parent = read(plan_dir / "manifest.json"), read(out / "results.json"), read(root / "protocol.json")
    assert result["complete"] and result["plan_sha256"] == PLAN_SHA and result["rubric_sha256"] == RUBRIC_SHA
    assert result["training_ready"] is False and result["model_forwards"] == result["backward_calls"] == result["optimizer_updates"] == 0
    assert result["native_reserved_used"] is False
    assert sha(root / "protocol.json") == plan["parent_reference_protocol_sha256"]
    expected = [r for r in parent["references"] if r["source"] == "dataset/thumbnails128x128"]
    assert [r["reference"] for r in plan["references"]] == expected
    assert len(expected) == len(result["references"]) == 510
    metadata_path = ROOT / "outputs/cctv_dgp_hq_metadata_v1/ffhq-dataset-v2.json"
    assert metadata_path.stat().st_size == 267793842 and sha(metadata_path, "md5") == "425ae20f06a4da1d4dc0f46d40ba5fd6"
    assert sha(metadata_path) == plan["metadata_sha256"]
    metadata = read(metadata_path)
    selected = {ref["id"]: metadata[str(int(Path(ref["source_file"]).stem))] for ref in expected}
    del metadata
    files, targets, photo_roles, pixel_roles, capture_geometry = {}, [], {}, {}, []
    total_source_bytes = 0
    for index, (ref, entry, row) in enumerate(zip(expected, plan["references"], result["references"])):
        original = selected[ref["id"]]
        assert row["reference_id"] == ref["id"] and row["original_role"] == ref["role"]
        assert row["source_review_status"] == entry["source_review_status"]
        assert entry["image_spec"] == original["image"] and entry["thumbnail_spec"] == original["thumbnail"]
        assert entry["attribution"] == original["metadata"] and entry["original_photo_spec"] == original["in_the_wild"]
        assert entry["official_category"] == original["category"]
        low = pixels(root / ref["native"])
        assert sha(root / ref["native"]) == parent["assets_sha256"][ref["native"]]
        assert hashlib.md5(low.tobytes()).hexdigest() == original["thumbnail"]["pixel_md5"] and low.shape == (128, 128, 3)
        source = ROOT / row["image_path_from_workspace"]
        target = out / row["target"]
        assert source.resolve().is_relative_to(ROOT.resolve()) and target.resolve().is_relative_to(out.resolve())
        assert source.stat().st_size == original["image"]["file_size"] and sha(source, "md5") == original["image"]["file_md5"]
        high = pixels(source)
        assert high.shape == (1024, 1024, 3) and hashlib.md5(high.tobytes()).hexdigest() == original["image"]["pixel_md5"]
        assert sha(source) == row["image_sha256"] and sha(target) == row["target_sha256"]
        with Image.open(source) as image:
            rebuilt = np.asarray(image.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS))
        assert np.array_equal(pixels(target), rebuilt)
        total_source_bytes += source.stat().st_size
        files[row["image_path_from_workspace"]] = sha(source)
        files[str(target.relative_to(ROOT)).replace("\\", "/")] = sha(target)
        targets.append(target)
        photo_roles.setdefault(original["metadata"]["photo_url"], []).append({"reference_id": ref["id"], "role": ref["role"]})
        pixel_roles.setdefault(hashlib.sha256(high.tobytes()).hexdigest(), []).append({"reference_id": ref["id"], "role": ref["role"]})
        quad = np.asarray(original["in_the_wild"].get("face_quad", []), dtype=np.float64)
        if quad.shape == (4, 2):
            edges = np.linalg.norm(quad-np.roll(quad, -1, axis=0), axis=1)
            capture_geometry.append({"reference_id": ref["id"], "original_photo_size": original["in_the_wild"]["pixel_size"], "alignment_quad_min_edge": float(edges.min())})
        if (index+1) % 100 == 0:
            print(f"Independent HQ audit {index+1}/510", flush=True)
    assert total_source_bytes == plan["expected_total_source_image_bytes"] == 695837473
    assert len(result["previews"]) == 32
    grid_cells = 0
    for page, info in enumerate(result["previews"]):
        path = out / info["path"]
        assert sha(path) == info["sha256"]
        grid = pixels(path)
        assert grid.shape == (1136, 1024, 3)
        slice_rows = result["references"][page*16:(page+1)*16]
        assert info["reference_ids"] == [r["reference_id"] for r in slice_rows]
        for local, row in enumerate(slice_rows):
            x, y = (local%4)*256, (local//4)*284+26
            assert np.array_equal(grid[y:y+256, x:x+256], pixels(out / row["target"]))
            grid_cells += 1
        files[str(path.relative_to(ROOT)).replace("\\", "/")] = sha(path)
    assert grid_cells == 510
    source_photo_overlaps = [rows for rows in photo_roles.values() if len({r["role"] for r in rows}) > 1]
    exact_image_overlaps = [rows for rows in pixel_roles.values() if len({r["role"] for r in rows}) > 1]
    outcome = {"complete": True, "plan_sha256": PLAN_SHA, "rubric_sha256": RUBRIC_SHA, "results_sha256": sha(out / "results.json"), "metadata_sha256": sha(metadata_path), "audit_source_sha256": sha(Path(__file__)), "thumbnail_pixels_checked": 510, "hq_file_and_pixel_checksums_checked": 510, "target_derivations_rebuilt": 510, "preview_cells_checked": grid_cells, "source_image_bytes_checked": total_source_bytes, "original_role_counts": {role: sum(r["role"] == role for r in expected) for role in ("train", "validation")}, "original_roles_preserved": True, "exact_image_overlaps_across_roles": exact_image_overlaps, "source_photo_overlaps_across_roles": source_photo_overlaps, "full_identity_overlap_established": False, "capture_geometry": capture_geometry, "artifacts_checked_sha256": files, "prior_failure_receipts_preserved": [p.name for p in out.glob("failure_*.json")], "elapsed_seconds": time.monotonic()-started, "source_review_complete": False, "source_reviews_pending": 494, "model_forwards": 0, "backward_calls": 0, "optimizer_updates": 0, "native_reserved_used": False, "model_improvement_established": False, "training_ready": False}
    with receipt.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(outcome, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({k: outcome[k] for k in ("complete", "thumbnail_pixels_checked", "hq_file_and_pixel_checksums_checked", "target_derivations_rebuilt", "preview_cells_checked", "source_image_bytes_checked", "exact_image_overlaps_across_roles", "source_photo_overlaps_across_roles", "elapsed_seconds", "model_forwards", "optimizer_updates", "training_ready")}))


if __name__ == "__main__":
    run()
