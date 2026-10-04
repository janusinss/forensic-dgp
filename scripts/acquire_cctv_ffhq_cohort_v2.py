"""Acquire the pinned full HQ counterpart catalog for source-only review."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import ssl
import threading
import time
import uuid

import numpy as np
from PIL import Image, ImageDraw

from acquire_cctv_ffhq_counterparts_v1 import digest, fetch, verify_blob, write

ROOT = Path(__file__).resolve().parents[1]
PLAN_SHA = "ad65e35c70104b72a38dfe3cdac1f1027029445313f9d5530dbb18af7cde9501"
RUBRIC_SHA = "1e0aa441917280168a87f73509e1846b15ada15ee4d8b0e62ec35bf7cc6bad3e"


def run():
    plan_dir = ROOT / "outputs/cctv_dgp_hq_cohort_plan_v2"
    output = ROOT / "outputs/cctv_dgp_hq_cohort_v2"
    root = ROOT / "outputs/cctv_dgp_vm_bundle_v1"
    assert digest(plan_dir / "manifest.json") == PLAN_SHA
    assert digest(plan_dir / "source_review_rubric.json") == RUBRIC_SHA
    plan = json.loads((plan_dir / "manifest.json").read_text())
    sample = json.loads((ROOT / "outputs/cctv_dgp_hq_counterparts_v1/local_independent_audit.json").read_text())
    assert digest(ROOT / "scripts/acquire_cctv_ffhq_counterparts_v1.py") == sample["acquisition_source_sha256"]
    assert digest(ROOT / "scripts/prepare_cctv_ffhq_cohort_v2.py") == plan["preparation_source_sha256"]
    assert digest(root / "protocol.json") == plan["parent_reference_protocol_sha256"]
    assert digest(ROOT / "outputs/cctv_dgp_hq_metadata_v1/ffhq-dataset-v2.json") == plan["metadata_sha256"]
    assert len(plan["references"]) == 510 and plan["training_ready"] is False
    assert plan["expected_new_source_image_bytes"] < plan["acquisition_max_new_bytes"] == 1024**3
    assert plan["acquisition_runtime_cap_seconds"] == 1800
    if (output / "results.json").exists():
        raise FileExistsError("Preserve completed cohort acquisition")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + 1800
    ca = ROOT / "scratch/gcloud-windows-roots-v1.pem"
    context = ssl.create_default_context(cafile=str(ca))
    stop = threading.Event()
    execution_name = "execution_" + uuid.uuid4().hex + ".json"
    write(output / execution_name, {"plan_sha256": PLAN_SHA, "rubric_sha256": RUBRIC_SHA, "execution_source_sha256": digest(Path(__file__)), "shared_acquisition_source_sha256": sample["acquisition_source_sha256"], "runtime_cap_seconds": 1800, "max_new_bytes": 1024**3, "workers": 4, "model_forwards": 0, "optimizer_updates": 0})

    def acquire(entry):
        if stop.is_set() or time.monotonic() >= deadline:
            raise TimeoutError("Acquisition stopped or time cap expired")
        ref = entry["reference"]
        if "reuse_verified_source" in entry:
            source = ROOT / entry["reuse_verified_source"]
            assert source.resolve().is_relative_to(ROOT.resolve())
            assert digest(source) == entry["reuse_source_sha256"]
            verify_blob(source, entry["image_spec"])
            transfer = {"reused_verified_file": True, "bytes": source.stat().st_size}
        else:
            source = output / "images1024" / (Path(ref["source_file"]).stem + ".png")
            # Per-file size is pinned (<=4MiB). No retries; a first failure stops
            # new work. At most four active file requests can finish thereafter.
            transfer = fetch(source, entry["image_spec"], context, deadline, [0])
        return ref["id"], source, transfer

    try:
        acquired = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            pending = [pool.submit(acquire, entry) for entry in plan["references"]]
            for future in as_completed(pending):
                try:
                    identifier, source, transfer = future.result()
                except Exception:
                    stop.set()
                    for remaining in pending:
                        remaining.cancel()
                    raise
                acquired[identifier] = (source, transfer)
                if len(acquired) % 20 == 0 or len(acquired) == 510:
                    print(f"HQ counterparts {len(acquired)}/510 verified; elapsed {time.monotonic()-started:.0f}s", flush=True)
        new_bytes = sum(transfer["bytes"] for _, transfer in acquired.values() if not transfer["reused_verified_file"])
        assert new_bytes <= plan["expected_new_source_image_bytes"] < 1024**3
        rows = []
        for entry in plan["references"]:
            if time.monotonic() >= deadline:
                raise TimeoutError("Time cap expired before catalog completion")
            ref = entry["reference"]
            source, transfer = acquired[ref["id"]]
            with Image.open(source) as image:
                target = image.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
            target_path = output / "targets256" / (ref["id"] + ".png")
            if target_path.exists():
                with Image.open(target_path) as existing:
                    assert np.array_equal(np.asarray(existing), np.asarray(target))
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target.save(target_path)
            rows.append({"reference_id": ref["id"], "original_role": ref["role"], "source": "dataset/thumbnails128x128", "image_path_from_workspace": str(source.relative_to(ROOT)).replace("\\", "/"), "image_sha256": digest(source), "target": str(target_path.relative_to(output)).replace("\\", "/"), "target_sha256": digest(target_path), "source_review_status": entry["source_review_status"], "transfer": transfer})
        previews = []
        # Source-only pages: 16 true-source targets each, labels preserve roles.
        for start in range(0, 510, 16):
            if time.monotonic() >= deadline:
                raise TimeoutError("Time cap expired before review pages completed")
            sheet = Image.new("RGB", (1024, 4*284), "#eeeeee")
            draw = ImageDraw.Draw(sheet)
            for local, row in enumerate(rows[start:start+16]):
                x, y = (local % 4)*256, (local // 4)*284
                draw.text((x+3, y+2), row["reference_id"] + " / " + row["original_role"], fill="black")
                with Image.open(output / row["target"]) as image:
                    sheet.paste(image, (x, y+26))
            name = f"source_review_{start//16+1:02d}.png"
            if (output / name).exists():
                raise FileExistsError("Preserve an existing review page")
            sheet.save(output / name)
            previews.append({"path": name, "sha256": digest(output / name), "reference_ids": [r["reference_id"] for r in rows[start:start+16]]})
        result = {"complete": True, "plan_sha256": PLAN_SHA, "rubric_sha256": RUBRIC_SHA, "execution": execution_name, "references": rows, "counterparts": 510, "original_role_counts": plan["original_role_counts"], "new_download_bytes": new_bytes, "elapsed_seconds": time.monotonic()-started, "previews": previews, "source_reviews_pending": 494, "independent_audit_pending": True, "source_review_complete": False, "training_ready": False, "model_forwards": 0, "backward_calls": 0, "optimizer_updates": 0, "native_reserved_used": False}
        write(output / "results.json", result)
        print(json.dumps({"complete": True, "counterparts": 510, "new_download_bytes": new_bytes, "elapsed_seconds": result["elapsed_seconds"], "source_review_pages": len(previews), "source_reviews_pending": 494, "training_ready": False}))
    except Exception as error:
        stop.set()
        write(output / ("failure_" + uuid.uuid4().hex + ".json"), {"complete": False, "error_type": type(error).__name__, "error": str(error), "execution": execution_name, "completed_file_transfers": len(locals().get("acquired", {})), "partial_files_preserved": True, "model_forwards": 0, "optimizer_updates": 0})
        raise


if __name__ == "__main__":
    run()
