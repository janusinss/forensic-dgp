"""Saved-output measurements and four-way sheets; no model, gradients or training."""
from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from render_restoration_checkpoint_comparison_v1 import metrics

ROOT = Path(__file__).resolve().parent
PARENT = ROOT / "outputs/cctv_dgp_checkpoint_comparison_v1"
OUT = PARENT / "cascade_v1"
ARMS = ["resize_control", "dgp_zamboanga_final", "codeformer_w1", "phase3_then_codeformer"]
VIEWS = ["raw", "native_png", "app_display"]
NULL = {"mse": None, "psnr_db": None, "ssim": None, "mae_255": None,
        "edge_mae": None, "exact_match": None}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rgb(path):
    with Image.open(path) as im:
        assert im.mode == "RGB" and im.size == (256, 256), str(path)
        return np.asarray(im).copy()


def unit(value):
    return value.astype(np.float32) / np.float32(255)


def output_values(folder, support, inp, receipt_row=None):
    from face_color_policy import preserve_input_palette
    paths = {name: folder / name for name in ["raw.npy", "native.png", "display.png"]}
    hashes = {name: sha(path) for name, path in paths.items()}
    if receipt_row is not None:
        assert hashes == receipt_row["files"], str(folder)
    raw = np.load(paths["raw.npy"], allow_pickle=False)
    assert raw.dtype == np.float32 and raw.shape == (256, 256, 3)
    assert np.isfinite(raw).all() and 0 <= raw.min() <= raw.max() <= 1
    native, display = rgb(paths["native.png"]), rgb(paths["display.png"])
    expected = np.floor(raw * np.float32(255)).astype(np.uint8)
    expected[~support] = inp[~support]
    assert np.array_equal(expected, native), (str(folder), "native conversion")
    expected, palette = preserve_input_palette(inp, np.zeros(support.shape, np.uint8), native, visible_restored=True)
    expected[~support] = inp[~support]
    assert np.array_equal(expected, display), (str(folder), "display conversion")
    if receipt_row is not None:
        assert palette == receipt_row["palette"]
    return [("raw", raw), ("native_png", unit(native)), ("app_display", unit(display))], hashes


def measure(plan):
    started = time.monotonic()
    original = read(PARENT / "plan.json")
    assert plan["cases"] == original["cases"]
    assert sha(PARENT / "plan.json") == plan["parent_plan_sha256"]
    for relative, digest in {**plan["parent_bindings"], **plan["bindings"]}.items():
        assert sha(ROOT / relative) == digest, relative
    receipt = read(OUT / "receipt.json")
    assert receipt["complete"] and receipt["plan_sha256"] == sha(OUT / "plan.json")
    assert receipt["state_before_sha256"] == receipt["state_after_sha256"]
    assert receipt["new_CodeFormer_calls"] == 88
    assert receipt["new_DGP_calls"] == receipt["optimizer_updates"] == receipt["gradient_queries"] == 0
    cascade_rows = {row["id"]: row for row in receipt["rows"]}
    assert len(cascade_rows) == len(receipt["rows"]) == 88
    assert [row["id"] for row in receipt["rows"]] == plan["execution_order"]
    prior_metrics = read(PARENT / "metrics.json")
    prior_rows = {(row["arm"], row["id"], row["view"]): row for row in prior_metrics["rows"]}
    prior_receipts = {}
    for arm in ARMS[1:3]:
        prior = read(PARENT / "results" / arm / "receipt.json")
        assert prior["complete"] and prior["plan_sha256"] == plan["parent_plan_sha256"]
        assert prior["state_before_sha256"] == prior["state_after_sha256"]
        prior_receipts[arm] = {row["id"]: row for row in prior["rows"]}
    rows, output_hashes, count = [], [], Counter()
    cases = {case["id"]: case for case in plan["cases"]}
    assert len(cases) == len(plan["execution_order"]) == 88
    for cid in plan["execution_order"]:
        case = cases[cid]
        inp = rgb(PARENT / case["input"])
        assert hashlib.sha256(inp.tobytes()).hexdigest() == case["input_pixel_sha256"]
        with Image.open(PARENT / case["observed"]) as im:
            observed = np.asarray(im).copy()
        assert observed.shape == (256, 256) and set(np.unique(observed)).issubset({0, 255})
        support = observed > 0
        assert int(support.sum()) == case["observed_pixels"]
        target = None
        if case["evidence"] == "synthetic_paired_development":
            target_pixels = rgb(PARENT / case["target"])
            assert hashlib.sha256(target_pixels.tobytes()).hexdigest() == case["target_pixel_sha256"]
            target = unit(target_pixels)
            count["paired_clear" if case["profile"] == "clear" else "paired_degraded"] += 1
        else:
            assert case["evidence"] == "native_unpaired" and "target" not in case
            assert not case["clean_reference_available"]
            count[case["source"]] += 1
        cr = cascade_rows[cid]
        intermediate = PARENT / "results/dgp_zamboanga_final" / cid / "native.png"
        assert sha(intermediate) == cr["intermediate_file_sha256"]
        assert hashlib.sha256(rgb(intermediate).tobytes()).hexdigest() == cr["intermediate_rgb_sha256"]
        for arm in ARMS:
            folder = (OUT if arm == "phase3_then_codeformer" else PARENT) / "results" / arm / cid
            rr = cr if arm == "phase3_then_codeformer" else prior_receipts.get(arm, {}).get(cid)
            values, hashes = output_values(folder, support, inp, rr)
            output_hashes.append({"arm": arm, "id": cid, "files": hashes})
            if arm == "resize_control":
                assert np.array_equal(values[0][1], unit(inp))
            seconds = cr["estimated_combined_forward_seconds"] if arm == "phase3_then_codeformer" else prior_rows[(arm, cid, "raw")]["forward_seconds"]
            for view, value in values:
                difference = (value - unit(inp))[support]
                paired = metrics(value, target, support) if target is not None else NULL.copy()
                result = {"arm": arm, "id": cid, "source": case["source"], "evidence": case["evidence"],
                          "profile": case["profile"], "view": view, "forward_seconds": seconds,
                          "input_change_mae_255": float(np.abs(difference).mean() * 255),
                          "mean_input_luma_change_255": float((difference @ np.array([.299, .587, .114])).mean() * 255),
                          **paired}
                if arm != "phase3_then_codeformer":
                    assert result == prior_rows[(arm, cid, view)], (arm, cid, view)
                rows.append(result)
        if len(output_hashes) % 40 == 0:
            print(json.dumps({"saved_cases_measured": len(output_hashes) // 4, "of": 88, "seconds": round(time.monotonic() - started, 2)}), flush=True)
    assert dict(count) == {"paired_degraded": 32, "QMUL-SurvFace": 24, "ChokePoint": 24, "paired_clear": 8}
    assert len(rows) == 1056 and len(output_hashes) == 352
    assert sum(row["arm"] == "phase3_then_codeformer" for row in rows) == 264
    native_rows = [row for row in rows if row["evidence"] == "native_unpaired"]
    assert len(native_rows) == 576 and all(all(row[key] is None for key in NULL) for row in native_rows)
    groups = []
    controls = {(row["id"], row["view"]): row for row in rows if row["arm"] == "resize_control"}
    for arm in ARMS:
        for view in VIEWS:
            for source in ["all_paired", "dataset/asian_faces", "dataset/thumbnails128x128"]:
                selected = [row for row in rows if row["arm"] == arm and row["view"] == view and row["evidence"] == "synthetic_paired_development" and (source == "all_paired" or row["source"] == source)]
                degraded = [row for row in selected if row["profile"] != "clear"]
                clear = [row for row in selected if row["profile"] == "clear"]
                group = {"arm": arm, "view": view, "source": source, "degraded_cases": len(degraded), "clear_cases": len(clear),
                         "mean_degraded_psnr_db": float(np.mean([row["psnr_db"] for row in degraded])),
                         "mean_degraded_ssim": float(np.mean([row["ssim"] for row in degraded])),
                         "mean_degraded_edge_mae": float(np.mean([row["edge_mae"] for row in degraded])),
                         "mean_clear_mae_255": float(np.mean([row["mae_255"] for row in clear])),
                         "degraded_cases_ssim_above_resize": sum(row["ssim"] > controls[(row["id"], view)]["ssim"] for row in degraded),
                         "mean_forward_seconds": float(np.mean([row["forward_seconds"] for row in selected]))}
                if arm != "phase3_then_codeformer":
                    previous = next(g for g in prior_metrics["groups"] if all(g[key] == group[key] for key in ["arm", "view", "source"]))
                    for key, value in previous.items():
                        if isinstance(value, float):
                            assert np.isclose(group[key], value, rtol=0, atol=1e-14), (arm, view, source, key)
                        else:
                            assert group[key] == value
                groups.append(group)
    write(OUT / "metrics.json", {"rows": rows, "groups": groups,
                               "method": "Exact parent metrics function: masked RGB PSNR; Gaussian SSIM on 11px-eroded observed support; Sobel gradient MAE. All references and support are ORIGINAL. Clear controls separate; native target scores null.",
                               "historical_one_percent_structure_gate_recomputed": False,
                               "new_neural_calls": 0, "optimizer_updates": 0, "gradient_queries": 0})
    with (OUT / "metrics.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write(OUT / "measurement_hash_audit.json", {"complete": True, "source_sha256": sha(__file__),
                                               "plan_sha256": sha(OUT / "plan.json"), "receipt_sha256": sha(OUT / "receipt.json"),
                                               "metric_function_source_sha256": sha(ROOT / "render_restoration_checkpoint_comparison_v1.py"),
                                               "metric_rows": 1056, "new_cascade_rows": 264, "retained_parent_rows_exact": 792,
                                               "saved_artifacts_verified": 1056, "output_hashes": output_hashes,
                                               "case_counts": dict(count), "original_inputs_support_targets_verified": True,
                                               "native_target_metrics_null": True, "neural_calls": 0, "optimizer_updates": 0,
                                               "gradient_queries": 0, "final_pixels_read": 0, "model_selected": False,
                                               "seconds": time.monotonic() - started})
    return rows, groups


def deltas(rows):
    lookup = {(row["arm"], row["id"], row["view"]): row for row in rows}
    fields = ["mse", "psnr_db", "ssim", "mae_255", "edge_mae", "input_change_mae_255", "mean_input_luma_change_255", "forward_seconds"]
    case_deltas = []
    for row in rows:
        if row["arm"] != "phase3_then_codeformer":
            continue
        direct = lookup[("codeformer_w1", row["id"], row["view"])]
        case_deltas.append({key: row[key] for key in ["id", "view", "source", "profile", "evidence"]} | {
            "cascade_minus_direct": {key: None if row[key] is None else row[key] - direct[key] for key in fields}})
    grouped = []
    for view in VIEWS:
        for source in ["all_paired", "dataset/asian_faces", "dataset/thumbnails128x128"]:
            selected = [row for row in rows if row["arm"] == "phase3_then_codeformer" and row["view"] == view and row["evidence"] == "synthetic_paired_development" and (source == "all_paired" or row["source"] == source)]
            for profile in ["all_degraded", "clear", "blur_lr24", "lowlight_lr32", "motion_lr48", "compound_lr24"]:
                sample = [row for row in selected if (row["profile"] != "clear" if profile == "all_degraded" else row["profile"] == profile)]
                if not sample:
                    continue
                entry = {"source": source, "view": view, "profile": profile, "cases": len(sample)}
                for key in fields:
                    values = [row[key] for row in sample]
                    previous = [lookup[("codeformer_w1", row["id"], view)][key] for row in sample]
                    entry[key] = {"cascade_mean": float(np.mean(values)), "direct_mean": float(np.mean(previous)),
                                  "cascade_minus_direct_mean": float(np.mean(np.subtract(values, previous))),
                                  "cascade_numerically_above_direct": sum(a > b for a, b in zip(values, previous)),
                                  "cascade_numerically_below_direct": sum(a < b for a, b in zip(values, previous)),
                                  "equal": sum(a == b for a, b in zip(values, previous))}
                grouped.append(entry)
    native = []
    for source in ["QMUL-SurvFace", "ChokePoint"]:
        for view in VIEWS:
            sample = [row for row in case_deltas if row["source"] == source and row["view"] == view]
            native.append({"source": source, "view": view, "cases": len(sample),
                           "mean_input_change_mae_255_cascade_minus_direct": float(np.mean([row["cascade_minus_direct"]["input_change_mae_255"] for row in sample])),
                           "mean_luma_change_255_cascade_minus_direct": float(np.mean([row["cascade_minus_direct"]["mean_input_luma_change_255"] for row in sample])),
                           "target_psnr_db": None, "target_ssim": None})
    write(OUT / "cascade_vs_direct_summary.json", {"case_deltas": case_deltas, "paired_source_profile_groups": grouped,
                                                  "native_input_change_only": native,
                                                  "delta_direction": "cascade minus direct CodeFormer; numerical above/below counts are not quality decisions",
                                                  "native_input_change_is_not_a_quality_score": True, "model_selected": False})


def font(size):
    return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)


def sheets(plan):
    directory = OUT / "sheets"
    directory.mkdir()
    cases = {case["id"]: case for case in plan["cases"]}
    columns = ["Input / resize", "Phase 3 final", "CodeFormer direct w=1", "Phase 3 -> CodeFormer w=1"]
    row_h, header, col_w = 310, 104, 272
    rendered = []
    for start in range(0, 88, 4):
        ids = plan["execution_order"][start:start + 4]
        block = [cases[cid] for cid in ids]
        paired = any("target" in case for case in block)
        im = Image.new("RGB", (col_w * (4 + int(paired)), header + row_h * len(block)), "#f6f3ed")
        draw = ImageDraw.Draw(im)
        draw.text((8, 7), f"Phase 3 PNG -> CodeFormer | Sheet {start // 4 + 1:02d} / 22 | native PNG at actual 256px", fill="#152b2f", font=font(19))
        draw.text((8, 34), "Common original input and support. No sharpening. Native CCTV has no clean reference. User chooses the model.", fill="#253738", font=font(14))
        for col, title in enumerate(columns + (["Original clean reference"] if paired else [])):
            draw.text((col * col_w + 8, 72), title, fill="#152b2f", font=font(16))
        for number, case in enumerate(block):
            yy = header + number * row_h
            caption = f"{case['id']} | {case['source']} | {case['profile']} | {case['evidence']}"
            draw.text((8, yy), caption, fill="#152b2f", font=font(14))
            files = [PARENT / case["input"], PARENT / "results/dgp_zamboanga_final" / case["id"] / "native.png",
                     PARENT / "results/codeformer_w1" / case["id"] / "native.png",
                     OUT / "results/phase3_then_codeformer" / case["id"] / "native.png"]
            if "target" in case:
                files.append(PARENT / case["target"])
            for col, path in enumerate(files):
                tile = Image.fromarray(rgb(path))
                assert tile.size == (256, 256)
                im.paste(tile, (col * col_w + 8, yy + 30))
            if paired and "target" not in case:
                draw.text((4 * col_w + 12, yy + 80), "Unpaired native CCTV", fill="#4c5d5e", font=font(16))
                draw.text((4 * col_w + 12, yy + 109), "No clean target", fill="#4c5d5e", font=font(16))
        path = directory / f"cascade-{start // 4 + 1:02d}.png"
        im.save(path)
        rendered.append({"path": path.relative_to(OUT).as_posix(), "sha256": sha(path), "cases": ids,
                         "columns": columns + (["Original clean reference (paired only)"] if paired else []),
                         "view": "native_png", "image_cell_pixels": 256, "resized_for_sheet": False,
                         "width": im.width, "height": im.height})
    assert len(rendered) == 22 and [cid for sheet in rendered for cid in sheet["cases"]] == plan["execution_order"]
    write(OUT / "visual_sheet_manifest.json", {"sheets": rendered, "cases": 88,
                                               "execution_order_matches_frozen_plan": True,
                                               "first_sheet_contains_four_user_examples": True,
                                               "actually_viewed": False})


def main():
    cv2.setNumThreads(1)
    plan = read(OUT / "plan.json")
    rows, groups = measure(plan)
    deltas(rows)
    sheets(plan)
    print(json.dumps({"complete": True, "metric_rows": len(rows), "groups": len(groups),
                      "sheets": 22, "new_neural_calls": 0, "model_selected": False}), flush=True)


if __name__ == "__main__":
    main()
