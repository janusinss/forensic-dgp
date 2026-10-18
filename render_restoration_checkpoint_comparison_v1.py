"""Audit saved comparison pixels and publish a local review gallery; no networks."""
from __future__ import annotations

import csv
import ast
import hashlib
import html
import json
import math
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/cctv_dgp_checkpoint_comparison_v1"
SHORT = {
    "resize_control": "Resize only", "dgp_epoch_1": "Epoch 1", "dgp_epoch_17": "Epoch 17 (old)",
    "dgp_epoch_5": "Epoch 5 (old)", "dgp_improved_epoch_10": "Improved 10",
    "dgp_improved_epoch_19": "Improved 19", "dgp_improved_epoch_2": "Improved 2",
    "dgp_improved_epoch_20": "Improved 20", "dgp_transfer_epoch_5": "Transfer 5 (old)",
    "dgp_zamboanga_final": "Phase 3 final", "current_research_dgp": "Current research",
    "mapped_deblurgan": "Mapped start weights", "codeformer_w1": "CodeFormer w=1",
}


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    assert not Path(path).exists(), f"Retain existing artifact: {path}"
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")


def rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB")).copy()


def unit(a):
    return a.astype(np.float32) / np.float32(255)


def metrics(a, target, support):
    a, b = a.astype(np.float64), target.astype(np.float64)
    mse = float(np.square(a - b)[support].mean())
    core = cv2.erode(support.astype(np.uint8), np.ones((11, 11), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    assert core.sum() > 0
    _, similarity = structural_similarity(a, b, data_range=1.0, channel_axis=2, gaussian_weights=True,
                                         sigma=1.5, use_sample_covariance=False, full=True)
    luma_weights = np.array([.299, .587, .114])
    la, lb = a @ luma_weights, b @ luma_weights
    ax, ay = cv2.Sobel(la, cv2.CV_64F, 1, 0, ksize=3, scale=1 / 8), cv2.Sobel(la, cv2.CV_64F, 0, 1, ksize=3, scale=1 / 8)
    bx, by = cv2.Sobel(lb, cv2.CV_64F, 1, 0, ksize=3, scale=1 / 8), cv2.Sobel(lb, cv2.CV_64F, 0, 1, ksize=3, scale=1 / 8)
    return {"mse": mse, "psnr_db": None if mse == 0 else -10 * math.log10(mse),
            "exact_match": mse == 0, "ssim": float(similarity[core].mean()),
            "mae_255": float(np.abs(a - b)[support].mean() * 255),
            "edge_mae": float((np.abs(ax - bx) + np.abs(ay - by))[core].mean() / 2)}


def audit_and_measure(plan):
    from face_color_policy import preserve_input_palette
    started = time.monotonic()
    execution = read(OUT / "execution_complete.json")
    assert execution["complete"] and execution["plan_sha256"] == sha(OUT / "plan.json")
    for name, digest in plan["bindings"].items():
        assert sha(ROOT / name) == digest, name
    cases = {}
    for case in plan["cases"]:
        inp = rgb(OUT / case["input"])
        assert hashlib.sha256(inp.tobytes()).hexdigest() == case["input_pixel_sha256"]
        observed = np.asarray(Image.open(OUT / case["observed"])) > 0
        target = unit(rgb(OUT / case["target"])) if "target" in case else None
        cases[case["id"]] = inp, observed, target
    arms = [{"id": "resize_control", "execution_id": "resize_control", "type": "resize_control"}] + plan["arms"]
    rows, checks, calls, signatures = [], 0, 0, {}
    legacy_ast = []
    for name in ["dgp_synthesizer.py", "fpn_mobilenet.py"]:
        package = OUT / "legacy_decoder_6017d97"
        old = ast.parse((package / (name + ".original")).read_text(encoding="utf-8"))
        new = ast.parse((package / name).read_text(encoding="utf-8"))
        old_forward = [ast.dump(n, include_attributes=False) for n in ast.walk(old) if isinstance(n, ast.FunctionDef) and n.name == "forward"]
        new_forward = [ast.dump(n, include_attributes=False) for n in ast.walk(new) if isinstance(n, ast.FunctionDef) and n.name == "forward"]
        assert old_forward == new_forward and len(old_forward) == 1
        legacy_ast.append({"source": name, "original_forward_AST_exact": True})
    for arm in arms:
        receipt = None
        if arm["id"] != "resize_control":
            receipt = read(OUT / "results" / arm["execution_id"] / "receipt.json")
            assert receipt["plan_sha256"] == sha(OUT / "plan.json") and receipt["complete"]
            assert receipt["state_before_sha256"] == receipt["state_after_sha256"]
            assert receipt["optimizer_updates"] == receipt["gradient_queries"] == 0
            saved_rows = {x["id"]: x for x in receipt["rows"]}
            assert len(saved_rows) == len(plan["cases"])
            signature = tuple(saved_rows[c["id"]]["files"]["raw.npy"] for c in plan["cases"])
            signatures.setdefault(signature, []).append(arm["id"])
            calls += receipt["neural_calls"]
        for case in plan["cases"]:
            inp, support, target = cases[case["id"]]
            path = OUT / "results" / arm["execution_id"] / case["id"]
            if receipt:
                rr = saved_rows[case["id"]]
                assert rr["input_pixel_sha256"] == case["input_pixel_sha256"]
                for name, digest in rr["files"].items():
                    assert sha(path / name) == digest
                raw = np.load(path / "raw.npy", allow_pickle=False)
                native, display = rgb(path / "native.png"), rgb(path / "display.png")
                assert raw.dtype == np.float32 and raw.shape == (256, 256, 3)
                assert np.isfinite(raw).all() and 0 <= raw.min() <= raw.max() <= 1
                expected = np.floor(raw * np.float32(255)).astype(np.uint8)
                expected[~support] = inp[~support]
                assert np.array_equal(native, expected), (arm["id"], case["id"], "native PNG")
                expected, palette = preserve_input_palette(inp, np.zeros(support.shape, np.uint8), native, visible_restored=True)
                expected[~support] = inp[~support]
                assert np.array_equal(display, expected) and palette == rr["palette"]
                seconds = rr["forward_seconds"]
            else:
                path.mkdir(parents=True)
                raw, native = unit(inp), inp.copy()
                display, _ = preserve_input_palette(inp, np.zeros(support.shape, np.uint8), native, visible_restored=True)
                display[~support] = inp[~support]
                Image.fromarray(native).save(path / "native.png")
                Image.fromarray(display).save(path / "display.png")
                np.save(path / "raw.npy", raw, allow_pickle=False)
                seconds = 0.0
            for view, value in [("raw", raw), ("native_png", unit(native)), ("app_display", unit(display))]:
                difference = (value - unit(inp))[support]
                paired = metrics(value, target, support) if target is not None else {"mse": None, "psnr_db": None, "ssim": None, "mae_255": None, "edge_mae": None, "exact_match": None}
                rows.append({"arm": arm["id"], "id": case["id"], "source": case["source"], "evidence": case["evidence"],
                             "profile": case["profile"], "view": view, "forward_seconds": seconds,
                             "input_change_mae_255": float(np.abs(difference).mean() * 255),
                             "mean_input_luma_change_255": float((difference @ np.array([.299, .587, .114])).mean() * 255),
                             **paired})
            checks += 1
        print(json.dumps({"saved_pixel_audit": arm["id"], "cases": len(plan["cases"]), "seconds": round(time.monotonic() - started, 2)}), flush=True)
    assert calls == 1056 and checks == 1144
    groups = []
    for arm in arms:
        for view in ["raw", "native_png", "app_display"]:
            for source in ["all_paired", "dataset/asian_faces", "dataset/thumbnails128x128"]:
                subset = [x for x in rows if x["arm"] == arm["id"] and x["view"] == view and x["evidence"] == "synthetic_paired_development" and (source == "all_paired" or x["source"] == source)]
                degraded = [x for x in subset if x["profile"] != "clear"]
                clear = [x for x in subset if x["profile"] == "clear"]
                control = {x["id"]: x for x in rows if x["arm"] == "resize_control" and x["view"] == view}
                groups.append({"arm": arm["id"], "view": view, "source": source, "degraded_cases": len(degraded), "clear_cases": len(clear),
                               "mean_degraded_psnr_db": float(np.mean([x["psnr_db"] for x in degraded])),
                               "mean_degraded_ssim": float(np.mean([x["ssim"] for x in degraded])),
                               "mean_degraded_edge_mae": float(np.mean([x["edge_mae"] for x in degraded])),
                               "mean_clear_mae_255": float(np.mean([x["mae_255"] for x in clear])),
                               "degraded_cases_ssim_above_resize": sum(x["ssim"] > control[x["id"]]["ssim"] for x in degraded),
                               "mean_forward_seconds": float(np.mean([x["forward_seconds"] for x in subset]))})
    write(OUT / "metrics.json", {"rows": rows, "groups": groups,
                                "method": "Masked RGB PSNR; Gaussian SSIM on 11px-eroded observed support; Sobel gradient MAE. Clear and degraded reported separately. Native scores null.",
                                "historical_one_percent_structure_gate_recomputed": False})
    with (OUT / "metrics.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write(OUT / "saved_output_audit.json", {"complete": True, "plan_sha256": sha(OUT / "plan.json"),
                                           "auditor_source_sha256": sha(Path(__file__)), "checkpoint_entries": 12, "unique_checkpoint_states": 12,
                                           "common_input_bindings": len(plan["bindings"]), "audited_model_case_sets_including_resize": checks,
                                           "raw_native_display_rows": len(rows), "measured_forward_calls": calls,
                                           "new_neural_calls_in_audit": 0, "gradient_queries": 0, "optimizer_updates": 0,
                                           "padding_and_conversion_exact": True, "checkpoint_hashes_unchanged": True,
                                           "raw_output_equivalence_groups": list(signatures.values()),
                                           "legacy_forward_AST_checks": legacy_ast,
                                           "native_psnr_ssim_absent": True, "final_pixels_read": 0, "visual_review_pending": True,
                                           "model_qualified": False, "seconds": time.monotonic() - started})
    return arms, groups


def font(size):
    return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)


def sheets(plan, arms):
    directory = OUT / "sheets"
    directory.mkdir()
    rendered = []
    for start in range(0, len(plan["cases"]), 4):
        block = plan["cases"][start:start + 4]
        paired = any("target" in x for x in block)
        width, row_h, header = 144, 166, 76
        im = Image.new("RGB", (width * (len(arms) + int(paired)), header + row_h * len(block)), "#f6f3ed")
        draw = ImageDraw.Draw(im)
        draw.text((8, 5), f"Restoration comparison {start // 4 + 1:02d} | native model PNG | all 12 checkpoints + resize", fill="#202b2b", font=font(16))
        for col, arm in enumerate(arms):
            draw.text((col * width + 7, 37), SHORT[arm["id"]], fill="#202b2b", font=font(12))
        if paired:
            draw.text((len(arms) * width + 7, 37), "Paired reference", fill="#202b2b", font=font(12))
        for row, case in enumerate(block):
            yy = header + row * row_h
            review = case.get("input_review", {})
            note = review.get("input_structure_review", review.get("input_review", "synthetic"))
            draw.text((7, yy), f"{case['id']} | {case['source']} | {note}", fill="#202b2b", font=font(12))
            for col, arm in enumerate(arms):
                path = OUT / "results" / arm["execution_id"] / case["id"] / "native.png"
                tile = Image.open(path).convert("RGB").resize((128, 128), Image.Resampling.NEAREST)
                im.paste(tile, (col * width + 7, yy + 25))
            if "target" in case:
                tile = Image.open(OUT / case["target"]).convert("RGB").resize((128, 128), Image.Resampling.NEAREST)
                im.paste(tile, (len(arms) * width + 7, yy + 25))
        path = directory / f"all-models-{start // 4 + 1:02d}.png"
        im.save(path)
        rendered.append({"path": path.relative_to(OUT).as_posix(), "sha256": sha(path), "cases": [x["id"] for x in block],
                         "columns": [x["id"] for x in arms], "view": "native_png", "thumbnail_pixels": 128})
    # Fixed inputs chosen from the pre-model input-only review/protocol ordering.
    focus = ["dev_16to23_01", "dev_ge40_02", "choke_dev_01_t067", "choke_dev_02_t067",
             "va_asian_09306_blur_lr24", "va_ffhq_28544_compound_lr24"]
    for number, cid in enumerate(focus, 1):
        case = next(x for x in plan["cases"] if x["id"] == cid)
        im = Image.new("RGB", (4 * 272, 4 * 291 + 75), "#f6f3ed")
        draw = ImageDraw.Draw(im)
        draw.text((8, 7), f"{cid} | model-native PNG at 256 pixels", fill="#202b2b", font=font(19))
        draw.text((8, 36), "No completion, sharpening or display adjustment. Native CCTV has no paired reference.", fill="#202b2b", font=font(13))
        for n, arm in enumerate(arms):
            xx, yy = (n % 4) * 272, 75 + (n // 4) * 291
            draw.text((xx + 8, yy), SHORT[arm["id"]], fill="#202b2b", font=font(15))
            im.paste(Image.open(OUT / "results" / arm["execution_id"] / cid / "native.png").convert("RGB"), (xx + 8, yy + 25))
        if "target" in case:
            draw.text((280, 948), "Paired reference", fill="#202b2b", font=font(15))
            im.paste(Image.open(OUT / case["target"]).convert("RGB"), (280, 973))
        path = directory / f"focus-{number:02d}.png"
        im.save(path)
        rendered.append({"path": path.relative_to(OUT).as_posix(), "sha256": sha(path), "cases": [cid],
                         "columns": [x["id"] for x in arms], "view": "native_png", "thumbnail_pixels": 256})
    write(OUT / "visual_sheet_manifest.json", {"sheets": rendered, "all_model_case_sets_covered": 1144,
                                              "focus_case_selection": "Fixed pre-output input-review/protocol IDs", "actually_viewed": False})


def gallery(plan, arms, groups):
    data = {"cases": plan["cases"], "arms": [{**a, "short": SHORT[a["id"]]} for a in arms], "groups": groups}
    template = (ROOT / "restoration_checkpoint_comparison_v1.template.html").read_text(encoding="utf-8")
    page = template.replace("__COMPARISON_DATA__", json.dumps(data, ensure_ascii=True).replace("<", "\\u003c"))
    (OUT / "comparison.html").write_text(page, encoding="utf-8")


def main():
    cv2.setNumThreads(1)
    plan = read(OUT / "plan.json")
    for name in ["request.json", "acquisition.json"]:
        shutil.copyfile(ROOT / "outputs/cctv_survface_acquisition_v1" / name, OUT / "source_terms" / ("QMUL_" + name))
    for name in ["CCTV_BENCHMARK_STATUS.md", "CCTV_NATIVE_SOURCE_EXTENSION_V1.md"]:
        shutil.copyfile(ROOT / name, OUT / "source_terms" / name)
    arms, groups = audit_and_measure(plan)
    sheets(plan, arms)
    gallery(plan, arms, groups)
    print(json.dumps({"complete": True, "gallery": str(OUT / "comparison.html"), "visual_sheets": 28, "model_qualified": False}), flush=True)


if __name__ == "__main__":
    main()
