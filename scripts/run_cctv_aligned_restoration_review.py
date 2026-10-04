"""Six frozen CPU forwards; compare aligned model inputs with warped cached controls."""
import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_cctv_native_comparison import sha, read, write, state_sha, DGP, DGP_PIN, CF, CF_PIN
from models import DGPSynthesizer
from pretrained_face_restoration import load_face_restorer

BASE = ROOT / "outputs/cctv_native_comparison_v1"
ALIGN = ROOT / "outputs/cctv_alignment_feasibility_v2"
OUT = ROOT / "outputs/cctv_alignment_restoration_comparison_v1"
IDS = ("dev_16to23_05", "dev_24to39_06", "dev_ge40_04")


def source_mask(case, matrix):
    width, height = case["native_width"], case["native_height"]
    side = max(width, height)
    canvas = np.zeros((side, side), dtype=np.uint8)
    x, y = (side-width)//2, (side-height)//2
    canvas[y:y+height, x:x+width] = 255
    common = np.asarray(Image.fromarray(canvas).resize((256, 256), Image.Resampling.NEAREST))
    return cv2.warpAffine(common, matrix, (256, 256), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def prepare():
    if OUT.exists():
        raise ValueError("Preserve earlier partial/completed aligned comparison")
    required = {
        "outputs/cctv_native_comparison_v1/results.json": "3be73af7b0cdae85f54adc8173fd8f48b42511a604fcd60244e3b1d2565a4c34",
        "outputs/cctv_native_comparison_v1/independent_verification.json": "c7801b90a817f20403abc3bd99743e8d0f8e782003b1e1d938bfff05be38671a",
        "outputs/cctv_alignment_feasibility_v2/results.json": "ff982ad5de0757854652dfe68c665ecc5087c40d0981c90de19441142567c348",
        "outputs/cctv_processing_evidence_v1.json": "0bd4b61f6fa2e50a04f20d769f31efca9b050fe5e9400198b548a3c8e7306a32",
        DGP: DGP_PIN, CF: CF_PIN,
    }
    for name, pin in required.items():
        if sha(ROOT / name) != pin:
            raise ValueError("Frozen prerequisite differs: " + name)
    subset = read(ROOT / "outputs/cctv_native_development_v2/frozen_subset.json")
    alignment, baseline = read(ALIGN / "results.json"), read(BASE / "results.json")
    available = {row["id"] for row in alignment["rows"] if row["space"] == "common_padded256_rgb" and row["method"] == "CANONICAL_5POINT"}
    if available != set(IDS):
        raise ValueError("Use all three feasible input-selected alignment cases")
    old = read(BASE / "frozen_protocol.json")
    files = [name for name in old["assets_sha256"] if name.startswith(("models/", "third_party/"))]
    files.extend(["pretrained_face_restoration.py", "scripts/run_cctv_native_comparison.py",
                  "scripts/run_cctv_aligned_restoration_review.py", "scripts/audit_cctv_aligned_restoration_review.py",
                  "outputs/cctv_native_development_v2/frozen_subset.json"])
    cases = []
    for case_id in IDS:
        source = next(c for c in subset["cases"] if c["id"] == case_id)
        frame = next(row for row in alignment["rows"] if row["id"] == case_id and row["space"] == "common_padded256_rgb")
        matrix = np.asarray(frame["transform"], dtype=np.float64)
        common = next(row for row in baseline["rows"] if row["id"] == case_id)["outputs"]["input"]
        with Image.open(BASE / common) as image:
            rgb = np.asarray(image.convert("RGB"))
        with Image.open(ALIGN / frame["image"]) as image:
            actual = np.asarray(image.convert("RGB"))
        if frame["errors_caught_by_legacy_helper"] or not np.array_equal(actual, cv2.warpAffine(rgb, matrix, (256, 256), borderMode=cv2.BORDER_REFLECT)):
            raise ValueError("Aligned pixels differ from independently checked transform")
        files.extend(["outputs/cctv_native_comparison_v1/"+common, "outputs/cctv_alignment_feasibility_v2/"+frame["image"]])
        controls = {name: f"outputs/cctv_native_comparison_v1/stages/{case_id}_{name}.npy" for name in ("dgp", "codeformer")}
        files.extend(controls.values())
        cases.append({"id": case_id, "source": source, "matrix": frame["transform"],
                      "common_input": "outputs/cctv_native_comparison_v1/"+common,
                      "aligned_input": "outputs/cctv_alignment_feasibility_v2/"+frame["image"], "cached_raw_controls": controls})
    assets = dict(required)
    assets.update({name: sha(ROOT / name) for name in files})
    protocol = {"date": "2026-10-03", "format": "dgp-cctv-aligned-restoration-review-v1",
        "frozen_before_aligned_outputs": True, "cases": cases, "assets_sha256": assets,
        "comparison": "Model(aligned uint8 common input) versus warp(cached model(common input) float32) in the same aligned frame",
        "support": "Native observed rectangle warped with nearest/zero border; reflected/padded pixels excluded from observational diagnostics",
        "quantization": "Save raw HWC float32; PNG=floor(raw*255); no display enhancement, inverse paste-back or selection",
        "budget": {"cpu_threads": 4, "dgp_forwards": 3, "codeformer_forwards": 3,
                   "wall_seconds_after_loading": 90, "optimizer_updates": 0, "alignment_forwards": 0},
        "criteria": ["Inspect actual clarity/structure changes against geometry-matched cached controls",
                     "Do not credit framing, reflection or sharper unverifiable features as recovered identity",
                     "Reject a universal default with any material structure/artifact regression; three cases alone cannot qualify whole-scope integration",
                     "Retain unsupported/fallback/insufficient cases and reserved evaluation; native CCTV has no clean paired PSNR/SSIM"],
        "interpretation": "Development geometry ablation; cached controls incur an extra warp, so differences include interpolation effects. No exact clean anatomy available.",
        "reserved_evaluation_used": False, "training": False, "application_change": False}
    OUT.mkdir()
    write(OUT / "frozen_protocol.json", protocol)
    print({"prepared": True, "protocol_sha256": sha(OUT / "frozen_protocol.json"), "new_forwards": 6})


def run():
    if (OUT / "execution.json").exists():
        raise ValueError("Preserve earlier completed/partial execution")
    policy = read(OUT / "frozen_protocol.json")
    for name, pin in policy["assets_sha256"].items():
        if sha(ROOT / name) != pin:
            raise ValueError("Frozen aligned-review asset differs: "+name)
    torch.set_num_threads(4)
    torch.manual_seed(20261003)
    dgp = DGPSynthesizer().cpu().eval().requires_grad_(False)
    dgp.load_state_dict(torch.load(ROOT / DGP, map_location="cpu", weights_only=True), strict=True)
    cf, provenance = load_face_restorer(ROOT / CF, "cpu")
    models = {"dgp": dgp, "codeformer": cf}
    before = {name: state_sha(model) for name, model in models.items()}
    counts = {name: 0 for name in models}
    handles = []
    for name, model in models.items():
        def count(module, args, output, name=name):
            counts[name] += 1
        handles.append(model.register_forward_hook(count))
    write(OUT / "execution.json", {"protocol_sha256": sha(OUT / "frozen_protocol.json"), "state_before": before,
          "budget": policy["budget"], "optimizer_constructed": False, "codeformer_provenance": provenance})
    for directory in ("stages", "images"):
        (OUT / directory).mkdir()
    start, rows, artifacts = time.monotonic(), [], {}
    for case in policy["cases"]:
        if time.monotonic()-start > 90:
            raise TimeoutError("Aligned review budget exceeded; preserve partial evidence")
        matrix = np.asarray(case["matrix"], dtype=np.float64)
        with Image.open(ROOT / case["aligned_input"]) as image:
            rgb = np.asarray(image.convert("RGB")).copy()
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()[None]/255
        with torch.inference_mode():
            predictions = {"dgp": dgp(tensor), "codeformer": cf(tensor, fidelity=1.0)}
        support = source_mask(case["source"], matrix)
        floats, outputs = {}, {}
        for name, prediction in predictions.items():
            if prediction.shape != (1, 3, 256, 256) or not torch.isfinite(prediction).all() or prediction.min() < 0 or prediction.max() > 1:
                raise ValueError("Invalid aligned restoration output")
            floats[name+"_aligned"] = prediction[0].permute(1, 2, 0).cpu().numpy().copy()
            cached = np.load(ROOT / case["cached_raw_controls"][name], allow_pickle=False)
            floats[name+"_control"] = cv2.warpAffine(cached, matrix, (256, 256), borderMode=cv2.BORDER_REFLECT)
        for arm, value in floats.items():
            file = f"stages/{case['id']}_{arm}.npy"
            with (OUT / file).open("xb") as stream:
                np.save(stream, value, allow_pickle=False)
            artifacts[file] = sha(OUT / file)
            file = f"images/{case['id']}_{arm}.png"
            Image.fromarray((value*255).astype(np.uint8)).save(OUT / file)
            artifacts[file] = sha(OUT / file)
            outputs[arm] = file
        for arm, pixels in (("input_aligned", rgb), ("observed_support", support)):
            file = f"images/{case['id']}_{arm}.png"
            Image.fromarray(pixels).save(OUT / file)
            artifacts[file] = sha(OUT / file)
            outputs[arm] = file
        observed = support == 255
        rows.append({"id": case["id"], "outputs": outputs, "observed_pixels": int(observed.sum()),
                     "observed_source_fraction": float(observed.mean()),
                     "same_frame_change_mae_diagnostic_only": {name: float(np.abs(floats[name+"_aligned"]-floats[name+"_control"])[observed].mean()) for name in models}})
        print(f"Aligned CCTV {len(rows)}/3: {case['id']}; elapsed={time.monotonic()-start:.1f}s", flush=True)
    seconds = time.monotonic()-start
    for handle in handles:
        handle.remove()
    after = {name: state_sha(model) for name, model in models.items()}
    if seconds > 90 or before != after or counts != {"dgp": 3, "codeformer": 3}:
        raise ValueError("Alignment review state/forward/time checks differ")
    columns = ("input_aligned", "dgp_control", "dgp_aligned", "codeformer_control", "codeformer_aligned")
    sheet = Image.new("RGB", (5*164, 3*190+26), (238, 238, 238))
    draw = ImageDraw.Draw(sheet)
    for j, arm in enumerate(columns):
        draw.text((j*164+2, 3), arm, fill=(0, 0, 0))
        for i, row in enumerate(rows):
            y = 26+i*190
            draw.text((j*164+2, y+2), row["id"], fill=(0, 0, 0))
            with Image.open(OUT / row["outputs"][arm]) as image:
                sheet.paste(image.resize((160, 160), Image.Resampling.BILINEAR), (j*164, y+28))
    sheet.save(OUT / "comparison.png")
    artifacts["comparison.png"] = sha(OUT / "comparison.png")
    report = {"complete": True, "protocol_sha256": sha(OUT / "frozen_protocol.json"), "rows": rows,
              "artifacts_sha256": artifacts, "model_forwards": counts, "state_after": after,
              "model_states_unchanged": True, "seconds_after_loading": seconds, "optimizer_updates": 0,
              "reserved_evaluation_used": False, "checkpoint_selected": False, "application_change": False,
              "PSNR": None, "SSIM": None}
    write(OUT / "results.json", report)
    print({"complete": True, "forwards": counts, "seconds": seconds, "results_sha256": sha(OUT / "results.json")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    args = parser.parse_args()
    prepare() if args.action == "prepare" else run()
