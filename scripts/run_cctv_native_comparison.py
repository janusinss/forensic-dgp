"""Finite, frozen CPU restoration comparison on native development CCTV crops."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import DGPSynthesizer
from pretrained_face_restoration import load_face_restorer

BASE = ROOT / "outputs/cctv_native_development_v2"
OUT = ROOT / "outputs/cctv_native_comparison_v1"
SUBSET_PIN = "c063985b258b808efaa897c88593cbdbcee2848d686d3d056219f8a8e555a98e"
REVIEW_PIN = "41912033f5070547e120a6b384024fa929eeaf3a7ecca408c0c67c53d37d36e2"
DGP = "checkpoints/dgp_zamboanga_final.pth"
DGP_PIN = "b6376f56c4161ef1f26a8f9efb0bb309b3014f4ecdeaca5399a47c1c4607586c"
CF = "outputs/codeformer_restoration_pretrained_v1/codeformer.pth"
CF_PIN = "1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7"
BINS = ("le15", "16to23", "24to39", "ge40")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + "\n")


def state_sha(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0" + str(value.dtype).encode() + b"\0")
        digest.update(str(tuple(value.shape)).encode() + b"\0" + value.numpy().tobytes())
    return digest.hexdigest()


def native_canvas(path):
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        side = max(width, height)
        canvas = Image.new("RGB", (side, side), (128, 128, 128))
        canvas.paste(rgb, ((side-width)//2, (side-height)//2))
    return canvas


def legacy_display(raw, common_input):
    """Only top-1 app.py color/CLAHE/unsharp; no legacy denoise/alignment."""
    value = (raw * 255).astype(np.uint8)
    lab_ref = cv2.cvtColor(common_input, cv2.COLOR_RGB2LAB).astype(np.float32)
    mean_ref, std_ref = float(lab_ref[:, :, 0].mean()), max(1e-5, float(lab_ref[:, :, 0].std()))
    lab = cv2.cvtColor(value, cv2.COLOR_RGB2LAB)
    mean_out, std_out = float(lab[:, :, 0].mean()), max(1e-5, float(lab[:, :, 0].std()))
    if mean_out < mean_ref:
        lab[:, :, 0] = np.clip((lab[:, :, 0]-mean_out)*(std_ref/std_out)+mean_ref, 0, 255).astype(np.uint8)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    blurred = cv2.GaussianBlur(rgb, (0, 0), 1.2)
    return cv2.addWeighted(rgb, 1.35, blurred, -0.35, 0)


def prepare():
    if OUT.exists():
        raise ValueError("Preserve completed/partial comparison")
    if sha(BASE / "frozen_subset.json") != SUBSET_PIN or sha(BASE / "input_review.json") != REVIEW_PIN:
        raise ValueError("Frozen native subset/input-only review differs")
    subset, review = read(BASE / "frozen_subset.json"), read(BASE / "input_review.json")
    cases = [row for row in subset["cases"] if row["role"] == "development"]
    if len(cases) != 24 or {r["id"] for r in review["rows"]} != {c["id"] for c in cases} or review["reserved_inputs_viewed"]:
        raise ValueError("Require 24 input-reviewed development cases; reserved inputs unviewed")
    files = ["scripts/run_cctv_native_comparison.py", "scripts/audit_cctv_native_comparison.py",
             "models/dgp_synthesizer.py", "models/fpn_mobilenet.py", "models/mobilenet_v2.py",
             "models/morphological_loss.py", "models/losses.py", "models/__init__.py",
             "pretrained_face_restoration.py", "app.py", DGP, CF,
             "outputs/cctv_native_development_v2/frozen_subset.json", "outputs/cctv_native_development_v2/input_review.json"]
    files.extend(path.relative_to(ROOT).as_posix() for path in sorted((ROOT / "third_party/codeformer").glob("*.py")))
    files.append("third_party/codeformer/LICENSE")
    for case in cases:
        path = BASE / case["source_file"]
        if sha(path) != case["source_sha256"]:
            raise ValueError("Native image bytes changed")
        files.append(path.relative_to(ROOT).as_posix())
    assets = {name: sha(ROOT / name) for name in files}
    if assets[DGP] != DGP_PIN or assets[CF] != CF_PIN:
        raise ValueError("Retained/official restoration weight fingerprint differs")
    protocol = {
        "format": "dgp-native-cctv-comparison-v1", "date": "2026-10-03",
        "frozen_before_model_outputs": True, "cases": cases, "assets_sha256": assets,
        "arms": ["bilinear256_common_input", "phase3_dgp_raw", "phase3_dgp_legacy_display_only", "codeformer_restoration_raw_w1"],
        "input": "Native source RGB; center-pad with RGB128 at native size; Pillow bilinear256. Same exact quantized input to both frozen models. No alignment or signal preconditioning.",
        "raw_output": "Float32 HWC [0,1] stages saved; display PNG uses floor(raw*255), not enhancement",
        "display_only_arm": "Current app top-1 illumination calibration, CLAHE1.8/8x8, Gaussian1.2 unsharp1.35. Applied only to saved DGP raw. Does not reproduce app denoise/alignment/top-k.",
        "codeformer": {"fidelity": 1.0, "adain": True, "internal_resolution": 512, "returned_resolution": 256},
        "budget": {"development_cases": 24, "dgp_forwards": 24, "codeformer_forwards": 24,
                   "optimizer_updates": 0, "device": "cpu", "torch_threads": 4, "wall_seconds_after_loading": 300},
        "criteria": read(BASE / "selection_policy.json")["review_criteria"],
        "interpretation": "Development comparison on native unaligned CCTV crops. No clean paired reference, PSNR/SSIM, verified identity accuracy or independent final review. Input-change diagnostics do not rank restoration quality.",
        "reserved_evaluation_used": False, "application_change": False, "training": False,
    }
    OUT.mkdir()
    write(OUT / "frozen_protocol.json", protocol)
    print(json.dumps({"prepared": True, "protocol_sha256": sha(OUT / "frozen_protocol.json"), "model_forwards_budget": 48}))


def run():
    if (OUT / "execution.json").exists():
        raise ValueError("Preserve completed/partial execution; no repeated run")
    protocol = read(OUT / "frozen_protocol.json")
    for name, pin in protocol["assets_sha256"].items():
        if sha(ROOT / name) != pin:
            raise ValueError("Frozen asset differs: " + name)
    torch.set_num_threads(4)
    torch.manual_seed(20261003)
    dgp = DGPSynthesizer().cpu().eval().requires_grad_(False)
    dgp.load_state_dict(torch.load(ROOT / DGP, map_location="cpu", weights_only=True), strict=True)
    cf, provenance = load_face_restorer(ROOT / CF, device="cpu")
    models = {"dgp": dgp, "codeformer": cf}
    if any(model.training or any(p.requires_grad for p in model.parameters()) for model in models.values()):
        raise ValueError("Only frozen evaluation models permitted")
    before = {name: state_sha(model) for name, model in models.items()}
    counts = {"dgp": 0, "codeformer": 0}

    def count(name):
        def hook(module, args, output):
            counts[name] += 1
        return hook

    handles = [model.register_forward_hook(count(name)) for name, model in models.items()]
    write(OUT / "execution.json", {"protocol_sha256": sha(OUT / "frozen_protocol.json"),
          "model_state_before": before, "codeformer_provenance": provenance,
          "torch_version": torch.__version__, "cpu_threads": torch.get_num_threads(),
          "optimizer_constructed": False, "models_eval_and_frozen": True,
          "reserved_evaluation_used": False, "budget": protocol["budget"]})
    (OUT / "images").mkdir()
    (OUT / "stages").mkdir()
    rows, artifacts = [], {}
    start = time.monotonic()
    for index, case in enumerate(protocol["cases"]):
        if time.monotonic() - start > 300:
            raise TimeoutError("Finite native comparison cap exceeded; partial evidence retained")
        canvas = native_canvas(BASE / case["source_file"])
        common = np.asarray(canvas.resize((256, 256), Image.Resampling.BILINEAR)).copy()
        tensor = torch.from_numpy(common).permute(2, 0, 1).float().unsqueeze(0) / 255
        with torch.inference_mode():
            generated = {"dgp": dgp(tensor), "codeformer": cf(tensor, fidelity=1.0)}
        floats = {}
        for name, value in generated.items():
            if value.shape != (1, 3, 256, 256) or not torch.isfinite(value).all() or value.min() < 0 or value.max() > 1:
                raise ValueError("Invalid model output")
            floats[name] = value[0].permute(1, 2, 0).cpu().numpy().copy()
            stage = f"stages/{case['id']}_{name}.npy"
            with (OUT / stage).open("xb") as stream:
                np.save(stream, floats[name], allow_pickle=False)
            artifacts[stage] = sha(OUT / stage)
        images = {"input": common, "dgp_raw": (floats["dgp"]*255).astype(np.uint8),
                  "dgp_display": legacy_display(floats["dgp"], common),
                  "codeformer_raw": (floats["codeformer"]*255).astype(np.uint8)}
        names = {}
        for arm, rgb in images.items():
            file = f"images/{case['id']}_{arm}.png"
            Image.fromarray(rgb).save(OUT / file)
            artifacts[file] = sha(OUT / file)
            names[arm] = file
        rows.append({"id": case["id"], "size_bin": case["size_bin"], "outputs": names,
                     "native_size": [case["native_width"], case["native_height"]],
                     "input_change_mae_diagnostic_only": {name: float(np.abs(value-common/255.0).mean()) for name, value in floats.items()},
                     "note": "Change from degraded input is not an error against clean truth or quality score"})
        print(f"Native CCTV {index+1}/24: {case['id']}; elapsed={time.monotonic()-start:.1f}s", flush=True)
    seconds = time.monotonic() - start
    if seconds > 300 or counts != {"dgp": 24, "codeformer": 24}:
        raise ValueError("Frozen comparison time/forward count differs")
    for handle in handles:
        handle.remove()
    after = {name: state_sha(model) for name, model in models.items()}
    if before != after:
        raise ValueError("Frozen inference changed model weights/buffers")
    for group in BINS:
        save_sheet([row for row in rows if row["size_bin"] == group], OUT / f"comparison_{group}.png", 160)
        artifacts[f"comparison_{group}.png"] = sha(OUT / f"comparison_{group}.png")
    save_sheet(rows[:10], OUT / "preview_10_rows.png", 160)
    artifacts["preview_10_rows.png"] = sha(OUT / "preview_10_rows.png")
    result = {"complete": True, "protocol_sha256": sha(OUT / "frozen_protocol.json"),
              "rows": rows, "artifacts_sha256": artifacts, "model_forwards": counts,
              "model_state_after": after, "model_states_unchanged": True, "seconds_after_loading": seconds,
              "optimizer_updates": 0, "reserved_evaluation_used": False, "PSNR": None, "SSIM": None,
              "checkpoint_selected": False, "application_change": False}
    write(OUT / "results.json", result)
    print(json.dumps({"complete": True, "forwards": counts, "seconds": seconds, "results_sha256": sha(OUT / "results.json")}))


def save_sheet(rows, path, cell):
    columns = ("input", "dgp_raw", "dgp_display", "codeformer_raw")
    sheet = Image.new("RGB", (len(columns)*(cell+4), len(rows)*(cell+32)+26), (238, 238, 238))
    draw = ImageDraw.Draw(sheet)
    for j, name in enumerate(columns):
        draw.text((j*(cell+4)+2, 3), name, fill=(0, 0, 0))
    for i, row in enumerate(rows):
        y = 26 + i*(cell+32)
        for j, arm in enumerate(columns):
            x = j*(cell+4)
            draw.text((x+2, y+1), row["id"], fill=(0, 0, 0))
            draw.text((x+2, y+14), f"native {row['native_size'][0]}x{row['native_size'][1]}", fill=(0, 0, 0))
            with Image.open(OUT / row["outputs"][arm]) as image:
                sheet.paste(image.resize((cell, cell), Image.Resampling.BILINEAR), (x, y+29))
    sheet.save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    args = parser.parse_args()
    prepare() if args.action == "prepare" else run()
