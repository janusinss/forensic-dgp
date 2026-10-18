"""User-requested, inference-only Phase3 PNG -> CodeFormer comparison."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parent
PARENT = ROOT / "outputs/cctv_dgp_checkpoint_comparison_v1"
OUT = PARENT / "cascade_v1"
SCRIPT = Path(__file__).resolve()
PARENT_SHA = "0899c33d142c490ad9118d81620508c09f640a50fc8d53bb050d305d80eec78a"
EXAMPLES = ["va_ffhq_37228_compound_lr24", "va_asian_01005_compound_lr24",
            "va_asian_01005_lowlight_lr32", "va_ffhq_37228_blur_lr24"]


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    path = Path(path)
    assert not path.exists(), f"Retain existing evidence: {path}"
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB")).copy()


def state_sha(model):
    h = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        assert torch.isfinite(tensor).all()
        a = tensor.detach().cpu().contiguous().numpy()
        h.update(json.dumps([key, str(a.dtype), list(a.shape)], separators=(",", ":")).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def prepare():
    assert not OUT.exists(), "Preserve existing comparison"
    assert sha(PARENT / "plan.json") == PARENT_SHA
    original = read(PARENT / "plan.json")
    assert read(PARENT / "saved_output_audit.json")["complete"]
    for name, digest in original["bindings"].items():
        assert sha(ROOT / name) == digest, name
    phase3 = read(PARENT / "results/dgp_zamboanga_final/receipt.json")
    direct = read(PARENT / "results/codeformer_w1/receipt.json")
    for receipt in [phase3, direct]:
        assert receipt["complete"] and receipt["plan_sha256"] == PARENT_SHA
        assert receipt["neural_calls"] == 88
        assert receipt["state_before_sha256"] == receipt["state_after_sha256"]
    bindings = {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path) for path in [
        SCRIPT, PARENT / "plan.json", PARENT / "saved_output_audit.json", PARENT / "metrics.json",
        PARENT / "results/dgp_zamboanga_final/receipt.json", PARENT / "results/codeformer_w1/receipt.json",
        ROOT / "pretrained_face_restoration.py", ROOT / "face_color_policy.py",
        ROOT / "render_restoration_checkpoint_comparison_v1.py",
        ROOT / "restoration_checkpoint_comparison_v1.template.html"]}
    cases = original["cases"]
    order = EXAMPLES + [c["id"] for c in cases if c["id"] not in EXAMPLES]
    assert len(order) == len(set(order)) == len(cases) == 88
    saved = {x["id"]: x for x in phase3["rows"]}
    for c in cases:
        for arm, receipt in [("dgp_zamboanga_final", phase3), ("codeformer_w1", direct)]:
            rr = next(x for x in receipt["rows"] if x["id"] == c["id"])
            for name, digest in rr["files"].items():
                path = PARENT / "results" / arm / c["id"] / name
                assert sha(path) == digest
                bindings[path.relative_to(ROOT).as_posix()] = digest
    OUT.mkdir()
    (OUT / "results").mkdir()
    weights = next(x for x in original["arms"] if x["id"] == "codeformer_w1")
    plan = {"format": "phase3-native-png-to-codeformer-v1", "date": "2026-10-15",
            "parent_plan_sha256": PARENT_SHA, "cases": cases, "execution_order": order,
            "parent_case_source": PARENT.relative_to(ROOT).as_posix(), "weights": weights,
            "bindings": bindings, "parent_bindings": original["bindings"],
            "intermediate": "Exact retained Phase3 native.png, including original restored padding; never display.png or silently substituted RAW.",
            "codeformer": {"fidelity": 1.0, "adain": True, "internal_size": 512, "output_size": 256, "alignment": "none; common original canvas"},
            "normalization": "NumPy float32 pixels / float32(255)",
            "output": "Retain float RAW; floor PNG then restore original padding; separate display policy based on ORIGINAL degraded input.",
            "device": "cpu", "threads": 4, "external_wall_cap_seconds": 1200,
            "training_updates": 0, "gradient_queries": 0, "new_DGP_calls_planned": 0, "new_CodeFormer_calls_planned": 88,
            "model_selection": "User chooses; no default or checkpoint promotion.",
            "evidence": "48 unpaired native crops; 32 degraded synthetic pairs and 8 clear controls. No final pixels."}
    write(OUT / "plan.json", plan)
    print(json.dumps({"prepared": True, "cases": 88, "plan_sha256": sha(OUT / "plan.json"), "new_neural_calls": 0}), flush=True)


def run():
    started = time.monotonic()
    p = read(OUT / "plan.json")
    for name, digest in {**p["parent_bindings"], **p["bindings"]}.items():
        assert sha(ROOT / name) == digest, name
    assert not (OUT / "results/phase3_then_codeformer").exists()
    from pretrained_face_restoration import load_face_restorer
    from face_color_policy import preserve_input_palette
    model, provenance = load_face_restorer(ROOT / p["weights"]["checkpoint"], device="cpu")
    model.eval().requires_grad_(False)
    before = state_sha(model)
    phase3_rows = {x["id"]: x for x in read(PARENT / "results/dgp_zamboanga_final/receipt.json")["rows"]}
    by_id = {c["id"]: c for c in p["cases"]}
    rows = []
    for i, cid in enumerate(p["execution_order"], 1):
        assert time.monotonic() - started < p["external_wall_cap_seconds"]
        c = by_id[cid]
        intermediate = PARENT / "results/dgp_zamboanga_final" / cid / "native.png"
        assert sha(intermediate) == phase3_rows[cid]["files"]["native.png"]
        stage = rgb(intermediate)
        original = rgb(PARENT / c["input"])
        support = np.asarray(Image.open(PARENT / c["observed"])) > 0
        x = torch.from_numpy((stage.astype(np.float32) / np.float32(255)).copy()).permute(2, 0, 1).unsqueeze(0)
        tick = time.monotonic()
        with torch.inference_mode():
            raw = model(x, fidelity=1.0)[0].permute(1, 2, 0).cpu().numpy().copy()
        seconds = time.monotonic() - tick
        assert raw.dtype == np.float32 and raw.shape == (256, 256, 3)
        assert np.isfinite(raw).all() and 0 <= raw.min() <= raw.max() <= 1
        assert all(not m.training for m in model.modules())
        png = np.floor(raw * np.float32(255)).astype(np.uint8)
        png[~support] = original[~support]
        display, palette = preserve_input_palette(original, np.zeros(support.shape, np.uint8), png, visible_restored=True)
        display[~support] = original[~support]
        folder = OUT / "results/phase3_then_codeformer" / cid
        folder.mkdir(parents=True)
        np.save(folder / "raw.npy", raw, allow_pickle=False)
        Image.fromarray(png).save(folder / "native.png")
        Image.fromarray(display).save(folder / "display.png")
        rows.append({"id": cid, "intermediate_file_sha256": sha(intermediate),
                     "intermediate_rgb_sha256": hashlib.sha256(stage.tobytes()).hexdigest(),
                     "CodeFormer_seconds": seconds, "retained_Phase3_seconds": phase3_rows[cid]["forward_seconds"],
                     "estimated_combined_forward_seconds": seconds + phase3_rows[cid]["forward_seconds"],
                     "palette": palette, "files": {n: sha(folder / n) for n in ["raw.npy", "native.png", "display.png"]}})
        if i <= 4 or i % 10 == 0 or i == 88:
            print(json.dumps({"cascade": i, "of": 88, "id": cid, "seconds": round(time.monotonic() - started, 2)}), flush=True)
    after = state_sha(model)
    assert before == after and sha(ROOT / p["weights"]["checkpoint"]) == p["weights"]["file_sha256"]
    write(OUT / "receipt.json", {"complete": True, "plan_sha256": sha(OUT / "plan.json"), "rows": rows,
                                 "state_before_sha256": before, "state_after_sha256": after, "provenance": provenance,
                                 "new_CodeFormer_calls": 88, "new_DGP_calls": 0, "optimizer_updates": 0, "gradient_queries": 0,
                                 "seconds": time.monotonic() - started, "device": "cpu", "torch": torch.__version__,
                                 "model_selected": False, "model_qualified": False})


def launch():
    assert not (OUT / "inference.log").exists()
    with (OUT / "inference.log").open("x", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, "-X", "utf8", "-B", "-u", str(SCRIPT), "--run"],
                                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=1200, check=False)
    print(json.dumps({"exit_code": result.returncode, "log": str(OUT / "inference.log")}), flush=True)
    assert result.returncode == 0


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.manual_seed(20261015)
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--prepare", action="store_true")
    choice.add_argument("--run", action="store_true")
    choice.add_argument("--launch", action="store_true")
    a = parser.parse_args()
    prepare() if a.prepare else run() if a.run else launch()
