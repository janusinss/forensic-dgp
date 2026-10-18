"""New, bounded CPU inference comparison. Never trains or replaces app weights.

The prepare command freezes checkpoint/source/DEV-input bindings before inference.
Each child executes one strictly loaded checkpoint, with a wall-clock timeout.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/cctv_dgp_checkpoint_comparison_v1"
SCRIPT = Path(__file__).resolve()
Q = ROOT / "outputs/cctv_native_development_v2"
C = ROOT / "outputs/cctv_chokepoint_native_development_v1"
P = ROOT / "outputs/cctv_dgp_mixed_vm_v9_r2"
NATIVE_PINS = {
    Q / "frozen_subset.json": "c063985b258b808efaa897c88593cbdbcee2848d686d3d056219f8a8e555a98e",
    Q / "input_review.json": "41912033f5070547e120a6b384024fa929eeaf3a7ecca408c0c67c53d37d36e2",
    C / "frozen_subset.json": "00cb11376bba4b7d615701382f5333bcd7c3dc4d98b5c5a5ac46875d5f12b590",
    C / "input_review.json": "e3443ffff4934906299e4f40195b655107a92fb88a562d7225c6caaaf4451408",
    P / "mixed_protocol_v9.json": "6cd9ac8d5f3bf27be773979c2f628e33f5d3cf09748452eeab91a65b05b01c70",
}
EXPECTED = {
    "dgp_epoch_1": "344cf2fc39c2a94791d9483ef30038d02cbac87de884ec084d9035c932f73870",
    "dgp_epoch_17": "f4d419c8f9824fa03cb0fd7e2ca7ebfcfe8afe7bedc6bd930a71e27288c5b500",
    "dgp_epoch_5": "ae314c965daccd9cf003fadad9139ffc1d65f98b12e3fba6f0bc105a88ac1e4e",
    "dgp_improved_epoch_10": "6b7fb90c74a10f388d24c0336d92c4db40886eb4e485d62abef0f1045b336020",
    "dgp_improved_epoch_19": "393e3e0b6266ec05d1bf7a4f82c9b08430318b18c4c329cfd68e727c96dd674e",
    "dgp_improved_epoch_2": "68bb1cea6293864d2f2144054a9a501e9e90351436ea0864ba866f44430d03af",
    "dgp_improved_epoch_20": "47386e2bb389073e7bc37f19047f72d5b344c53218c6aec5c1e794b5d935fa4a",
    "dgp_transfer_epoch_5": "349844877793bfd90a66fe89b50ba04af73ff123b1eee00ad50141de5d9a7cda",
    "dgp_zamboanga_final": "b6376f56c4161ef1f26a8f9efb0bb309b3014f4ecdeaca5399a47c1c4607586c",
    "current_research_dgp": "646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b",
    "mapped_deblurgan": "330a656717bd3752a8ddccde8b17f17790365271ea3d46fd9794aa6a403c1cef",
    "codeformer_w1": "1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7",
}


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    path = Path(path)
    assert not path.exists(), f"Retain existing evidence: {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def image(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB")).copy()


def state_sha(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        assert isinstance(tensor, torch.Tensor) and torch.isfinite(tensor).all(), name
        a = tensor.detach().cpu().contiguous().numpy()
        h.update(json.dumps([name, str(a.dtype), list(a.shape)], separators=(",", ":")).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def bind(path, bindings):
    bindings[rel(path)] = sha(path)


def frozen_legacy():
    """Retain exact historical sources; only imports/init downloads are adapted."""
    package = OUT / "legacy_decoder_6017d97"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    receipt = {"git_revision": subprocess.check_output(["git", "rev-parse", "6017d97"], cwd=ROOT, text=True).strip(),
               "forward_math_changed": False, "strict_full_state_required": True, "files": []}
    for name in ["dgp_synthesizer.py", "fpn_mobilenet.py"]:
        original = subprocess.check_output(["git", "show", f"6017d97:models/{name}"], cwd=ROOT)
        (package / (name + ".original")).write_bytes(original)
        source = original.decode("utf-8")
        if name == "fpn_mobilenet.py":
            assert source.count("mobilenet_v2(pretrained=True)") == 1
            changed = source.replace("mobilenet_v2(pretrained=True)", "mobilenet_v2(weights=None)")
            reason = "Disable network/initial pretrained weights; all 352 checkpoint entries load strictly before use."
        else:
            assert source.count("from .morphological_loss import MorphologicalFANLoss") == 1
            changed = source.replace("from .morphological_loss import MorphologicalFANLoss",
                                     "from models.morphological_loss import MorphologicalFANLoss")
            reason = "Resolve unused top-k loss import from isolated package; only forward is called."
        (package / name).write_bytes(changed.encode("utf-8"))
        receipt["files"].append({"name": name, "original_sha256": sha(package / (name + ".original")),
                                 "adapted_sha256": sha(package / name), "adaptation": reason})
    write(package / "provenance.json", receipt)


def load_model(arm):
    assert sha(ROOT / arm["checkpoint"]) == arm["file_sha256"]
    if arm["architecture"] == "legacy_fpn_transposed_6017d97":
        sys.path.insert(0, str(OUT))
        cls = importlib.import_module("legacy_decoder_6017d97.dgp_synthesizer").DGPSynthesizer
        net = cls()
        net.load_state_dict(torch.load(ROOT / arm["checkpoint"], map_location="cpu", weights_only=True), strict=True)
        model = net.eval().requires_grad_(False)
        prov = {"internal_resolution": 256, "input_range": "RGB [0,1]", "normalization": "stored BatchNorm evaluation statistics",
                "historical_graph": "6017d97", "strict": True}
    elif arm["architecture"] == "codeformer":
        from pretrained_face_restoration import load_face_restorer
        model, prov = load_face_restorer(ROOT / arm["checkpoint"], device="cpu")
        prov.update({"fidelity": 1.0, "common_unaligned_crop": True,
                     "comparison_limit": "512 internal canvas returned to common 256; no publisher alignment pipeline"})
    else:
        from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
        model, prov = load_frozen_dgp_restorer(ROOT / arm["checkpoint"], expected_sha256=arm["file_sha256"], device="cpu")
    model.eval().requires_grad_(False)
    assert all(not x.requires_grad for x in model.parameters())
    return model, prov


def prepare():
    assert not OUT.exists(), "Comparison root exists; preserve its evidence"
    assert shutil.disk_usage(ROOT).free >= 2 * 1024**3
    for path, digest in NATIVE_PINS.items():
        assert sha(path) == digest, str(path)
    OUT.mkdir()
    frozen_legacy()
    bindings = {}
    for path in NATIVE_PINS:
        bind(path, bindings)
    for path in [SCRIPT, ROOT / "dgp_frozen_inference_v2.py", ROOT / "dgp_face_restoration.py",
                 ROOT / "cctv_dgp_frozen_norm.py", ROOT / "face_color_policy.py",
                 ROOT / "pretrained_face_restoration.py", ROOT / "models/dgp_synthesizer.py",
                 ROOT / "models/fpn_mobilenet.py", ROOT / "models/mobilenet_v2.py",
                 ROOT / "third_party/codeformer/codeformer_arch.py", ROOT / "third_party/codeformer/LICENSE"]:
        bind(path, bindings)
    for path in (OUT / "legacy_decoder_6017d97").iterdir():
        if path.is_file():
            bind(path, bindings)
    arms, seen = [], {}
    for path in sorted((ROOT / "checkpoints").glob("dgp*.pth")) + [
            ROOT / "outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth",
            ROOT / "weights/mapped_deblurgan.pth",
            ROOT / "outputs/codeformer_restoration_pretrained_v1/codeformer.pth"]:
        aid = path.stem
        if "face_code_fit_vm_v12_r2" in str(path):
            aid = "current_research_dgp"
        if aid == "codeformer":
            aid = "codeformer_w1"
        digest = sha(path)
        assert digest == EXPECTED[aid], aid
        state = torch.load(path, map_location="cpu", weights_only=True)
        if aid == "codeformer_w1":
            state = state["params_ema"]
            architecture = "codeformer"
        else:
            architecture = "legacy_fpn_transposed_6017d97" if len(state) == 352 else "current_fpn_residual_frozen_norm"
            assert len(state) in (352, 622)
        tensor_digest = state_sha(state)
        key = (architecture, tensor_digest)
        execution_id = seen.setdefault(key, aid)
        arm = {"id": aid, "checkpoint": rel(path), "file_sha256": digest, "bytes": path.stat().st_size,
               "tensor_state_sha256": tensor_digest, "state_entries": len(state), "architecture": architecture,
               "execution_id": execution_id, "type": "pretrained_comparison" if aid in ("mapped_deblurgan", "codeformer_w1") else "own_trained_checkpoint"}
        model, provenance = load_model(arm)
        arm["strict_load_verified_without_forward"] = True
        arm["provenance"] = provenance
        del model, state
        arms.append(arm)
        bind(path, bindings)
        print(json.dumps({"strict_load": aid, "execution_id": execution_id, "state_entries": arm["state_entries"]}), flush=True)
    assert len(arms) == 12
    from dgp_face_restoration import prepare_crop
    cases = []

    def save_case(case, rgb, observed, target=None):
        assert rgb.shape == (256, 256, 3) and rgb.dtype == np.uint8 and observed.shape == (256, 256)
        folder = OUT / "cases" / case["id"]
        folder.mkdir(parents=True)
        for name, pixels in [("input", rgb), ("observed", observed.astype(np.uint8) * 255), ("target", target)]:
            if pixels is not None:
                path = folder / (name + ".png")
                Image.fromarray(pixels).save(path)
                case[name] = path.relative_to(OUT).as_posix()
                bind(path, bindings)
        case["input_pixel_sha256"] = hashlib.sha256(rgb.tobytes()).hexdigest()
        case["observed_pixels"] = int(observed.sum())
        cases.append(case)

    reviews = {x["id"]: x for x in load_json(Q / "input_review.json")["rows"]}
    for item in load_json(Q / "frozen_subset.json")["cases"]:
        if item["role"] != "development":
            continue
        path = Q / item["source_file"]
        assert sha(path) == item["source_sha256"]
        bind(path, bindings)
        rgb, support, _, geometry = prepare_crop(image(path))
        save_case({"id": item["id"], "source": "QMUL-SurvFace", "evidence": "native_unpaired",
                   "identity": f"QMUL/{item['global_person_id']}", "native_size": [item["native_width"], item["native_height"]],
                   "profile": item["size_bin"], "input_review": reviews[item["id"]], "geometry": geometry,
                   "clean_reference_available": False}, rgb, support)
    reviews = {x["id"]: x for x in load_json(C / "input_review.json")["rows"]}
    for item in load_json(C / "frozen_subset.json")["cases"]:
        if item["role"] != "development":
            continue
        path = C / item["native_crop"]
        assert sha(path) == item["native_crop_sha256"]
        rgb, support, _, geometry = prepare_crop(image(path))
        assert np.array_equal(rgb, image(C / item["input"]))
        assert np.array_equal(support, np.asarray(Image.open(C / item["observed"])) > 0)
        for name, digest in [("native_crop", item["native_crop_sha256"]), ("input", item["input_sha256"]), ("observed", item["observed_sha256"])]:
            assert sha(C / item[name]) == digest
            bind(C / item[name], bindings)
        save_case({"id": item["id"], "source": "ChokePoint", "evidence": "native_unpaired",
                   "identity": f"ChokePoint/{item['source_person_id']}", "native_size": [item["native_width"], item["native_height"]],
                   "profile": item["camera"], "input_review": reviews[item["id"]], "geometry": geometry,
                   "clean_reference_available": False}, rgb, support)
    paired = load_json(P / "mixed_protocol_v9.json")
    selected = []
    for source in sorted({x["source"] for x in paired["references"]}):
        selected.extend([x for x in paired["references"] if x["role"] == "validation" and x["source"] == source][:4])
    assert len(selected) == 8
    train_files = {x["source_file"] for x in paired["references"] if x["role"] == "train"}
    for ref in selected:
        assert ref["source_file"] not in train_files
        target = image(P / ref["target"])
        canonical = paired["canonical_target_rgb_sha256"][ref["id"]]
        assert hashlib.sha256(target.tobytes()).hexdigest() == canonical
        support = np.asarray(Image.open(P / ref["observed"])) > 0
        selected_cases = [x for x in paired["validation_cases"] if x["reference_id"] == ref["id"]]
        assert len(selected_cases) == 5
        for item in selected_cases:
            for name in [ref["target"], ref["observed"], item["input"]]:
                assert sha(P / name) == paired["assets_sha256"][name]
                bind(P / name, bindings)
            save_case({"id": item["id"], "source": ref["source"], "evidence": "synthetic_paired_development",
                       "identity": ref["id"], "native_size": ref["native_size"], "profile": item["profile"],
                       "evaluation_source_kind": ref["evaluation_source_kind"], "target_pixel_sha256": canonical,
                       "clean_reference_available": True, "used_in_previous_development": True},
                      image(P / item["input"]), support, target)
    assert len(cases) == 88
    terms = OUT / "source_terms"
    terms.mkdir()
    for path in [C / "LICENSE_SOURCE.html", C / "DERIVATIVE_NOTICE.txt", ROOT / "third_party/codeformer/LICENSE"]:
        destination = terms / ("codeformer_LICENSE" if path.name == "LICENSE" else path.name)
        shutil.copyfile(path, destination)
        bind(destination, bindings)
    plan = {"format": "restoration-checkpoint-comparison-v1", "date": "2026-10-15", "device": "CPU", "threads": 4,
            "arms": arms, "cases": cases, "bindings": bindings,
            "input_policy": "Identical retained RGB PNG 256 inputs; native neutral center pad and bilinear resize; NumPy float32 divided by float32(255).",
            "output_policy": "Unchanged native float32 RAW; floor-to-uint8 native PNG with only unobserved padding restored; separate app grayscale display PNG.",
            "selection_policy": "All 24 QMUL DEV + all 24 ChokePoint DEV; first four validation references per paired source in immutable protocol order, all five profiles.",
            "resize_control": "Exact prepared bilinear input, without a neural forward",
            "no_training": True, "optimizer_updates": 0, "gradient_queries": 0, "final_pixels_read": 0,
            "limits": {"per_dgp_seconds": 300, "per_codeformer_seconds": 900, "total_seconds": 2400, "min_initial_free_bytes": 2 * 1024**3},
            "interpretation": ["Development selection evidence only, never an independent final evaluation.",
                               "Native CCTV has no aligned clean target; PSNR/SSIM must be null for native cases.",
                               "Older checkpoint training overlap is not independently excluded; filenames are not verified lifetime epoch histories.",
                               "Folder name asian_faces is source metadata, not inferred ethnicity. No Zamboanga performance claim.",
                               "No mask detector, completion, top-k score, sharpening, CLAHE, per-model tuning or app promotion."]}
    write(OUT / "plan.json", plan)
    print(json.dumps({"prepared": True, "checkpoint_entries": len(arms), "unique_states": len(seen), "cases": len(cases), "plan_sha256": sha(OUT / "plan.json"), "neural_calls": 0}), flush=True)


def run_arm(aid):
    started = time.monotonic()
    plan = load_json(OUT / "plan.json")
    assert sha(SCRIPT) == plan["bindings"][rel(SCRIPT)]
    arm = next(x for x in plan["arms"] if x["id"] == aid)
    assert arm["execution_id"] == aid
    for name, digest in plan["bindings"].items():
        assert sha(ROOT / name) == digest, name
    folder = OUT / "results" / aid
    assert not folder.exists(), f"Do not overwrite retained inference: {aid}"
    folder.mkdir(parents=True)
    model, provenance = load_model(arm)
    before = state_sha(model.state_dict())
    from face_color_policy import preserve_input_palette
    rows, calls = [], 0
    with torch.inference_mode():
        for case in plan["cases"]:
            pixels = image(OUT / case["input"])
            support = np.asarray(Image.open(OUT / case["observed"])) > 0
            canonical = pixels.astype(np.float32) / np.float32(255)
            x = torch.from_numpy(canonical.copy()).permute(2, 0, 1).unsqueeze(0)
            tick = time.monotonic()
            raw = model(x)[0].permute(1, 2, 0).cpu().numpy().copy()
            elapsed = time.monotonic() - tick
            calls += 1
            assert raw.shape == (256, 256, 3) and raw.dtype == np.float32
            assert np.isfinite(raw).all() and raw.min() >= 0 and raw.max() <= 1
            assert not any(m.training for m in model.modules())
            native_png = np.floor(raw * np.float32(255)).astype(np.uint8)
            native_png[~support] = pixels[~support]
            display, palette = preserve_input_palette(pixels, np.zeros(support.shape, np.uint8), native_png, visible_restored=True)
            display[~support] = pixels[~support]
            path = folder / case["id"]
            path.mkdir()
            np.save(path / "raw.npy", raw, allow_pickle=False)
            Image.fromarray(native_png).save(path / "native.png")
            Image.fromarray(display).save(path / "display.png")
            row = {"id": case["id"], "forward_seconds": elapsed, "palette": palette,
                   "input_pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                   "files": {n: sha(path / n) for n in ["raw.npy", "native.png", "display.png"]}}
            rows.append(row)
            if calls == 1 or calls % 20 == 0 or calls == len(plan["cases"]):
                print(json.dumps({"arm": aid, "forwards": calls, "of": len(plan["cases"]), "seconds": round(time.monotonic() - started, 2)}), flush=True)
    after = state_sha(model.state_dict())
    assert before == after, "Inference mutated checkpoint state"
    assert sha(ROOT / arm["checkpoint"]) == arm["file_sha256"]
    write(folder / "receipt.json", {"complete": True, "arm": aid, "plan_sha256": sha(OUT / "plan.json"),
                                   "provenance": provenance, "runtime": {"torch": torch.__version__, "device": "cpu", "threads": torch.get_num_threads()},
                                   "neural_calls": calls, "optimizer_updates": 0, "gradient_queries": 0,
                                   "state_before_sha256": before, "state_after_sha256": after, "rows": rows,
                                   "seconds": time.monotonic() - started, "automatic_app_promotion": False})


def run_all():
    plan = load_json(OUT / "plan.json")
    started = time.monotonic()
    for arm in plan["arms"]:
        if arm["execution_id"] != arm["id"]:
            continue
        receipt = OUT / "results" / arm["id"] / "receipt.json"
        if receipt.exists():
            old = load_json(receipt)
            assert old["complete"] and old["plan_sha256"] == sha(OUT / "plan.json")
            print(json.dumps({"retained_complete": arm["id"]}), flush=True)
            continue
        remaining = plan["limits"]["total_seconds"] - (time.monotonic() - started)
        cap = min(remaining, plan["limits"]["per_codeformer_seconds" if arm["architecture"] == "codeformer" else "per_dgp_seconds"])
        assert cap > 0, "Total inference wall cap reached"
        log = OUT / (arm["id"] + ".log")
        assert not log.exists(), "Retain prior attempt log"
        with log.open("x", encoding="utf-8") as stream:
            result = subprocess.run([sys.executable, "-X", "utf8", "-B", "-u", str(SCRIPT), "--arm", arm["id"]],
                                    cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=cap, check=False)
        print(json.dumps({"arm": arm["id"], "exit_code": result.returncode, "log": rel(log), "elapsed_seconds": round(time.monotonic() - started, 2)}), flush=True)
        assert result.returncode == 0, f"Inference stopped; retain {log}"
    write(OUT / "execution_complete.json", {"complete": True, "plan_sha256": sha(OUT / "plan.json"),
                                          "unique_executed_arms": len({x["execution_id"] for x in plan["arms"]}),
                                          "cases_per_arm": len(plan["cases"]), "optimizer_updates": 0,
                                          "seconds": time.monotonic() - started, "quality_or_promotion_implied": False})


def main():
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.manual_seed(20261015)
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--prepare", action="store_true")
    choice.add_argument("--run-all", action="store_true")
    choice.add_argument("--arm")
    a = parser.parse_args()
    if a.prepare:
        prepare()
    elif a.arm:
        run_arm(a.arm)
    else:
        run_all()


if __name__ == "__main__":
    main()
