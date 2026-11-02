"""Finite, admission-gated MI-GAN FFHQ256 comparison. Preparation never imports torch."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
HISTORY = Path(r"C:\xampp\htdocs\YEAR 4\Testing")
ASSETS = ROOT / "outputs/completion_migan256_assets_v1"
OUT = ROOT / "outputs/completion_migan256_comparison_v1"
OLD_PROTOCOL = HISTORY / "outputs/completion_mat_mirror_comparison_v1/protocol.json"
REVISION = "2b793c5ece43f4253e32d4afc257120a5deed6f5"
SOURCE_SHA = "1e35a181e516742ae32af935848e6a9a4382f528ea7096db6658f14cd8b9ace0"
WEIGHT_SHA = "3cab897cfcc3fcbe224510cedad06a634ae8bf5bd7120428a32fd7224f4237f4"
LIMITS = {"requests": 32, "case_forwards": 28, "parity_forwards": 1,
          "maximum_forwards": 29, "empty_bypasses": 4, "case_seconds": 120,
          "worker_seconds": 600, "external_seconds": 630,
          "maximum_output_bytes": 150 * 1024**2}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def progress(value):
    value["updated_UTC"] = datetime.now(timezone.utc).isoformat()
    target = OUT / "progress.json"
    temporary = OUT / f".progress-{os.getpid()}.tmp"
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    os.replace(temporary, target)


def image_array(path, mode):
    import numpy as np
    from PIL import Image
    with Image.open(path) as image:
        return np.asarray(image.convert(mode)).copy()


def bound_file(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file(), f"Unsafe/missing file: {relative}")
    return path


def check_pixels(image, mask):
    import numpy as np
    require(image.shape == (256, 256, 3) and image.dtype == np.uint8, "Input must be RGB256 uint8")
    require(mask.shape == (256, 256) and mask.dtype == np.uint8, "Mask must be L256 uint8")
    require(bool(np.isin(mask, [0, 255]).all()), "Mask must be exactly binary 0/255")


def numpy_input(image, removal):
    """Official operation order, provided separately for non-neural mathematical tests."""
    import numpy as np
    known = (1 - removal.astype(np.float32))[None, None]
    rgb = image.astype(np.float32).transpose(2, 0, 1)[None] * 2 / 255 - 1
    return np.concatenate([known - 0.5, rgb * known], axis=1)


def compose(image, removal, raw):
    import numpy as np
    require(raw.shape == (256, 256, 3) and raw.dtype == np.float32, "Invalid raw prediction")
    require(bool(np.isfinite(raw).all()), "Nonfinite raw prediction")
    decoded = np.clip(raw * np.float32(0.5) + np.float32(0.5), 0, 1)
    # Official demo truncates after multiplying by 255, rather than rounding.
    generated8 = (decoded * np.float32(255)).astype(np.uint8)
    final = image.copy()
    final[removal] = generated8[removal]
    completed = image.astype(np.float32) / np.float32(255)
    completed[removal] = decoded[removal]
    require(np.array_equal(final[~removal], image[~removal]), "Outside pixels changed")
    return decoded, completed, final


def asset_bindings():
    acquisition = read(ASSETS / "acquisition.json")
    require(acquisition.get("complete") is True and acquisition["source_revision"] == REVISION, "Acquisition incomplete")
    require(acquisition["files"]["migan_inference.py"]["sha256"] == SOURCE_SHA, "Wrong source acquisition")
    require(acquisition["files"]["migan_256_ffhq.pt"]["sha256"] == WEIGHT_SHA, "Wrong weight acquisition")
    require(acquisition["files"]["migan_256_ffhq.pt"]["bytes"] == 25231129, "Wrong weight size")
    files = {"acquisition.json": sha(ASSETS / "acquisition.json")}
    for name, item in acquisition["files"].items():
        path = bound_file(ASSETS, name)
        require(path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], f"Asset drift: {name}")
        files[name] = item["sha256"]
    return files


def prepare():
    require(not OUT.exists(), "Comparison directory exists; preparation never overwrites")
    assets = asset_bindings()
    old = read(OLD_PROTOCOL)
    require(old["status"] == "frozen_before_generation" and old["assisted_only"], "Wrong historical protocol")
    require(len(old["cases"]) == 32 and len(old["excluded_before_generation"]) == 4, "Wrong historical counts")
    cases = []
    nonempty = 0
    ids = set()
    for original in old["cases"]:
        case = dict(original)
        require(case["id"] not in ids and Path(case["id"]).name == case["id"], "Unsafe/duplicate case ID")
        ids.add(case["id"])
        case["bindings_sha256"] = {}
        for key in ("input", "reviewed", "output", "stages", "metadata"):
            path = bound_file(HISTORY, case[key])
            digest = sha(path)
            require(digest == old["sources_sha256"][case[key]], f"Historical case drift: {case[key]}")
            case["bindings_sha256"][key] = digest
        image = image_array(HISTORY / case["input"], "RGB")
        mask = image_array(HISTORY / case["reviewed"], "L")
        check_pixels(image, mask)
        case["mask_pixels"] = int((mask == 255).sum())
        nonempty += bool(case["mask_pixels"])
        require(case["operator_input_review"] == "usable", "Unreviewed input")
        cases.append(case)
    require(nonempty == 28, "Expected 28 nonempty and four exact bypasses")
    expected_families = {"face_mask", "sunglasses", "strong_lens_glare", "hand", "obstructing_hair", "scarf", "other_object", "uncovered_control", "clear_glasses_control"}
    require(expected_families <= {case["family"] for case in cases}, "Missing covering family/control")
    protocol = {"format": "migan-ffhq256-reviewed-completion-v1", "status": "frozen_before_generation",
                "frozen_UTC": datetime.now(timezone.utc).isoformat(), "source_revision": REVISION,
                "runner_sha256": sha(__file__), "assets_sha256": assets,
                "historical_protocol": str(OLD_PROTOCOL), "historical_protocol_sha256": sha(OLD_PROTOCOL),
                "source_root": str(HISTORY), "cases": cases,
                "excluded_before_generation": old["excluded_before_generation"],
                "parity_case_id": next(case["id"] for case in cases if case["mask_pixels"]),
                "budgets": LIMITS, "threads": 4, "device": "cpu", "precision": "FP32",
                "seed": 240, "internal_resolution": 256, "delivered_resolution": 256,
                "normalization": "K=1-M; RGB.float32*2/255-1; cat[K-.5,RGB*K] in that order",
                "decode": "clip(raw*.5+.5,0,1)*255 then truncation to uint8; direct original uint8 copy outside M",
                "mask": "Historical reviewed binary removal mask unchanged; no expansion or feather",
                "restoration": "Off; retain historical CodeFormer Off PNG as fixed baseline",
                "assisted_only": True, "hidden_ground_truth": None, "hidden_PSNR_SSIM": None,
                "scope": "Previously exposed development photographs and synthetic degradations; native suffix is not CCTV",
                "pretraining_overlap": "Unknown", "new_training": False, "optimizer_updates": 0,
                "review_criteria": ["Plausible anatomy and appearance at delivered256 across every case",
                                    "Record leftovers, seams, skin/gaze inconsistencies and unsupported cases",
                                    "Exact original bytes outside the fixed reviewed mask; exact empty-mask bypass",
                                    "All seven covering families and both control types retained",
                                    "No identity recovery, hidden-pixel metrics, automatic qualification or app promotion"],
                "goal_complete": False}
    OUT.mkdir()
    write_new(OUT / "protocol.json", protocol)
    print(json.dumps({"prepared": True, "protocol_sha256": sha(OUT / "protocol.json"),
                      "requests": 32, "maximum_forwards": 29, "neural_calls": 0}, indent=2))


def validate(protocol_sha, admission_path, admission_sha=None):
    require(sha(OUT / "protocol.json") == protocol_sha, "Protocol SHA mismatch")
    protocol = read(OUT / "protocol.json")
    require(protocol["runner_sha256"] == sha(__file__), "Runner changed after preparation")
    require(protocol["budgets"] == LIMITS and protocol["threads"] == 4, "Budget changed")
    require(protocol["assets_sha256"] == asset_bindings(), "Assets changed")
    require(protocol["historical_protocol_sha256"] == sha(OLD_PROTOCOL), "Historical protocol changed")
    require(protocol["source_root"] == str(HISTORY) and len(protocol["cases"]) == 32, "Source scope changed")
    for case in protocol["cases"]:
        for key, digest in case["bindings_sha256"].items():
            require(sha(bound_file(HISTORY, case[key])) == digest, f"Case drift: {case['id']}/{key}")
    receipt = read(admission_path)
    if admission_sha is not None:
        require(sha(admission_path) == admission_sha, "Admission changed")
    require(receipt.get("complete") is True and receipt.get("independent") is True, "Independent admission required")
    require(receipt.get("safe_state_dict") is True, "Safe archive admission required")
    for key, digest in {"source_sha256": SOURCE_SHA, "checkpoint_sha256": WEIGHT_SHA,
                        "runner_sha256": sha(__file__), "protocol_sha256": protocol_sha}.items():
        require(receipt.get(key) == digest, f"Admission binding missing: {key}")
    return protocol


def state_sha(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def worker(protocol_sha, admission_path, admission_sha):
    started = time.monotonic()
    protocol = validate(protocol_sha, admission_path, admission_sha)
    require(os.environ.get("MIGAN_FINITE_SUPERVISED") == protocol_sha, "Worker requires its owning supervisor")
    # No torch or model-module import occurs before independent admission validation.
    import numpy as np
    from PIL import Image, ImageDraw
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    torch.manual_seed(240)
    write_new(OUT / "execution.json", {"complete": False, "pid": os.getpid(), "protocol_sha256": protocol_sha,
                                      "admission_sha256": admission_sha, "torch": str(torch.__version__),
                                      "device": "cpu", "threads": 4, "grad_enabled": False,
                                      "optimizer_constructed": False, "optimizer_updates": 0})
    rows, forward_events = [], []
    model, before = None, None
    active = {"phase": "loading", "worker_pid": os.getpid(), "finished_requests": 0,
              "forward_started": 0, "forward_finished": 0, "worker_started_unix": time.time()}
    progress(active)
    try:
        spec = importlib.util.spec_from_file_location("admitted_migan_inference_v1", ASSETS / "migan_inference.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # weights_only is explicit: no arbitrary Python pickle-object loading fallback exists.
        state = torch.load(ASSETS / "migan_256_ffhq.pt", map_location="cpu", weights_only=True)
        require(isinstance(state, dict) and 1 <= len(state) <= 500, "Expected bounded plain state dictionary")
        require(all(type(key) is str and isinstance(value, torch.Tensor) for key, value in state.items()), "Non-tensor state")
        require(sum(value.numel() for value in state.values()) <= 10000000, "State tensor cap")
        require(all(value.device.type == "cpu" and bool(torch.isfinite(value).all()) for value in state.values()), "Invalid CPU state")
        model = module.Generator(resolution=256).cpu().float().eval().requires_grad_(False)
        model.load_state_dict(state, strict=True)
        del state
        require(not model.training and not any(value.requires_grad for value in model.parameters()), "Model not frozen")
        before = state_sha(model)
        for folder in ("inputs", "masks", "baseline", "images", "stages", "preview"):
            (OUT / folder).mkdir()

        def forward(image, removal, identity, parity=False):
            require(len(forward_events) < LIMITS["maximum_forwards"], "Forward cap")
            require(time.monotonic() - started < LIMITS["worker_seconds"], "Worker time cap")
            event = {"number": len(forward_events) + 1, "id": identity, "parity": parity,
                     "started_unix": time.time(), "finished": False}
            forward_events.append(event)
            active.update(phase="forward", active_case=identity,
                          active_forward_started_unix=event["started_unix"], forward_started=len(forward_events))
            if parity:
                active["active_started_unix"] = event["started_unix"]
            progress(active)
            began = time.monotonic()
            with torch.inference_mode():
                known = torch.from_numpy((1 - removal.astype(np.float32)))[None, None]
                rgb = torch.from_numpy(image).permute(2, 0, 1)[None].float() * 2 / 255 - 1
                network_input = torch.cat([known - 0.5, rgb * known], dim=1)
                raw_tensor = model(network_input)
                require(raw_tensor.shape == (1, 3, 256, 256) and raw_tensor.dtype == torch.float32, "Wrong output geometry/precision")
                raw = raw_tensor[0].permute(1, 2, 0).cpu().numpy().copy()
            event.update(finished=True, seconds=time.monotonic() - began, finished_unix=time.time())
            active.update(phase="saving", active_forward_started_unix=None,
                          forward_finished=sum(item["finished"] for item in forward_events))
            progress(active)
            require(event["seconds"] <= LIMITS["case_seconds"], "Forward exceeded120s")
            return raw

        for number, case in enumerate(protocol["cases"], 1):
            require(number <= LIMITS["requests"], "Request cap")
            case_started = time.monotonic()
            active.update(phase="case", active_case=case["id"], active_started_unix=time.time())
            progress(active)
            image = image_array(HISTORY / case["input"], "RGB")
            mask8 = image_array(HISTORY / case["reviewed"], "L")
            check_pixels(image, mask8)
            removal = mask8 == 255
            name = case["id"]
            for folder, key in (("inputs", "input"), ("masks", "reviewed"), ("baseline", "output")):
                shutil.copyfile(HISTORY / case[key], OUT / folder / (name + ".png"))
            raw = forward(image, removal, name) if removal.any() else None
            if raw is None:
                decoded = completed = image.astype(np.float32) / np.float32(255)
                final = image.copy()
            else:
                decoded, completed, final = compose(image, removal, raw)
            artifacts = {}
            for label, array in (("raw256", raw), ("decoded256", decoded), ("completed256", completed)):
                if array is not None:
                    target = OUT / "stages" / (name + "_" + label + ".npy")
                    with target.open("xb") as stream:
                        np.save(stream, array, allow_pickle=False)
                    artifacts[label] = target.relative_to(OUT).as_posix()
                    artifacts[label + "_sha256"] = sha(target)
                else:
                    artifacts[label] = artifacts[label + "_sha256"] = None
            output = OUT / "images" / (name + ".png")
            with output.open("xb") as stream:
                Image.fromarray(final).save(stream, format="PNG")
            require(np.array_equal(image_array(output, "RGB")[~removal], image[~removal]), "Saved outside bytes changed")
            elapsed = time.monotonic() - case_started
            require(elapsed <= LIMITS["case_seconds"], "Case exceeded120s")
            row = {"id": name, "family": case["family"], "condition": case["condition"], "mode": "assisted",
                   "output": output.relative_to(OUT).as_posix(), "output_sha256": sha(output),
                   "mask_pixels": int(removal.sum()), "empty_mask_bypass": raw is None,
                   "exact_outside_reviewed_mask": True, "seconds": elapsed,
                   "hidden_PSNR_SSIM": None, "exact_hidden_identity_claim": False, **artifacts}
            rows.append(row)
            write_new(OUT / f"progress_{number:02d}.json", row)
            active.update(phase="between_cases", finished_requests=number, active_started_unix=None)
            progress(active)
            print(json.dumps({"finished": number, "of": 32, "id": name, "seconds": elapsed,
                              "forward_count": len(forward_events)}), flush=True)
        # One predeclared replay only, independent of generated appearance or quality.
        first = next(case for case in protocol["cases"] if case["id"] == protocol["parity_case_id"])
        image = image_array(HISTORY / first["input"], "RGB")
        removal = image_array(HISTORY / first["reviewed"], "L") == 255
        replay = forward(image, removal, first["id"], parity=True)
        saved = np.load(OUT / "stages" / (first["id"] + "_raw256.npy"), allow_pickle=False)
        require(np.array_equal(saved, replay), "Fixed first-case replay differs")
        with (OUT / "stages/parity_first_raw256.npy").open("xb") as stream:
            np.save(stream, replay, allow_pickle=False)
        after = state_sha(model)
        require(before == after, "Generator state changed")
        require(len(rows) == 32 and len(forward_events) == 29 and sum(row["empty_mask_bypass"] for row in rows) == 4, "Final counts differ")
        require(sha(ASSETS / "migan_256_ffhq.pt") == WEIGHT_SHA, "Checkpoint changed")
        validate(protocol_sha, admission_path, admission_sha)
        active.update(phase="rendering", active_started_unix=None)
        progress(active)
        for page in range(8):
            sheet = Image.new("RGB", (1024, 1148), "#16181c")
            draw = ImageDraw.Draw(sheet)
            for col, label in enumerate(("input256", "reviewed removal", "CodeFormer Off", "MI-GAN FFHQ256")):
                draw.text((col * 256 + 4, 5), label, fill="white")
            for slot, case in enumerate(protocol["cases"][page * 4:(page + 1) * 4]):
                image = image_array(OUT / "inputs" / (case["id"] + ".png"), "RGB")
                removal = image_array(OUT / "masks" / (case["id"] + ".png"), "L") == 255
                overlay = image.astype(np.float32)
                overlay[removal] = overlay[removal] * .6 + np.array([255, 120, 20], dtype=np.float32) * .4
                arrays = (image, np.rint(overlay).astype(np.uint8), image_array(OUT / "baseline" / (case["id"] + ".png"), "RGB"), image_array(OUT / "images" / (case["id"] + ".png"), "RGB"))
                for col, array in enumerate(arrays):
                    sheet.paste(Image.fromarray(array), (col * 256, 28 + slot * 280))
                draw.text((4, 286 + slot * 280), case["id"] + " / assisted", fill="white")
            with (OUT / "preview" / f"page_{page+1:02d}.png").open("xb") as stream:
                sheet.save(stream, format="PNG")
        require(time.monotonic() - started <= LIMITS["worker_seconds"], "Worker600s cap")
        require(sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file()) <= LIMITS["maximum_output_bytes"], "Output byte cap")
        report = {"complete": True, "protocol_sha256": protocol_sha, "admission_sha256": admission_sha,
                  "requests": 32, "pretrained_forwards": 28, "parity_forwards": 1, "total_forwards": 29,
                  "empty_mask_bypasses": 4, "rows": rows, "forward_events": forward_events,
                  "generator_state_before": before, "generator_state_after": after, "state_unchanged": True,
                  "weight_sha256_before": WEIGHT_SHA, "weight_sha256_after": sha(ASSETS / "migan_256_ffhq.pt"),
                  "parity_first_raw_exact": True, "parity_raw_sha256": sha(OUT / "stages/parity_first_raw256.npy"),
                  "optimizer_constructed": False, "optimizer_updates": 0, "gradient_calls": 0,
                  "seconds": time.monotonic() - started, "assisted_only": True,
                  "automatic_quality_qualification": False, "assisted_quality_qualification": False,
                  "independent_final_review": False, "goal_complete": False}
        write_new(OUT / "results.json", report)
        active.update(phase="finished", complete=True, finished_requests=32, active_started_unix=None)
        progress(active)
    except BaseException as exc:
        write_new(OUT / "failure.json", {"complete": False, "cause": repr(exc), "traceback": traceback.format_exc(),
                                        "finished_rows": rows, "forward_events": forward_events,
                                        "generator_state_before": before, "seconds": time.monotonic() - started,
                                        "optimizer_updates": 0, "goal_complete": False})
        raise


def supervise(protocol_sha, admission_path):
    validate(protocol_sha, admission_path)
    require(not (OUT / "execution.json").exists() and not (OUT / "supervisor_started.json").exists(), "Run exists; never rerun/overwrite")
    admission_sha = sha(admission_path)
    start = time.monotonic()
    child = None
    reason = None
    write_new(OUT / "supervisor_started.json", {"protocol_sha256": protocol_sha, "pid": os.getpid(),
                                               "admission_sha256": admission_sha, "started_unix": time.time()})
    env = dict(os.environ)
    env["MIGAN_FINITE_SUPERVISED"] = protocol_sha
    with (OUT / "worker.log").open("x", encoding="utf-8") as log:
        child = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "--_worker",
                                  "--protocol-sha", protocol_sha, "--admission", str(admission_path),
                                  "--admission-sha", admission_sha], cwd=str(ROOT), env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                elapsed = time.monotonic() - start
                if elapsed > LIMITS["external_seconds"]:
                    reason = "External630s total cap"
                if (OUT / "progress.json").exists():
                    current = read(OUT / "progress.json")
                    began = current.get("active_started_unix")
                    if began is not None and time.time() - began > LIMITS["case_seconds"]:
                        reason = "External120s active case/forward cap"
                    if current.get("forward_started", 0) > 29 or current.get("finished_requests", 0) > 32:
                        reason = "External request/forward cap"
                if reason:
                    child.kill()  # Only this supervisor's Popen child; no process-name/tree kills.
                    break
                time.sleep(.2)
        except BaseException:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=10)
            raise
        returncode = child.wait(timeout=10)
    write_new(OUT / "supervisor_receipt.json", {"complete": returncode == 0 and reason is None,
                                               "child_pid": child.pid, "returncode": returncode,
                                               "stopped_reason": reason, "seconds": time.monotonic() - start,
                                               "only_owned_child_terminated": bool(reason), "goal_complete": False})
    require(returncode == 0 and reason is None, f"Finite worker failed: {reason or returncode}; see retained worker.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--protocol-sha")
    parser.add_argument("--admission", type=Path, default=ASSETS / "source_admission.json")
    parser.add_argument("--admission-sha")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        require(args.protocol_sha is not None, "Explicit prepared protocol SHA required")
        if args.run:
            supervise(args.protocol_sha, args.admission.resolve())
        else:
            require(args.admission_sha is not None, "Explicit admission SHA required")
            worker(args.protocol_sha, args.admission.resolve(), args.admission_sha)


if __name__ == "__main__":
    main()
