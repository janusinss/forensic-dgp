"""Training-only oracle for a prior-rendered difference; no learned fitting."""
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cctv_dgp_pilot import state_hash
from dgp_face_code_conditioner_v11 import load_teacher, to_prior_canvas, from_prior_canvas
from dgp_face_workflow_v3 import canonical_tensor, DEFAULT_DGP, DGP_SHA256
from dgp_frozen_inference_v2 import load_frozen_dgp_restorer

OUT = ROOT / "outputs/cctv_dgp_latent_delta_feasibility_v20"
MIXED = ROOT / "outputs/cctv_dgp_mixed_vm_v9_r2"
PARENT = ROOT / "outputs/cctv_dgp_broader_codes_vm_v16_r2/broader_codes_protocol_v16_r2.json"
PARENT_SHA = "4f64cf2204458f8fadcab1824fc4bc293c65803e123fdebb304f7b5bd3a502a0"
TEACHER = ROOT / "outputs/codeformer_teacher_pretrained_v1/vqgan_code1024.pth"
TEACHER_SHA = "4d1c6741b3cffcbdc2cd1a12b2c3c2442282e042d5de66909cb643d4fa31b20f"
PROFILES = ["clear", "blur_lr24", "lowlight_lr32", "motion_lr48", "compound_lr24"]
ARMS = ["resize", "dgp", "input_vq", "target_vq_oracle", "prior_delta_oracle"]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def pixels(path, mode="RGB"):
    with Image.open(path) as image:
        return np.asarray(image.convert(mode)).copy()


def raw(value):
    if value.shape != (1, 3, 256, 256) or value.dtype != torch.float32:
        raise ValueError("Unexpected RGB tensor")
    if not torch.isfinite(value).all() or value.min() < 0 or value.max() > 1:
        raise ValueError("Nonfinite/out-of-range RGB")
    return value[0].permute(1, 2, 0).numpy().copy()


def metrics(image, target, observed):
    mse = float(np.square((image.astype(np.float64) - target.astype(np.float64)) / 255.)[observed].mean())
    interior = cv2.erode(observed.astype(np.uint8), np.ones((7, 7), np.uint8)) != 0
    interior[:3] = interior[-3:] = False
    interior[:, :3] = interior[:, -3:] = False
    if not interior.any():
        raise ValueError("Empty metric support")
    ssim = float(np.mean([structural_similarity(image[..., c], target[..., c], data_range=255,
                         full=True)[1][interior].mean() for c in range(3)]))
    return {"MSE_observed": mse, "PSNR_observed": float(-10 * np.log10(max(mse, 1e-12))),
            "SSIM_valid_observed_windows": ssim}


def main():
    if OUT.exists():
        raise ValueError("Preserve original/partial diagnostic; no repeat")
    if sha(PARENT) != PARENT_SHA or sha(DEFAULT_DGP) != DGP_SHA256 or sha(TEACHER) != TEACHER_SHA:
        raise ValueError("Frozen dependency changed")
    p = read(PARENT)
    ids = [p["train_preview_reference_ids"][index] for index in (0, 5)]
    refs = {r["id"]: r for r in p["references"]}
    cases, sources = [], [Path(__file__), PARENT, DEFAULT_DGP, TEACHER,
        ROOT / "dgp_face_code_conditioner_v11.py", ROOT / "dgp_face_workflow_v3.py",
        ROOT / "dgp_frozen_inference_v2.py", ROOT / "dgp_face_restoration.py",
        ROOT / "cctv_dgp_frozen_norm.py", ROOT / "cctv_dgp_pilot.py",
        ROOT / "CCTV_DGP_STRUCTURE_V18_RESULTS.md", ROOT / "CCTV_DGP_PROCESSING_DIAGNOSTICS_V3.md",
        ROOT / "CCTV_NATIVE_SOURCE_EXTENSION_V1.md"]
    sources.extend(sorted((ROOT / "models").glob("*.py")))
    sources.extend(sorted((ROOT / "third_party/codeformer").glob("*.py")))
    sources.append(ROOT / "third_party/codeformer/LICENSE")
    for rid in ids:
        ref = refs[rid]
        if ref["role"] != "train":
            raise ValueError("Require training-role source")
        for profile in PROFILES:
            selected = [c for c in p["training_cases"] if c["reference_id"] == rid and c["profile"] == profile]
            if len(selected) != 1:
                raise ValueError("Changed source/profile cohort")
            case = {**selected[0], "role": "train", "target": ref["target"], "observed": ref["observed"]}
            for name in (case["input"], case["target"], case["observed"]):
                path = MIXED / name
                if sha(path) != p["data_assets_sha256"][name]:
                    raise ValueError("Changed input/target/support")
                sources.append(path)
            cases.append(case)
    if len(cases) != 10 or len({c["source"] for c in cases}) != 2:
        raise ValueError("Require ten paired synthetic training cases")
    OUT.mkdir()
    (OUT / "raw").mkdir()
    (OUT / "images").mkdir()
    write(OUT / "plan.json", {"date": "2026-10-05", "frozen_before_inference": True,
        "cases": cases, "arms": ARMS, "sources_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(sources))},
        "hypothesis": "A correction expressed through a fixed face generator may preserve DGP structure better than an unconstrained learned RGB residual. Before fitting, measure whether a clean-target-informed prior difference has the required capacity.",
        "composition": "clamp(retained_DGP + clean_target_VQ_RGB - camera_input_VQ_RGB,0,1); native128 padding restored only for PNG delivery",
        "prior": "Official clean-image VQGAN teacher encoder/nearest quantizer/generator, all frozen. No CodeFormer restoration transformer, code head, feature fusion or ADAIN.",
        "clean_target_usage": "Oracle only, unavailable to actual restoration. This is not a fresh-image learned model and cannot qualify native output or adoption.",
        "difference_from_closed_recipes": "Use the fixed generator's RGB difference relative to the input VQ reconstruction, anchored to unchanged DGP. No free RGB decoder, token CE fit, w1 cascade or normalization substitution.",
        "budget": {"dgp_forwards": 10, "teacher_encoder_forwards": 12, "teacher_quantizer_forwards": 12,
                   "teacher_generator_forwards": 12, "wall_seconds": 240, "external_timeout_seconds": 300,
                   "device": "cpu", "threads": 4},
        "prospective_feasibility_guards": {"clear_raw_and_png_exact_DGP": True,
            "each_degraded_case_MSE_not_worse_than_DGP_tolerance": 1e-12,
            "each_degraded_case_SSIM_not_worse_than_DGP_tolerance": 1e-6,
            "stop_before_conditioner_or_VM_pilot_if_any_failure": True},
        "native_used": False, "reserved_used": False, "validation_used": False, "training": False,
        "parameter_updates": False, "app_changes": False, "checkpoint_selection": False,
        "scope": "Two existing training photographs/five synthetic profiles each; source folders are provenance, not ethnicity. Capacity oracle, not generalization or identity accuracy."})
    started = time.monotonic()
    handles = []
    counts = {"dgp": 0, "teacher_encoder": 0, "teacher_quantizer": 0, "teacher_generator": 0}
    def clock():
        if time.monotonic() - started > 240:
            raise TimeoutError("Finite four-minute diagnostic cap")
    try:
        torch.set_num_threads(4)
        dgp, _ = load_frozen_dgp_restorer(DEFAULT_DGP, expected_sha256=DGP_SHA256, device="cpu")
        teacher = load_teacher(TEACHER, TEACHER_SHA, device="cpu")
        models = {"dgp": dgp, "teacher": teacher}
        if any(net.training or any(p.requires_grad for p in net.parameters()) for net in models.values()):
            raise ValueError("Frozen inference only")
        before = {name: state_hash(net) for name, net in models.items()}
        for name, module in {"dgp": dgp, "teacher_encoder": teacher.encoder,
                             "teacher_quantizer": teacher.quantize, "teacher_generator": teacher.generator}.items():
            def count(module, args, output, key=name):
                counts[key] += 1
            handles.append(module.register_forward_hook(count))
        write(OUT / "execution.json", {"plan_sha256": sha(OUT / "plan.json"), "model_state_before": before,
              "optimizer_constructed": False, "torch": torch.__version__, "threads": 4})
        targets, rows = {}, []
        with torch.inference_mode():
            for rid in ids:
                clock()
                image = canonical_tensor(pixels(MIXED / refs[rid]["target"]), "cpu")
                features = teacher.encoder(to_prior_canvas(image))
                quantized, _, _ = teacher.quantize(features)
                targets[rid] = from_prior_canvas(teacher.generator(quantized))
                np.save(OUT / "raw" / (rid + "_target_vq.npy"), raw(targets[rid]), allow_pickle=False)
            for index, case in enumerate(cases):
                clock()
                camera, target = pixels(MIXED / case["input"]), pixels(MIXED / case["target"])
                observed = pixels(MIXED / case["observed"], "L") != 0
                image = canonical_tensor(camera, "cpu")
                base = dgp(image)
                features = teacher.encoder(to_prior_canvas(image))
                quantized, _, _ = teacher.quantize(features)
                prior_input = from_prior_canvas(teacher.generator(quantized))
                prior_target = targets[case["reference_id"]]
                oracle = (base + (prior_target - prior_input)).clamp(0, 1)
                values = {"dgp": base, "input_vq": prior_input,
                          "target_vq_oracle": prior_target, "prior_delta_oracle": oracle}
                outputs, scores, raw_paths = {}, {}, {}
                for arm in ARMS:
                    if arm == "resize":
                        delivered = camera
                    else:
                        value = raw(values[arm]); raw_name = f"raw/{case['id']}_{arm}.npy"
                        np.save(OUT / raw_name, value, allow_pickle=False); raw_paths[arm] = raw_name
                        delivered = np.floor(value * np.float32(255)).astype(np.uint8)
                        delivered[~observed] = camera[~observed]
                    name = f"images/{case['id']}_{arm}.png"
                    Image.fromarray(delivered).save(OUT / name)
                    outputs[arm], scores[arm] = name, metrics(delivered, target, observed)
                clear_parity = bool(torch.equal(oracle, base)) if case["profile"] == "clear" else None
                rows.append({"id": case["id"], "reference_id": case["reference_id"], "source": case["source"],
                             "profile": case["profile"], "outputs": outputs, "raw": raw_paths, "metrics": scores,
                             "clear_raw_exact_DGP": clear_parity})
                write(OUT / f"progress_{index + 1:02d}.json", {"cases": index + 1, "counts": counts.copy(),
                      "seconds": time.monotonic() - started})
                print(json.dumps({"case": index + 1, "total": 10, "profile": case["profile"],
                                  "seconds": time.monotonic() - started}), flush=True)
        after = {name: state_hash(net) for name, net in models.items()}
        if before != after or counts != {"dgp": 10, "teacher_encoder": 12, "teacher_quantizer": 12, "teacher_generator": 12}:
            raise ValueError("Frozen state/forward count differs")
        failures = []
        for row in rows:
            if row["profile"] == "clear":
                if not row["clear_raw_exact_DGP"] or not np.array_equal(pixels(OUT / row["outputs"]["dgp"]), pixels(OUT / row["outputs"]["prior_delta_oracle"])):
                    failures.append({"id": row["id"], "gate": "clear_exact_DGP"})
                continue
            base, candidate = row["metrics"]["dgp"], row["metrics"]["prior_delta_oracle"]
            if candidate["MSE_observed"] > base["MSE_observed"] + 1e-12:
                failures.append({"id": row["id"], "gate": "degraded_MSE_preservation"})
            if candidate["SSIM_valid_observed_windows"] < base["SSIM_valid_observed_windows"] - 1e-6:
                failures.append({"id": row["id"], "gate": "degraded_SSIM_preservation"})
        sheets = []
        for rid in ids:
            group = [row for row in rows if row["reference_id"] == rid]
            sheet = Image.new("RGB", (1340, 1504), "white"); draw = ImageDraw.Draw(sheet)
            for column, arm in enumerate(ARMS):
                draw.text((268 * column + 5, 5), arm, fill="black")
            for index, row in enumerate(group):
                y = 48 + 292 * index
                draw.text((5, y - 23), row["id"], fill="black")
                for column, arm in enumerate(ARMS):
                    with Image.open(OUT / row["outputs"][arm]) as image:
                        sheet.paste(image, (268 * column + 5, y))
            name = rid + "_oracle.png"; sheet.save(OUT / name)
            sheets.append({"path": name, "sha256": sha(OUT / name), "cases": [row["id"] for row in group]})
        clock()
        artifacts = {p.relative_to(OUT).as_posix(): sha(p) for p in sorted(OUT.rglob("*")) if p.is_file()}
        write(OUT / "results.json", {"complete": True, "date": "2026-10-05", "plan_sha256": sha(OUT / "plan.json"),
            "seconds": time.monotonic() - started, "records": rows, "sheets": sheets,
            "model_forwards": counts, "model_state_after": after, "model_states_unchanged": True,
            "prospective_feasibility_failures": failures, "stop_before_conditioner_or_VM_pilot": bool(failures),
            "paired_synthetic_training_only": True, "target_informed_oracle_not_restoration": True,
            "optimizer_updates": 0, "backward_calls": 0, "native_used": False, "reserved_used": False,
            "validation_used": False, "app_changes": False, "trained_contribution_verified": False,
            "visual_review_pending": True, "goal_complete": False, "artifacts_sha256": artifacts})
        print(json.dumps({"complete": True, "seconds": time.monotonic() - started,
                          "failures": len(failures), "new_training_recipe_allowed_by_this_check": not bool(failures)}), flush=True)
    except BaseException as error:
        if not (OUT / "failure.json").exists():
            write(OUT / "failure.json", {"error_type": type(error).__name__, "message": str(error),
                  "seconds": time.monotonic() - started, "counts": counts,
                  "optimizer_updates": 0, "backward_calls": 0})
        raise
    finally:
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    main()
