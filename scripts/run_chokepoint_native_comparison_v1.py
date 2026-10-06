"""Compare frozen restorers on24 input-reviewed native CCTV crops; no training."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cctv_dgp_pilot import state_hash
from cctv_input_quality import observed_quality_signals
from dgp_face_workflow_v3 import canonical_tensor, DEFAULT_DGP, DGP_SHA256
from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
from pretrained_face_restoration import load_face_restorer, WEIGHTS_SHA256

BASE = ROOT / "outputs/cctv_chokepoint_native_development_v1"
OUT = ROOT / "outputs/cctv_chokepoint_native_comparison_v1"
PHASE3 = ROOT / "checkpoints/dgp_zamboanga_final.pth"
PHASE3_SHA = "b6376f56c4161ef1f26a8f9efb0bb309b3014f4ecdeaca5399a47c1c4607586c"
CF = ROOT / "outputs/codeformer_restoration_pretrained_v1/codeformer.pth"
ARMS = ("resize", "phase3", "identity_v2", "codeformer_w1", "identity_v2_auto")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def pixels(path, mode="RGB"):
    with Image.open(path) as image:
        return np.asarray(image.convert(mode)).copy()


def main():
    if OUT.exists():
        raise ValueError("Preserve completed/partial comparison; no automatic repeat")
    subset, review = read(BASE / "frozen_subset.json"), read(BASE / "input_review.json")
    if review["subset_sha256"] != sha(BASE / "frozen_subset.json") or not review["reviewed_before_model_outputs"]:
        raise ValueError("Require the frozen input-only review")
    cases = [case for case in subset["cases"] if case["role"] == "development"]
    if len(cases) != 24 or review["usable_cases"] != 24 or any(row["input_review"] != "usable" for row in review["rows"]):
        raise ValueError("Preserve selected cohort and input decisions; no replacements")
    paths = [Path(__file__), BASE / "frozen_subset.json", BASE / "input_review.json",
             BASE / "selection_plan.json", BASE / "independent_geometry_audit.json", PHASE3, DEFAULT_DGP, CF,
             ROOT / "cctv_dgp_pilot.py", ROOT / "cctv_input_quality.py", ROOT / "face_workflow.py",
             ROOT / "dgp_face_workflow_v3.py", ROOT / "dgp_face_restoration.py",
             ROOT / "dgp_frozen_inference_v2.py", ROOT / "cctv_dgp_frozen_norm.py",
             ROOT / "pretrained_face_restoration.py"]
    paths.extend(sorted((ROOT / "models").glob("*.py")))
    paths.extend(sorted((ROOT / "third_party/codeformer").glob("*.py")))
    paths.extend([ROOT / "third_party/codeformer/LICENSE", BASE / "LICENSE_SOURCE.html", BASE / "DERIVATIVE_NOTICE.txt"])
    for case in cases:
        for key in ("native_crop", "input", "observed"):
            path = BASE / case[key]
            if sha(path) != case[key + "_sha256"]:
                raise ValueError("Changed native/prepared source")
            paths.append(path)
    if sha(PHASE3) != PHASE3_SHA or sha(DEFAULT_DGP) != DGP_SHA256 or sha(CF) != WEIGHTS_SHA256:
        raise ValueError("Checkpoint fingerprint differs")
    OUT.mkdir()
    for folder in ("raw", "images"):
        (OUT / folder).mkdir()
    (OUT / "LICENSE_SOURCE.html").write_bytes((BASE / "LICENSE_SOURCE.html").read_bytes())
    (OUT / "DERIVATIVE_NOTICE.txt").write_text(
        "THESIS RESEARCH DERIVATIVES: Saved outputs/contact sheets are frozen-model restoration estimates "
        "from our ChokePoint native face crops, not original released photographs or recovered identity. "
        "Retain LICENSE_SOURCE.html. Acknowledge NICTA and Wong et al., CVPR Workshops2011, "
        "DOI10.1109/CVPRW.2011.5981881. Noncommercial research only.\n", encoding="utf-8", newline="\n")
    write(OUT / "plan.json", {"date": "2026-10-05", "frozen_before_model_outputs": True,
          "cases": cases, "arms": list(ARMS),
          "common_input": "Same exact uint8 native eye-based crop, native128 padding and Pillow bilinear256; same NumPy float32/255 tensor to all three models. No extra alignment, enhancement or synthetic degradation.",
          "raw_and_display": "Save each model's untouched float32 HWC[0,1]. PNG floor(raw*float32(255)), original padding restored; no display enhancement. Auto is an exact cached current-policy alias of input or retained DGP PNG, not another forward.",
          "phase3": {"checkpoint": PHASE3.relative_to(ROOT).as_posix(), "sha256": PHASE3_SHA,
                     "normalization": "Stored evaluation statistics with disposable kernel buffer copies"},
          "primary_dgp": {"checkpoint": DEFAULT_DGP.relative_to(ROOT).as_posix(), "sha256": DGP_SHA256,
                          "normalization": "Retained stored evaluation statistics; no per-image substitution"},
          "pretrained_comparison": {"model": "Official CodeFormer restoration, separate from inpainting",
             "weights_sha256": WEIGHTS_SHA256, "fidelity": 1.0, "adain": True,
             "internal_resolution": 512, "returned_resolution": 256,
             "alignment_limit": "Unrotated common face crops, not the complete recommended detection/FFHQ alignment pipeline. This comparison cannot establish performance of that full pipeline."},
          "auto": "Reuse the unchanged observed-support suggest_restoration policy; no threshold fitting or structure-classifier claim",
          "criteria": read(BASE / "selection_plan.json")["input_review_criteria"],
          "budget": {"phase3_forwards": 24, "identity_v2_forwards": 24, "codeformer_forwards": 24,
                     "maximum_total_forwards": 72, "wall_seconds_including_loading": 300,
                     "outer_timeout_seconds": 360, "device": "cpu", "threads": 4},
          "paired_reference": False, "PSNR": None, "SSIM": None, "identity_accuracy": None,
          "reserved_source13_and_original_qmul32_used": False, "source_country_inferred": False,
          "ethnicity_inferred": False, "zamboanga_validation": False, "independent_final_review": False,
          "app_changed": False, "training": False, "checkpoint_adoption": False,
          "sources_sha256": {path.relative_to(ROOT).as_posix(): sha(path) for path in paths}})
    started = time.monotonic()
    counts = {"phase3": 0, "identity_v2": 0, "codeformer": 0}
    records, handles = [], []
    try:
        torch.set_num_threads(4)
        phase3, p3 = load_frozen_dgp_restorer(PHASE3, expected_sha256=PHASE3_SHA, device="cpu")
        primary, pd = load_frozen_dgp_restorer(DEFAULT_DGP, expected_sha256=DGP_SHA256, device="cpu")
        codeformer, pc = load_face_restorer(CF, device="cpu")
        models = {"phase3": phase3, "identity_v2": primary, "codeformer": codeformer}
        if any(model.training or any(p.requires_grad for p in model.parameters()) for model in models.values()):
            raise ValueError("Only frozen inference permitted")
        before = {name: state_hash(model) for name, model in models.items()}
        for name, model in models.items():
            def count(module, args, output, model_name=name):
                counts[model_name] += 1
            handles.append(model.register_forward_hook(count))
        write(OUT / "execution.json", {"plan_sha256": sha(OUT / "plan.json"), "model_state_before": before,
             "provenance": {"phase3": p3, "identity_v2": pd, "codeformer": pc},
             "torch_version": torch.__version__, "threads": 4, "optimizer_constructed": False})
        for index, case in enumerate(cases):
            if time.monotonic() - started > 300:
                raise TimeoutError("Finite comparison cap; preserve partial outputs")
            common, observed = pixels(BASE / case["input"]), pixels(BASE / case["observed"], "L") != 0
            tensor = canonical_tensor(common, "cpu")
            with torch.inference_mode():
                outputs = {"phase3": phase3(tensor), "identity_v2": primary(tensor),
                           "codeformer_w1": codeformer(tensor, fidelity=1.0)}
            images, paths, changes = {"resize": common}, {}, {}
            for arm, value in outputs.items():
                if value.shape != (1, 3, 256, 256) or value.dtype != torch.float32 or not torch.isfinite(value).all() or value.min() < 0 or value.max() > 1:
                    raise ValueError("Invalid frozen model output")
                raw = value[0].permute(1, 2, 0).numpy().copy()
                raw_path = f"raw/{case['id']}_{arm}.npy"
                np.save(OUT / raw_path, raw, allow_pickle=False)
                image = np.floor(raw * np.float32(255)).astype(np.uint8)
                image[~observed] = common[~observed]
                images[arm] = image
                changes[arm] = float(np.abs(image.astype(np.float64) - common.astype(np.float64))[observed].mean())
            signals = observed_quality_signals(common, observed, np.zeros_like(observed))
            chosen = "identity_v2" if signals["suggest_restoration"] else "resize"
            images["identity_v2_auto"] = images[chosen]
            for arm, image in images.items():
                name = f"images/{case['id']}_{arm}.png"
                Image.fromarray(image).save(OUT / name)
                paths[arm] = name
            records.append({"id": case["id"], "source_person_id": case["source_person_id"],
                 "native_size": [case["native_width"], case["native_height"]], "outputs": paths,
                 "auto_selected": chosen, "quality_signals": signals,
                 "observed_MAE_change_255_diagnostic_only": changes,
                 "input_change_is_quality_or_identity_metric": False})
            write(OUT / f"progress_{index + 1:02d}.json", {"cases": index + 1, "forwards": counts.copy(),
                  "seconds": time.monotonic() - started})
            print(json.dumps({"case": index + 1, "total": 24, "id": case["id"], "auto": chosen,
                              "seconds": time.monotonic() - started}), flush=True)
        after = {name: state_hash(model) for name, model in models.items()}
        if before != after or counts != {"phase3": 24, "identity_v2": 24, "codeformer": 24}:
            raise ValueError("Frozen state/forward count differs")
        for first in range(0, 24, 4):
            page = Image.new("RGB", (1340, 1192), "white")
            draw = ImageDraw.Draw(page)
            for col, arm in enumerate(ARMS):
                draw.text((268 * col + 5, 5), arm, fill="black")
            for index, row in enumerate(records[first:first + 4]):
                y = 24 + 292 * index
                draw.text((5, y), row["id"] + f" / native{row['native_size']} / Auto={row['auto_selected']}", fill="black")
                for col, arm in enumerate(ARMS):
                    page.paste(Image.fromarray(pixels(OUT / row["outputs"][arm])), (268 * col + 5, y + 24))
            page.save(OUT / f"comparison-{first // 4 + 1:02d}.png")
        seconds = time.monotonic() - started
        if seconds > 300:
            raise TimeoutError("Comparison completion cap")
        write(OUT / "results.json", {"complete": True, "date": "2026-10-05", "seconds": seconds,
              "plan_sha256": sha(OUT / "plan.json"), "records": records, "model_forwards": counts,
              "model_state_after": after, "model_states_unchanged": True, "optimizer_updates": 0,
              "backward_calls": 0, "native_evidence_unpaired": True, "PSNR": None, "SSIM": None,
              "reserved_source13_and_original_qmul32_used": False, "app_changed": False,
              "visual_review_pending": True, "independent_final_review": False, "restoration_qualified": False,
              "artifacts_sha256": {path.relative_to(OUT).as_posix(): sha(path) for path in OUT.rglob("*") if path.is_file()}})
        print(json.dumps({"complete": True, "seconds": seconds, "forwards": counts}), flush=True)
    except Exception as error:
        write(OUT / "failure.json", {"complete": False, "error": str(error), "type": type(error).__name__,
             "seconds": time.monotonic() - started, "completed_cases": len(records), "forwards": counts,
             "partial_outputs_preserved": True, "training_calls": 0})
        raise
    finally:
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    main()
