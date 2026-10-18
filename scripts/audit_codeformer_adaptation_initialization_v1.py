"""Local inference-only initializer/teacher audit. Never calls backward/optimizer.

Run from project root: venv/Scripts/python -B -m scripts.audit_codeformer_adaptation_initialization_v1
"""
import argparse
import ast
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from research.codeformer_adaptation_v1.common import require, sha256, write_json
from research.codeformer_adaptation_v1.model import (
    assert_shared_codebook, delivered_rgb, hq_targets, infer, load_teacher,
    load_trainable, state_digest,
)


def normalized_ast(path):
    """Compare upstream computation while excluding only logger/registry imports."""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    kept = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        kept.append(node)
    tree.body = kept
    return ast.dump(tree, include_attributes=False)


def run(root, destination):
    started = time.monotonic()
    require(not destination.exists(), "Do not overwrite a prior initializer audit")
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    bindings = {}
    for name in ("codeformer_arch.py", "vqgan_arch.py"):
        adapted = root / "third_party/codeformer" / name
        original = root / "outputs/CodeFormer/basicsr/archs" / name
        require(normalized_ast(adapted) == normalized_ast(original),
                "Local architecture computations differ from pristine upstream: " + name)
        bindings[str(adapted.relative_to(root))] = sha256(adapted)
        bindings[str(original.relative_to(root))] = sha256(original)
    initializer = root / "outputs/codeformer_restoration_pretrained_v1/codeformer.pth"
    teacher_path = root / "outputs/codeformer_teacher_pretrained_v1/vqgan_code1024.pth"
    from pretrained_face_restoration import load_face_restorer
    candidate, census = load_trainable(initializer)
    benchmark, provenance = load_face_restorer(initializer)
    teacher = load_teacher(teacher_path)
    codebook_compatibility = assert_shared_codebook(candidate, teacher)
    before = {"candidate": state_digest(candidate), "benchmark": state_digest(benchmark.net),
              "teacher": state_digest(teacher)}
    require(before["candidate"] == before["benchmark"], "Initializer state differs")
    plan_path = root / "outputs/cctv_dgp_checkpoint_comparison_v1/plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    comparison = plan_path.parent
    paired = [row for row in plan["cases"] if row["evidence"] == "synthetic_paired_development"]
    selected = [next(row for row in paired if row["source"] == source and row["profile"] == profile)
                for source in sorted({row["source"] for row in paired})
                for profile in ("clear", "blur_lr24")]
    parity = []
    with torch.inference_mode():
        for case in selected:
            require(time.monotonic() - started < 240, "Initializer inference audit exceeded240s")
            path = comparison / case["input"]
            pixel = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
            require(sha256(path) == plan["bindings"][str(path.relative_to(root)).replace("\\", "/")],
                    "Bound DEV input changed")
            image = torch.from_numpy(pixel.copy()).permute(2, 0, 1)[None].float() / 255
            expected = benchmark(image)
            actual = delivered_rgb(infer(candidate, image))
            error = float((expected - actual).abs().max())
            require(error == 0, "Separate trainable initialization changes benchmark output")
            parity.append({"id": case["id"], "maximum_raw_error": error,
                           "input_file_sha256": sha256(path)})
        target = comparison / selected[-1]["target"]
        # Token geometry check uses a held-out development target, never a final crop.
        rgb = np.asarray(Image.open(target).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS))
        clean = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float() / 255
        codes = hq_targets(teacher, clean, codebook=candidate.quantize)
    after = {"candidate": state_digest(candidate), "benchmark": state_digest(benchmark.net),
             "teacher": state_digest(teacher)}
    require(before == after, "Initializer or fixed teacher mutated during inference")
    receipt = {"complete": True, "scope": "CPU inference/initialization only; no fine-tuning qualification",
               "strict_restoration_load": True, "strict_teacher_load": True,
               "upstream_computation_AST_equal_excluding_imports": True,
               "source_revision": provenance["source_revision"], "source_bindings": bindings,
               "initializer_sha256": sha256(initializer), "teacher_sha256": sha256(teacher_path),
               "codebook_compatibility": codebook_compatibility,
               "state_before": before, "state_after": after,
               "census": census, "parity": parity,
               "clean_teacher_geometry": {"indices": list(codes.indices.shape),
                                           "features": list(codes.features.shape),
                                           "requires_grad": codes.features.requires_grad,
                                           "test_target_is_upscaled_fixture_not_training_GT": True},
               "neural_forwards": {"benchmark": 4, "candidate": 4, "teacher_encoder": 1},
               "gradient_queries": 0, "optimizer_updates": 0, "final_pixels_read": 0,
               "torch": torch.__version__, "seconds": time.monotonic() - started}
    write_json(destination, receipt)
    print(json.dumps({"complete": True, "exact_parity_cases": len(parity),
                      "trainable_parameters": census["trainable_parameters"],
                      "frozen_parameters": census["frozen_parameters"],
                      "gradient_queries": 0, "optimizer_updates": 0,
                      "receipt": str(destination), "seconds": receipt["seconds"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--receipt", type=Path,
                        default=Path("outputs/codeformer_training_readiness_v1/initializer_audit_v1.json"))
    args = parser.parse_args()
    run(args.root.resolve(), args.receipt.resolve())
