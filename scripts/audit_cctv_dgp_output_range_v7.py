"""Training-only arithmetic bound on the frozen DGP RGB residual; no model calls.

The pinned forward computes clamp(x + 0.5*tanh(z), 0, 1). Targets outside
[max(0, x-0.5), min(1, x+0.5)] are unreachable at those pixels, regardless
of training. This is an oracle error floor, not a generated face or quality claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

PROTOCOL_SHA = "0fe7d036bf8567bf0860790df553afd627ac50bd5096cfda29cc7d09d71d320c"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def range_floor(low, target, support):
    lower, upper = np.maximum(0., low - .5), np.minimum(1., low + .5)
    distance = np.maximum(lower - target, 0.) + np.maximum(target - upper, 0.)
    values = distance[support]
    if not len(values):
        raise ValueError("Empty fixed support")
    return {"mse_floor": float(np.square(values).mean()),
            "mae_floor": float(values.mean()),
            "unreachable_rgb_fraction": float((values > 1e-12).mean()),
            "unreachable_pixel_fraction": float((values > 1e-12).any(axis=1).mean())}


def run(root):
    started = time.monotonic()
    if sha(root / "targets_protocol_v6.json") != PROTOCOL_SHA:
        raise ValueError("Frozen V6 cohort differs")
    p = json.loads((root / "targets_protocol_v6.json").read_text())
    refs = {r["id"]: r for r in p["references"]}
    verified = set()

    def asset(name):
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Asset escaped root")
        if name not in verified:
            if sha(path) != p["assets_sha256"][name]:
                raise ValueError("Frozen asset differs: " + name)
            verified.add(name)
        return path

    model_path = asset("models/dgp_synthesizer.py")
    text = model_path.read_text()
    for expression in ("res = torch.tanh(final) + x_norm", "res = torch.clamp(res, min=-1.0, max=1.0)",
                       "out = (res + 1.0) / 2.0", "x_norm = x_up * 2.0 - 1.0"):
        if expression not in text:
            raise ValueError("Documented output-range derivation no longer matches pinned source")
    cases = sum((p["training_epochs"][str(epoch)] for epoch in (1, 2)), [])
    training_ids = {r["id"] for r in p["references"] if r["role"] == "train"}
    if len(cases) != 782 or len(training_ids) != 391 or {c["reference_id"] for c in cases} != training_ids:
        raise ValueError("Training-only cohort differs")
    cached, rows = {}, []
    yy, xx = np.mgrid[:256, :256]
    for case in cases:
        if time.monotonic() - started > 90:
            raise TimeoutError("Training-only range audit exceeded 90 seconds")
        ref = refs[case["reference_id"]]
        if ref["role"] != "train":
            raise ValueError("Non-training reference requested")
        if ref["id"] not in cached:
            target = np.asarray(Image.open(asset(ref["identity_target"])).convert("RGB"), dtype=np.float64) / 255
            support = np.asarray(Image.open(asset(ref["observed"]))) > 0
            matrix = np.asarray(ref["matrix112"], dtype=np.float64)
            u, v = matrix[0, 0]*xx + matrix[0, 1]*yy + matrix[0, 2], matrix[1, 0]*xx + matrix[1, 1]*yy + matrix[1, 2]
            face = support & (u >= 0) & (u < 112) & (v >= 0) & (v < 112)
            cached[ref["id"]] = target, support, face
        target, support, face = cached[ref["id"]]
        low = np.asarray(Image.open(asset(case["input"])).convert("RGB"), dtype=np.float64) / 255
        rows.append({"id": case["id"], "reference_id": ref["id"], "profile": case["profile"],
                     "source": ref["source"], "input_sha256": p["assets_sha256"][case["input"]],
                     "observed": range_floor(low, target, support), "fixed_face_footprint": range_floor(low, target, face)})
    groups = {}
    for profile in sorted({r["profile"] for r in rows}):
        selected = [r for r in rows if r["profile"] == profile]
        groups[profile] = {"cases": len(selected), **{
            region: {key: float(np.mean([r[region][key] for r in selected])) for key in selected[0][region]}
            for region in ("observed", "fixed_face_footprint")}}
    return {"complete": True, "protocol_sha256": PROTOCOL_SHA, "pinned_dgp_source_sha256": sha(model_path),
            "training_cases": len(rows), "training_references": len(cached), "assets_checked": len(verified),
            "groups": groups, "rows": rows, "seconds": time.monotonic() - started,
            "model_forwards": 0, "backward_calls": 0, "optimizer_updates": 0,
            "validation_used": False, "native_used": False, "native_reserved_used": False,
            "model_improvement_established": False,
            "limitation": "Best possible pointwise float-RGB floor under the pinned residual bound; not attainable restoration, saturation measurement, gradient evidence, or causal attribution of V6 regressions"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print({k: v for k, v in result.items() if k != "rows"})
