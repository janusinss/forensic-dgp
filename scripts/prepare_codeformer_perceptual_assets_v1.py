"""Finite verified asset preparation and CPU forward-only LPIPS parity.

No pip installation, gradients, optimizer updates, training, or VM connection.
Downloads official sources with normal TLS verification; all originals retained.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import ssl
import sys
import time
import urllib.request
import zipfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
ASSETS = REPO / "outputs/codeformer_perceptual_assets_v1"
WHEEL_URL = "https://files.pythonhosted.org/packages/9b/13/1df50c7925d9d2746702719f40e864f51ed66f307b20ad32392f1ad2bb87/lpips-0.1.4-py3-none-any.whl"
WHEEL_SHA = "fd537af5828b69d2e6ffc0a397bd506dbc28ca183543617690844c08e102ec5e"
VGG_URL = "https://download.pytorch.org/models/vgg16-397923af.pth"
LICENSES = {
    "LICENSE_LPIPS.txt": "https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/LICENSE",
    "LICENSE_TORCHVISION.txt": "https://raw.githubusercontent.com/pytorch/vision/main/LICENSE",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def download(url, path, expected_sha=None, prefix=None, maximum_bytes=600_000_000):
    if path.exists():
        digest = sha(path)
        if (expected_sha and digest != expected_sha) or (prefix and not digest.startswith(prefix)):
            raise ValueError(f"Existing asset hash differs: {path}")
        return {"path": path.name, "bytes": path.stat().st_size, "sha256": digest, "url": url}
    part = path.with_name(path.name + ".partial")
    start = time.monotonic()
    total = 0
    request = urllib.request.Request(url, headers={"User-Agent": "ForensicThesisAssetAudit/1.0"})
    with urllib.request.urlopen(request, timeout=30, context=ssl.create_default_context()) as response:
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > maximum_bytes:
            raise ValueError("Download size exceeds declared finite cap")
        with part.open("xb") as stream:
            while data := response.read(1024 * 1024):
                if time.monotonic() - start > 600:
                    raise TimeoutError("Asset download exceeded 600-second cap")
                total += len(data)
                if total > maximum_bytes:
                    raise ValueError("Download exceeded byte cap")
                stream.write(data)
        if declared and total != int(declared):
            raise ValueError("Download length differs")
    digest = sha(part)
    if (expected_sha and digest != expected_sha) or (prefix and not digest.startswith(prefix)):
        raise ValueError("Downloaded asset hash differs")
    part.rename(path)
    print(json.dumps({"downloaded": path.name, "bytes": total, "sha256": digest,
                      "seconds": time.monotonic() - start}), flush=True)
    return {"path": path.name, "bytes": total, "sha256": digest, "url": url}


def prepare(root):
    import torch
    from research.codeformer_adaptation_v1.perceptual import ASSET_SCHEMA
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in (manifest["original_vgg16"], manifest["wheel"], manifest["features"],
                      manifest["calibration"], *manifest["licenses"]):
            path = root / entry["path"]
            if path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
                raise ValueError("Retained asset manifest mismatch")
        print(json.dumps({"assets_already_verified": True, "manifest_sha256": sha(manifest_path)}), flush=True)
        return manifest
    wheel = download(WHEEL_URL, root / "lpips-0.1.4-py3-none-any.whl", WHEEL_SHA, maximum_bytes=100_000)
    original = download(VGG_URL, root / "vgg16-397923af.pth", prefix="397923af")
    notices = [download(url, root / filename, maximum_bytes=30_000) for filename, url in LICENSES.items()]
    extracted = root / "reference_wheel"
    with zipfile.ZipFile(root / wheel["path"]) as package:
        for member in package.infolist():
            destination = (extracted / member.filename).resolve()
            if not destination.is_relative_to(extracted.resolve()):
                raise ValueError("Unsafe package member")
        package.extractall(extracted)
        raw_calibration = package.read("lpips/weights/v0.1/vgg.pth")
    calibration_path = root / "lpips_vgg_v01.pth"
    with calibration_path.open("xb") as stream:
        stream.write(raw_calibration)
    state = torch.load(root / original["path"], map_location="cpu", weights_only=True)
    features = {key: value for key, value in state.items() if key.startswith("features.")}
    if len(features) != 26 or not all(torch.isfinite(value).all() for value in features.values()):
        raise ValueError("Unexpected VGG16 feature tensor set")
    compact_path = root / "vgg16_features.pth"
    torch.save({"params": features, "source_sha256": original["sha256"],
                "conversion": "Exact features.* tensors; classifier excluded"}, compact_path)
    reloaded = torch.load(compact_path, map_location="cpu", weights_only=True)
    if set(reloaded["params"]) != set(features) or not all(
            torch.equal(reloaded["params"][key], state[key]) for key in features):
        raise ValueError("Compact feature tensor equality failed")
    del state, features, reloaded
    manifest = {"schema": ASSET_SCHEMA, "wheel": wheel, "original_vgg16": original,
                "features": {"path": compact_path.name, "bytes": compact_path.stat().st_size,
                             "sha256": sha(compact_path), "exact_source_tensor_count": 26},
                "calibration": {"path": calibration_path.name, "bytes": calibration_path.stat().st_size,
                                "sha256": sha(calibration_path),
                                "wheel_member": "lpips/weights/v0.1/vgg.pth"},
                "licenses": notices, "full_original_retained": True,
                "package_installations": 0, "optimizer_updates": 0,
                "normalization": "Standard LPIPS RGB[-1,1], normalize=False, v0.1 scaling once",
                "source_references": ["https://github.com/richzhang/PerceptualSimilarity",
                                      "https://pypi.org/project/lpips/0.1.4/",
                                      "https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py"],
                "checksum_scope": "Wheel SHA is PyPI-published; VGG16 full SHA is observed and publisher prefix checked"}
    write_new(manifest_path, manifest)
    return manifest


def parity(root, receipt_name="inference_parity.json"):
    import torch
    from research.codeformer_adaptation_v1.perceptual import load_perceptual
    torch.set_num_threads(4)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    # The official reference source must still be byte-identical to the pinned wheel.
    with zipfile.ZipFile(root / manifest["wheel"]["path"]) as package:
        for name in ("lpips/__init__.py", "lpips/lpips.py", "lpips/pretrained_networks.py"):
            if (root / "reference_wheel" / name).read_bytes() != package.read(name):
                raise ValueError("Extracted LPIPS reference source differs from the pinned wheel")
    sys.path.insert(0, str(root / "reference_wheel"))
    lpips = importlib.import_module("lpips")
    if not Path(lpips.__file__).resolve().is_relative_to((root / "reference_wheel").resolve()):
        raise ValueError("LPIPS reference is not the exact extracted wheel")
    reference = lpips.LPIPS(net="vgg", version="0.1", lpips=True, spatial=False,
                            pnet_rand=True, pnet_tune=False, use_dropout=True,
                            model_path=str(root / manifest["calibration"]["path"]),
                            eval_mode=True, verbose=False).eval().requires_grad_(False)
    features = torch.load(root / manifest["features"]["path"], map_location="cpu", weights_only=True)["params"]
    for group, indices in enumerate((range(4), range(4, 9), range(9, 16), range(16, 23), range(23, 30)), 1):
        part = {key.removeprefix("features."): value for key, value in features.items()
                if int(key.split(".")[1]) in indices}
        getattr(reference.net, f"slice{group}").load_state_dict(part, strict=True)
    calibration = torch.load(root / manifest["calibration"]["path"], map_location="cpu", weights_only=True)
    if not all(torch.equal(reference.state_dict()[key], value) for key, value in calibration.items()):
        raise ValueError("Official reference calibration loading differs")
    candidate = load_perceptual(root)
    candidate.train(True)
    if any(module.training for module in candidate.modules()) or any(p.requires_grad for p in candidate.parameters()):
        raise ValueError("Perceptual eval/freeze contract failed")
    cases = []
    with torch.inference_mode():
        for index, size in enumerate((64, 64, 128)):
            coordinate = torch.linspace(-1, 1, size)
            yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
            clean = torch.stack((xx, yy, torch.sin((index + 1) * 3 * xx) * torch.cos(2 * yy))).unsqueeze(0)
            changed = (torch.roll(clean, shifts=index + 1, dims=-1) * (0.9 - 0.05 * index)
                       + 0.04 * torch.sin(xx * 17).view(1, 1, size, size)).clamp(-1, 1)
            expected = reference(changed, clean, normalize=False).mean()
            actual = candidate(changed, clean)
            alternative = reference((changed + 1) / 2, (clean + 1) / 2, normalize=True).mean()
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(alternative, expected, rtol=1e-5, atol=1e-6)
            cases.append({"fixture": index, "size": size, "official": float(expected),
                          "compact": float(actual), "absolute_error": float((actual - expected).abs()),
                          "normalization_error": float((alternative - expected).abs())})
        identity = candidate(clean, clean)
        if identity.item() != 0:
            raise ValueError("Identity LPIPS is not zero")
    receipt = {"complete": True, "comparison": "Exact extracted lpips0.1.4 reference with same verified VGG16/calibration",
               "manifest_sha256": sha(root / "manifest.json"), "fixtures": cases,
               "identity_distance": float(identity), "absolute_tolerance": 1e-6,
               "relative_tolerance": 1e-5, "eval_forced_after_train_true": True,
               "all_parameters_frozen": True, "forward_only": True,
               "gradient_queries": 0, "backwards": 0, "optimizer_updates": 0,
               "reference_pnet_rand": "True only to suppress downloading; every VGG feature strictly overwritten from verified pretrained tensors before any forward",
               "torch_version": torch.__version__, "reference_source": str(Path(lpips.__file__).relative_to(root)),
               "source_bindings": [{"path": str(path.relative_to(REPO)), "sha256": sha(path)}
                                   for path in (Path(__file__).resolve(),
                                                REPO / "research/codeformer_adaptation_v1/perceptual.py")],
               "reference_wheel_sources_verified_exact": True}
    write_new(root / receipt_name, receipt)
    print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ASSETS)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--parity", action="store_true")
    parser.add_argument("--parity-receipt", default="inference_parity.json")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(REPO / "outputs"):
        raise ValueError("Asset root must stay inside project outputs")
    if Path(args.parity_receipt).name != args.parity_receipt or not args.parity_receipt.endswith(".json"):
        raise ValueError("Parity receipt must be a JSON filename within the asset directory")
    if args.prepare:
        prepare(root)
    if args.parity:
        parity(root, args.parity_receipt)
    if not (args.prepare or args.parity):
        parser.error("Select --prepare and/or --parity")


if __name__ == "__main__":
    main()
