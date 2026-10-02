"""Prepare verified LaMa generator-only inference assets; no training dependencies."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import typing
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]
REVISION = "786f5936b27fb3dacd2b1ad799e4de968ea697e7"
ARCHIVE_SHA = "f1b358ca24093b93a106183b98a3dea6e8ed09f3b43ea7251eb2c81e7b4575f6"
ORIGINAL_SHA = "fccb7adffd53ec0974ee5503c3731c2c2f1e7e07856fd9228cdcc0b46fd5d423"
MIRROR_REVISION = "05cb2be7f8dbe6ca7c6e78f4fc827a4b2baaa4a9"


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


class MetadataOnly:
    """Opaque discarded metadata; never import/execute Lightning or OmegaConf."""

    def __new__(cls, *args, **kwargs):
        if args or kwargs:
            raise ValueError("Unexpected constructor in discarded checkpoint metadata")
        return super().__new__(cls)


class CallbackKey:
    def __new__(cls, *args, **kwargs):
        raise ValueError("The callback class may only be an inert dictionary key")


def metadata_defaultdict(default_factory=None, *args):
    """Discard the default factory of unused optimizer/configuration mappings."""
    if default_factory not in (None, dict, list, int):
        raise ValueError("Unexpected discarded-metadata default factory")
    return dict(*args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "scratch/lama_source_v1")
    parser.add_argument("--assets", type=Path, default=ROOT / "outputs/lama_pretrained_v1")
    args = parser.parse_args()
    assets, vendor = args.assets.resolve(), ROOT / "third_party/lama"
    if not assets.is_relative_to(ROOT) or vendor.exists() or (assets / "generator.pth").exists():
        raise ValueError("Preserve prior assets/source; use a fresh prepared location")
    if sha(assets / "big-lama.zip") != ARCHIVE_SHA:
        raise ValueError("Published LaMa archive SHA256 mismatch")
    source_revision = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
    if source_revision != REVISION:
        raise ValueError("LaMa source revision mismatch")
    upstream = {}
    for name in ("LICENSE", "saicinpainting/training/modules/ffc.py",
                 "saicinpainting/training/modules/base.py", "saicinpainting/training/modules/squeeze_excitation.py",
                 "saicinpainting/training/trainers/default.py", "saicinpainting/evaluation/data.py"):
        upstream[name] = subprocess.check_output(["git", "-C", str(args.source), "show", f"{REVISION}:{name}"])
    original = assets / "big-lama/models/best.ckpt"
    with zipfile.ZipFile(assets / "big-lama.zip") as archive:
        for name in ("big-lama/config.yaml", "big-lama/models/best.ckpt"):
            destination = assets / name
            if not destination.exists():
                archive.extract(name, assets)
    if sha(original) != ORIGINAL_SHA:
        raise ValueError("Original LaMa checkpoint SHA256 mismatch")
    # The published Lightning checkpoint contains configuration objects/callback
    # class keys. Replace *only* those metadata globals with inert types in the
    # restricted weights-only unpickler. Unknown globals remain forbidden.
    opaque_names = ("omegaconf.nodes.AnyNode", "omegaconf.base.Metadata", "omegaconf.base.ContainerMetadata",
                    "omegaconf.dictconfig.DictConfig", "omegaconf.listconfig.ListConfig")
    permitted = [(MetadataOnly, name) for name in opaque_names] + [
        (CallbackKey, "pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint"),
        dict, list, int, (metadata_defaultdict, "collections.defaultdict"), typing.Any,
    ]
    unsafe = set(torch.serialization.get_unsafe_globals_in_checkpoint(original))
    expected = set(opaque_names) | {"pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint",
                                  "builtins.dict", "builtins.list", "builtins.int", "collections.defaultdict", "typing.Any"}
    if unsafe != expected:
        raise ValueError(f"Unexpected published-checkpoint metadata: {sorted(unsafe ^ expected)}")
    with torch.serialization.safe_globals(permitted):
        checkpoint = torch.load(original, map_location="cpu", weights_only=True)
    generator = {key.removeprefix("generator."): value for key, value in checkpoint["state_dict"].items()
                 if key.startswith("generator.")}
    if not generator or not all(isinstance(value, torch.Tensor) and torch.isfinite(value).all() for value in generator.values()):
        raise ValueError("Invalid generator tensor state")
    torch.save({"format": "dgp-lama-inference-v1", "generator": generator,
                "original_checkpoint_sha256": ORIGINAL_SHA, "archive_sha256": ARCHIVE_SHA,
                "source_revision": REVISION}, assets / "generator.pth")
    vendor.mkdir(parents=True)
    (vendor / "__init__.py").write_text("", encoding="utf-8")
    (vendor / "LICENSE").write_bytes(upstream["LICENSE"])
    ffc = upstream["saicinpainting/training/modules/ffc.py"].decode()
    ffc = ffc.split("\nclass FFCNLayerDiscriminator", 1)[0]
    ffc = ffc.replace("from saicinpainting.training.modules.base import get_activation, BaseDiscriminator", "from .helpers import get_activation")
    ffc = ffc.replace("from saicinpainting.training.modules.spatial_transform import LearnableSpatialTransformWrapper", "from .helpers import LearnableSpatialTransformWrapper")
    ffc = ffc.replace("from saicinpainting.training.modules.squeeze_excitation import SELayer", "from .squeeze_excitation import SELayer")
    ffc = ffc.replace("from saicinpainting.utils import get_shape\n", "")
    (vendor / "ffc.py").write_text(ffc, encoding="utf-8", newline="\n")
    (vendor / "squeeze_excitation.py").write_bytes(upstream["saicinpainting/training/modules/squeeze_excitation.py"])
    base = upstream["saicinpainting/training/modules/base.py"].decode()
    activation = next(node for node in ast.parse(base).body if isinstance(node, ast.FunctionDef) and node.name == "get_activation")
    helpers = "import torch.nn as nn\n\n\n" + ast.get_source_segment(base, activation)
    helpers += '\n\n\nclass LearnableSpatialTransformWrapper(nn.Module):\n    def __init__(self, *args, **kwargs):\n        raise ValueError("Spatial-transform configurations are outside the pinned Big-LaMa inference path")\n'
    (vendor / "helpers.py").write_text(helpers, encoding="utf-8", newline="\n")
    provenance = {
        "source_url": "https://github.com/advimman/lama", "source_revision": REVISION,
        "license": "Apache-2.0; source license retained", "model": "Big-LaMa Places2, not face-specific",
        "mirror_url": f"https://huggingface.co/smartywu/big-lama/resolve/{MIRROR_REVISION}/big-lama.zip",
        "mirror_revision": MIRROR_REVISION, "archive_bytes": (assets / "big-lama.zip").stat().st_size,
        "archive_sha256": ARCHIVE_SHA, "original_checkpoint_sha256": ORIGINAL_SHA,
        "config_sha256": sha(assets / "big-lama/config.yaml"), "generator_sha256": sha(assets / "generator.pth"),
        "generator_state_items": len(generator),
        "metadata_policy": "weights_only=True with inert placeholders exclusively for discarded OmegaConf metadata and callback key; no original callback/config imports",
        "upstream_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in upstream.items()},
        "vendor_sha256": {p.name: sha(p) for p in sorted(vendor.iterdir()) if p.is_file()},
        "source_changes": "Generator/FFT code unchanged; relative imports, unused discriminator/get_shape removal; exact get_activation and SELayer retained; unused spatial-transform path explicitly rejected",
        "optimizer_constructed": False, "optimizer_updates": 0, "model_forwards": 0,
    }
    (assets / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    (vendor / "README.md").write_text(
        "# Vendored LaMa generator\n\nApache-2.0 source from https://github.com/advimman/lama at " + REVISION +
        ". Copyright 2021 Samsung Research. Retain LICENSE. FFC source also credits https://github.com/pkumivision/FFC.\n\n"
        "The pinned Big-LaMa generator/FFT operations are unchanged. Imports use minimal local helpers; discriminator and unused get_shape import are removed. "
        "Spatial-transform variants are rejected because the pinned config does not use them. Training code/dependencies are not bundled. "
        "Preparation provenance and upstream hashes are saved in outputs/lama_pretrained_v1/provenance.json. "
        "The prepared generator tensors derive from the original archive; no training occurs.\n", encoding="utf-8")
    print(json.dumps({"prepared": True, "generator_sha256": provenance["generator_sha256"],
                      "state_items": len(generator), "model_forwards": 0}))


if __name__ == "__main__":
    main()
