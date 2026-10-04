"""Inference-only loader preserving the corrected CCTV pilot normalization.

The historical adapter remains unchanged. This loader does not select a model
or promote a checkpoint; the caller must supply an audited weight fingerprint.
"""
from pathlib import Path

from cctv_dgp_frozen_norm import install_frozen_instance_norm
from dgp_face_restoration import load_dgp_restorer, sha

NORMALIZATION_SOURCE_SHA256 = "61c7ddc6f002a5ab87856ae43923e0ec8a1f5613c718d0fd6f68f1cab2936060"
ADAPTER_SOURCE_SHA256 = "2d468f53a2b3fab9111d7ab6ccce752b59dab516cb67f286911073b7d4865ab4"
POLICY = "dgp256-observed-canvas-frozen-normalization-v2"


def load_frozen_dgp_restorer(path, *, expected_sha256, device="cpu"):
    """Load explicitly fingerprinted weights with five frozen normalization layers."""
    root = Path(__file__).resolve().parent
    for filename, fingerprint in (
        ("dgp_face_restoration.py", ADAPTER_SOURCE_SHA256),
        ("cctv_dgp_frozen_norm.py", NORMALIZATION_SOURCE_SHA256),
    ):
        if sha(root / filename) != fingerprint:
            raise ValueError("Changed frozen inference dependency: " + filename)
    model, provenance = load_dgp_restorer(path, device=device, expected_sha256=expected_sha256)
    if install_frozen_instance_norm(model.net) != 5:
        raise ValueError("Expected exactly five corrected DGP normalization layers")
    return model, {
        **provenance,
        "policy": POLICY,
        "input_geometry_policy": provenance["policy"],
        "normalization": "stored evaluation statistics; disposable kernel copies",
        "normalization_layers": 5,
        "normalization_source_sha256": NORMALIZATION_SOURCE_SHA256,
        "adapter_source_sha256": ADAPTER_SOURCE_SHA256,
        "local_training_permitted": False,
    }
