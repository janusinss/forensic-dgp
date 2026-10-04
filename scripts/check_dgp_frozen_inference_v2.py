"""Finite CPU inference check; no training or native evaluation inputs."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from dgp_face_restoration import PHASE3_SHA256, load_dgp_restorer, sha
from dgp_frozen_inference_v2 import load_frozen_dgp_restorer


def state_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def run():
    output = ROOT / "outputs/dgp_frozen_inference_v2_real_check.json"
    if output.exists():
        raise ValueError("Preserve existing inference check; do not repeat automatically")
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(20261004)
    checkpoint = ROOT / "checkpoints/dgp_zamboanga_final.pth"
    old, _ = load_dgp_restorer(checkpoint, device="cpu", expected_sha256=PHASE3_SHA256)
    fixed, provenance = load_frozen_dgp_restorer(checkpoint, device="cpu", expected_sha256=PHASE3_SHA256)
    initial = state_fingerprint(fixed)
    assert state_fingerprint(old) == initial
    errors = []
    for size in (1, 6):
        if time.monotonic() - started > 120:
            raise TimeoutError("Finite120-second forward-only check exceeded")
        old.net.load_state_dict(fixed.net.state_dict(), strict=True)
        # Artificial RGB signals exercise batch semantics, not output usefulness.
        x = torch.linspace(0.2, 0.8, size * 3 * 256 * 256).reshape(size, 3, 256, 256)
        expected, actual = old(x), fixed(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert state_fingerprint(fixed) == initial
        assert all(not value.requires_grad and value.grad is None for value in fixed.parameters())
        errors.append({"batch_size": size, "maximum_output_difference": float((actual - expected).abs().max())})
    assert sha(checkpoint) == PHASE3_SHA256
    receipt = {
        "complete": True, "seconds": time.monotonic() - started,
        "device": "cpu", "torch": torch.__version__, "checkpoint_sha256": PHASE3_SHA256,
        "corrected_state_before": initial, "corrected_state_after": state_fingerprint(fixed),
        "checks": errors, "real_checkpoint_forward_calls": 4,
        "image_exposures": 14, "inputs": "artificial RGB signals; no dataset images",
        "provenance": provenance, "local_backward_calls": 0, "local_optimizer_updates": 0,
        "native_reserved_used": False, "model_improvement_established": False,
        "production_promoted": False,
        "source_sha256": {name: sha(ROOT / name) for name in (
            "scripts/check_dgp_frozen_inference_v2.py", "dgp_frozen_inference_v2.py",
            "dgp_face_restoration.py", "cctv_dgp_frozen_norm.py")},
    }
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    run()
