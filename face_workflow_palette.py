"""Palette-preserving extension of the frozen, audited inference route."""
import hashlib
import time

from face_color_policy import preserve_input_palette
from face_workflow import FaceWorkflow

POLICY = "reviewed-face-workflow-v2"


class PaletteFaceWorkflow(FaceWorkflow):
    def configuration(self):
        return {**super().configuration(), "policy":POLICY}

    def generate(self, rgb, mask, restoration="auto"):
        started = time.perf_counter()
        output, metadata = super().generate(rgb, mask, restoration)
        fixed, color = preserve_input_palette(rgb, mask, output, metadata["restoration_applied"])
        metadata.update({"base_policy":metadata["policy"], "policy":POLICY,
                         "pre_palette_output_rgb_sha256":metadata["output_rgb_sha256"],
                         "color_policy":color, "elapsed_seconds":time.perf_counter()-started,
                         "output_rgb_sha256":hashlib.sha256(fixed.tobytes()).hexdigest()})
        return fixed, metadata
