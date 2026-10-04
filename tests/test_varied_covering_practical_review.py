"""Practical review scope/refusal counterexamples; no models or optimizers."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_varied_covering_practical_review as review
from tests.test_varied_covering_results_audit import temporary_workspace


class PracticalReviewTests(unittest.TestCase):
    def protocol(self):
        cases = json.loads((review.ROOT / "outputs/real_camera_inference_review_v2/frozen_protocol.json").read_text())["cases"]
        return {"cases": cases, "reproduction_cases": [{"name": f"real/{i:03d}"} for i in range(16)],
                "threshold": .5, "proposal_margin_256_pixels": 3, "device": "cpu", "threads": 4,
                "budget": {"reproduction_forward_images": 34, "practical_forward_images": 72,
                           "total_forward_images": 106, "cached_proposals": 72, "practical_metric_records": 144,
                           "generator_forward_images": 0, "restorer_forward_images": 0, "optimizer_updates": 0,
                           "wall_seconds_excluding_loading": 300},
                "checkpoint_selection": False, "promoted": False, "training": False}

    def test_missing_return_refuses_before_any_protocol_or_model_read(self):
        with temporary_workspace() as root, patch.object(review, "ROOT", root), patch.object(review, "read") as read:
            with self.assertRaisesRegex(ValueError, "Audited VM return required"):
                review.run()
            read.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])

    def test_dropping_a_hard_family_or_rejection_case_is_refused(self):
        for field, value in (("family", "uncovered_control"), ("expected_rejection", False), ("synthetically_degraded", 1)):
            protocol = self.protocol()
            index = next(i for i, row in enumerate(protocol["cases"]) if row["expected_rejection"])
            protocol["cases"][index][field] = value
            with self.subTest(field=field):
                with self.assertRaises(ValueError): review.validate_protocol(protocol)

    def test_threshold_margin_and_training_scope_cannot_change(self):
        for field, value in (("threshold", .4), ("proposal_margin_256_pixels", 5),
                             ("promoted", True), ("checkpoint_selection", True), ("training", True)):
            protocol = self.protocol(); protocol[field] = value
            with self.subTest(field=field):
                with self.assertRaises(ValueError): review.validate_protocol(protocol)

    def test_false_cannot_substitute_for_zero_update_budget(self):
        protocol = self.protocol()
        review.validate_protocol(protocol)
        protocol["budget"]["optimizer_updates"] = False
        with self.assertRaisesRegex(ValueError, "type differs"): review.validate_protocol(protocol)

    def test_nonfinite_soft_range_wrong_shape_and_dtype_are_refused(self):
        original = np.full((256, 256), .5, np.float32)
        self.assertIs(review.validate_probability(original), original)
        for changed in (original.astype(np.float64), original[:255], original + .6,
                        np.full_like(original, np.nan), np.full_like(original, np.inf)):
            with self.assertRaises(ValueError): review.validate_probability(changed)


if __name__ == "__main__": unittest.main()
