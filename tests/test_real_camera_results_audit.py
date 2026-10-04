"""Counterexamples for independent return auditing; no models or optimizers."""
import copy
import unittest
import numpy as np

from scripts.audit_real_camera_results import recount, audit_steps, exact


class CameraReturnAuditContract(unittest.TestCase):
    def test_ignored_predictions_do_not_become_false_clear_errors(self):
        target = np.zeros((256, 256), dtype=bool)
        valid = np.ones_like(target); valid[:5] = False
        prediction = ~valid
        counted = recount(prediction, target, valid)
        self.assertEqual(counted["fp"], 0)
        self.assertEqual(counted["negative_false_positive_cases"], 0)
        self.assertEqual(counted["ignored_positive_pixels"], 1280)
        prediction[10, 10] = True
        counted = recount(prediction, target, valid)
        self.assertEqual(counted["fp"], 1)
        self.assertEqual(counted["negative_false_positive_cases"], 1)

    def test_missing_step_and_changed_input_condition_are_rejected(self):
        batch = {"real": [{"index": 0, "degraded": False}], "fixture": [0]}
        schedule = [[batch for _ in range(56)] for _ in range(2)]
        rows = [{"arm": "native83", "experiment_epoch": i // 56 + 1, "step": i % 56 + 1,
                 "experiment_updates": i + 1, "optimizer_state_step": 673 + i, "cumulative_model_updates": 883 + i,
                 **copy.deepcopy(batch), "loss": 1., "real_loss": 1., "fixture_loss": 1., "pre_clip_norm": 1.}
                for i in range(112)]
        with self.assertRaises(ValueError):
            audit_steps(rows[1:], schedule, "native83")
        rows[71]["real"][0]["degraded"] = True
        with self.assertRaises(ValueError):
            audit_steps(rows, schedule, "native83")

    def test_false_cannot_replace_integer_zero_in_provenance(self):
        with self.assertRaises(ValueError):
            exact({"updates": False}, {"updates": 0}, "counter")


if __name__ == "__main__":
    unittest.main()
