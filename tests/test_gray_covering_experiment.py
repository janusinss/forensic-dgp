"""Training-boundary and archived-trace regressions; no models or weight updates."""
import ast
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from gray_covering_experiment import (ARMS, EPOCHS, STEPS, FAMILIES, FORWARD_BUDGET,
                                     fixed_schedules, prepare_view, fit_decisions)

ROOT = Path(__file__).resolve().parents[1]


class GrayCoveringExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        metadata = json.loads((ROOT / "dataset/detector_supported_review_v3/manifest.json").read_text())
        cls.rows = [r for r in metadata["supported_records"] if r["split"] == "train"]
        cls.fixtures = json.loads((ROOT / "outputs/reflection_coverage_data_v1/manifest.json").read_text())["cases"]
        cls.schedule = fixed_schedules(cls.rows, cls.fixtures)

    def test_first_128_rgb_steps_match_executed_varied_v2_trace(self):
        prior = json.loads((ROOT / "outputs/varied_covering_protocol_v2/protocol.json").read_text())
        for old_epoch, new_epoch in zip(prior["schedules"]["varied133"], self.schedule["rgb133"]):
            self.assertEqual(old_epoch, [{"real": [{k: row[k] for k in ("index", "degraded")} for row in batch["real"]],
                                         "fixture": batch["fixture"]} for batch in new_epoch])
            self.assertFalse(any(row["grayscale"] for batch in new_epoch for row in batch["real"]))

    def test_matched_training_slots_and_all_color_camera_conditions(self):
        self.assertEqual(EPOCHS * STEPS, 512)
        self.assertEqual(FORWARD_BUDGET, 11440)
        self.assertEqual(self.schedule, fixed_schedules(self.rows, self.fixtures))
        for a_epoch, b_epoch in zip(self.schedule["rgb133"], self.schedule["gray133"]):
            self.assertEqual(len(a_epoch), 64)
            for a, b in zip(a_epoch, b_epoch):
                self.assertEqual(a["fixture"], b["fixture"])
                self.assertEqual([(r["index"], r["degraded"]) for r in a["real"]],
                                 [(r["index"], r["degraded"]) for r in b["real"]])
                self.assertEqual([self.rows[r["index"]]["kind"] for r in b["real"]],
                                 ["covered", "covered", "uncovered", "uncovered"])
        for arm in ARMS:
            seen = {(r["index"], r["degraded"], r["grayscale"])
                    for epoch in self.schedule[arm] for batch in epoch for r in batch["real"]}
            colors = (False, True) if arm == "gray133" else (False,)
            self.assertEqual(seen, {(i, d, g) for i in range(133) for d in (False, True) for g in colors})

    def test_held_out_and_unreviewed_strata_refused(self):
        changed = copy.deepcopy(self.rows)
        changed[101]["split"] = "validation"
        with self.assertRaises(ValueError): fixed_schedules(changed, self.fixtures)
        changed[101]["split"] = "train"
        changed[101]["occlusion_stratum"] = "unreviewed"
        with self.assertRaises(ValueError): fixed_schedules(changed, self.fixtures)

    def test_color_conversion_keeps_target_and_unknown_support_objects(self):
        import torch
        x = torch.stack([torch.full((8, 8), v) for v in (1., .5, .1)])
        target = torch.zeros((1, 8, 8)); target[:, 2:6, 2:6] = 1
        valid = torch.ones_like(target); valid[:, 0] = 0
        before = x.clone(), target.clone(), valid.clone()
        gray, kept_target, kept_valid = prepare_view((x, target, valid), True)
        self.assertTrue(torch.allclose(gray, torch.full_like(gray, .6039), atol=1e-7))
        self.assertIs(kept_target, target); self.assertIs(kept_valid, valid)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip((x, target, valid), before)))
        original = prepare_view((x, target, valid), False)
        self.assertIs(original[0], x)

    def test_invalid_input_and_nonboolean_view_refused(self):
        import torch
        x = torch.ones((3, 8, 8)); y = torch.zeros((1, 8, 8)); valid = torch.ones_like(y)
        with self.assertRaises(ValueError): prepare_view((x, y, valid), 1)
        x[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError): prepare_view((x, y, valid), True)

    def test_local_entry_point_refuses_before_files_models_or_optimizers(self):
        from argparse import Namespace
        from scripts.train_gray_covering_vm import main
        tree = ast.parse((ROOT / "scripts/train_gray_covering_vm.py").read_text())
        body = next(n.body for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        self.assertEqual(body[0].value.func.id, "require_vm_gpu")
        out = ROOT / "outputs/gray_covering_vm"
        before = out.exists()
        with patch("native_expert.sys.platform", "win32"):
            with self.assertRaisesRegex(RuntimeError, "no local training"):
                main(Namespace(dry_run=True, batch_size=1, existing=None, camera=None, varied=None))
        self.assertEqual(out.exists(), before)

    def test_grayscale_gain_cannot_hide_rgb_or_clear_regression_or_promote(self):
        metrics = {"iou": .6, "missed_fraction": .2, "visible_false_positive": .01,
                   "empty_mask_cases": 0, "negative_false_positive_cases": 0}
        groups = {k: dict(metrics) for k in ("old_native", "old_degraded", "reflection")}
        for family in FAMILIES:
            for condition in ("native", "degraded"):
                for suffix in ("", "_grayscale"):
                    groups[f"cofw_{family}_{condition}{suffix}"] = dict(metrics)
        final = {arm: {"groups": copy.deepcopy(groups)} for arm in ARMS}
        for condition in ("native", "degraded"):
            final["gray133"]["groups"][f"cofw_hair_{condition}_grayscale"]["iou"] = .7
        decision = fit_decisions(final)
        self.assertTrue(decision["all_training_fit_checks_pass"])
        self.assertFalse(decision["promoted"]); self.assertIsNone(decision["selected_checkpoint"])
        final["gray133"]["groups"]["old_native"]["iou"] = .5
        self.assertFalse(fit_decisions(final)["all_training_fit_checks_pass"])
        final["gray133"]["groups"]["old_native"]["iou"] = .6
        final["gray133"]["groups"]["reflection"]["negative_false_positive_cases"] = 1
        self.assertFalse(fit_decisions(final)["all_training_fit_checks_pass"])


if __name__ == "__main__":
    unittest.main()
