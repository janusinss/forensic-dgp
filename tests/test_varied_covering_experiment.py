"""Finite sampling/selection and local-training refusal contracts; no models."""
import ast
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from varied_covering_experiment import ARMS, FAMILIES, fixed_schedules, fit_decisions, preflight_case

ROOT = Path(__file__).resolve().parents[1]


class VariedCoveringExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        metadata = json.loads((ROOT / "dataset/detector_supported_review_v3/manifest.json").read_text())
        cls.rows = [r for r in metadata["supported_records"] if r["split"] == "train"]
        cls.fixture = json.loads((ROOT / "outputs/reflection_coverage_data_v1/manifest.json").read_text())["cases"]

    def test_matched_domain_and_condition_slots_cover_all_sources(self):
        schedule = fixed_schedules(self.rows, self.fixture)
        self.assertEqual(schedule, fixed_schedules(self.rows, self.fixture))
        for epoch in range(2):
            self.assertEqual(len(schedule["existing91"][epoch]), 64)
            for a, b in zip(schedule["existing91"][epoch], schedule["varied133"][epoch]):
                self.assertEqual(a["fixture"], b["fixture"])
                self.assertEqual([r["degraded"] for r in a["real"]], [r["degraded"] for r in b["real"]])
                self.assertEqual(a["real"][0], b["real"][0])
                self.assertEqual(a["real"][2], b["real"][2])
                for batch in (a, b):
                    self.assertEqual([self.rows[r["index"]]["kind"] for r in batch["real"]],
                                     ["covered", "covered", "uncovered", "uncovered"])
                    self.assertEqual([self.fixture[i]["style"] == "clear" for i in batch["fixture"]],
                                     [False, False, True, True])
        for arm, size in zip(ARMS, (91, 133)):
            seen = {(r["index"], r["degraded"]) for epoch in schedule[arm] for batch in epoch for r in batch["real"]}
            self.assertEqual(seen, {(i, d) for i in range(size) for d in (False, True)})

    def test_held_out_or_missing_family_refused(self):
        changed = copy.deepcopy(self.rows)
        changed[91]["split"] = "validation"
        with self.assertRaises(ValueError):
            fixed_schedules(changed, self.fixture)

    def test_preflight_resolves_reviewed_hair_by_identifier(self):
        cases = json.loads((ROOT / "outputs/cofw_camera_pairs_v2/manifest.json").read_text())["cases"]
        selected = preflight_case(cases)
        self.assertEqual(selected["case_id"], 202)
        self.assertEqual(selected["source_id"], "cofw_train_0868")
        self.assertEqual(selected["family"], "obstructing_hair")
        changed = copy.deepcopy(self.rows)
        changed[91]["occlusion_stratum"] = "unreviewed"
        with self.assertRaises(ValueError):
            fixed_schedules(changed, self.fixture)

    def test_fit_cannot_promote_and_clear_failure_remains_failure(self):
        metric = {"iou": .6, "missed_fraction": .2, "visible_false_positive": .01,
                  "empty_mask_cases": 0, "negative_false_positive_cases": 0}
        groups = {k: dict(metric) for k in ("old_native", "old_degraded", "reflection")}
        groups.update({f"cofw_{name}_{condition}": dict(metric) for name in FAMILIES for condition in ("native", "degraded")})
        final = {arm: {"groups": copy.deepcopy(groups)} for arm in ARMS}
        for name in FAMILIES:
            for condition in ("native", "degraded"):
                final["varied133"]["groups"][f"cofw_{name}_{condition}"]["iou"] = .7
        decision = fit_decisions(final)
        self.assertTrue(decision["all_training_fit_checks_pass"])
        self.assertFalse(decision["promoted"])
        self.assertIsNone(decision["selected_checkpoint"])
        final["varied133"]["groups"]["reflection"]["negative_false_positive_cases"] = 1
        self.assertFalse(fit_decisions(final)["all_training_fit_checks_pass"])

    def test_vm_guard_precedes_every_main_operation(self):
        path = ROOT / "scripts/train_varied_covering_vm.py"
        tree = ast.parse(path.read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        self.assertIsInstance(main.body[0], ast.Expr)
        self.assertEqual(main.body[0].value.func.id, "require_vm_gpu")

    def test_actual_local_entry_point_refuses_before_output_creation(self):
        from argparse import Namespace
        from scripts.train_varied_covering_vm import main
        output = ROOT / "outputs/varied_covering_vm"
        before = output.exists()
        with patch("native_expert.sys.platform", "win32"):
            with self.assertRaisesRegex(RuntimeError, "no local training"):
                main(Namespace(dry_run=True, batch_size=1, existing=None, camera=None))
        self.assertEqual(output.exists(), before)

    def test_unknown_logits_do_not_change_four_real_four_fixture_loss(self):
        import torch
        from scripts.train_varied_covering_vm import loss_for_batch
        logits = torch.full((4, 1, 8, 8), -1.)
        target = torch.zeros_like(logits)
        target[:2, :, 2:6, 2:6] = 1
        valid = torch.ones_like(logits)
        valid[:, :, 0, :] = 0
        changed = logits.clone()
        changed[:, :, 0, :] = 100
        original = loss_for_batch(logits, target, valid, logits, target, valid)
        altered = loss_for_batch(changed, target, valid, changed, target, valid)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(original, altered)))
        self.assertTrue(torch.equal(original[0], .5 * original[1] + .5 * original[2]))


if __name__ == "__main__":
    unittest.main()
