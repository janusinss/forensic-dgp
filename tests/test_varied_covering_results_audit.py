"""Return-audit counterexamples only; no VM results, models or optimizers."""
import copy
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import subprocess
import shutil
import sys
import tarfile
import unittest
from unittest.mock import patch
import uuid

import numpy as np

from scripts import audit_varied_covering_results as audit

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def temporary_workspace():
    """Use inherited workspace permissions; Windows mode700 temp ACLs are unusable here."""
    parent = ROOT / "scratch"
    parent.mkdir(exist_ok=True)
    directory = parent / ("varied-return-audit-" + uuid.uuid4().hex)
    directory.mkdir()
    try:
        yield directory
    finally:
        resolved = directory.resolve()
        assert resolved.is_relative_to(parent.resolve()) and parent.resolve().is_relative_to(ROOT.resolve())
        shutil.rmtree(resolved)


class VariedCoveringReturnAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads((ROOT / "outputs/varied_covering_protocol_v2/protocol.json").read_text())

    def logs(self, arm):
        return [{"arm": arm, "experiment_epoch": i // 64 + 1, "step": i % 64 + 1,
                 "experiment_updates": i + 1, "optimizer_state_step": i + 1,
                 "cumulative_model_updates": 995 + i,
                 **copy.deepcopy(self.protocol["schedules"][arm][i // 64][i % 64]),
                 "loss": .5, "real_loss": .25, "fixture_loss": .75, "pre_clip_norm": 1.}
                for i in range(128)]

    def final_metrics(self):
        metric = {"iou": .6, "missed_fraction": .2, "visible_false_positive": .01,
                  "empty_mask_cases": 0, "negative_false_positive_cases": 0}
        groups = {key: dict(metric) for key in ("old_native", "old_degraded", "reflection",
                                                "cofw_clear_native", "cofw_clear_degraded")}
        groups.update({f"cofw_{name}_{condition}": dict(metric) for name in audit.FAMILIES
                       for condition in ("native", "degraded")})
        final = {arm: {"groups": copy.deepcopy(groups)} for arm in audit.ARMS}
        for name in audit.FAMILIES:
            for condition in ("native", "degraded"):
                final["varied133"]["groups"][f"cofw_{name}_{condition}"]["iou"] = .7
        return final

    def test_exact_file_budget_excludes_optimizer_exports(self):
        names = audit.expected_files()
        self.assertEqual(len(names), 1653)
        self.assertEqual(sum(name.endswith(".png") and "/real/" in name for name in names), 798)
        self.assertEqual(sum(name.endswith(".png") and "/fixture/" in name for name in names), 840)
        self.assertEqual(sum(name.endswith(".pth") for name in names), 2)
        self.assertFalse(any("optimizer" in name for name in names))

    def test_links_traversal_extra_duplicate_and_oversize_members_refused(self):
        names = audit.expected_files()
        name = audit.RESULT + "baseline.json"
        member = tarfile.TarInfo(audit.PREFIX + name)
        member.size = 2
        self.assertEqual(audit.validate_member(member, names, set()), name)
        bad_members = []
        for path in ("../escape", "/absolute", "./bundle_inventory.json", "C:/absolute",
                     "bundle_inventory.json\\escape", "unexpected.txt"):
            bad_members.append(tarfile.TarInfo(audit.PREFIX + path))
        bad_members.append(tarfile.TarInfo("wrong_root/" + name))
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE):
            changed = copy.copy(member)
            changed.type = kind
            changed.linkname = "../../escape"
            bad_members.append(changed)
        oversized = copy.copy(member)
        oversized.size = 2 * 1024**2 + 1
        bad_members.append(oversized)
        for changed in bad_members:
            with self.subTest(path=changed.name, kind=changed.type):
                with self.assertRaises(ValueError):
                    audit.validate_member(changed, names, set())
        with self.assertRaises(ValueError):
            audit.validate_member(member, names, {name.casefold()})

    def test_missing_return_refuses_before_extraction_or_quality_report(self):
        with temporary_workspace() as directory:
            root = Path(directory)
            with patch.multiple(audit, ARCHIVE=root / "varied-covering-results.tar.gz",
                                EXTRACT=root / "extract", OUT=root / "audit"):
                with self.assertRaisesRegex(ValueError, "Download varied-covering-results"):
                    audit.main()
            self.assertEqual(list(root.iterdir()), [])

    def test_valid_checksum_does_not_accept_partial_return_or_crlf_sidecar(self):
        with temporary_workspace() as directory:
            root = Path(directory)
            archive = root / "varied-covering-results.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                member = tarfile.TarInfo(audit.PREFIX + audit.RESULT + "baseline.json")
                member.size = 2
                tar.addfile(member, io.BytesIO(b"{}"))
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            sidecar = archive.with_name(archive.name + ".sha256")
            with patch.multiple(audit, ARCHIVE=archive, EXTRACT=root / "extract", OUT=root / "audit"):
                sidecar.write_bytes(f"{digest}  {archive.name}\r\n".encode("ascii"))
                with self.assertRaisesRegex(ValueError, "Transfer/LF checksum differs"):
                    audit.extract_return()
                sidecar.write_bytes(f"{digest}  {archive.name}\n".encode("ascii"))
                with self.assertRaisesRegex(ValueError, "Return membership/total budget differs"):
                    audit.extract_return()
            self.assertFalse((root / "extract").exists())
            self.assertFalse((root / "audit").exists())

    def test_every_logged_update_matches_both_frozen_schedules(self):
        for arm in audit.ARMS:
            checked = audit.audit_steps(self.logs(arm), self.protocol["schedules"][arm], arm)
            self.assertEqual(checked["logged_updates"], 128)
            self.assertEqual(checked["first_fresh_moment_step"], 1)
            self.assertEqual(checked["last_fresh_moment_step"], 128)
            self.assertEqual(checked["cumulative_model_updates"], 1122)

    def test_missing_steps_old_moments_changed_inputs_and_invalid_losses_refused(self):
        arm = "varied133"
        rows, schedule = self.logs(arm), self.protocol["schedules"][arm]
        with self.assertRaises(ValueError):
            audit.audit_steps(rows[:-1], schedule, arm)
        mutations = (
            lambda row: row.update(optimizer_state_step=785),
            lambda row: row.update(cumulative_model_updates=1250),
            lambda row: row.update(experiment_updates=True),
            lambda row: row.update(loss=float("nan")),
            lambda row: row.update(pre_clip_norm=float("inf")),
            lambda row: row.update(real_loss=-.25),
            lambda row: row.update(loss=.75),
            lambda row: row["real"][0].update(degraded=not row["real"][0]["degraded"]),
            lambda row: row["real"][0].update(degraded=int(row["real"][0]["degraded"])),
            lambda row: row["fixture"].__setitem__(0, 999),
        )
        for number, mutation in enumerate(mutations):
            changed = copy.deepcopy(rows)
            mutation(changed[73])
            with self.subTest(counterexample=number):
                with self.assertRaises(ValueError):
                    audit.audit_steps(changed, schedule, arm)

    def test_passing_training_fit_never_selects_or_promotes_checkpoint(self):
        decision = audit.fit_checks(self.final_metrics())
        self.assertTrue(decision["all_training_fit_checks_pass"])
        self.assertFalse(decision["original_425_case_gates_evaluated"])
        self.assertTrue(decision["historical_gate_failure_unchanged"])
        self.assertFalse(decision["promoted"])
        self.assertIsNone(decision["selected_checkpoint"])

    def test_clear_errors_family_misses_and_retention_loss_remain_failures(self):
        changes = (
            ("existing91", "cofw_clear_native", "negative_false_positive_cases", 1),
            ("varied133", "cofw_hair_degraded", "iou", .6),
            ("varied133", "cofw_hand_native", "empty_mask_cases", 1),
            ("varied133", "old_native", "visible_false_positive", .02),
            ("varied133", "reflection", "missed_fraction", .3),
        )
        for arm, group, key, value in changes:
            final = self.final_metrics()
            final[arm]["groups"][group][key] = value
            with self.subTest(group=group, changed=key):
                self.assertFalse(audit.fit_checks(final)["all_training_fit_checks_pass"])

    def test_unknown_pixels_are_context_not_false_positive_or_coverage_evidence(self):
        valid = np.ones((256, 256), dtype=bool)
        valid[:5] = False
        prediction = ~valid
        target = np.zeros_like(valid)
        counted = audit.recount(prediction, target, valid)
        self.assertEqual(counted["fp"], 0)
        self.assertEqual(counted["negative_false_positive_cases"], 0)
        self.assertEqual(counted["ignored_positive_pixels"], 1280)
        target[10:12, 10:12] = True
        self.assertEqual(audit.recount(prediction, target, valid)["empty_mask_cases"], 1)
        prediction[10, 10] = True
        self.assertEqual(audit.recount(prediction, target, valid)["empty_mask_cases"], 0)

    def test_tensor_and_metadata_inspection_rejects_frozen_changes_and_false_reset(self):
        import torch
        state = {f"network.encoder.bn{i}.{name}": torch.zeros(1)
                 for i in range(30) for name in ("running_mean", "running_var", "num_batches_tracked")}
        state.update({"reference_visible_head.weight": torch.zeros(1), "reference_visible_head.bias": torch.zeros(1)})
        state.update({prefix + "weight": torch.zeros(1) for prefix in
                      ("network.encoder.", "network.decoder.", "network.segmentation_head.")})
        source = {"model": state, "selection": {"selected": False}, "epoch": 44}
        payload = {**source, "model": {key: value.clone() for key, value in state.items()},
                   "epoch": 46, "additional_epoch": 36, "optimizer_updates": 1122,
                   "fresh_optimizer_updates": 128, "experiment_updates": 128,
                   "source_varied_checkpoint_counters": self.protocol["source_counters"], "optimizer_reset": True,
                   "varied_covering_arm": "varied133", "varied_covering_protocol_sha256": audit.PROTOCOL_SHA,
                   "detector_only": True, "source_varied_selection": source["selection"],
                   "selection": {"selected": False, "training_fit_only": True, "historical_gates_not_evaluated": True}}
        for prefix in ("network.encoder.", "network.decoder.", "network.segmentation_head."):
            payload["model"][prefix + "weight"] += 1
        checked = audit.checkpoint_audit(payload, source, self.protocol, "varied133")
        self.assertEqual(checked["frozen_tensors_verified"], 92)
        changed = copy.deepcopy(payload)
        changed["model"]["reference_visible_head.weight"] += 1
        with self.assertRaisesRegex(ValueError, "Frozen head/BN tensors changed"):
            audit.checkpoint_audit(changed, source, self.protocol, "varied133")
        changed = copy.deepcopy(payload)
        changed["optimizer_reset"] = 1
        with self.assertRaisesRegex(ValueError, "type differs"):
            audit.checkpoint_audit(changed, source, self.protocol, "varied133")
        changed = copy.deepcopy(payload)
        changed["model"]["network.decoder.weight"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Invalid detector state tensor"):
            audit.checkpoint_audit(changed, source, self.protocol, "varied133")

    def test_direct_cli_help_works_away_from_project_root(self):
        with temporary_workspace() as directory:
            checked = subprocess.run([sys.executable, str(ROOT / "scripts/audit_varied_covering_results.py"), "--help"],
                                     cwd=directory, capture_output=True, text=True, timeout=15)
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertIn("Independent finite-pilot return verification", checked.stdout)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
