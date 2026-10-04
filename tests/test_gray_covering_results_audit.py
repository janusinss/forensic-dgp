"""Integrity counterexamples using labeled test fixtures; no VM results or training."""
import copy
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import unittest
from unittest.mock import patch
import uuid

import numpy as np
from PIL import Image

from scripts import audit_gray_covering_results as audit

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def temporary_workspace():
    parent = ROOT / "scratch"
    parent.mkdir(exist_ok=True)
    directory = parent / ("gray-return-audit-test-" + uuid.uuid4().hex)
    directory.mkdir()
    try:
        yield directory
    finally:
        resolved = directory.resolve()
        assert resolved.is_relative_to(parent.resolve()) and parent.resolve().is_relative_to(ROOT.resolve())
        shutil.rmtree(resolved)


class GrayCoveringReturnAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = audit.read(ROOT / "outputs/gray_covering_protocol_v1/protocol.json")

    def logs(self, arm):
        return [{"arm": arm, "experiment_epoch": i // 64 + 1, "step": i % 64 + 1,
                 "experiment_updates": i + 1, "optimizer_state_step": i + 1, "cumulative_model_updates": 995 + i,
                 **copy.deepcopy(self.protocol["schedules"][arm][i // 64][i % 64]),
                 "loss": .5, "real_loss": .25, "fixture_loss": .75, "pre_clip_norm": 1.}
                for i in range(512)]

    def final_metrics(self):
        metric = {"iou": .6, "missed_fraction": .2, "visible_false_positive": .01,
                  "empty_mask_cases": 0, "negative_false_positive_cases": 0}
        rgb = ("old_native", "old_degraded", "cofw_clear_native", "cofw_clear_degraded") + tuple(
            f"cofw_{n}_{c}" for n in audit.FAMILIES for c in ("native", "degraded"))
        groups = {g: dict(metric) for g in (*rgb, *(g + "_grayscale" for g in rgb), "reflection")}
        final = {arm: {"groups": copy.deepcopy(groups)} for arm in audit.ARMS}
        for condition in ("native", "degraded"):
            final["gray133"]["groups"][f"cofw_hair_{condition}_grayscale"]["iou"] = .7
        return final

    def tensor_fixture(self):
        import torch
        state = {f"network.encoder.bn{i}.{name}": torch.zeros(1)
                 for i in range(30) for name in ("running_mean", "running_var", "num_batches_tracked")}
        state.update({"reference_visible_head.weight": torch.zeros(1), "reference_visible_head.bias": torch.zeros(1)})
        for prefix, count in (("network.encoder.", 31), ("network.decoder.", 30), ("network.segmentation_head.", 31)):
            state.update({prefix + f"test_parameter_{i}": torch.zeros(1) for i in range(count)})
        source = {"model": state, "selection": {"selected": False}, "epoch": 44,
                  "additional_epoch": 34, "optimizer_updates": 994, "fresh_optimizer_updates": 784,
                  "experiment_updates": 112}
        changed = {k: v.clone() for k, v in state.items()}
        for k, value in changed.items():
            if "test_parameter_" in k:
                value += 1
        return source, changed

    def checkpoint_fixture(self, intermediate=False):
        source, state = self.tensor_fixture()
        payload = {**source, "model": state, "epoch": 46 if intermediate else 52,
                   "additional_epoch": 36 if intermediate else 42,
                   "optimizer_updates": 1122 if intermediate else 1506,
                   "fresh_optimizer_updates": 128 if intermediate else 512,
                   "experiment_updates": 128 if intermediate else 512,
                   "gray_covering_arm": "rgb133" if intermediate else "gray133",
                   "gray_covering_protocol_sha256": audit.PROTOCOL_SHA, "optimizer_reset": True,
                   "detector_only": True, "selection": {"selected": False, "training_fit_only": True}}
        if not intermediate:
            payload.update(source_varied_checkpoint_counters=self.protocol["source_counters"],
                           source_varied_selection=source["selection"])
            payload["selection"]["historical_gates_not_evaluated"] = True
        return source, payload

    def test_exact_return_budget_includes_intermediate_and_gray_masks_but_no_moments(self):
        names = audit.expected_files()
        self.assertEqual(len(names), 3267)
        for domain, expected in (("real", 1064), ("gray", 1064), ("fixture", 1120)):
            self.assertEqual(sum(n.endswith(".png") and f"/{domain}/" in n for n in names), expected)
        self.assertEqual(sum(n.endswith(".pth") for n in names), 3)
        self.assertFalse(any("optimizer" in n for n in names))

    def test_unsafe_duplicate_extra_and_oversize_archive_members_refused(self):
        names = audit.expected_files()
        name = audit.RESULT + "baseline.json"
        good = tarfile.TarInfo(audit.PREFIX + name)
        good.size = 2
        self.assertEqual(audit.validate_member(good, names, set()), name)
        bad = [tarfile.TarInfo(audit.PREFIX + path) for path in
               ("../escape", "/absolute", "./bundle_inventory.json", "C:/absolute", "back\\slash", "unexpected.json")]
        bad.append(tarfile.TarInfo("wrong_root/" + name))
        for kind in (tarfile.DIRTYPE, tarfile.SYMTYPE, tarfile.LNKTYPE):
            member = copy.copy(good)
            member.type, member.linkname = kind, "../../escape"
            bad.append(member)
        member = copy.copy(good)
        member.size = 2 * 1024**2 + 1
        bad.append(member)
        member = copy.copy(good)
        member.size = -1
        bad.append(member)
        for member in bad:
            with self.subTest(name=member.name, type=member.type, size=member.size):
                with self.assertRaises(ValueError):
                    audit.validate_member(member, names, set())
        with self.assertRaises(ValueError):
            audit.validate_member(good, names, {name.casefold()})

    def test_missing_partial_and_crlf_return_never_create_success_evidence(self):
        with temporary_workspace() as directory:
            archive = directory / "synthetic-envelope-only.tar.gz"
            with patch.multiple(audit, ARCHIVE=archive, EXTRACT=directory / "extract", OUT=directory / "audit"):
                with self.assertRaisesRegex(ValueError, "Download gray-covering-results"):
                    audit.main()
                with tarfile.open(archive, "w:gz") as tar:
                    member = tarfile.TarInfo(audit.PREFIX + audit.RESULT + "baseline.json")
                    member.size = 2
                    tar.addfile(member, io.BytesIO(b"{}"))
                digest = hashlib.sha256(archive.read_bytes()).hexdigest()
                sidecar = archive.with_name(archive.name + ".sha256")
                sidecar.write_bytes(f"{digest}  {archive.name}\r\n".encode())
                with self.assertRaisesRegex(ValueError, "Transfer/LF"):
                    audit.extract_return()
                sidecar.write_bytes(f"{digest}  {archive.name}\n".encode())
                with self.assertRaisesRegex(ValueError, "Return membership differs"):
                    audit.extract_return()
                self.assertFalse((directory / "extract").exists())
                self.assertFalse((directory / "audit").exists())

    def test_inventory_corruption_is_rejected_before_any_extraction(self):
        with temporary_workspace() as directory:
            names = {"bundle_inventory.json": b"{}", "inputs/gray_covering_protocol.json": b"{}",
                     audit.RESULT + "tiny-test-only.json": b"not claimed as VM results"}
            inventory_name = audit.RESULT + "return_inventory.json"
            archive = directory / "synthetic-envelope-only.tar.gz"

            def save(corrupt):
                pins = {name: hashlib.sha256(blob).hexdigest() for name, blob in names.items()}
                if corrupt:
                    pins[audit.RESULT + "tiny-test-only.json"] = "0" * 64
                data = {**names, inventory_name: json.dumps(pins).encode()}
                with tarfile.open(archive, "w:gz") as tar:
                    for name, blob in data.items():
                        item = tarfile.TarInfo(audit.PREFIX + name)
                        item.size = len(blob)
                        tar.addfile(item, io.BytesIO(blob))
                digest = hashlib.sha256(archive.read_bytes()).hexdigest()
                archive.with_name(archive.name + ".sha256").write_bytes(f"{digest}  {archive.name}\n".encode())

            digest = hashlib.sha256(b"{}").hexdigest()
            with patch.multiple(audit, ARCHIVE=archive, EXTRACT=directory / "extract", OUT=directory / "audit",
                                PROTOCOL_SHA=digest, INVENTORY_SHA=digest), patch.object(audit, "expected_files",
                                return_value=set(names) | {inventory_name}):
                save(corrupt=True)
                with self.assertRaisesRegex(ValueError, "Returned inventory hash differs"):
                    audit.extract_return()
                self.assertFalse((directory / "extract").exists())
                save(corrupt=False)
                _, hashes = audit.extract_return()
                self.assertEqual(len(hashes), 4)
                self.assertEqual((directory / "extract" / (audit.RESULT + "tiny-test-only.json")).read_bytes(),
                                 names[audit.RESULT + "tiny-test-only.json"])
                self.assertFalse((directory / "audit").exists())
                with self.assertRaisesRegex(ValueError, "Preserve earlier"):
                    audit.extract_return()

    def test_independent_schedule_keeps_heldout_out_and_reproduces_gray_exposure(self):
        registry = audit.read(ROOT / "dataset/detector_supported_review_v3/manifest.json")
        rows = [r for r in registry["supported_records"] if r["split"] == "train"]
        fixtures = audit.read(ROOT / audit.ASSET_PATHS["fixture_manifest"])["cases"]
        schedule = audit.independent_schedules(rows, fixtures)
        audit.exact(schedule, self.protocol["schedules"], "Frozen schedule")
        exposure = {(r["index"], r["degraded"], r["grayscale"]) for epoch in schedule["gray133"]
                    for batch in epoch for r in batch["real"]}
        self.assertEqual(exposure, {(i, d, g) for i in range(133) for d in (False, True) for g in (False, True)})
        self.assertEqual(sum(r["grayscale"] for epoch in schedule["gray133"] for b in epoch for r in b["real"]), 970)
        changed = copy.deepcopy(rows)
        changed[94]["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "Training membership"):
            audit.independent_schedules(changed, fixtures)

    def test_every_logged_update_and_both_input_schedules_verified(self):
        for arm in audit.ARMS:
            checked = audit.audit_steps(self.logs(arm), self.protocol["schedules"][arm], arm)
            self.assertEqual(checked["logged_updates"], 512)
            self.assertEqual(checked["last_fresh_moment_step"], 512)
            self.assertEqual(checked["cumulative_model_updates"], 1506)

    def test_old_moments_wrong_gray_inputs_extra_steps_and_invalid_losses_rejected(self):
        rows, schedule = self.logs("gray133"), self.protocol["schedules"]["gray133"]
        for changed in (rows[:-1], rows + [copy.deepcopy(rows[-1])]):
            with self.assertRaises(ValueError):
                audit.audit_steps(changed, schedule, "gray133")
        mutations = (
            lambda r: r.update(optimizer_state_step=785),
            lambda r: r.update(cumulative_model_updates=1250),
            lambda r: r.update(experiment_updates=True),
            lambda r: r.update(loss=float("nan")),
            lambda r: r.update(pre_clip_norm=float("inf")),
            lambda r: r.update(fixture_loss=-.75),
            lambda r: r.update(loss=.75),
            lambda r: r["real"][0].update(grayscale=not r["real"][0]["grayscale"]),
            lambda r: r["real"][0].update(degraded=int(r["real"][0]["degraded"])),
            lambda r: r["fixture"].__setitem__(0, 999),
        )
        for number, mutate in enumerate(mutations):
            changed = copy.deepcopy(rows)
            mutate(changed[201])
            with self.subTest(counterexample=number):
                with self.assertRaises(ValueError):
                    audit.audit_steps(changed, schedule, "gray133")

    def test_clear_errors_ignored_regions_and_family_regressions_remain_distinct(self):
        final = self.final_metrics()
        decision = audit.fit_checks(final)
        self.assertTrue(decision["all_training_fit_checks_pass"])
        self.assertIsNone(decision["selected_checkpoint"])
        self.assertFalse(decision["promoted"])
        self.assertFalse(decision["original_425_case_gates_evaluated"])
        changes = (("rgb133", "old_native_grayscale", "negative_false_positive_cases", 1),
                   ("gray133", "cofw_hair_native_grayscale", "iou", .6),
                   ("gray133", "cofw_hand_degraded_grayscale", "iou", .5),
                   ("gray133", "cofw_cloth_degraded_grayscale", "empty_mask_cases", 1),
                   ("gray133", "reflection", "visible_false_positive", .02),
                   ("gray133", "cofw_object_native", "missed_fraction", .3))
        for arm, group, key, value in changes:
            altered = copy.deepcopy(final)
            altered[arm]["groups"][group][key] = value
            with self.subTest(group=group, key=key):
                self.assertFalse(audit.fit_checks(altered)["all_training_fit_checks_pass"])
        valid = np.ones((256, 256), dtype=bool)
        valid[:5] = False
        target = np.zeros_like(valid)
        prediction = ~valid
        counted = audit.recount(prediction, target, valid)
        self.assertEqual(counted["fp"], 0)
        self.assertEqual(counted["negative_false_positive_cases"], 0)
        self.assertEqual(counted["ignored_positive_pixels"], 1280)
        target[20, 20] = True
        self.assertEqual(audit.recount(prediction, target, valid)["empty_mask_cases"], 1)

    def test_frozen_changes_nonfinite_tensors_and_false_checkpoint_metadata_rejected(self):
        for intermediate in (False, True):
            source, payload = self.checkpoint_fixture(intermediate)
            arm = payload["gray_covering_arm"]
            checked = audit.checkpoint_audit(payload, source, self.protocol, arm, intermediate)
            self.assertEqual(checked["tensor_states"], 184)
            self.assertEqual(checked["frozen_tensors_verified"], 92)
            mutations = (
                lambda p: p["model"]["reference_visible_head.weight"].add_(1),
                lambda p: p.update(optimizer_reset=1),
                lambda p: p["model"]["network.decoder.test_parameter_0"].fill_(float("nan")),
                lambda p: p.update(fresh_optimizer_updates=999),
                lambda p: p["selection"].update(selected=True),
            )
            for number, mutate in enumerate(mutations):
                altered = copy.deepcopy(payload)
                mutate(altered)
                with self.subTest(intermediate=intermediate, counterexample=number):
                    with self.assertRaises(ValueError):
                        audit.checkpoint_audit(altered, source, self.protocol, arm, intermediate)

    def test_rgb128_receipt_cannot_hide_one_changed_tensor(self):
        _, payload = self.checkpoint_fixture(intermediate=True)
        reference = {"model": copy.deepcopy(payload["model"]), "varied_covering_arm": "varied133",
                     "optimizer_updates": 1122, "fresh_optimizer_updates": 128}
        receipt = {"complete": True, "reference_model_sha256": audit.sha(ROOT / audit.ASSET_PATHS["rgb128_reference"]),
                   "tensors_compared": 184, "different_tensors": [], "fresh_updates": 128,
                   "cumulative_model_updates": 1122, "all_model_states_equal": True, "optimizer_reset": True}
        self.assertTrue(audit.audit_reproduction(payload, reference, receipt)["all_model_states_equal"])
        changed = copy.deepcopy(payload)
        changed["model"]["network.decoder.test_parameter_0"] += .0001
        with self.assertRaisesRegex(ValueError, "failed exact prior-run"):
            audit.audit_reproduction(changed, reference, receipt)
        changed = copy.deepcopy(receipt)
        changed["all_model_states_equal"] = 1
        with self.assertRaisesRegex(ValueError, "type differs"):
            audit.audit_reproduction(payload, reference, changed)

    def test_final_budget_hash_and_selection_claims_are_checked_independently(self):
        decision = audit.fit_checks(self.final_metrics())
        hashes = {audit.RESULT + n: "1" * 64 for n in ("preview.png", "preview_grayscale.png")}
        hashes.update({audit.RESULT + arm + "/last.pth": "2" * 64 for arm in audit.ARMS})
        complete = {"complete": True, "protocol_sha256": audit.PROTOCOL_SHA, "updates_per_arm": 512,
                    "total_optimizer_updates": 1024, "moment_steps_per_arm": 512, "cumulative_model_updates_per_arm": 1506,
                    "optimizer_reset": True, "epochs_per_arm": 8, "actual_model_forward_images": 11440,
                    "held_out_forward_images": 0, "fit_decision": decision, "original_425_case_gates_evaluated": False,
                    "promoted": False, "selected_checkpoint": None, "preview_sha256": "1" * 64,
                    "preview_grayscale_sha256": "1" * 64, "rgb128_reproduction_equal": True,
                    "original_assets_unchanged": True, "seconds": 99.,
                    "checkpoints": {arm: {"model_sha256": "2" * 64, "optimizer_sha256": "3" * 64,
                                         "optimizer_retained_on_vm_not_in_return_archive": True} for arm in audit.ARMS}}
        audit.audit_completion(complete, decision, hashes)
        mutations = (
            lambda r: r.update(total_optimizer_updates=256),
            lambda r: r.update(actual_model_forward_images=11441),
            lambda r: r.update(held_out_forward_images=1),
            lambda r: r.update(promoted=True),
            lambda r: r.update(selected_checkpoint="gray133/last.pth"),
            lambda r: r.update(rgb128_reproduction_equal=1),
            lambda r: r.update(seconds=1800.),
            lambda r: r["checkpoints"]["gray133"].update(model_sha256="4" * 64),
            lambda r: r["checkpoints"]["gray133"].update(optimizer_retained_on_vm_not_in_return_archive=False),
        )
        for number, mutate in enumerate(mutations):
            changed = copy.deepcopy(complete)
            mutate(changed)
            with self.subTest(counterexample=number):
                with self.assertRaises(ValueError):
                    audit.audit_completion(changed, decision, hashes)

    def test_cuda_preflight_is_separate_from_actual_training_and_pins_dependencies(self):
        deps = {name: {"version": version, "module_path":
                      "/home/janusdominic0/forensic-dgp/coverage_vm_bundle/outputs/face_occlusion_dependencies/test/__init__.py"}
                for name, version in (("segmentation-models-pytorch", "0.5.0"), ("timm", "1.0.15"),
                                      ("huggingface-hub", "0.29.3"), ("safetensors", "0.5.3"), ("PyYAML", "6.0.2"))}
        preflight = {"complete": True, "batch_size": 1, "protocol_sha256": audit.PROTOCOL_SHA, "gpu": "NVIDIA L4",
                     "cuda_available": True, "fresh_optimizer_planned": True, "optimizer_constructed": False,
                     "optimizer_updates": 0, "actual_model_forward_images": 1, "held_out_forward_images": 0,
                     "grayscale_input": True, "loss": .1, "dependencies": deps}
        execution = {"protocol_sha256": audit.PROTOCOL_SHA, "inventory_sha256": audit.INVENTORY_SHA,
                     "gpu": "NVIDIA L4", "torch": "2.9.1+cu129", "arms": list(audit.ARMS), "updates_per_arm": 512,
                     "total_update_budget": 1024, "optimizer_reset": True, "source_model_updates": 994,
                     "fresh_moment_start_step": 0, "held_out_forward_images": 0, "promoted": False}
        audit.audit_execution(preflight, execution)
        mutations = (
            lambda r: r.update(optimizer_updates=1),
            lambda r: r.update(cuda_available=False),
            lambda r: r.update(grayscale_input=False),
            lambda r: r.update(loss=float("nan")),
            lambda r: r["dependencies"]["timm"].update(version="unknown"),
            lambda r: r["dependencies"]["timm"].update(module_path="C:/local/test.py"),
        )
        for number, mutate in enumerate(mutations):
            changed = copy.deepcopy(preflight)
            mutate(changed)
            with self.subTest(counterexample=number):
                with self.assertRaises(ValueError):
                    audit.audit_execution(changed, execution)
        changed = copy.deepcopy(execution)
        changed["fresh_moment_start_step"] = 784
        with self.assertRaises(ValueError):
            audit.audit_execution(preflight, changed)

    def test_preview_reconstruction_matches_frozen_producer_without_model_construction(self):
        import torch
        from scripts import train_gray_covering_vm as producer
        cases = audit.read(ROOT / "outputs/cofw_camera_pairs_v2/manifest.json")["cases"]
        ids = self.protocol["preview_case_ids"]
        with temporary_workspace() as directory:
            out = directory / audit.RESULT
            for condition in ("real", "gray"):
                for folder, old in (("initial_masks", "initial_masks"), ("rgb133/final_masks", "existing91/final_masks"),
                                    ("gray133/final_masks", "varied133/final_masks")):
                    target = out / folder / condition
                    target.mkdir(parents=True)
                    for case_id in ids:
                        shutil.copyfile(audit.PREVIOUS / old / "real" / f"{case_id:03d}.png", target / f"{case_id:03d}.png")
            with patch.object(torch.nn.Module, "__init__", side_effect=AssertionError("No model construction")), \
                 patch.object(audit, "EXTRACT", directory):
                for grayscale in (False, True):
                    producer.preview(cases, ids, out, grayscale)
                    checked = audit.recreate_preview(cases, ids, grayscale)
                    self.assertEqual(checked.size, (960, 2222))
                path = out / "preview_grayscale.png"
                with Image.open(path) as image:
                    data = np.asarray(image).copy()
                data[1, 1] = (255, 0, 0)
                Image.fromarray(data).save(path)
                with self.assertRaisesRegex(ValueError, "preview pixels differ"):
                    audit.recreate_preview(cases, ids, True)
            self.assertFalse((directory / "verification.json").exists())

    def test_direct_cli_help_works_outside_project_and_missing_return_is_specific(self):
        with temporary_workspace() as directory:
            checked = subprocess.run([sys.executable, str(ROOT / "scripts/audit_gray_covering_results.py"), "--help"],
                                     cwd=directory, capture_output=True, text=True, timeout=15)
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertIn("no detector construction or training", " ".join(checked.stdout.split()))
            self.assertEqual(list(directory.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
