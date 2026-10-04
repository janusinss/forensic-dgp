"""Freeze a source-only review; no model imports, training or split reassignment."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "outputs/cctv_dgp_hq_source_review_v3"
CATALOG = ROOT / "outputs/cctv_dgp_hq_cohort_v2"
PLAN = ROOT / "outputs/cctv_dgp_hq_cohort_plan_v2"
RESULT_SHA = "7388a67b655d8259db8a30cefcdfa5b520597e80fd71bfd1d5952051fac98370"
AUDIT_SHA = "d4989390d2c69bcd3c3abb8169aa1c02b4ad6a50aed3c5575b544aa93e96c0c0"
RUBRIC_SHA = "1e0aa441917280168a87f73509e1846b15ada15ee4d8b0e62ec35bf7cc6bad3e"
PLAN_SHA = "ad65e35c70104b72a38dfe3cdac1f1027029445313f9d5530dbb18af7cde9501"
SAMPLE_SHA = "681eaa46641bd5529a59577f05f6287aa0299051f4d6f9c785aed6084be25ff0"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(
            hashlib, "file_digest"
        ) else hashlib.sha256(stream.read()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_once(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def main() -> None:
    started = time.monotonic()
    outputs = [REVIEW / name for name in (
        "final_decisions.json", "eligible_references.json", "source_review_audit.json"
    )]
    assert not any(path.exists() for path in outputs), "Existing frozen review: preserve it"
    assert sha(CATALOG / "results.json") == RESULT_SHA
    assert sha(CATALOG / "local_independent_audit.json") == AUDIT_SHA
    assert sha(PLAN / "manifest.json") == PLAN_SHA
    assert sha(PLAN / "source_review_rubric.json") == RUBRIC_SHA
    assert sha(PLAN / "sample_visual_review.json") == SAMPLE_SHA
    catalog = read(CATALOG / "results.json")
    prior = read(CATALOG / "local_independent_audit.json")
    assert catalog["complete"] and prior["complete"]
    assert prior["exact_image_overlaps_across_roles"] == []
    assert prior["source_photo_overlaps_across_roles"] == []
    rows = {row["reference_id"]: row for row in catalog["references"]}
    assert len(rows) == 510
    pages = {page["path"]: page for page in catalog["previews"]}
    entries, reviewed_pages, raw_uncertain, files = [], [], set(), {}
    for path in sorted(REVIEW.glob("pages_*.json")):
        review = read(path)
        assert review["rubric_sha256"] == RUBRIC_SHA
        assert review["catalog_results_sha256"] == RESULT_SHA
        assert review["model_outputs_used"] is False
        assert review["native_reserved_used"] is False
        assert all(review[key] == 0 for key in (
            "local_model_forwards", "local_backward_calls", "local_optimizer_updates"
        ))
        local_ids = []
        for page in review["pages"]:
            assert page == pages[page["path"]]
            assert sha(CATALOG / page["path"]) == page["sha256"]
            reviewed_pages.append(page["path"])
            local_ids.extend(page["reference_ids"])
        assert [entry["reference_id"] for entry in review["entries"]] == local_ids
        entries.extend(review["entries"])
        raw_uncertain.update(entry["reference_id"] for entry in review["entries"]
                             if entry["decision"] == "uncertain_review")
        files[path.name] = sha(path)
    assert len(entries) == len({entry["reference_id"] for entry in entries}) == 510
    assert set(rows) == {entry["reference_id"] for entry in entries}
    assert len(reviewed_pages) == len(set(reviewed_pages)) == 32
    assert set(reviewed_pages) == set(pages)
    observations = {}
    for path in sorted(REVIEW.glob("original_resolution_checks_*.json")):
        inspection = read(path)
        assert inspection["full_source_resolution"] == [1024, 1024]
        assert inspection["model_outputs_used"] is False
        assert all(inspection[key] == 0 for key in (
            "local_model_forwards", "local_backward_calls", "local_optimizer_updates"
        ))
        assert not (set(observations) & set(inspection["observations"]))
        observations.update(inspection["observations"])
        files[path.name] = sha(path)
    assert set(observations) == raw_uncertain and len(observations) == 38
    decisions = {"accept_clean_restoration", "exclude_covering", "exclude_reference_quality"}
    for entry in entries:
        rid = entry["reference_id"]
        row = rows[rid]
        assert entry["original_role"] == row["original_role"]
        assert entry["source_sha256"] == row["image_sha256"]
        assert entry["target_sha256"] == row["target_sha256"]
        assert rid in pages[entry["review_page"]]["reference_ids"]
        source = (ROOT / row["image_path_from_workspace"]).resolve()
        assert source.is_relative_to(ROOT)
        target = (CATALOG / row["target"]).resolve()
        assert target.is_relative_to(CATALOG)
        assert sha(source) == entry["source_sha256"]
        assert sha(target) == entry["target_sha256"]
        with Image.open(source) as image:
            assert image.size == (1024, 1024)
        with Image.open(target) as image:
            assert image.size == (256, 256) and image.mode == "RGB"
        entry["source_path_from_workspace"] = row["image_path_from_workspace"]
        entry["target_path_from_workspace"] = target.relative_to(ROOT).as_posix()
        entry["initial_decision"] = entry["decision"]
        if rid in observations:
            entry.update(observations[rid])
            entry["additional_full_resolution_inspection"] = True
        else:
            entry["additional_full_resolution_inspection"] = False
        assert entry["decision"] in decisions and entry["reason"]
    final = {entry["reference_id"]: entry for entry in entries}
    for sample in read(PLAN / "sample_visual_review.json")["entries"]:
        actual = final[sample["reference_id"]]
        assert actual["decision"] == sample["decision"]
        assert actual["source_sha256"] == sample["source_sha256"]
    counts = dict(Counter(entry["decision"] for entry in entries))
    by_role = {role: dict(Counter(entry["decision"] for entry in entries
                                if entry["original_role"] == role))
               for role in ("train", "validation")}
    accepted = [entry.copy() for entry in entries
                if entry["decision"] == "accept_clean_restoration"]
    base = {"version": 3, "date": "2026-10-04", "rubric_sha256": RUBRIC_SHA,
            "catalog_results_sha256": RESULT_SHA, "catalog_independent_receipt_sha256": AUDIT_SHA,
            "original_roles_preserved": True, "source_review_complete": True,
            "model_outputs_used_for_decisions": False, "native_reserved_used": False,
            "model_forwards": 0, "backward_calls": 0, "optimizer_updates": 0,
            "training_ready": False, "model_improvement_established": False,
            "limitation": "Assistant development source review, not independent final human quality review; full historical identity overlap is unestablished. A finite controlled VM protocol is still required."}
    write_once(outputs[0], {**base, "decisions": counts, "by_role": by_role, "entries": entries})
    write_once(outputs[1], {**base, "eligible_sources": len(accepted),
                           "eligible_by_role": dict(Counter(entry["original_role"] for entry in accepted)),
                           "references": accepted})
    files.update({path.name: sha(path) for path in outputs[:2]})
    receipt = {**base, "complete": True, "reviewed_sources": 510, "reviewed_pages": 32,
               "original_resolution_inspections": 38, "initial_sample_decisions_reproduced": 16,
               "source_and_target_sha256_pairs_checked": 510, "decisions": counts,
               "by_role": by_role, "review_input_sha256": files,
               "execution_source_sha256": sha(Path(__file__)),
               "elapsed_seconds": time.monotonic() - started}
    write_once(outputs[2], receipt)
    print(json.dumps({"complete": True, "reviewed": 510, "decisions": counts,
                      "eligible_by_role": dict(Counter(entry["original_role"] for entry in accepted)),
                      "training_ready": False, "elapsed_seconds": receipt["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
