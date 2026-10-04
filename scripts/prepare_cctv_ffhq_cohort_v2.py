"""Freeze full-counterpart acquisition and source review; never train or fetch."""
import json
from pathlib import Path

from acquire_cctv_ffhq_counterparts_v1 import bind_thumbnail, digest, write

ROOT = Path(__file__).resolve().parents[1]


def run():
    root = ROOT / "outputs/cctv_dgp_vm_bundle_v1"
    sample = ROOT / "outputs/cctv_dgp_hq_counterparts_v1"
    destination = ROOT / "outputs/cctv_dgp_hq_cohort_plan_v2"
    if destination.exists():
        raise FileExistsError("Preserve the existing plan; use a new version")
    protocol = json.loads((root / "protocol.json").read_text())
    independent = json.loads((sample / "local_independent_audit.json").read_text())
    result = json.loads((sample / "results.json").read_text())
    metadata_path = ROOT / "outputs/cctv_dgp_hq_metadata_v1/ffhq-dataset-v2.json"
    assert independent["complete"] and digest(metadata_path) == independent["metadata_sha256"]
    assert digest(sample / "results.json") == independent["results_sha256"]
    assert digest(root / "protocol.json") == (root / "protocol.sha256").read_text().strip()
    with metadata_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    refs = [r for r in protocol["references"] if r["source"] == "dataset/thumbnails128x128"]
    assert len(refs) == 510 and sum(r["role"] == "train" for r in refs) == 451 and sum(r["role"] == "validation" for r in refs) == 59
    existing = {r["reference_id"]: r for r in result["counterparts"]}
    rows = []
    for ref in refs:
        item = metadata[str(int(Path(ref["source_file"]).stem))]
        assert digest(root / ref["native"]) == protocol["assets_sha256"][ref["native"]]
        bind_thumbnail(ref, item, root)
        row = {"reference": ref, "official_category": item["category"], "image_spec": item["image"], "thumbnail_spec": item["thumbnail"], "original_photo_spec": item["in_the_wild"], "attribution": item["metadata"], "thumbnail_pixel_match": True, "source_review_status": "pending"}
        prior = existing.get(ref["id"])
        if prior:
            source = sample / prior["image"]
            assert digest(source) == prior["image_sha256"]
            row["reuse_verified_source"] = str(source.relative_to(ROOT)).replace("\\", "/")
            row["reuse_source_sha256"] = prior["image_sha256"]
        rows.append(row)

    rubric = {"version": 2, "date": "2026-10-04", "purpose": "Source-only target suitability before new model training/evaluation", "decisions": ["accept_clean_restoration", "exclude_covering", "exclude_reference_quality", "uncertain_review"], "criteria": ["Frontal or mildly turned recognizable face, with no covering hiding eye/nose/mouth features.", "Ordinary clear glasses and non-obstructing hair remain valid; minor reflections leaving features visible are not strong-glare exclusions.", "Natural eye state, facial expression, skin texture and facial hair must not be normalized to an invented appearance.", "Reject obscuring sunglasses, feature-crossing hair/hands/objects, strong lens glare, facial watermarks, severe source blur/darkness or other missing clean target information.", "1024 aligned dimensions do not prove 1024 captured-face detail; inspect original-photo geometry and visible source quality."], "review_all_candidates_before_training": True, "do_not_use_model_metric_failures_to_exclude_references": True, "preserve_original_roles": True, "no_automatic_ethnicity_inference": True, "no_validation_to_training_reassignment": True, "identity_overlap_limit": "Exact files/pixels/photo URLs can be checked; full identity overlap across earlier 80,000 training sources is not established."}
    exclusions = {"tr_ffhq_25564": "Sunglasses hide both eyes; not an uncovered restoration target.", "tr_ffhq_49766": "Windblown hair crosses eye/nose/mouth regions; no clean hidden-region reference."}
    comparisons = {r["reference_id"]: r for r in independent["source_comparison_summaries"]}
    rows_by_id = {r["reference"]["id"]: r for r in rows}
    sample_review = []
    for ref in result["counterparts"]:
        identifier = ref["reference_id"]
        decision = "exclude_covering" if identifier in exclusions else "accept_clean_restoration"
        note = exclusions.get(identifier, "Visible facial features and source detail are adequate for a clean restoration candidate. Clear frames/non-obstructing hair or face-framing clothing are retained.")
        if identifier == "tr_ffhq_00084":
            note = "Inspected the 1024 source: small reflections sit above visible irises; ordinary clear glasses are retained."
        sample_review.append({"reference_id": identifier, "original_role": ref["original_role"], "decision": decision, "reason": note, "source_sha256": ref["image_sha256"], "comparison": comparisons[identifier]})
        rows_by_id[identifier]["source_review_status"] = decision
    expected_bytes = sum(r["image_spec"]["file_size"] for r in rows)
    missing_bytes = sum(r["image_spec"]["file_size"] for r in rows if "reuse_verified_source" not in r)
    manifest = {"version": 2, "date": "2026-10-04", "status": "Prepared acquisition/review plan; not a training protocol", "preparation_source_sha256": digest(Path(__file__)), "parent_reference_protocol_sha256": digest(root / "protocol.json"), "metadata_sha256": independent["metadata_sha256"], "sample_independent_receipt_sha256": digest(sample / "local_independent_audit.json"), "candidates": 510, "original_role_counts": {"train": 451, "validation": 59}, "counterparts_already_verified": 16, "remaining_counterparts": 494, "all510_thumbnail_pixels_bound_to_official_metadata": True, "expected_total_source_image_bytes": expected_bytes, "expected_new_source_image_bytes": missing_bytes, "acquisition_runtime_cap_seconds": 1800, "acquisition_max_new_bytes": 1024*1024*1024, "reuse_verified_metadata_and_source_files": True, "source_review_complete": False, "training_ready": False, "native_reserved_used": False, "model_forwards": 0, "optimizer_updates": 0, "references": rows}
    assert missing_bytes < manifest["acquisition_max_new_bytes"]
    assert len(sample_review) == 16 and sum(r["decision"] == "accept_clean_restoration" for r in sample_review) == 14
    write(destination / "manifest.json", manifest)
    write(destination / "source_review_rubric.json", rubric)
    write(destination / "sample_visual_review.json", {"complete": True, "reviewer": "Assistant source-only development review", "reviewed_counterparts": 16, "accepted_clean_candidates": 14, "excluded_coverings": 2, "reviewed_preview_pages": [r["path"] for r in result["previews"]], "additional1024_images_inspected": ["tr_ffhq_00084", "tr_ffhq_49766"], "model_output_review": False, "entries": sample_review})
    print(json.dumps({"prepared": True, "full_cohort_candidates": 510, "all_thumbnail_bindings_checked": 510, "expected_total_source_image_bytes": expected_bytes, "expected_new_source_image_bytes": missing_bytes, "remaining_images": 494, "remaining_source_reviews": 494, "manifest_sha256": digest(destination / "manifest.json"), "rubric_sha256": digest(destination / "source_review_rubric.json"), "sample_review_sha256": digest(destination / "sample_visual_review.json"), "sample_audit_sha256": digest(sample / "local_independent_audit.json"), "training_ready": False}, indent=2))


if __name__ == "__main__":
    run()
