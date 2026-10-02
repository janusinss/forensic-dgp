"""Copy the prior registry unchanged and append eight reviewed train-only sources."""
import copy
import json
from collections import Counter
from pathlib import Path
import shutil

from supported_real_data import load_supported_manifest, sha

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / "dataset/detector_supported_review_v1/manifest.json"
PROPOSALS = ROOT / "outputs/mendeley_mask_proposals_v3"
OUT = ROOT / "dataset/detector_supported_review_v2"


def main():
    if OUT.exists():
        raise ValueError("Preserve existing dataset version")
    if sha(PREVIOUS) != "860ea1dfba8fa442cab19bf3ab41f6a678109dcdb067c38ec0bd79ce43b51ace":
        raise ValueError("Fixed previous registry changed")
    metadata, previous_rows = load_supported_manifest(PREVIOUS)
    decisions_path = PROPOSALS / "annotation_decisions.json"
    if sha(decisions_path) != "2242891ee5a496498f7420429c28f5f5ab570ff877bcdaff0c738a683fc574fb":
        raise ValueError("Reviewed annotation decision changed")
    decisions = json.loads(decisions_path.read_text())
    proposals = json.loads((PROPOSALS / "manifest.json").read_text())
    verification = json.loads((PROPOSALS / "independent_verification.json").read_text())
    assert decisions["visual_review_complete"] and verification["complete"]
    assert decisions["proposal_sha256"] == verification["manifest_sha256"] == sha(PROPOSALS / "manifest.json")
    assert decisions["independent_verification_sha256"] == sha(PROPOSALS / "independent_verification.json")
    accepted = {row["source_id"]: row for row in decisions["accepted"]}
    assert set(accepted) == {row["source_id"] for row in proposals["records"]} and len(accepted) == 8
    OUT.mkdir()
    rows = copy.deepcopy(metadata["supported_records"])
    for row in rows:
        for field in ("image", "mask", "valid", "source_valid"):
            target = (OUT / row[field]).resolve()
            assert target.is_relative_to(OUT.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(PREVIOUS.parent / row[field], target)
            assert sha(target) == row[field + "_sha256"]
    for proposal in proposals["records"]:
        index = proposal["source_id"]
        decision = accepted[index]
        assert decision["files"] == proposal["files"] and decision["source_sha256"] == proposal["source_sha256"]
        assert decision["split"] == "train" and decision["requires_valid_support"]
        original = ROOT / proposal["source"]
        assert sha(original) == proposal["source_sha256"]
        original_copy = OUT / "sources" / f"mendeley_{index}.jpg"
        original_copy.parent.mkdir(exist_ok=True)
        shutil.copyfile(original, original_copy)
        row = {"reviewed": True, "support_required": True, "data_origin": "mendeley_supported_extension",
               "source_id": index, "source": original_copy.relative_to(ROOT).as_posix(),
               "source_sha256": proposal["source_sha256"], "group": proposal["group"], "split": "train",
               "kind": proposal["kind"], "reviewer": decisions["reviewer"],
               "annotation_quality": proposal["annotation_quality"], "review_rationale": proposal["scope"],
               "occlusion_stratum": proposal["family"], "native_size": proposal["native_size"],
               "native_polygons": proposal["native_polygons"], "affine": proposal["affine"],
               "native_unknown_boundary_radius": proposal["native_unknown_boundary_radius"],
               "native_unknown_crop_rows": proposal["native_unknown_crop_rows"],
               "annotation_scope": "Facial covering overlap only; uncertainty/padding excluded with explicit support",
               "geometry": "Uniform full-source pixel-center affine to 256; true padding RGB96, uncertain observed RGB retained",
               "source_url": proposals["source_url"], "source_doi": proposals["doi"],
               "source_license": proposals["license_listed_by_publisher"], "source_attribution": proposals["attribution"],
               "uncovered_face_reference": None, "high_resolution_reference": False,
               "identity_disjointness": "not verified; related tail captures grouped train-only",
               "approved_proposal": copy.deepcopy(proposal)}
        for key, role in (("image", "images"), ("mask", "masks"), ("valid", "valid"), ("source_valid", "source_valid")):
            asset = proposal["files"][role]
            target = OUT / "mendeley" / role / f"{index}.png"
            assert target.resolve().is_relative_to(OUT.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            assert sha(PROPOSALS / asset["path"]) == asset["sha256"]
            shutil.copyfile(PROPOSALS / asset["path"], target)
            row[key] = target.relative_to(OUT).as_posix()
            row[key + "_sha256"] = sha(target)
            assert row[key + "_sha256"] == asset["sha256"]
        rows.append(row)
    result = copy.deepcopy(metadata)
    counts = {split: dict(Counter(r["kind"] for r in rows if r["split"] == split))
              for split in ("train", "validation", "test")}
    result.update(date="2026-10-02", supported_records=rows, counts=counts,
                  parent_supported_manifest_sha256=sha(PREVIOUS), parent_supported_records_preserved=115,
                  mendeley_added_training_records=8, training_recipe_ready=False,
                  current_extension_builder_sha256=sha(__file__),
                  mendeley_lineage_sha256={name: sha(PROPOSALS / name) for name in
                                           ("manifest.json", "annotation_decisions.json", "independent_verification.json", "refinement.json")},
                  new_training_sources_not_in_phase4_split=True,
                  previous_records_preserved=105, new_training_records=18,
                  model_forward_passes=0, optimizer_updates_locally=0, promoted=False)
    result["limitations"] += [
        "The eight Mendeley records share a related capture cohort, not eight independent subjects.",
        "Native Mendeley sources are92x112; labels serve only covering detection, not restoration/inpainting targets.",
        "One obstructing-hair and three hand examples are diagnostic coverage, not broad-family generalization.",
        "New scarf/general-object training coverage is not supplied by this extension.",
    ]
    destination = OUT / "manifest.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _, checked = load_supported_manifest(destination)
    assert len(checked) == 123 and counts == {"train": {"covered": 58, "uncovered": 33},
                                            "validation": {"covered": 15, "uncovered": 10},
                                            "test": {"covered": 4, "uncovered": 3}}
    print(json.dumps({"records": len(checked), "counts": counts, "preserved_previous_records": 115,
                      "manifest_sha256": sha(destination), "training_recipe_ready": False}))


if __name__ == "__main__":
    main()
