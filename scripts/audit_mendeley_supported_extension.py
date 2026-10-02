"""Verify active registry, previous split/bytes and added annotations without fitting."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from supported_real_data import load_supported_manifest

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "dataset/detector_supported_review_v1"
NEW = ROOT / "dataset/detector_supported_review_v2"
PROPOSALS = ROOT / "outputs/mendeley_mask_proposals_v3"
OUT = ROOT / "outputs/mendeley_supported_dataset_validation_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise ValueError("Preserve prior dataset verification")
    old, _ = load_supported_manifest(OLD / "manifest.json")
    new, active = load_supported_manifest(NEW / "manifest.json")
    assert sha(OLD / "manifest.json") == "860ea1dfba8fa442cab19bf3ab41f6a678109dcdb067c38ec0bd79ce43b51ace"
    assert new["parent_supported_manifest_sha256"] == sha(OLD / "manifest.json")
    assert new["supported_records"][:115] == old["supported_records"]
    previous_sources = {row["source_sha256"] for row in old["supported_records"]}
    checked = set()
    for row in old["supported_records"]:
        for key in ("image", "mask", "valid", "source_valid"):
            assert (OLD / row[key]).read_bytes() == (NEW / row[key]).read_bytes()
            checked.add(row[key])
    for split in ("validation", "test"):
        assert [r for r in new["supported_records"] if r["split"] == split] == [r for r in old["supported_records"] if r["split"] == split]
    decisions = json.loads((PROPOSALS / "annotation_decisions.json").read_text())
    proposal_data = json.loads((PROPOSALS / "manifest.json").read_text())
    proposals = {r["source_id"]: r for r in proposal_data["records"]}
    for filename, expected in new["mendeley_lineage_sha256"].items():
        assert sha(PROPOSALS / filename) == expected
    assert json.loads((PROPOSALS / "independent_verification.json").read_text())["complete"]
    added = new["supported_records"][115:]
    assert len(added) == len(decisions["accepted"]) == 8
    assert {r["group"] for r in added} == {"mendeley-upload-tail-related-capture-training-only-v1"}
    assert not previous_sources.intersection(r["source_sha256"] for r in added)
    for row in added:
        proposal = proposals[row["source_id"]]
        assert row["approved_proposal"] == proposal and row["split"] == "train"
        assert row["native_polygons"] == proposal["native_polygons"] and row["affine"] == proposal["affine"]
        assert row["native_unknown_crop_rows"] == proposal["native_unknown_crop_rows"]
        assert row["source_license"] == "CC BY 4.0" and row["source_doi"] == "10.17632/s57wnx78vh.1"
        assert sha(ROOT / row["source"]) == sha(ROOT / proposal["source"]) == row["source_sha256"]
        for key, role in (("image", "images"), ("mask", "masks"), ("valid", "valid"), ("source_valid", "source_valid")):
            assert (NEW / row[key]).read_bytes() == (PROPOSALS / proposal["files"][role]["path"]).read_bytes()
            checked.add(row[key])
        checked.add(Path(row["source"]).relative_to(NEW.relative_to(ROOT)).as_posix())
    # Original mannequin is still present with full supervision; native and
    # test/validation support must not be relaxed to improve a future gate.
    mannequin = [r for r in active if r["split"] == "validation"][8]
    assert Path(mannequin["image"]).name == "new_covered_40.png"
    with Image.open(mannequin["valid_path"]) as mask:
        assert np.all(np.asarray(mask) == 255)
    assert len(active) == 123 and sum(r["split"] == "train" for r in active) == 91
    assert new["training_recipe_ready"] is False and new["optimizer_updates_locally"] == 0
    files = {p.relative_to(NEW).as_posix() for p in NEW.rglob("*") if p.is_file()}
    assert files == checked | {"manifest.json"}
    record = {"format": "dgp-mendeley-supported-extension-verification-v1", "date": "2026-10-02",
              "complete": True, "registry_sha256": sha(NEW / "manifest.json"),
              "previous_registry_sha256": sha(OLD / "manifest.json"), "auditor_sha256": sha(__file__),
              "data_files_verified": len(checked), "previous_records_preserved": 115,
              "new_train_records": 8, "train_count": 91, "validation_count": 25, "test_count": 7,
              "counts": new["counts"], "old_active_metadata_and_bytes_exact": True,
              "original_mannequin_and_support_preserved": True, "new_sources_grouped_train_only": True,
              "training_recipe_ready": False, "model_forwards": 0, "optimizer_updates": 0,
              "independent_expert_mask_accuracy_verified": False, "identity_disjointness_verified": False}
    OUT.mkdir()
    destination = OUT / "verification.json"
    destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"complete": True, "records": 123, "data_files_verified": len(checked),
                      "verification_sha256": sha(destination)}))


if __name__ == "__main__":
    main()
