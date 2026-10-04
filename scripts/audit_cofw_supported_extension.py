"""Verify COFW dataset admission and old held-out bytes; no model/optimizer."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from supported_real_data import load_supported_manifest

OLD = ROOT / "dataset/detector_supported_review_v2"
NEW = ROOT / "dataset/detector_supported_review_v3"
PROPOSALS = ROOT / "outputs/cofw_covering_proposals_v3"
OUT = ROOT / "outputs/cofw_supported_dataset_validation_v1"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    if OUT.exists(): raise ValueError("Preserve completed dataset audit")
    old,new = read(OLD / "manifest.json"),read(NEW / "manifest.json")
    assert sha(OLD / "manifest.json") == new["parent_supported_manifest_sha256"] == "3054c864bf7612fdcd5d0f55d47130b70f2895b6a0590f1341a60bbb011e2c7e"
    assert sha(NEW / "manifest.json") == "abab07e152941c4ae24d8aea3755fa7aaedb18965b25f6d0d1b6a7e00094aadd"
    _,rows = load_supported_manifest(NEW / "manifest.json")
    assert len(rows) == 165 and new["supported_records"][:123] == old["supported_records"]
    assert new["training_recipe_ready"] is False and new["promoted"] is False
    checked,external = set(),set()
    for row in old["supported_records"]:
        for field in ("image","mask","valid","source_valid"):
            assert (OLD / row[field]).read_bytes() == (NEW / row[field]).read_bytes()
            checked.add(row[field])
        original = ROOT / row["source"].replace("\\","/")
        assert sha(original) == row["source_sha256"]
        external.add(original.relative_to(ROOT).as_posix())
    for split in ("validation","test"):
        assert [r for r in new["supported_records"] if r["split"] == split] == [r for r in old["supported_records"] if r["split"] == split]
    proposal,decisions = read(PROPOSALS / "manifest.json"),read(PROPOSALS / "annotation_decisions.json")
    audit = read(ROOT / "outputs/cofw_covering_data_validation_v3/verification.json")
    assert audit["complete"] and decisions["visual_review_complete"]
    assert new["cofw_proposal_sha256"] == decisions["proposal_sha256"] == audit["proposal_sha256"] == sha(PROPOSALS / "manifest.json")
    assert new["cofw_annotation_decisions_sha256"] == sha(PROPOSALS / "annotation_decisions.json")
    assert decisions["independent_verification_sha256"] == new["cofw_independent_geometry_verification_sha256"] == sha(ROOT / "outputs/cofw_covering_data_validation_v3/verification.json")
    data = {p["id"]:p for p in proposal["records"]}
    source = read(ROOT / "outputs/cofw_annotation_sources_v1/manifest.json")
    seen,counts,core_families = set(),Counter(),Counter()
    for row in new["supported_records"][123:]:
        p = data[row["source_id"]]
        assert row["split"] == row["publisher_split"] == "train" and row["reviewed"] and row["support_required"]
        assert row["approved_proposal"] == p and row["source_license"] == "CC BY 4.0"
        assert row["source_matrix_sha256"] == source["matrix_sha256"]
        assert row["source_id"] in decisions["accepted_train_only_ids"]
        assert row["native_unknown_points_xy"] == p["native_unknown_points_xy"]
        original = ROOT / row["source"]
        assert sha(original) == p["source_sha256"] == row["source_sha256"]
        with Image.open(original) as image:
            rgb = np.asarray(image.convert("RGB")); width,height = image.size
        digest = hashlib.sha256(f"RGB:{width}x{height}:".encode()+rgb.tobytes()).hexdigest()
        assert digest == row["native_rgb_pixel_sha256"] == p["native_rgb_sha256"]
        seen.add(digest); checked.add(original.relative_to(NEW).as_posix())
        for key,role in (("image","images"),("mask","masks"),("valid","valid"),("source_valid","source_valid")):
            assert (NEW / row[key]).read_bytes() == (PROPOSALS / p["files"][role]["path"]).read_bytes()
            checked.add(row[key])
        counts[row["kind"]] += 1; core_families[row["occlusion_stratum"]] += 1
    assert len(seen) == 42 and counts == {"covered":31,"uncovered":11}
    assert sum(len(r["native_unknown_points_xy"]) for r in new["supported_records"][123:]) == 12
    files = {p.relative_to(NEW).as_posix() for p in NEW.rglob("*") if p.is_file()}
    assert files == checked | {"manifest.json"}
    mannequin = [r for r in rows if r["split"] == "validation"][8]
    assert Path(mannequin["image"]).name == "new_covered_40.png"
    with Image.open(mannequin["valid_path"]) as image: assert np.all(np.asarray(image) == 255)
    result = {"format":"dgp-cofw-supported-extension-independent-verification-v1","date":"2026-10-03",
              "complete":True,"auditor_sha256":sha(__file__),"registry_sha256":sha(NEW / "manifest.json"),
              "parent_registry_sha256":sha(OLD / "manifest.json"),"data_files_verified":len(checked),
              "preserved_previous_raw_source_dependencies":sorted(external),"previous_records_preserved":123,
              "new_train_records":42,"counts":new["counts"],"new_primary_family_counts":dict(core_families),
              "old_active_metadata_and_bytes_exact":True,"old_validation_test_records_and_support_exact":True,
              "original_mannequin_preserved":True,"new_native_unknown_raster_pixels":12,
              "actual_three_tensor_reader_verified":True,"training_recipe_ready":False,
              "independent_expert_pixel_accuracy_verified":False,"identity_disjointness_verified":False,
              "optimizer_updates":0,"model_forwards":0,"promoted":False}
    OUT.mkdir()
    with (OUT / "verification.json").open("x",encoding="utf-8",newline="\n") as stream:
        stream.write(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"complete":True,"records":165,"data_files_verified":len(checked),
                      "counts":new["counts"],"verification_sha256":sha(OUT / "verification.json")}))


if __name__ == "__main__":
    main()
