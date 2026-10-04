"""Append source-reviewed COFW TRAIN annotations; preserve prior registry/splits."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "dataset/detector_supported_review_v2"
PROPOSALS = ROOT / "outputs/cofw_covering_proposals_v3"
VERIFICATION = ROOT / "outputs/cofw_covering_data_validation_v3/verification.json"
OUT = ROOT / "dataset/detector_supported_review_v3"
PARENT_SHA = "3054c864bf7612fdcd5d0f55d47130b70f2895b6a0590f1341a60bbb011e2c7e"
LABEL_SHA = "e180481c516a6025a162d5c3e135ff41212f9a120a628e7ccf8ec5b6029a8733"
AUDIT_SHA = "e16625fa2f66d51c33a8ee5481dec821648d531522a93d4f54236e187938faf1"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def write(path,value):
    with path.open("x",encoding="utf-8",newline="\n") as stream:
        stream.write(json.dumps(value,indent=2)+"\n")


def main():
    if OUT.exists() or (PROPOSALS / "annotation_decisions.json").exists():
        raise ValueError("Preserve existing data/annotation decisions")
    assert sha(PARENT / "manifest.json") == PARENT_SHA
    assert sha(PROPOSALS / "manifest.json") == LABEL_SHA and sha(VERIFICATION) == AUDIT_SHA
    parent = json.loads((PARENT / "manifest.json").read_text())
    proposal = json.loads((PROPOSALS / "manifest.json").read_text())
    verification = json.loads(VERIFICATION.read_text())
    assert verification["complete"] is True and verification["proposal_sha256"] == LABEL_SHA
    assert not verification["exact_native_or_crop_rgb_overlaps"] and not verification["failures"]
    assert len(parent["supported_records"]) == 123 and len(proposal["records"]) == 42
    assert all(r["publisher_split"] == "train" for r in proposal["records"])
    decisions = {
        "format":"dgp-cofw-assistant-pilot-annotation-decisions-v1","date":"2026-10-03",
        "proposal_sha256":LABEL_SHA,"independent_verification_sha256":AUDIT_SHA,
        "reviewer":"Assistant source/native/overlay inspection; no independent expert adjudication",
        "visual_review_complete":True,"new_model_forwards":0,"optimizer_updates":0,
        "scope":"Reviewed real covering detection only; not clean-face/generator supervision or qualified model",
        "reviewed_source_queue_sheets":4,"reviewed_native_coordinate_sheets":7,
        "reviewed_final_overlay_sheets":proposal["sheets"],
        "accepted_train_only_ids":[r["id"] for r in proposal["records"]],
        "ordinary_retention_controls":"Clear glasses, ordinary hairstyle, beard/moustache, scarf below chin and incidental peripheral person/background are not positive coverings",
        "pose_scope":"Frontal/mild-turn application scope unchanged. Some training sources have extra roll; no deployment pose claim.",
        "limitations":[
            "Facial overlap under opaque coverings is approximate; exact hidden contour/anatomy is unknown.",
            "Dense hair edges/strands and hand margins remain approximate, not pixel-perfect expert labels.",
            "COFW source images may contain repeated identities or overlap pretraining; source hashes prove only the declared exact comparison.",
            "No new nearly-hidden training family or independent target-population validation is certified.",
        ],
    }
    write(PROPOSALS / "annotation_decisions.json",decisions)
    OUT.mkdir()
    rows = copy.deepcopy(parent["supported_records"])
    for row in rows:
        for field in ("image","mask","valid","source_valid"):
            relative = Path(row[field])
            assert not relative.is_absolute() and ".." not in relative.parts
            target = OUT / relative
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists(): shutil.copyfile(PARENT / relative,target)
            assert sha(target) == row[field+"_sha256"]
    for p in proposal["records"]:
        source = ROOT / p["source"]
        assert sha(source) == p["source_sha256"]
        copied = OUT / "sources" / f"{p['id']}.png"
        copied.parent.mkdir(exist_ok=True)
        shutil.copyfile(source,copied)
        row = {
            "reviewed":True,"support_required":True,"split":"train","kind":p["kind"],
            "group":"cofw-author-training-selected-cohort-only-v1","data_origin":"cofw_supported_extension",
            "source_id":p["id"],"source":copied.relative_to(ROOT).as_posix(),"source_sha256":sha(copied),
            "native_rgb_pixel_sha256":p["native_rgb_sha256"],"matlab_train_index":p["matlab_train_index"],
            "publisher_split":"train","source_matrix_sha256":proposal["records"][0].get("matrix_sha256",verification["queue_sha256"]),
            "occlusion_stratum":p["family"],"additional_covering_families":p["additional_families"],
            "reviewer":decisions["reviewer"],"annotation_quality":p["annotation_quality"],
            "native_size":p["native_size"],"crop_native_size":p["crop_native_size"],
            "crop_xyxy_zero_based_half_open":p["crop_xyxy_zero_based_half_open"],
            "native_polygons":p["native_polygons"],"native_preserved_holes":p["native_preserved_holes"],
            "native_unknown_polygons":p["native_unknown_polygons"],
            "native_unknown_points_xy":p["native_unknown_points_xy"],
            "native_unknown_crop_edges":p["native_unknown_crop_edges"],"affine":p["affine"],
            "source_url":proposal["source_url"],"source_doi":proposal["doi"],"source_license":"CC BY 4.0",
            "source_attribution":proposal["attribution"],"uncovered_face_reference":None,
            "identity_disjointness":"not verified; original author train images grouped training-only",
            "annotation_scope":proposal["target_semantics"],"geometry":"Uniform cropped-source pixel-center affine; padding RGB96; uncertain observed pixels retain RGB",
            "approved_proposal":copy.deepcopy(p),
        }
        for key,role in (("image","images"),("mask","masks"),("valid","valid"),("source_valid","source_valid")):
            asset = p["files"][role]
            assert sha(PROPOSALS / asset["path"]) == asset["sha256"]
            target = OUT / "cofw" / role / f"{p['id']}.png"
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(PROPOSALS / asset["path"],target)
            row[key] = target.relative_to(OUT).as_posix(); row[key+"_sha256"] = sha(target)
        rows.append(row)
    # Record the actual original MAT fingerprint, not an image-queue hash.
    source_selection = json.loads((ROOT / "outputs/cofw_annotation_sources_v1/manifest.json").read_text())
    for row in rows[123:]: row["source_matrix_sha256"] = source_selection["matrix_sha256"]
    counts = {s:dict(Counter(r["kind"] for r in rows if r["split"] == s)) for s in ("train","validation","test")}
    assert counts == {"train":{"covered":89,"uncovered":44},"validation":{"covered":15,"uncovered":10},"test":{"covered":4,"uncovered":3}}
    result = copy.deepcopy(parent)
    result.update(date="2026-10-03",supported_records=rows,counts=counts,training_recipe_ready=False,
                  parent_supported_manifest_sha256=PARENT_SHA,parent_supported_records_preserved=123,
                  cofw_added_training_records=42,cofw_builder_sha256=sha(__file__),
                  cofw_annotation_decisions_sha256=sha(PROPOSALS / "annotation_decisions.json"),
                  cofw_proposal_sha256=LABEL_SHA,cofw_independent_geometry_verification_sha256=AUDIT_SHA,
                  cofw_native_unknown_points=12,cofw_original_train_only=True,
                  model_forward_passes=0,optimizer_updates_locally=0,promoted=False)
    result["limitations"] += decisions["limitations"]
    write(OUT / "manifest.json",result)
    print(json.dumps({"records":len(rows),"counts":counts,"previous_records_preserved":123,
                      "manifest_sha256":sha(OUT / "manifest.json"),"training_recipe_ready":False}))


if __name__ == "__main__":
    main()
