"""Record completed development visual review and independently check downloads."""
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "outputs/dgp_app_covering_review_v3"
CONTEXT = ROOT / "outputs/dgp_app_completion_context_v3"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rgb(source):
    with Image.open(source) as im:
        return np.asarray(im.convert("RGB")).copy()


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


# These observations were entered after actually viewing the two input sheets,
# six route-output sheets and eight context sheets. They are development review
# by the implementing assistant, not independent final human assessment.
OBSERVATIONS = {
    "00_cloth_mask": ("Fragmented covering/background proposal; large cloth area missed.",
                      "Main cloth removed; plausible lower-face estimate. DGP softens visible eyes/skin; completed-context change does not repair this."),
    "01_pink_mask": ("Automatic proposal is not reliable full covering removal.",
                     "Main mask removed; estimated lower face has stubble and boundary remnants. Hidden anatomy is unknown; DGP softens visible appearance."),
    "02_mask_with_clear_glasses": ("Sparse false marks around eyewear and incomplete mask proposal.",
                                  "Difficult three-quarter pose excluded before generation in all modes; no supported-pose claim."),
    "03_dark_sunglasses": ("Empty automatic proposal despite opaque eyewear.",
                          "Reviewed full eyewear region replaced with estimated narrow/stylized eyes; hidden accuracy unavailable. DGP softens surrounding visible face."),
    "04_sunglasses": ("Automatic eyewear area missed.",
                     "Reviewed opaque eyewear removed; estimated eye area has noticeable rectangular transition. Completed-context restoration does not resolve it."),
    "05_white_glare": ("Strong glare area missed or incompletely marked.",
                      "Assisted glare replacement largely keeps the visible frame. DGP softens remaining facial boundaries; original-photo Auto selects On based on blur signal."),
    "06_mirrored_glare": ("Partial area proposal; incomplete full mirrored eyewear coverage.",
                         "Reviewed opaque eyewear removed; plausible eye estimate. DGP changes/softens visible appearance; exact hidden identity is unknown."),
    "07_hand_over_mask": ("Main hand/mask combination incompletely detected.",
                         "Assisted central hand/mask removed; plausible lower-face estimate differs between original/degraded photos. Boundary remnants can remain."),
    "08_uncovered": ("Original-photo false background/face marks; reviewed mask is empty.",
                     "Off preserves the entire original image. On/context alter and soften visible facial detail; context raw is exactly the current On raw."),
    "09_clear_glasses": ("Empty automatic area is appropriate for this clear-glasses control.",
                         "Empty-mask Off preserves every pixel, including clear frames and non-obstructing hair. On/context soften visible detail; no protected-mask support proves On preservation."),
    "val_18_hand_eyes": ("Hands are fragmentary or missed by automatic proposal.",
                         "Wide/stylized estimated eyes, prominent remaining side fingers and cheek/nose joins. Context change retains these completion defects."),
    "val_25_hand_mouth": ("Lower-face hand obstruction incompletely detected.",
                          "Main hand area replaced with plausible lower face; hidden anatomy unverified. DGP softens visible eyes; context variant is similar."),
    "val_362_hair_eye": ("Obstructing hair nearly absent from automatic proposal.",
                        "Estimated eye looks in a different direction from the visible eye. DGP softens the visible eye; context change does not fix gaze inconsistency. Off preserves hair outside the reviewed footprint."),
    "val_244_knit_scarf": ("Scarf proposal is partial.",
                         "Central nose/mouth revealed with an estimated beard; outer/lower scarf remains. Distinguish clothing outside the face from boundary-removal failure. DGP softens visible facial detail."),
    "val_336_scarf_gloves": ("Scarf/glove obstruction incompletely detected.",
                           "Orange covering/glove texture remains around chin/cheek and the generated patch; poor boundary transition. Context change does not repair completion."),
    "val_6_flower_mouth": ("Flower is largely missed by automatic proposal.",
                          "Main flower removed. Degraded-input completion adds stubble/changes estimated lower-face appearance; no clean hidden reference. DGP/context preserve neither a new useful structural gain nor exact visible texture."),
    "val_7_leaf_eye": ("Core leaf proposal is incomplete.",
                      "Main leaf removed; estimated eye plausible but hand/edge remains outside the frozen footprint. Off keeps unremoved pixels exact. Context has no convincing gain."),
    "val_26_nearly_hidden_hands": ("Near-total central covering is largely missed automatically.",
                                 "Input-only review requires a less-covered/clearer crop; all generation modes reject. This is operator policy, not automatic usability accuracy."),
}


def main():
    parent = read(PARENT / "results.json")
    proof = read(PARENT / "saved_output_audit.json")
    plan = read(PARENT / "plan.json")
    context = read(CONTEXT / "results.json")
    context_proof = read(CONTEXT / "saved_output_audit.json")
    flow = read(ROOT / "scratch/dgp-v3-family-flow.json")
    assert parent["complete"] and proof["complete"] and context["complete"] and context_proof["complete"]
    assert proof["results_sha256"] == sha(PARENT/"results.json")
    assert context_proof["results_sha256"] == sha(CONTEXT/"results.json")
    assert flow["complete"] and flow["errors"] == [] and len(flow["records"]) == 4
    case_map = {c["id"]: c for c in plan["cases"]}
    downloads = []
    for record in flow["records"]:
        if not record.get("output256"):
            continue
        file = ROOT / "scratch" / ("dgp-v3-family-"+record["id"]+".png")
        target = PARENT / "images" / (record["id"]+"_on.png")
        assert sha(file) == sha(target) == record["downloadSha256"]
        downloads.append({"id": record["id"], "download_sha256": sha(file), "exact_cached_png": True})
    archive = ROOT/"scratch/dgp-v3-family-glare.zip"
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist()) == {"original.png", "removal-mask-original.png", "input-256.png",
            "removal-mask.png", "estimate.png", "dgp-raw-float32.npy", "processing.json", "README.txt"}
        assert z.testzip() is None
        case = case_map["05_white_glare_native"]
        np.testing.assert_array_equal(rgb(io.BytesIO(z.read("original.png"))), rgb(ROOT/case["input"]))
        np.testing.assert_array_equal(rgb(io.BytesIO(z.read("input-256.png"))), rgb(ROOT/case["input"]))
        for name in ("removal-mask-original.png", "removal-mask.png"):
            np.testing.assert_array_equal(rgb(io.BytesIO(z.read(name))), rgb(ROOT/case["reviewed"]))
        np.testing.assert_array_equal(rgb(io.BytesIO(z.read("estimate.png"))), rgb(PARENT/"images/05_white_glare_native_on.png"))
        raw = np.load(io.BytesIO(z.read("dgp-raw-float32.npy")), allow_pickle=False)
        with np.load(PARENT/"stages/05_white_glare_native_on.npz", allow_pickle=False) as stages:
            np.testing.assert_array_equal(raw, stages["dgp"][0].transpose(1, 2, 0))
        meta = json.loads(z.read("processing.json"))
        assert hashlib.sha256(raw.tobytes()).hexdigest() == meta["raw_dgp"]["sha256"]
        assert not meta["native_usefulness_qualified"] and not meta["covering_families_qualified"]
        assert meta["mask_source"] == "assisted_reviewed"
    download_audit = {"complete": True, "browser_receipt_sha256": sha(ROOT/"scratch/dgp-v3-family-flow.json"),
        "pngs": downloads, "bundle_sha256": sha(archive), "bundle_input_mask_raw_result_exact": True,
        "model_forwards": 0, "training_calls": 0, "scope": flow["scope"]}
    write(ROOT/"scratch/dgp-v3-family-download-audit.json", download_audit)
    reviews = []
    for row in parent["rows"]:
        base = row["id"].removesuffix("_native").removesuffix("_degraded")
        automatic, assisted = OBSERVATIONS[base]
        reviews.append({"id": row["id"], "family": row["family"], "condition": row["condition"],
            "input_review": row["input_review"], "automatic_observation": automatic,
            "assisted_observation": assisted, "hidden_accuracy": None, "family_qualified": False})
    main_sheets = [PARENT/(condition+"_"+str(i)+".png")
                   for condition in ("original_photo", "synthetic_degraded_photo") for i in (1, 2, 3)]
    inputs = [PARENT/"inputs-original-photos.png", PARENT/"inputs-degraded-photos.png"]
    context_sheets = [CONTEXT/s["file"] for s in context_proof["sheets"]]
    for file in inputs + main_sheets + context_sheets:
        assert file.is_file()
    visual = {"date": "2026-10-05", "complete": True,
        "reviewer": "Implementing assistant; direct development visual inspection",
        "input_review_before_inference": True,
        "input_sheets": [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in inputs],
        "route_output_sheets": [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in main_sheets],
        "route_sheet_display": "Requested original; 1608x1764 source sheets displayed at 1504x1649. All six were inspected; no original-resolution claim for these sheets.",
        "context_sheets": [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in context_sheets],
        "context_sheet_display": "All eight 1072x1192 sheets inspected at original resolution, with 256-pixel cells",
        "cases": reviews, "route_generated_outputs_reviewed": 96, "context_outputs_reviewed": 32,
        "metric_scope": context_proof["metric_scope"],
        "source_limit": "Reused development photographs; legacy _native suffix means original photograph, not native CCTV",
        "automatic_generation_quality": "No automatic proposals approved for generation; proposal-only diagnostic. Automatic family qualification fails.",
        "assisted_quality": "Plausible completion examples coexist with incomplete removal, poor joins and gaze inconsistency; full-family scope remains unqualified.",
        "context_decision": "Do not adopt; 1/16 paired nonremoved MSE and 3/16 SSIM improve over Off, no convincing visual structural gain, completion unchanged.",
        "protected_support_clarification_sha256": sha(PARENT/"protected_support_clarification.json"),
        "state_before_after_unchanged": parent["state_before"] == parent["state_after"] and context["state_before"] == context["state_after"],
        "native_cctv_used": False, "reserved_native_used": False, "hidden_metrics": None,
        "family_qualification": False, "independent_final_review": False,
        "no_neural_calls_in_recording_script": True, "training_calls": 0}
    write(PARENT/"visual_review.json", visual)
    write(CONTEXT/"visual_review.json", {"date": visual["date"], "complete": True,
        "results_sha256": sha(CONTEXT/"results.json"), "saved_output_audit_sha256": sha(CONTEXT/"saved_output_audit.json"),
        "sheets": visual["context_sheets"], "display": visual["context_sheet_display"],
        "observations": {k: v[1] for k, v in OBSERVATIONS.items() if k not in ("02_mask_with_clear_glasses", "val_26_nearly_hidden_hands")},
        "decision": visual["context_decision"], "app_adopted": False, "independent_final_review": False,
        "source_limit": visual["source_limit"], "model_forwards": 0, "training_calls": 0})
    print(json.dumps({"complete": True, "route_reviewed_outputs": 96, "context_reviewed_outputs": 32,
                      "download_pngs_exact": len(downloads), "bundle_exact": True, "family_qualification": False}))


if __name__ == "__main__":
    main()
