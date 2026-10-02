"""Freeze the returned direct-occlusion head on the same exposed practical cases."""
import json
from pathlib import Path

from scripts.prepare_xseg_mask_comparison import sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT / "outputs/practical_direct_detector_v1"
MODEL=ROOT / "outputs/downloaded_reflection_coverage/outputs/reflection_coverage_vm/reflective/epoch_42.pth"
MODEL_SHA="a51f20e8fa12cee7dec574195debc5ada9b8cfeeb289b12bad11a6d39a9acf25"


def main():
    if OUT.exists():raise ValueError("Preserve previous comparison")
    base=ROOT / "outputs/xseg_mask_comparison_v2"
    p=json.loads((base / "frozen_protocol.json").read_text());r=json.loads((base / "results.json").read_text())
    if sha(base / "frozen_protocol.json")!="9fd70d1edef468e6fb6fe951e3080290e9e63bef0fc65c5fffb99a17fadb9516" or sha(base / "results.json")!="f844ec2740352f1ec5ce15fb32b06bc280fe72e6f41fea8d6b7e51f53da518df" or sha(MODEL)!=MODEL_SHA:
        raise ValueError("Fixed evidence/model changed")
    old_root=ROOT / "outputs/practical_gallery_v2"
    old_path=old_root / "frozen_native_protocol_v1.json"
    old=json.loads(old_path.read_text())
    if old["models"]["candidate"]["sha256"]!=MODEL_SHA:raise ValueError("Cached detector identity differs")
    native={case["id"]:case for case in old["cases"]};rows={row["id"]:row for row in r["rows"]}
    assets={};cases=[]
    def asset(path):
        rel=Path(path).relative_to(ROOT).as_posix();assets[rel]=sha(path);return rel
    for case in p["cases"]:
        item=dict(case)
        for key in ("input","reviewed","protected"):
            if item[key]:asset(ROOT / item[key])
        item["baseline_cached"]=asset(base / rows[item["id"]]["baseline_mask"])
        cached=native.get(item["id"])
        if cached:
            if sha(old_root / cached["input"])!=sha(ROOT / item["input"]):raise ValueError("Cached native source differs")
            item["direct_raw_cached"]=asset(old_root / cached["cached_original_masks"]["candidate"])
        else:item["direct_raw_cached"]=None
        item.pop("probability_cached",None);cases.append(item)
    for path in (MODEL,base / "frozen_protocol.json",base / "results.json",old_path,
                 ROOT / "face_occlusion_adapter.py",ROOT / "face_workflow.py",
                 ROOT / "scripts/run_practical_direct_detector.py",ROOT / "scripts/run_xseg_mask_comparison.py",
                 ROOT / "REFLECTION_COVERAGE_RESULTS.md"):
        asset(path)
    protocol={"format":"dgp-practical-direct-detector-v1","date":"2026-10-02","frozen_before_new_forwards":True,
              "model":asset(MODEL),"model_sha256":MODEL_SHA,"cases":cases,"assets_sha256":assets,
              "preparer_sha256":sha(__file__),"architecture":"pretrained resnet18-unet direct occlusion head",
              "input":"RGB float32/255 NCHW; no extra normalization, padding, crop or source-aware routing",
              "proposal":"Direct covering probability >=0.5 then3px dilation; no fixed-face-ellipse clipping",
              "hypothesis":"A learned covering target avoids the background/clear-glasses complement error while transferring to the newly reviewed hands/hair/scarves/objects. Masks alone do not prove output improvement.",
              "budget":{"new_detector_forwards":26,"cached_native_raw_masks":10,"wall_seconds_after_loading":300,
                        "completion_forwards":0,"restoration_forwards":0,"optimizer_updates":0},
              "criterion":p["criterion"],"metrics":p["metrics"],"scope":p["scope"],
              "historical_selection":"This checkpoint failed the original synthetic retention gates; that immutable failure remains. This practical diagnostic does not qualify best.pth or authorize automatic app selection.",
              "training_admitted":False,"promoted":False,"device":"cpu","threads":4}
    if len(cases)!=36 or sum(case["direct_raw_cached"] is not None for case in cases)!=10:raise ValueError("Expected fixed36/10cached")
    OUT.mkdir();(OUT / "frozen_protocol.json").write_text(json.dumps(protocol,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({"cases":36,"new_forwards":26,"cached_native_masks":10,"protocol_sha256":sha(OUT / "frozen_protocol.json")}))


if __name__=="__main__":main()
