"""Prepare a finite CodeFormer completion pilot packet. No model imports or runs.

The VM checkpoint stays external and read-only. This builder writes a new packet,
not an archive, and refuses an existing destination. Python3.10-compatible hashes.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import shutil
import sys

import numpy as np
from PIL import Image
import PIL

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"C:\xampp\htdocs\YEAR 4\Testing")
PARENT = ROOT / "outputs/completion_codeformer_finetune_v1_packet"
VM_NAME = "cctv_completion_codeformer_finetune_v1_vm"
STAGE = PARENT / VM_NAME
WORKER = ROOT / "scripts/cctv_completion_codeformer_finetune_v1_vm.py"
WORKER_SHA = "8f3a69bc3209590a6919adfa67afa9dbd3da4857e9180c337123a6ee65e332cb"
CHECKPOINT_SHA = "b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797"
CHECKPOINT_RELATIVE = "../cctv_completion_codeformer_gradient_v1_vm/weights/codeformer_inpainting.pth"
EVIDENCE = {
    "training_data_readiness.json": (ROOT / "outputs/completion_codeformer_quality_triage_v1/training_data_readiness.json", "87dabe0b2c81350d91f9c0a00666c44dd8b514d1ac23d83c2b571f4cef80f0d6"),
    "root_input_review.json": (ROOT / "outputs/completion_codeformer_finetune_v1_input_review/root_input_review.json", "c103673ca233d28a5a3a82855611886dc7543ddd8922271ea943de02f2456295"),
    "gradient_transport_audit.json": (ROOT / "outputs/completion_codeformer_gradient_v1_return/transport_audit.json", "94e24ec881e708f6b6dd2eedf2a3675b1a492515e876e5570155bae01b4c564f"),
    "gradient_independent_result_audit.json": (ROOT / "outputs/completion_codeformer_gradient_v1_return/independent_result_audit.json", "a24d315d770acd7fbfe49454535776c15b06f4215206e0f6d411a5b542ce659c"),
    "gradient_repaired_protocol.json": (ROOT / "outputs/completion_python310_repair_v1/payload/protocol.json", "e79183a9f4e3ea50e9ac1d6705d9667f046364132e1f1be0628befd024835d9a"),
    "original_source_admission.json": (ROOT / "outputs/completion_codeformer_manual_v1/source_admission.json", "188e54fd998ad97e9bef84607b8e92643f555565bce5a5c64c4a7a759f2c61ab"),
}
DEV_PROTOCOL = DATA_ROOT / "outputs/completion_mat_mirror_comparison_v1/protocol.json"
DEV_PROTOCOL_SHA = "d3b35fe043c198007515f3dc0a221b7c7b369b7edd8d7b9bd45122cc550cf76f"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new(path, obj):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def pin(path):
    path = Path(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha(path)}


def check_pin(record):
    path = Path(record["path"])
    need(path.is_file() and not path.is_symlink(), "Missing/linked source: " + str(path))
    need(path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], "Source pin differs: " + str(path))


def target_path(relative):
    name = PurePosixPath(relative)
    need(not name.is_absolute() and relative and "\\" not in relative and ":" not in relative
         and all(part not in (".", "..") for part in name.parts), "Unsafe packet member")
    path = STAGE.joinpath(*name.parts)
    need(path.resolve().is_relative_to(STAGE.resolve()), "Packet member escaped destination")
    return path


def copy_new(source, relative, expected_sha=None):
    source = Path(source)
    need(source.is_file() and not source.is_symlink(), "Missing/linked source: " + str(source))
    digest = sha(source)
    if expected_sha is not None:
        need(digest == expected_sha, "Source SHA differs: " + str(source))
    target = target_path(relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, target.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, 8 * 1024**2)
    need(sha(target) == digest, "Copied packet bytes differ")
    return {"path": relative, "bytes": target.stat().st_size, "sha256": digest}


def save_png(array, relative):
    target = target_path(relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        Image.fromarray(array).save(stream, format="PNG")
    with Image.open(target) as image:
        need(np.array_equal(np.array(image), array), "Saved PNG differs")
    return {"path": relative, "bytes": target.stat().st_size, "sha256": sha(target)}


def source_only_import(path, name):
    """Only import admitted stdlib-only worker contract or NumPy/Pillow mask helper."""
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    need("torch" not in sys.modules, "Preparation must not import Torch")
    return module


def inspect_vm_python(path):
    source = Path(path).read_text(encoding="utf-8-sig")
    ast.parse(source, filename=str(path), feature_version=(3, 10))
    need("file_digest" not in source, "Python3.10 incompatible hash API in VM Python: " + str(path))


def prepare():
    need("torch" not in sys.modules, "Start preparation in a fresh non-neural interpreter")
    need(not PARENT.exists(), "Packet preparation already exists; preserve it and choose a new version")
    need(sha(WORKER) == WORKER_SHA, "Reviewed worker source changed")
    for path, expected in EVIDENCE.values():
        need(sha(path) == expected, "Evidence changed: " + str(path))
    need(sha(DEV_PROTOCOL) == DEV_PROTOCOL_SHA, "Historical DEV protocol changed")
    worker = source_only_import(WORKER, "admitted_completion_finetune_contract")
    need(worker.VM_NAME == VM_NAME, "Worker VM name differs")
    contract = worker.CONTRACT
    need(contract["fit_references"] == 32 and contract["monitor_references"] == 8
         and contract["optimizer_steps_max"] == 64 and contract["output_bytes"] == 1536 * 1024**2, "Worker bounds differ")
    helper = source_only_import(ROOT / "scripts/run_completion_codeformer_hq_train_probe_v1.py", "admitted_hq_fixed_masks")
    need(tuple(worker.KINDS) == tuple(helper.KINDS[:4]), "Geometry kinds differ")
    readiness = read(EVIDENCE["training_data_readiness.json"][0])
    need(readiness["complete"] and readiness["independent"], "Independent TRAIN readiness required")
    pool = readiness["genuine_HQ_candidate_sources"]["training_sources"]
    need(len(pool) == 391, "Expected391 eligible historical TRAIN references")
    selected = sorted(pool, key=lambda row: row["reference_id"])[:40]
    need(all(row["original_role"] == row["legacy_completion_split"] == "train"
             and row["source_review_decision"] == "accept_clean_restoration" for row in selected), "Only clean TRAIN admitted")
    review = read(EVIDENCE["root_input_review.json"][0])
    need(review["fit"] == [row["reference_id"] for row in selected[:32]]
         and review["monitor"] == [row["reference_id"] for row in selected[32:]], "Root input review selection mismatch")
    original_admission = read(EVIDENCE["original_source_admission.json"][0])
    need(original_admission["complete"] and original_admission["independent"], "Original source admission missing")
    checkpoint = original_admission["checkpoint"]
    check_pin(checkpoint)
    need(checkpoint["bytes"] == 370786611 and checkpoint["sha256"] == CHECKPOINT_SHA, "Wrong original inpainting checkpoint")
    for row in selected:
        check_pin(row["source"]); check_pin(row["target256"])
        need(row["source_size"] == [1024, 1024] and row["target_size"] == [256, 256], "Wrong source resolution")
    # All new executable sources must exist before any output is created.
    sources = ["scripts/cctv_completion_codeformer_finetune_v1_vm.py",
               "scripts/export_completion_codeformer_finetune_v1.py", "scripts/run_completion_codeformer_finetune_v1.sh",
               "pretrained_completion.py", "completion.py", *worker.SOURCE_MODULES.values(),
               "third_party/codeformer/LICENSE", "third_party/codeformer/NOTICE.md"]
    sources = list(dict.fromkeys(sources))
    admitted_source_pins = {Path(item["path"]).resolve(): item for item in original_admission["source_files"]}
    for relative in sources:
        path = ROOT / relative
        need(path.is_file(), "Source not ready: " + relative)
        if relative.endswith(".py"): inspect_vm_python(path)
        if path.resolve() in admitted_source_pins: check_pin(admitted_source_pins[path.resolve()])
    need("file_digest" not in WORKER.read_text(encoding="utf-8"), "Hash compatibility regression")
    development = read(DEV_PROTOCOL)
    need(len(development["cases"]) == 32 and len(development["excluded_before_generation"]) == 4, "Public development membership differs")
    for row in development["cases"]:
        for key in ("input", "reviewed"):
            need(sha(DATA_ROOT / row[key]) == development["sources_sha256"][row[key]], "Historical development source drift")
    PARENT.mkdir(); STAGE.mkdir()
    files, references, fit_cases, monitor_cases, dev_cases = [], [], [], [], []
    for relative in sources:
        files.append(copy_new(ROOT / relative, relative))
    for name, (path, expected) in EVIDENCE.items():
        files.append(copy_new(path, "evidence/" + name, expected))
    files.append(copy_new(DEV_PROTOCOL, "evidence/public_development_protocol.json", DEV_PROTOCOL_SHA))
    source_admission = ROOT / "outputs/completion_codeformer_finetune_v1_source_review/independent_source_admission.json"
    need(source_admission.is_file(), "Independent fine-tune source admission missing")
    source_review = read(source_admission)
    need(source_review["complete"] and source_review["independent"], "Fine-tune source review incomplete")
    files.append(copy_new(source_admission, "evidence/finetune_source_admission.json"))
    for page in review["pages"]:
        need(sha(page["path"]) == page["sha256"], "Root input-review page changed")
        files.append(copy_new(page["path"], "evidence/input_review/" + Path(page["path"]).name, page["sha256"]))
    for position, row in enumerate(selected):
        identity = row["reference_id"]; role = "fit" if position < 32 else "monitor"
        path256 = "data/clean256/" + identity + ".png"; path512 = "data/clean512/" + identity + ".png"
        files.append(copy_new(row["target256"]["path"], path256, row["target256"]["sha256"]))
        with Image.open(row["source"]["path"]) as image:
            need(image.format == "PNG" and image.mode == "RGB" and image.size == (1024, 1024), "Exact native source must be RGB1024 PNG")
            clean512 = np.array(image.resize((512, 512), Image.Resampling.LANCZOS), copy=True)
            direct256 = np.array(image.resize((256, 256), Image.Resampling.LANCZOS), copy=True)
        with Image.open(row["target256"]["path"]) as image:
            need(image.mode == "RGB" and image.size == (256, 256) and np.array_equal(np.array(image), direct256), "Retained256 target derivation mismatch")
        files.append(save_png(clean512, path512))
        references.append({"id": identity, "role": role, "original_role": "train", "legacy_completion_split": "train",
                           "clean256": path256, "clean512": path512, "original_source1024": row["source"],
                           "retained_target256": row["target256"], "original_size": [1024, 1024],
                           "clean_review_decision": row["source_review_decision"], "attribution": row["attribution"],
                           "clean512_derivation": "Pillow RGB LANCZOS1024-to512 directly from admitted original; clean256 exact retained PNG"})
        for kind in worker.KINDS:
            relative = "data/masks256/" + identity + "__" + kind + ".png"
            mask = helper.fixed_mask(kind)
            need(mask.shape == (256, 256) and mask.dtype == np.uint8 and np.isin(mask, (0,255)).all(), "Invalid fixed geometry")
            files.append(save_png(mask, relative))
            case = {"id": identity + "__" + kind, "reference_id": identity, "mask_kind": kind,
                    "mask": relative, "mask_pixels": int((mask == 255).sum()), "role": role}
            (fit_cases if role == "fit" else monitor_cases).append(case)
    active_development = 0
    for original in development["cases"]:
        identity = original["id"]
        input_relative = "data/development/input256/" + identity + ".png"
        mask_relative = "data/development/masks256/" + identity + ".png"
        files.append(copy_new(DATA_ROOT / original["input"], input_relative, development["sources_sha256"][original["input"]]))
        files.append(copy_new(DATA_ROOT / original["reviewed"], mask_relative, development["sources_sha256"][original["reviewed"]]))
        with Image.open(target_path(input_relative)) as image:
            need(image.mode == "RGB" and image.size == (256, 256), "Public development RGB256 differs")
        with Image.open(target_path(mask_relative)) as image:
            need(image.mode == "L" and image.size == (256, 256), "Public development L256 differs")
            mask = np.array(image, copy=True)
        need(np.isin(mask, (0,255)).all(), "Public development mask not binary")
        active_development += bool(mask.any())
        dev_cases.append({"id": identity, "input256": input_relative, "mask": mask_relative,
                          "family": original["family"], "condition": original["condition"], "mask_pixels": int((mask==255).sum()),
                          "role": "previously_exposed_public_development_unpaired", "operator_input_review": original["operator_input_review"],
                          "historical_input": original["input"], "historical_reviewed_mask": original["reviewed"]})
    need(len(references)==40 and len(fit_cases)==128 and len(monitor_cases)==32 and len(dev_cases)==32 and active_development==28, "Fixed data recount failed")
    families = Counter(row["family"] for row in dev_cases)
    protocol = {"format": "completion-codeformer-finetune-v1", "status": "prepared_not_run_unqualified",
                "frozen_UTC": datetime.now(timezone.utc).isoformat(), "contract": contract,
                "checkpoint": {"path": CHECKPOINT_RELATIVE, "bytes": 370786611, "sha256": CHECKPOINT_SHA},
                "references": references, "fit_cases": fit_cases, "monitor_cases": monitor_cases,
                "development_cases": dev_cases, "development_excluded_before_generation": development["excluded_before_generation"],
                "schedule": worker.training_schedule(fit_cases), "files": sorted(files, key=lambda item:item["path"]),
                "selection": "First40 eligible391 historicalTRAIN references sorted by zero-paddedreferenceID; first32 fit,next8 monitor; all reviewed prospectively without output/error ranking",
                "data_derivation": {"Pillow": PIL.__version__, "NumPy": np.__version__, "clean512": "Direct LANCZOS1024-to512",
                                    "clean256": "Retained exactPNG; independently recounted directLANCZOS1024-to256",
                                    "masks": "Same four prospective HQ-probe fixed256 coordinates; unchanged perreference; no actual hidden target for realDEV"},
                "data_limits": ["Clear paired TRAIN contexts only in this first pilot; no degraded paired fit/monitor simulation",
                                "Monitor is originalTRAIN source-disjoint from fit by ID/file, not established independent identity orFINAL",
                                "CodeFormer FFHQ pretraining overlap unknown; historical project exposure preserved",
                                "PublicDEV32 retains original/synthetic-degraded photographs; native suffix is not CCTV; no hiddenPSNR/SSIM",
                                "OriginalVAL and reservedFINAL pixels are not accessed or transferred"],
                "terms": {"experiment": "Noncommercial thesis/research; retain CodeFormer S-Lab and per-photo FFHQ source licenses/attribution",
                          "source_license_counts": dict(Counter(row["attribution"]["license"] for row in selected)),
                          "no_blanket_license_or_identity_clearance": True},
                "development_family_counts": dict(families), "empty_controls": "40fit/monitor exact zero-mask bypasses at each evaluation plus four realDEV emptycontrols; not additional neuralcalls",
                "state_retention": "One atomic terminal full trainable state plus bothAdam states, parameter names/order, allRNGs, sampler/counters. u8/u16 images/metrics are not recoverable snapshots and cannot be rescued as checkpoints.",
                "export": {"stem": "cctv-completion-codeformer-finetune-v1", "cap_bytes":1536*1024**2, "cap_seconds":900,
                           "original_weight_archive_included":False, "terminal_state_max_count":1},
                "manual_execution_only": True, "optimizer_updates_locally":0, "local_neural_calls":0,
                "automatic_resume_or_promotion":False, "quality_qualified":False, "goal_complete":False}
    need(len({item["path"] for item in files})==len(files), "Duplicate file binding")
    actual = {path.relative_to(STAGE).as_posix() for path in STAGE.rglob("*") if path.is_file()}
    need(actual == {row["path"] for row in files}, "Unlisted prepared packet files")
    for row in files:
        need(sha(target_path(row["path"]))==row["sha256"], "Prepared packet source drift")
    write_new(STAGE / "protocol.json", protocol)
    protocol_sha=sha(STAGE/"protocol.json")
    with (PARENT/"protocol.sha256").open("x",encoding="ascii",newline="\n") as stream:stream.write(protocol_sha+"\n")
    write_new(PARENT/"preparation.json", {"complete":True,"builder":pin(__file__),"worker_sha256":WORKER_SHA,
               "stage":str(STAGE),"protocol_sha256":protocol_sha,"files":len(files),"payload_bytes":sum(item["bytes"] for item in files),
               "references":40,"fit_cases":128,"monitor_cases":32,"public_development_cases":32,"public_development_nonempty":28,
               "original_weight_copied":False,"checkpoint_local_verification":pin(checkpoint["path"]),
               "local_runtime_verify_transfer":"Not executed: relative external checkpoint is intentionally the VM layout; independent local packet-byte audit required",
               "torch_imported":False,"model_constructed":False,"neural_calls":0,"optimizer_updates":0,"archive_created":False,"goal_complete":False})
    need("torch" not in sys.modules, "Unexpected Torch import during preparation")
    print(json.dumps({"complete":True,"stage":str(STAGE),"protocol_sha256":protocol_sha,"files":len(files),"payload_bytes":sum(item["bytes"] for item in files),"neural_calls":0},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare",action="store_true",required=True)
    parser.parse_args(); prepare()


if __name__=="__main__":
    main()
