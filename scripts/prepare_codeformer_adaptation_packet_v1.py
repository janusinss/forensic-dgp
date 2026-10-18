"""Assemble a self-contained, hash-bound packet; never connect or train.

Run only after independent source/data review. Full archive re-read verifies each
member; immutable large inputs need not be sent back with later result exports.
"""
import argparse
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import time

from research.codeformer_adaptation_v1.common import require, sha256, write_json
from research.codeformer_adaptation_v1.worker import DEFAULT_PROTOCOL, bounded, verify_transfer

ROOT = Path(__file__).resolve().parents[1]
NAME = "cctv_codeformer_adaptation_v1_vm"
ARCHIVE = "cctv-codeformer-adaptation-v1-execution.tar.gz"
COMPARISON_SHA = "0899c33d142c490ad9118d81620508c09f640a50fc8d53bb050d305d80eec78a"
MODEL_PINS = {
    "outputs/codeformer_restoration_pretrained_v1/codeformer.pth":
        "1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7",
    "outputs/codeformer_teacher_pretrained_v1/vqgan_code1024.pth":
        "4d1c6741b3cffcbdc2cd1a12b2c3c2442282e042d5de66909cb643d4fa31b20f",
    "outputs/codeformer_perceptual_assets_v1/vgg16_features.pth":
        "a5b412426cfb06d2196179871dff81a905f97688cd46f2f16df1b63d5b251279",
    "outputs/codeformer_perceptual_assets_v1/lpips_vgg_v01.pth":
        "a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868",
}


def export_companion(name):
    """Return only source/manifests/notices, following the worker's exact policy."""
    if name.startswith(("data/", "weights/")) or name.endswith(".pth"):
        return False
    path = Path(name)
    return path.suffix.lower() in {".py", ".md", ".txt", ".json", ".csv",
                                   ".yaml", ".yml", ".sh", ".sha256"} or \
        path.name.upper() in {"LICENSE", "COPYING", "NOTICE", "AUTHORS"}


def preflight(root):
    """Fail before staging if any required byte/source/role prerequisite is absent."""
    data = root / "outputs/codeformer_training_readiness_v1/data_review"
    pins = {**MODEL_PINS,
        "outputs/codeformer_training_readiness_v1/data_review/hq_train_candidates_v1.json":
            "7af5f313743361ae435629171c9e92e2a4fbcf119bdcac8221338a6fd457a493",
        "outputs/codeformer_training_readiness_v1/data_review/ffhq_sources_with_roles_v1.json":
            "4b77dc04c3da7520caf2e2fca5836727d9183e539d4fdc1817e06df50ab7b73c",
        "outputs/cctv_dgp_checkpoint_comparison_v1/plan.json": COMPARISON_SHA}
    for relative, digest in pins.items():
        require(sha256(bounded(root, relative)) == digest, "Frozen input differs: " + relative)
    ledger = json.loads((data / "ffhq_sources_with_roles_v1.json").read_text(encoding="utf-8"))
    for row in ledger["sources"]:
        if row["decision"] != "accept_clean_restoration":
            continue
        require(sha256(bounded(root, row["source_path"])) == row["source_sha256"],
                "Missing/changed clean source: " + row["id"])
        if row["historical_role"] == "validation":
            require(sha256(bounded(root, row["existing_target_path"])) == row["existing_target_sha256"],
                    "Validation target changed")
    plan_path = root / "outputs/cctv_dgp_checkpoint_comparison_v1/plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    for case in plan["cases"]:
        require(case["evidence"] in ("native_unpaired", "synthetic_paired_development"),
                "Unknown comparison evidence kind")
        for key in ("input", "observed", "target"):
            if key in case:
                path = plan_path.parent / case[key]
                relative = path.relative_to(root).as_posix()
                require(sha256(bounded(root, relative)) == plan["bindings"][relative],
                        "Retained DEV bytes changed: " + relative)
    required = ["pretrained_face_restoration.py", "research/__init__.py", "third_party/__init__.py",
        "SYSTEM_WORKFLOW_AND_GOAL.md", "PRACTICAL_OUTPUT_SCOPE.md", "CCTV_CODEFORMER_FINETUNING_PLAN_V1.md",
        "CCTV_CODEFORMER_ADAPTATION_V1_PREPARATION.md",
        *["research/codeformer_adaptation_v1/" + n for n in
          ("__init__.py", "common.py", "data.py", "model.py", "perceptual.py", "worker.py")],
        *["third_party/codeformer/" + n for n in
          ("__init__.py", "codeformer_arch.py", "vqgan_arch.py", "registry.py", "LICENSE", "NOTICE.md")],
        *["outputs/codeformer_perceptual_assets_v1/" + n for n in
          ("manifest.json", "LICENSE_LPIPS.txt", "LICENSE_TORCHVISION.txt", "NOTICE.md")],
        "outputs/codeformer_restoration_pretrained_v1/provenance.json",
        "outputs/codeformer_teacher_pretrained_v1/acquisition.json",
        "outputs/CodeFormer/README.md", "outputs/CodeFormer/LICENSE"]
    for relative in required:
        bounded(root, relative)


def assemble(root):
    started = time.monotonic()
    output = root / "outputs" / NAME
    require(not output.exists(), "Keep the existing frozen packet; use a new version")
    require(shutil.disk_usage(root).free >= 5 * 1024**3, "Require5GiB local assembly reserve")
    preflight(root)
    output.mkdir()
    bindings = {}
    copied_sources = {}

    def copy(source, relative, expected=None):
        source = bounded(root, Path(source).relative_to(root).as_posix())
        require(source.is_file() and not source.is_symlink(), "Missing regular packet input")
        digest = sha256(source)
        require(expected is None or digest == expected, "Original source fingerprint differs: " + str(source))
        destination = bounded(output, relative, must_exist=False)
        require(not destination.exists(), "Duplicate packet path: " + relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        require(sha256(destination) == digest, "Copied bytes differ")
        bindings[relative] = digest
        copied_sources[relative] = {"original": source.relative_to(root).as_posix(),
                                    "sha256": digest, "bytes": destination.stat().st_size}

    sources = ["pretrained_face_restoration.py", "research/__init__.py",
               *["research/codeformer_adaptation_v1/" + name for name in
                 ("__init__.py", "common.py", "data.py", "model.py", "perceptual.py", "worker.py")],
               "third_party/__init__.py",
               *["third_party/codeformer/" + name for name in
                 ("__init__.py", "codeformer_arch.py", "vqgan_arch.py", "registry.py", "LICENSE", "NOTICE.md")]]
    for relative in sources:
        copy(root / relative, relative)
    for path in sorted((root / "tests").glob("test_codeformer_adaptation_*_v1.py")):
        copy(path, "tests/" + path.name)
    data_review = root / "outputs/codeformer_training_readiness_v1/data_review"
    ledger = json.loads((data_review / "ffhq_sources_with_roles_v1.json").read_text(encoding="utf-8"))
    candidate = json.loads((data_review / "hq_train_candidates_v1.json").read_text(encoding="utf-8"))
    require(sha256(data_review / "hq_train_candidates_v1.json") ==
            "7af5f313743361ae435629171c9e92e2a4fbcf119bdcac8221338a6fd457a493", "TRAIN audit changed")
    require(sha256(data_review / "ffhq_sources_with_roles_v1.json") ==
            "4b77dc04c3da7520caf2e2fca5836727d9183e539d4fdc1817e06df50ab7b73c", "Role ledger changed")
    approved = [deepcopy(row) for row in ledger["sources"] if row["decision"] == "accept_clean_restoration"]
    require(sum(row["historical_role"] == "train" for row in approved) == 391 and
            sum(row["historical_role"] == "validation" for row in approved) == 53, "Reference roles differ")
    train_ids = {row["id"] for row in candidate["candidates"]}
    references = []
    for row in approved:
        old_source = row["source_path"]
        destination = "data/sources/" + row["id"] + ".png"
        copy(root / old_source, destination, row["source_sha256"])
        row["original_source_path"] = old_source
        row["source_path"] = destination
        row["selected_for_this_finite_pilot"] = row["historical_role"] == "train"
        if row["historical_role"] == "train":
            require(row["id"] in train_ids, "Unapproved TRAIN ID")
            row["suitable_as_hq_train_candidate"] = True
            row["original_existing_target_path"] = row.pop("existing_target_path")
        else:
            destination = "data/targets/" + row["id"] + ".png"
            copy(root / row["existing_target_path"], destination, row["existing_target_sha256"])
            row["original_existing_target_path"] = row["existing_target_path"]
            row["existing_target_path"] = destination
        references.append(row)
    write_json(output / "references.json", {"format": "codeformer-adaptation-reference-manifest-v1",
        "references": references,
        "historical_roles_preserved": True, "no_pretrained_identity_nonoverlap_claim": True,
        "training_target": "RGB1024-to-Pillow-LANCZOS512; genuine source detail",
        "validation_target": "Retained directRGB1024-to-Pillow-LANCZOS256",
        "FFHQ_pretraining_overlap": "possible/unknown; no guaranteed unseen FFHQ evaluation",
        "identity_overlap": "Only exact file/photo overlap excluded; subject overlap unestablished"})
    bindings["references.json"] = sha256(output / "references.json")
    old_plan = root / "outputs/cctv_dgp_checkpoint_comparison_v1/plan.json"
    plan = json.loads(old_plan.read_text(encoding="utf-8"))
    cases = []
    for old in plan["cases"]:
        row = {key: old[key] for key in ("id", "source", "identity", "profile", "native_size")}
        row["original_evidence"] = old["evidence"]
        row["evidence"] = "native_unpaired" if old["evidence"] == "native_unpaired" else "paired_synthetic"
        for original_key, new_key, new_hash in (("input", "prepared_path", "prepared_sha256"),
                                                ("observed", "observed_path", "observed_sha256"),
                                                ("target", "target_path", "target_sha256")):
            if original_key not in old:
                continue
            source = old_plan.parent / old[original_key]
            relative = "data/comparison/" + old["id"] + "/" + source.name
            original_relative = source.relative_to(root).as_posix()
            copy(source, relative, plan["bindings"][original_relative])
            row[new_key], row[new_hash] = relative, bindings[relative]
        if "input_review" in old:
            row["input_review"] = old["input_review"]
        if "geometry" in old:
            row["geometry"] = old["geometry"]
        cases.append(row)
    write_json(output / "comparison_cases.json", {"cases": cases,
        "original_plan_sha256": sha256(old_plan), "repeated_development_exposure": True,
        "reserved_final_cases": 0, "native_clean_reference_available": False})
    bindings["comparison_cases.json"] = sha256(output / "comparison_cases.json")

    assets = {}
    for label, source, relative in (
        ("initializer", "outputs/codeformer_restoration_pretrained_v1/codeformer.pth", "weights/codeformer.pth"),
        ("teacher", "outputs/codeformer_teacher_pretrained_v1/vqgan_code1024.pth", "weights/vqgan_code1024.pth"),
        ("perceptual_features", "outputs/codeformer_perceptual_assets_v1/vgg16_features.pth", "perceptual/vgg16_features.pth"),
        ("perceptual_calibration", "outputs/codeformer_perceptual_assets_v1/lpips_vgg_v01.pth", "perceptual/lpips_vgg_v01.pth"),
    ):
        copy(root / source, relative, MODEL_PINS[source])
        assets[label] = {"path": relative, "sha256": bindings[relative],
                         "bytes": (output / relative).stat().st_size}
    for name in ("manifest.json", "LICENSE_LPIPS.txt", "LICENSE_TORCHVISION.txt", "NOTICE.md"):
        copy(root / "outputs/codeformer_perceptual_assets_v1" / name, "perceptual/" + name)
    for name in ("provenance.json",):
        copy(root / "outputs/codeformer_restoration_pretrained_v1" / name, "provenance/restoration_" + name)
    copy(root / "outputs/codeformer_teacher_pretrained_v1/acquisition.json", "provenance/teacher_acquisition.json")
    for path in sorted((root / "outputs/codeformer_training_readiness_v1").rglob("*")):
        if not path.is_file() or path.suffix not in (".json", ".md", ".log"):
            continue
        if any(part in ("runtime_inventory",) for part in path.parts):
            continue
        if path.stat().st_size <= 2 * 1024**2:
            copy(path, "provenance/readiness/" + path.relative_to(root / "outputs/codeformer_training_readiness_v1").as_posix())
    for name in ("SYSTEM_WORKFLOW_AND_GOAL.md", "PRACTICAL_OUTPUT_SCOPE.md", "CCTV_CODEFORMER_FINETUNING_PLAN_V1.md",
                 "CCTV_CODEFORMER_ADAPTATION_V1_PREPARATION.md"):
        copy(root / name, "instructions/" + name)
    for name in ("README.md", "LICENSE"):
        copy(root / "outputs/CodeFormer" / name, "provenance/upstream_CodeFormer_" + name)

    script = '''#!/usr/bin/env bash
set -u -o pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
PROTOCOL_SHA="${1:?Pass the frozen protocol SHA256}"
test ! -e trainer.log || { echo 'Retain existing log; do not repeat this packet'; exit 1; }
python -B -u -m research.codeformer_adaptation_v1.worker --root . --protocol-sha "$PROTOCOL_SHA" --verify-transfer || exit 1
set +e
timeout --signal=TERM --kill-after=180 10800 python -B -u -m research.codeformer_adaptation_v1.worker --root . --protocol-sha "$PROTOCOL_SHA" --run 2>&1 | tee trainer.log
TRAINER_STATUS=${PIPESTATUS[0]}
echo "Trainer exit code: $TRAINER_STATUS"
timeout --signal=TERM --kill-after=30 600 python -B -u -m research.codeformer_adaptation_v1.worker --root . --protocol-sha "$PROTOCOL_SHA" --export 2>&1 | tee export.log
EXPORT_STATUS=${PIPESTATUS[0]}
echo "Export exit code: $EXPORT_STATUS"
test "$TRAINER_STATUS" -eq 0 && test "$EXPORT_STATUS" -eq 0
'''
    path = output / "scripts/run_adaptation.sh"
    path.parent.mkdir()
    path.write_text(script, encoding="utf-8", newline="\n")
    bindings["scripts/run_adaptation.sh"] = sha256(path)
    readme = """# CodeFormer adaptation V1 — manual experiment, not a trained result

Only the human's explicit `bash scripts/run_adaptation.sh <protocol-sha>` starts
this experiment. Unpacking, importing or --verify-transfer performs no training.
Use the exact protocol SHA and commands from the local VM instructions document.

The fixed recipe allows1200updates, effective batch4 and4800reference exposures.
It uses391TRAIN sources,53separate validation references and88old comparison
cases. Native CCTV stays unpaired. No final identities or completion training.
Input/output256; internal512; w1; AdaINTrue; no display processing. Initializer,
generator/codebook, teacher/perceptual asset fingerprints remain recorded.

The initial manually run prerequisite measures gradients, actual parameter
changes, timing and memory. Preparation does not claim those VM checks passed.
Require16GiB free after installation and retain4GiB reserve. Optimization caps
at7200seconds; the worker caps at10800seconds. Stops retain failure evidence.
Successful export is not model qualification. Do not repeat a stopped recipe.

`references.json` retains per-photo attribution and historical data roles.
`instructions/` includes current method/preparation and historical instruction
snapshots for provenance. This packet's frozen protocol and run script identify
the finite experiment. Preserve every original checkpoint and prior failed gate.

The shared environment must supplyPython3.10+,Torch with CUDA/BF16 onNVIDIA L4,
NumPy,Pillow andOpenCV. No worker network download or package installation occurs.
Runtime versions and actual generated validation inputs are saved in the return.
The compact LPIPS assets avoid a TorchVision/lpips package dependency on theVM.

Return archive+SHA sidecar+export receipt for independent local audit and visual
review against untouched CodeFormer, DGP and resizing. No app default is changed.
"""
    (output / "READ_ME_FIRST.md").write_text(readme, encoding="utf-8", newline="\n")
    bindings["READ_ME_FIRST.md"] = sha256(output / "READ_ME_FIRST.md")
    protocol = deepcopy(DEFAULT_PROTOCOL)
    protocol.update(assets=assets, files_sha256=bindings,
                    export_files=[name for name in sorted(bindings) if export_companion(name)],
                    attribution="CodeFormer S-Lab; LPIPS Zhang et al.; pretrained ImageNet VGG16; per-photo FFHQ credits in references.json",
                    implementation="Separate Stage-III-inspired adaptation; not full upstream reproduction",
                    exposure_counts={"max_optimizer_updates": 1200, "effective_batch": 4,
                        "maximum_reference_exposures": 4800, "unique_TRAIN_references": 391,
                        "reference_pass_equivalents": 4800 / 391,
                        "unique_new_epoch_count_not_claimed": True},
                    scientific_qualification={"historical_gates_preserved": True,
                        "one_percent_historical_gate_recomputed_here": False,
                        "clear_guard_is_abort_only": True, "user_acceptance_required": True,
                        "independent_RAW_PNG_native_review_required": True,
                        "no_final_evaluation_or_app_promotion": True})
    write_json(output / "protocol.json", protocol)
    pin = sha256(output / "protocol.json")
    _, verification = verify_transfer(output, pin)
    require(len(cases) == 88 and len(references) == 444, "Packet cohort incomplete")
    receipt = {"complete": True, "packet": output.relative_to(root).as_posix(),
        "protocol_sha256": pin, "files": len(bindings) + 1,
        "bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file()),
        "verification": verification, "copy_bindings": copied_sources,
        "training_started": False, "gradient_queries": 0, "optimizer_updates": 0,
        "seconds": time.monotonic() - started}
    write_json(root / "outputs/codeformer_adaptation_v1_assembly.json", receipt)
    print(json.dumps({key: value for key, value in receipt.items() if key != "copy_bindings"}))


def archive(root):
    started = time.monotonic()
    packet = root / "outputs" / NAME
    protocol_sha = sha256(packet / "protocol.json")
    protocol, _ = verify_transfer(packet, protocol_sha)
    members = {name: digest for name, digest in protocol["files_sha256"].items()}
    members["protocol.json"] = protocol_sha
    archive_path = root / "outputs" / ARCHIVE
    require(not archive_path.exists(), "Retain prior execution archive")
    with tarfile.open(archive_path, "x:gz", compresslevel=1) as destination:
        for relative, digest in sorted(members.items()):
            require(sha256(packet / relative) == digest, "Source changed before archive")
            destination.add(packet / relative, arcname=NAME + "/" + relative, recursive=False)
    found = {}
    with tarfile.open(archive_path, "r:gz") as source:
        for entry in source:
            require(entry.isfile() and entry.name.startswith(NAME + "/"), "Unsafe archive member")
            relative = entry.name[len(NAME) + 1:]
            require(relative in members and relative not in found, "Extra/duplicate archive member")
            stream = source.extractfile(entry)
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(4 * 1024**2), b""):
                digest.update(block)
            require(digest.hexdigest() == members[relative], "Archived payload differs")
            found[relative] = digest.hexdigest()
    require(found == members, "Archive namespace differs")
    digest = sha256(archive_path)
    with archive_path.with_name(ARCHIVE + ".sha256").open("x", encoding="ascii", newline="\n") as stream:
        stream.write(digest + "  " + ARCHIVE + "\n")
    receipt = {"complete": True, "archive": archive_path.relative_to(root).as_posix(),
        "archive_sha256": digest, "bytes": archive_path.stat().st_size,
        "all_members_streamed_and_verified": True, "members": len(found),
        "protocol_sha256": protocol_sha, "training_started": False,
        "seconds": time.monotonic() - started}
    write_json(root / "outputs/codeformer_adaptation_v1_archive.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    archive(args.root.resolve()) if args.archive else assemble(args.root.resolve())
