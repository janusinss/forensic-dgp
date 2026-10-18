"""One pre-archive staging repair, preserving the first assembly stop verbatim.

Only export-companion policy/source changes are permitted. No training, data,
loss, schedule or stop controls change. A sealed archive makes this inapplicable.
"""
from copy import deepcopy
import json
from pathlib import Path
import shutil

from research.codeformer_adaptation_v1.common import require, sha256, write_json
from research.codeformer_adaptation_v1.worker import DEFAULT_PROTOCOL, verify_transfer
from scripts.prepare_codeformer_adaptation_packet_v1 import ARCHIVE, NAME, export_companion


def main():
    root = Path.cwd().resolve()
    packet = root / "outputs" / NAME
    require(not (root / "outputs" / ARCHIVE).exists(), "A sealed archive cannot be revised in place")
    evidence = root / "outputs/codeformer_training_readiness_v1/preseal_companion_repair_v1"
    require(not evidence.exists(), "Keep the previous repair attempt")
    old_protocol_path = packet / "protocol.json"
    old_protocol = json.loads(old_protocol_path.read_text(encoding="utf-8"))
    for key, value in DEFAULT_PROTOCOL.items():
        require(old_protocol[key] == value, "Scientific/finite controls must remain unchanged")
    evidence.mkdir()
    affected = ["protocol.json", "research/codeformer_adaptation_v1/worker.py",
                "tests/test_codeformer_adaptation_worker_v1.py"]
    before = {}
    for relative in affected:
        source = packet / relative
        before[relative] = sha256(source)
        if relative != "protocol.json":
            require(before[relative] == old_protocol["files_sha256"][relative], "Unrecorded stage mutation")
        destination = evidence / "before" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        require(sha256(destination) == before[relative], "Before backup differs")
    write_json(evidence / "initial_assembly_stop.json", {
        "error_type": "ValueError", "error": "Only source/manifest/license companions may be exported outside outputs",
        "stage": "Initial packet validate_protocol after copying; no execution archive existed",
        "cause": "Source/notice export allowlist rejected extensionless license and readonly verification log companions",
        "fix": "Explicit extensionless source notices; export only companions accepted by worker; retain every input locally",
        "gradient_queries": 0, "optimizer_updates": 0, "not_a_training_or_model_quality_failure": True,
        "old_protocol_sha256": before["protocol.json"]})
    log = root / "outputs/codeformer_training_readiness_v1/combined_static_tests_sealed_v1.log"
    require("Ran 36 tests" in log.read_text(encoding="utf-8") and log.read_text(encoding="utf-8").rstrip().endswith("OK"),
            "Current sealed-source tests did not pass")
    protocol = deepcopy(old_protocol)
    for relative in affected[1:]:
        shutil.copyfile(root / relative, packet / relative)
        protocol["files_sha256"][relative] = sha256(packet / relative)
    verification = {"complete": True, "unit_tests": 36, "failures": 0,
        "source_sha256": {name: sha256(root / name) for name in affected[1:]},
        "log_sha256": sha256(log), "gradient_queries": 0, "optimizer_updates": 0}
    verification_path = evidence / "sealed_source_verification.json"
    write_json(verification_path, verification)
    relative = "provenance/readiness/preseal_sealed_source_verification.json"
    shutil.copyfile(verification_path, packet / relative)
    protocol["files_sha256"][relative] = sha256(packet / relative)
    protocol["export_files"] = [name for name in sorted(protocol["files_sha256"]) if export_companion(name)]
    write_json(old_protocol_path, protocol)
    pin = sha256(old_protocol_path)
    _, checked = verify_transfer(packet, pin)
    require({key: protocol[key] for key in DEFAULT_PROTOCOL} == DEFAULT_PROTOCOL,
            "Preseal repair changed experiment controls")
    actual_files = {p.relative_to(packet).as_posix() for p in packet.rglob("*") if p.is_file()}
    require(actual_files == set(protocol["files_sha256"]) | {"protocol.json"}, "Unexpected stage namespace")
    receipt = {"complete": True, "protocol_sha256": pin, "before_sha256": before,
        "after_sha256": {name: sha256(packet / name) for name in affected},
        "scientific_and_finite_controls_unchanged": True, "verification": checked,
        "no_training_or_archive_existed": True, "gradient_queries": 0, "optimizer_updates": 0,
        "packet_files": len(actual_files), "packet_bytes": sum(p.stat().st_size for p in packet.rglob("*") if p.is_file())}
    write_json(evidence / "receipt.json", receipt)
    write_json(root / "outputs/codeformer_adaptation_v1_assembly.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
