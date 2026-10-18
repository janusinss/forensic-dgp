"""Publish a verified PREPARED/UNRUN milestone, preserving exact prior bytes."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from research.codeformer_adaptation_v1.common import require, sha256, write_json


def main():
    root = Path.cwd().resolve()
    evidence = root / "outputs/codeformer_adaptation_v1_publication"
    require(not evidence.exists(), "Preserve prior preparation publication")
    archive = json.loads((root / "outputs/codeformer_adaptation_v1_archive.json").read_text())
    reviews = root / "outputs/codeformer_training_readiness_v1/packet_review/full_packet_audit_v1"
    independent = json.loads((reviews / "independent_full_archive_audit_v1.json").read_text())
    require(archive["complete"] and independent["complete"] and
            archive["archive_sha256"] == independent["archive_sha256"] and
            archive["protocol_sha256"] == independent["protocol_sha256"], "Independent archive evidence differs")
    require(sha256(root / archive["archive"]) == archive["archive_sha256"], "Execution archive changed")
    pin = archive["protocol_sha256"]
    status = f"""# CodeFormer adaptation V1 verified for manual transfer —15October2026

**PREPARED / UNRUN; the complete thesis/model goal remains active.**
The finite restoration-first adaptation uses391approved1024FFHQ TRAIN sources,
53separate validation sources and88retained comparison views (48native/unpaired,
40synthetic/paired). It preserves the original CodeFormer/DGP states and data
roles. No actual training, gradients, new candidate qualification or app/default
change occurred. Further completion/detector development remains deferred.

[Five manual upload/install/tmux/run/download steps](CCTV_CODEFORMER_ADAPTATION_V1_VM.md).
[Complete design, finite stops and acceptance boundaries](CCTV_CODEFORMER_ADAPTATION_V1_PREPARATION.md).

Execution archive: `{archive['archive']}`,{archive['bytes']:,}bytes.
SHA256:`{archive['archive_sha256']}`.
Protocol SHA256:`{pin}`.
All779regular archive members were independently streamed and matched, including
774copied payloads audited against their retained originals.36local tests pass;
the trainable loader matches untouched CodeFormer exactly on four CPU cases,
and LPIPS matches its official reference on three fixtures. These establish
preparation/inference integrity, not learned restoration quality or L4 feasibility.

The planned run has1200updates, effective batch4,4800exposures and snapshots at
0/200/600/1200. Require16GiB free after installation; retain4GiB reserve. First20
updates measure actual gradients/changes, timing projection and VRAM. Optimization
caps at7200seconds, worker10800, export600; no automatic launch/resume. RAW/PNG
clear-preservation stops are additional abort guards, not historical-gate
replacement or qualification. Retain every stopped state and audit returns.

Fresh authenticated API inventory confirms the existing L4 VM is running after
the user's start. Guest free space, packages and active workloads remain
unverified pending its trusted SSH host fingerprint. No guest command or
training ran in this preparation. The manual guide contains a dependency check;
all actual learning remains user-manual in tmux.

The earlier local offload remains complete:12.39GB additional historical output
copies were removed after verified Drive recovery, separate from the old52.03GB.
Current research assets and active comparison/app files stay local. This packet
adds necessary training/reference copies; it is not a full VM migration backup.

Independent evidence: `outputs/codeformer_training_readiness_v1/`,
`outputs/codeformer_adaptation_v1_archive.json` and this publication's exact
before-byte backups. Current restoration goal and exact prior document follow.

---

"""
    evidence.mkdir()
    prefixes = {
        "PROJECT_HANDOFF.md": status,
        "SYSTEM_WORKFLOW_AND_GOAL.md": status,
        "PRACTICAL_OUTPUT_SCOPE.md": status,
        "CCTV_CODEFORMER_FINETUNING_PLAN_V1.md": status,
        "CCTV_CODEFORMER_ADAPTATION_V1_PREPARATION.md":
            "# Execution packet independently verified —15October2026\n\n"
            "**PREPARED / UNRUN.**36local tests and independent779-member archive/copy audits pass.\n"
            "No actual gradients, optimizer updates or restoration qualification.\n"
            "[Manual commands](CCTV_CODEFORMER_ADAPTATION_V1_VM.md).\n"
            "The complete finite design below is unchanged; its earlier packaging status is historical.\n\n---\n\n",
    }
    records = {}
    for name, prefix in prefixes.items():
        path = root / name
        before = path.read_bytes()
        backup = evidence / "before" / name
        backup.parent.mkdir(exist_ok=True)
        backup.write_bytes(before)
        require(sha256(backup) == sha256(path), "Before backup differs")
        new_prefix = prefix.encode("utf-8")
        path.write_bytes(new_prefix + before)
        require(path.read_bytes()[len(new_prefix):] == before, "Historical suffix changed")
        records[name] = {"before_sha256": sha256(backup), "after_sha256": sha256(path),
                         "unchanged_suffix_offset": len(new_prefix), "exact_prior_bytes_retained": True}
    name = "CCTV_CODEFORMER_ADAPTATION_V1_VM.md"
    path = root / name
    before = path.read_bytes()
    (evidence / "before" / name).write_bytes(before)
    text = before.decode("utf-8")
    old = "Status: **PACKET ASSEMBLED; final archive verification in progress; NOT LAUNCHED.**"
    require(text.count(old) == 1, "Manual guide status marker differs")
    text = text.replace(old, "Status: **VERIFIED / PREPARED / UNRUN.** Independent full-copy and archive audits pass.")
    text += (f"\nExecution archive SHA256: `{archive['archive_sha256']}`.\n"
             f"Protocol SHA256: `{pin}`. SHA sidecars use Linux LF line endings.\n")
    path.write_text(text, encoding="utf-8", newline="\n")
    records[name] = {"before_sha256": sha256(evidence / "before" / name), "after_sha256": sha256(path),
                     "earlier_draft_retained": True, "commands_unchanged": True}
    receipt = {"complete": True, "timestamp_UTC": datetime.now(timezone.utc).isoformat(),
        "documents": records, "archive_sha256": archive["archive_sha256"], "protocol_sha256": pin,
        "archive_bytes": archive["bytes"], "local_free_bytes": shutil.disk_usage(root).free,
        "preparation_complete": True, "training_started": False, "guest_inventory_verified": False,
        "model_qualified": False, "goal_complete": False,
        "gradient_queries": 0, "optimizer_updates": 0}
    write_json(evidence / "publication.json", receipt)
    print(json.dumps({key: value for key, value in receipt.items() if key != "documents"}))


if __name__ == "__main__":
    main()
