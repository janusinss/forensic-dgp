"""Retain publication R0 and clarify the mean-of-fractions wording only."""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

from publish_u64_role_gradients_return_v1 import ROOT, PUBLICATION, DOCUMENTS, pin


OLD = ("Low-light gradients contribute 62.23% of the mean per-photo sum of role-objective\n"
       "gradient norms. They oppose localized/global/clear objectives on 14/15/12 of\n")
NEW = ("Low-light's share of the sum of role-objective gradient norms averages 62.23%\n"
       "across the 16 matched photographs. Those gradients oppose the\n"
       "localized/global/clear objectives on 14/15/12 of\n")


def main():
    previous_path = PUBLICATION / "publication_receipt_v1.json"
    destination = PUBLICATION / "publication_receipt_v1_r1.json"
    if destination.exists():
        raise ValueError("Retain existing R1; do not overwrite")
    previous = json.loads(previous_path.read_bytes())
    updated = deepcopy(previous)
    retained = []
    for row in previous["changes"]:
        current = ROOT / row["after"]["path"]
        if pin(current) != row["after"]:
            raise ValueError("R0 current publication pin differs")
        old_bytes = (ROOT / row["backup"]["path"]).read_bytes()
        data = current.read_bytes()
        if data[row["prefix_bytes"]:] != old_bytes:
            raise ValueError("R0 historical suffix differs")
    for row in updated["changes"]:
        current = ROOT / row["after"]["path"]
        if current.name not in DOCUMENTS:
            continue
        original = current.read_bytes()
        offset = row["prefix_bytes"]
        prefix, suffix = original[:offset], original[offset:]
        before, after = OLD.encode("utf-8"), NEW.encode("utf-8")
        if prefix.count(before) != 1:
            raise ValueError("Expected exactly one mean wording in new prefix")
        new_prefix = prefix.replace(before, after)
        backup = PUBLICATION / (current.name + ".publication_r0_retained")
        if backup.exists():
            raise ValueError("R0 retained copy already exists")
        backup.write_bytes(original)
        if backup.read_bytes() != original:
            raise ValueError("R0 retained bytes differ")
        current.write_bytes(new_prefix + suffix)
        if current.read_bytes()[len(new_prefix):] != suffix:
            raise ValueError("Prior document suffix changed")
        retained.append(pin(backup))
        row["prefix_bytes"] = len(new_prefix)
        row["after"] = pin(current)
    for value in updated["consumed_before_after_exact"]:
        if pin(ROOT / value["path"]) != value:
            raise ValueError("Original consumed evidence changed")
    updated["created_UTC"] = datetime.now(timezone.utc).isoformat()
    updated["original_publication_source"] = updated["source"]
    updated["source"] = pin(Path(__file__).resolve())
    updated["revision"] = "R1: clarify mean of per-photo norm fractions; no numbers, evidence or status change"
    updated["supersedes_retained_R0_receipt"] = pin(previous_path)
    updated["retained_R0_published_documents"] = retained
    updated["mean_share_definition"] = "mean across 16 photographs of lowlight_objective_norm / sum(four_role_objective_norms)"
    destination.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"complete": True, "clarified_documents": len(retained),
                      "receipt": str(destination), "all_historical_bytes_retained": True}))


if __name__ == "__main__":
    main()
