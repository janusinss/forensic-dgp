"""Versioned native-grid refinement; retain the first eight proposal previews."""
import json
from pathlib import Path

from scripts import prepare_mendeley_mask_proposals as base

REFINED = {
    11353: [[(21, 55), (25, 52), (42, 58), (54, 62), (63, 64), (68, 68),
             (69, 76), (67, 85), (64, 91), (60, 96), (47, 99), (34, 99),
             (23, 97), (17, 86), (18, 73)]],
    11347: [[(33, 22), (43, 22), (42, 37), (39, 51), (35, 67), (32, 83),
             (29, 100), (25, 98), (22, 86), (18, 66), (18, 50), (23, 35)]],
    11341: [base.PROPOSALS[-1][2][0],
            [(13, 73), (22, 70), (36, 69), (47, 71), (61, 66), (64, 65),
             (64, 78), (62, 89), (56, 99), (48, 106), (36, 111),
             (25, 108), (16, 99), (13, 89)]],
}


def main():
    first = base.ROOT / "outputs/mendeley_mask_proposals_v1/manifest.json"
    if base.sha(first) != "e623a0338d71e1c8f41ee31a2ebbc006d1dd5a40a5233fad52cf2f214eaae822":
        raise ValueError("First proposal evidence changed")
    base.OUT = base.ROOT / "outputs/mendeley_mask_proposals_v2"
    base.PROPOSALS = [(image_id, family, REFINED.get(image_id, polygons), rationale)
                      for image_id, family, polygons, rationale in base.PROPOSALS]
    base.main()
    lineage = {"format": "dgp-native-proposal-refinement-v1", "date": "2026-10-02",
               "previous_manifest_sha256": base.sha(first),
               "new_manifest_sha256": base.sha(base.OUT / "manifest.json"),
               "wrapper_sha256": base.sha(__file__), "base_builder_sha256": base.sha(base.__file__),
               "changed_source_ids": sorted(REFINED),
               "rationale": "Native overlay review found excessive right-hand/background and combined-mask cheek coverage; tighten those footprints and hair's lower visible-skin edge. No held-out or model-informed boundary edits.",
               "source_images_changed": False, "uncertainty_radius_changed": False,
               "training_admitted": False, "model_forwards": 0, "optimizer_updates": 0}
    (base.OUT / "refinement.json").write_text(json.dumps(lineage, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
