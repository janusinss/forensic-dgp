"""Preserve V1 and correct four source-visible covering boundaries in V2."""
import json
from pathlib import Path

from scripts import prepare_cofw_covering_proposals as builder

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "outputs/cofw_covering_proposals_v1"
OUT = ROOT / "outputs/cofw_covering_proposals_v2"
PARENT_SHA = "232e47d3009612f1b1cd0710bef3df2d5ec159518bb90263da872ca9ec666115"
CHANGES = {
    975: [[(43,147),(57,142),(60,120),(65,107),(70,104),(74,108),(76,132),(81,150),(89,176),(83,199),(81,218),(45,222),(47,205),(40,195),(37,181),(38,167)]],
    1338: [[(88,27),(116,27),(138,44),(139,59),(130,73),(124,92),(116,111),(113,123),(106,136),(106,127),(109,115),(112,99),(112,87),(116,76),(106,87),(96,83),(92,74),(94,67),(87,74),(82,68),(86,57),(77,63),(75,59),(84,47)]],
    1143: [[(3,39),(13,41),(19,32),(29,30),(50,35),(66,38),(73,44),(80,43),(92,46),(112,48),(122,54),(122,67),(119,80),(113,86),(105,89),(95,88),(89,85),(85,77),(79,65),(73,58),(69,66),(60,76),(46,78),(31,74),(19,64),(14,53)]],
    930: [[(26,88),(50,74),(70,58),(78,57),(90,69),(112,78),(123,96),(117,124),(106,147),(101,161),(52,161),(33,143),(23,125),(17,106)],
          [(10,98),(28,105),(29,110),(11,104),(7,99)],[(15,124),(29,135),(36,138),(34,146),(16,133)],
          [(119,83),(122,90),(114,106),(106,110),(104,104)]],
}
REASONS = {
    975: "V1 exceeded the observed hand into visible left cheek; tighten palm/outer finger boundary without labelling ordinary hair.",
    1338: "Extend the inferred facial-overlap edge along observed hair strands over the lower cheek; preserve outer hairstyle.",
    1143: "V1 right-lens lower edge overran the rim into visible cheek; tighten that observed rim.",
    930: "The observed respirator reaches the bottom crop; represent its clipped edge explicitly and exclude that source row from supervision.",
}


def main():
    if OUT.exists() or builder.sha(PARENT / "manifest.json") != PARENT_SHA:
        raise ValueError("Preserve proposal versions and require V1 parent")
    builder.OUT = OUT
    builder.POLYGONS = {**builder.POLYGONS, **CHANGES}
    builder.main()
    lineage = {"format":"dgp-cofw-source-label-refinement-v2","date":"2026-10-03",
               "parent_manifest_sha256":PARENT_SHA,"manifest_sha256":builder.sha(OUT / "manifest.json"),
               "builder_sha256":builder.sha(builder.__file__),"refiner_sha256":builder.sha(__file__),
               "changed_train_indices":list(CHANGES),"source_review_rationale":REASONS,
               "native_images_changed":False,"publisher_splits_changed":False,
               "new_model_forwards":0,"optimizer_updates":0,"training_admitted":False,
               "earlier_label_draft_preserved":True}
    with (OUT / "refinement.json").open("x",encoding="utf-8",newline="\n") as stream:
        stream.write(json.dumps(lineage,indent=2)+"\n")
    print(json.dumps({"sources":42,"native_boundary_refinements":4,"training_admitted":False,
                      "manifest_sha256":lineage["manifest_sha256"]}))


if __name__ == "__main__":
    main()
