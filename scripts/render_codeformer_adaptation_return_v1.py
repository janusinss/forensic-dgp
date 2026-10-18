"""Render every audited saved case; no model, enhancement or training call."""
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from research.codeformer_adaptation_v1.common import require, sha256, write_json


def main():
    root = Path.cwd()
    review = root / "outputs/codeformer_adaptation_v1_return_review"
    source = review / "metrics_review/actual_return_v1/visual_manifest.json"
    manifest = json.loads(source.read_text(encoding="utf-8"))
    require(manifest["complete"], "Saved metrics review must finish before rendering")
    entries = manifest["cases"]
    by_update = {}
    for row in entries:
        by_update.setdefault(row["snapshot"], {})[row["id"]] = row
    require(set(by_update) == {0, 200} and set(by_update[0]) == set(by_update[200]) and
            len(by_update[0]) == 406, "Expected two complete406-case snapshots")
    out = review / "visual_sheets"
    require(not out.exists(), "Retain prior comparison renders")
    out.mkdir()
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
    title = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 12)
    groups = {}
    for row in by_update[200].values():
        key = ("native_" + row["source"] if row["evidence"] == "native_unpaired" else
               ("old_paired_" + row["source"] if row["id"].startswith("comparison_") else "HQ_validation"))
        groups.setdefault(key, []).append(row)
    pages = []
    metrics = json.loads((review / "metrics_review/actual_return_v1/metrics.json").read_text())["rows"]
    lookup = {(r["snapshot"], r["id"], r["view"]): r for r in metrics}
    columns = ("Original / resize", "Paired target / native note", "Untouched CodeFormer", "Our update200: STOPPED")
    tile, row_height, rows_per_page = 192, 234, 8
    all_shown = []
    for group, rows in sorted(groups.items()):
        rows.sort(key=lambda row: (row["profile"], row["id"]))
        for offset in range(0, len(rows), rows_per_page):
            selected = rows[offset:offset + rows_per_page]
            page = Image.new("RGB", (4 * tile, 86 + row_height * len(selected)), "#f4f1ea")
            draw = ImageDraw.Draw(page)
            draw.text((8, 5), group + " | native PNG | preservation-stop evidence", fill="#171717", font=title)
            draw.text((8, 30), "All cases retained. No sharpening, palette correction or extra model pass. Native CCTV has no clean target.",
                      fill="#171717", font=small)
            for col, label in enumerate(columns):
                draw.text((col * tile + 4, 55), label, fill="#171717", font=font)
            for line, row in enumerate(selected):
                y = 82 + line * row_height
                old = by_update[0][row["id"]]
                label = row["id"].removeprefix("validation_").removeprefix("comparison_")
                draw.text((4, y), label, fill="#171717", font=small)
                paths = [row["input"], row["target"], old["native_png"], row["native_png"]]
                for col, path in enumerate(paths):
                    if path is None:
                        draw.rectangle((col * tile, y + 18, (col + 1) * tile - 1, y + 18 + tile), fill="#e1ded6")
                        draw.text((col * tile + 12, y + 75), "Unpaired native CCTV", fill="#171717", font=font)
                        draw.text((col * tile + 12, y + 97), "No clean reference", fill="#171717", font=font)
                        continue
                    path = Path(path)
                    expected = (row.get("input_sha256") if col == 0 else
                                row.get("target_sha256") if col == 1 else
                                old["native_png_sha256"] if col == 2 else row["native_png_sha256"])
                    if expected:
                        require(sha256(path) == expected, "Audited render source changed")
                    with Image.open(path) as image:
                        image = image.convert("RGB")
                        require(image.size == (256, 256), "Unexpected reviewed geometry")
                        page.paste(image.resize((tile, tile), Image.Resampling.LANCZOS), (col * tile, y + 18))
                if row["evidence"] == "paired_synthetic":
                    a = lookup[(0, row["id"], "native_png")]["mae"]
                    b = lookup[(200, row["id"], "native_png")]["mae"]
                    draw.text((3 * tile + 4, y + 212), "MAE delta%+.2f /255" % ((b - a) * 255),
                              fill="#8a2020" if b > a else "#205c38", font=small)
                all_shown.append(row["id"])
            index = len(pages) + 1
            name = "comparison_%02d.png" % index
            page.save(out / name)
            pages.append({"file": name, "sha256": sha256(out / name), "group": group,
                          "case_ids": [row["id"] for row in selected],
                          "output_view": "native_png", "tile_size": tile})
    require(len(all_shown) == len(set(all_shown)) == 406 and set(all_shown) == set(by_update[0]),
            "Comparison sheets omit or duplicate a case")
    write_json(out / "manifest.json", {"complete": True, "pages": pages, "cases": 406,
        "snapshot_outputs_covered": 812, "source_manifest_sha256": sha256(source),
        "resampling_only_for_review_thumbnails": "Pillow LANCZOS256-to192; native PNG originals remain256",
        "neural_calls": 0, "gradient_queries": 0, "optimizer_updates": 0,
        "visual_review_pending": True, "model_qualified": False})
    index = ["# CodeFormer V1: all returned comparison images", "",
             "Stopped at200updates. These are development outputs, not an accepted replacement.", "",
             "Each page: input, clean target (unavailable for native), untouched CodeFormer, update200.", ""]
    for page in pages:
        index.append("- [%s: %s](%s)" % (page["file"], page["group"], page["file"]))
    (out / "INDEX.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    print(json.dumps({"complete": True, "pages": len(pages), "cases": len(all_shown),
                      "outputs": 812, "folder": str(out)}))


if __name__ == "__main__":
    main()
