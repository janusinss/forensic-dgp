"""Saved-PNG-only, exact-pixel visual coverage of the finite readout return.

No Torch, checkpoint, neural, gradient, training, or VM operations. Archive
members are never extracted to disk. Only declared record JSON and PNG bytes
are decoded. Generated cells are primary exactly once; repeated references are
explicit context. This renderer does not provide a human visual verdict.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import time

from PIL import Image, ImageDraw, ImageFont


STAGE = "cctv_codeformer_u96_decoder_readout_v1_vm"
OUT_PREFIX = "outputs/u96_decoder_readout_v1/"
ARCHIVE_SHA256 = "73c8f4bcca5814cb845058cbc7d553ad3ea1f29e97967449cad66ac848fed2c4"
SNAPSHOTS = (0, 1, 4, 16, 32, 64)
EXPECTED_KINDS = {"reference": 120, "snapshot": 336, "training": 256}
CELL = 256
GAP = 12
LEFT = 16
TOP = 124
ROW = 314
BG = (246, 244, 239)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=pairs,
                      parse_constant=lambda value: need(False, "Nonfinite JSON"))


def safe_relative(name):
    need(isinstance(name, str) and name and "\\" not in name,
         "Expected POSIX relative path")
    value = PurePosixPath(name)
    need(not value.is_absolute() and all(part not in ("..", ".", "")
                                       for part in name.split("/"))
         and ":" not in name, "Unsafe archive/source path")
    return value


def rgb256(data):
    with Image.open(io.BytesIO(data)) as picture:
        need(picture.format == "PNG" and picture.mode == "RGB"
             and picture.size == (CELL, CELL), "Expected exact RGB256 PNG")
        picture.load()
        return picture.copy()


def bind_bytes(data, pin):
    need(len(data) == pin["bytes"] and digest(data) == pin["sha256"],
         "PNG/JSON file pin differs")


def read_asset(root, pin):
    safe_relative(pin["path"])
    base = Path(root).resolve(strict=True)
    path = base.joinpath(*PurePosixPath(pin["path"]).parts)
    need(base in path.resolve(strict=True).parents and path.is_file(),
         "Asset escaped sealed stage")
    for item in (path, *path.parents):
        if item == base:
            break
        need(not item.is_symlink(), "Symlink asset ancestry")
    need(path.stat().st_size <= 1024 * 1024, "Asset too large")
    data = path.read_bytes()
    bind_bytes(data, pin)
    return data


def load_return(archive, protocol_bytes, *, max_seconds=180):
    """Read only 712 record JSONs, generated PNGs, and embedded protocol."""
    start = time.monotonic()
    need(file_sha(archive) == ARCHIVE_SHA256, "Returned archive SHA differs")
    records, pngs, seen = {}, {}, set()
    embedded = None
    total = 0
    with tarfile.open(archive, "r|gz") as source:
        for item in source:
            need(time.monotonic() - start < max_seconds, "Renderer read cap")
            safe_relative(item.name)
            need(item.name.startswith(STAGE + "/") and item.name not in seen,
                 "Unexpected or duplicate archive namespace")
            seen.add(item.name)
            need(item.isfile() and not item.issym() and not item.islnk(),
                 "Archive has non-regular member")
            total += item.size
            need(total <= 4 * 1024**3, "Archive payload cap exceeded")
            rel = item.name[len(STAGE) + 1:]
            is_record = (rel.startswith(OUT_PREFIX) and
                         rel.endswith("/record.json") and
                         rel[len(OUT_PREFIX):].split("/", 1)[0]
                         in ("reference", "snapshots", "training"))
            is_png = rel.startswith(OUT_PREFIX) and rel.endswith("/output.png")
            if rel == "protocol.json" or is_record or is_png:
                need(item.size <= (2 * 1024 * 1024 if rel == "protocol.json"
                                   else 1024 * 1024), "Selected member too large")
                handle = source.extractfile(item)
                need(handle is not None, "Missing archive member bytes")
                data = handle.read(item.size + 1)
                need(len(data) == item.size, "Short archive read")
                if rel == "protocol.json":
                    embedded = data
                elif is_record:
                    records[rel] = (strict_json(data), digest(data), len(data))
                else:
                    pngs[rel] = data
    need(embedded == protocol_bytes, "Embedded and sealed protocols differ")
    need(len(records) == 712 and len(pngs) == 712,
         "Expected all 712 generated records and PNGs")
    need(Counter(row[0]["kind"] for row in records.values()) == EXPECTED_KINDS,
         "Generated kind census differs")
    output_paths = set()
    for record_path, (row, _sha, _size) in records.items():
        need(row["complete"] is True and row["split"] == "TRAIN",
             "Incomplete or unexpected split")
        pin = row["outputs"]["png"]
        safe_relative(pin["path"])
        need(pin["path"] not in output_paths and pin["path"] in pngs,
             "Missing/duplicate generated PNG binding")
        output_paths.add(pin["path"])
        bind_bytes(pngs[pin["path"]], pin)
        need(record_path == str(PurePosixPath(pin["path"]).parent / "record.json"),
             "Record/PNG namespace differs")
    need(output_paths == set(pngs), "Orphan generated PNG")
    return records, pngs, len(seen), total


def build_plan(records, protocol):
    """All generated files get exactly one primary placement."""
    declared = {row["id"]: row for row in protocol["cases"] + protocol["guards"]}
    need(len(declared) == 120, "Expected120 distinct declared cases")
    refs, snapshots, training = {}, defaultdict(dict), defaultdict(list)
    for path, (record, _sha, _size) in records.items():
        item = declared.get(record["id"])
        need(item is not None and record["input"] == item["input"]
             and record.get("target") == item.get("target"),
             "Record input/target differs from sealed case")
        need(record["role"] == item["role"], "Role differs")
        kind = record["kind"]
        if kind == "reference":
            need(record["id"] not in refs and record["update"] == 0,
                 "Duplicate/noninitial reference")
            refs[record["id"]] = path
        elif kind == "snapshot":
            need(record["update"] in SNAPSHOTS, "Unexpected snapshot update")
            need(record["update"] not in snapshots[record["id"]],
                 "Duplicate snapshot case/update")
            snapshots[record["id"]][record["update"]] = path
        elif kind == "training":
            need(item.get("optimization_permitted") is True
                 and 1 <= record["update"] <= 64 and record["slot"] in (1,2,3,4),
                 "Unexpected training case/update/slot")
            training[record["id"]].append(path)
        else:
            need(False, "Unexpected generated kind")
    need(len(refs) == 120 and len(snapshots) == 56 and len(training) == 64,
         "Case layout count differs")
    need(set(snapshots).isdisjoint(training) and set(snapshots) | set(training) == set(refs),
         "Snapshot/optimization partition differs")
    sort_key = lambda identifier: (declared[identifier].get("cohort", "legacy"),
                                   declared[identifier]["role"], identifier)
    pages = []
    ids = sorted(snapshots, key=sort_key)
    for group, updates in enumerate((SNAPSHOTS[:3], SNAPSHOTS[3:])):
        for start in range(0, len(ids), 8):
            rows = []
            for identifier in ids[start:start+8]:
                need(set(snapshots[identifier]) == set(SNAPSHOTS),
                     "Incomplete snapshot schedule")
                cells = [{"record": refs[identifier], "primary": group == 0,
                          "label": "Frozen U96 reference"}]
                cells.extend({"record": snapshots[identifier][update], "primary": True,
                              "label": f"Snapshot u{update:04d}"} for update in updates)
                rows.append({"id": identifier, "generated": cells})
            pages.append({"name": f"snapshots_group{group+1}_{start//8+1:02d}",
                          "kind": "snapshot", "updates": list(updates), "rows": rows})
    ids = sorted(training, key=sort_key)
    for start in range(0, len(ids), 8):
        rows = []
        for identifier in ids[start:start+8]:
            appearances = sorted(training[identifier], key=lambda path: records[path][0]["update"])
            need(len(appearances) == 4 and len({records[path][0]["update"] for path in appearances}) == 4,
                 "Expected four distinct training appearances")
            cells = [{"record": refs[identifier], "primary": True,
                      "label": "Frozen U96 reference"}]
            cells.extend({"record": path, "primary": True,
                          "label": f"Before step {records[path][0]['update']:02d} / slot{records[path][0]['slot']}"}
                         for path in appearances)
            rows.append({"id": identifier, "generated": cells})
        pages.append({"name": f"optimization_exposures_{start//8+1:02d}",
                      "kind": "training", "rows": rows})
    validate_coverage(pages, set(records))
    return pages, declared


def validate_coverage(pages, expected_records):
    primary = [cell["record"] for page in pages for row in page["rows"]
               for cell in row["generated"] if cell["primary"]]
    need(len(primary) == len(expected_records) and set(primary) == expected_records
         and len(primary) == len(set(primary)), "Generated primary coverage differs")
    need(all(1 <= len(page["rows"]) <= 8 for page in pages), "Page row cap differs")


def put_cell(sheet, picture, x, y):
    need(picture.mode == "RGB" and picture.size == (CELL, CELL), "Cell pixel contract differs")
    sheet.paste(picture, (x, y))
    need(sheet.crop((x, y, x+CELL, y+CELL)).tobytes() == picture.tobytes(),
         "Pasted cell pixels differ")


def render(archive, stage, destination):
    start = time.monotonic()
    stage, destination = Path(stage), Path(destination)
    protocol_bytes = (stage / "protocol.json").read_bytes()
    protocol = strict_json(protocol_bytes)
    need(protocol["stage_name"] == STAGE, "Sealed stage differs")
    need(not destination.exists(), "Preserve previous visual namespace")
    records, pngs, member_count, payload_bytes = load_return(archive, protocol_bytes)
    pages, declared = build_plan(records, protocol)
    destination.mkdir(parents=True)
    font = ImageFont.load_default(size=14)
    small = ImageFont.load_default(size=12)
    manifest = {"format": "saved-PNG-exact256-coverage-v1", "complete": False,
                "archive": {"path": str(Path(archive).resolve()), "bytes": Path(archive).stat().st_size,
                            "sha256": ARCHIVE_SHA256},
                "protocol": {"path": str((stage/'protocol.json').resolve()),
                             "bytes": len(protocol_bytes), "sha256": digest(protocol_bytes)},
                "renderer": {"path": str(Path(__file__).resolve()), "sha256": file_sha(__file__)},
                "coverage_expected": EXPECTED_KINDS, "cell_pixels": [256,256],
                "PNG_only_not_RAW_visualization": True,
                "enhancement_resampling_color_conversion": "none",
                "human_visual_review": "PENDING_ROOT_ALL_SHEETS",
                "model_qualified": False, "neural_calls": 0, "torch_imports": 0,
                "archive_members_examined": member_count, "archive_payload_bytes": payload_bytes,
                "sheets": []}
    for page_number, page in enumerate(pages, 1):
        need(time.monotonic() - start < 240, "Overall render cap exceeded")
        columns = 2 + len(page["rows"][0]["generated"])
        width = LEFT * 2 + columns * CELL + (columns-1)*GAP
        height = TOP + len(page["rows"])*ROW + 22
        sheet = Image.new("RGB", (width,height), BG)
        drawing = ImageDraw.Draw(sheet)
        drawing.text((LEFT,12), f"U96 decoder readout return | {page_number:02d}/{len(pages):02d} | {page['name']}", fill=(0,0,0), font=font)
        drawing.text((LEFT,34), "Exact saved 256x256 PNG cells; no enhancement/resampling. TRAIN only, reused monitor cases; no independent FINAL/native claim.", fill=(0,0,0), font=small)
        drawing.text((LEFT,54), "Training cells are BEFORE the named optimizer step. Frozen U96 is comparison context, never a clean photographic target.", fill=(0,0,0), font=small)
        drawing.text((LEFT,74), "Legacy dark rows: second column is retained R2 behavior, NOT ground truth. Other target cells are genuine aligned clean photographs.", fill=(115,34,20), font=small)
        sheet_binding = {"page": page_number, "path": page['name']+'.png', "kind": page['kind'],
                         "dimensions": [width,height], "rows": []}
        for row_index, row in enumerate(page["rows"]):
            item = declared[row["id"]]
            y = TOP + row_index*ROW
            drawing.text((LEFT,y), f"{row['id']} | {item.get('cohort','legacy preservation guard')}", fill=(0,0,0), font=small)
            row_binding = {"id": row["id"], "role": item["role"], "cohort": item.get("cohort","legacy"), "cells": []}
            target_pin = item.get("target")
            target_note = "Genuine clean target"
            if target_pin is None:
                need(item["role"] == "dark", "Missing target outside legacy dark guard")
                target_pin = item["outputs"]["png"]
                target_note = "R2 behavior / NOT clean GT"
            cells = [{"pin": item["input"], "label": "Observed input", "origin": "sealed_asset", "primary": False},
                     {"pin": target_pin, "label": target_note, "origin": "sealed_asset", "primary": False}]
            for generated in row["generated"]:
                path = generated["record"]
                record, record_sha, record_size = records[path]
                cells.append({"pin": record["outputs"]["png"], "label": generated["label"],
                              "origin": "returned_generated_PNG", "primary": generated["primary"],
                              "record": {"path": path, "bytes": record_size, "sha256": record_sha},
                              "RAW_not_displayed": record["outputs"]["raw"],
                              "update": record["update"], "slot": record.get("slot"),
                              "generated_kind": record["kind"]})
            for column, cell in enumerate(cells):
                x = LEFT + column*(CELL+GAP)
                label = cell["label"] + (" [repeat]" if cell["origin"] == "returned_generated_PNG" and not cell["primary"] else "")
                drawing.text((x,y+17), label, fill=(0,0,0), font=small)
                data = (pngs[cell["pin"]["path"]] if cell["origin"] == "returned_generated_PNG"
                        else read_asset(stage, cell["pin"]))
                picture = rgb256(data)
                put_cell(sheet, picture, x, y+36)
                binding = {**cell, "rectangle": [x,y+36,x+CELL,y+36+CELL],
                           "RGB8_sha256": digest(picture.tobytes())}
                row_binding["cells"].append(binding)
            sheet_binding["rows"].append(row_binding)
        path = destination / sheet_binding["path"]
        sheet.save(path, format="PNG", optimize=False)
        with Image.open(path) as reopened:
            need(reopened.mode == "RGB" and reopened.size == sheet.size, "Saved sheet mode/size differs")
            for row in sheet_binding["rows"]:
                for cell in row["cells"]:
                    need(digest(reopened.crop(tuple(cell["rectangle"])).tobytes()) == cell["RGB8_sha256"],
                         "Saved sheet cell pixels differ")
        sheet_binding.update(bytes=path.stat().st_size, sha256=file_sha(path))
        manifest["sheets"].append(sheet_binding)
        print(json.dumps({"rendered": page_number, "of": len(pages), "sheet": path.name}), flush=True)
    all_cells = [cell for sheet in manifest["sheets"] for row in sheet["rows"] for cell in row["cells"]]
    generated = [cell for cell in all_cells if cell["origin"] == "returned_generated_PNG"]
    primary = [cell for cell in generated if cell["primary"]]
    need(len(primary) == 712 and len({cell['record']['path'] for cell in primary}) == 712, "Final primary coverage differs")
    need(Counter(cell["generated_kind"] for cell in primary) == EXPECTED_KINDS, "Final primary kind census differs")
    manifest.update(complete=True, planned_sheets=len(pages), generated_unique_primary_cells=len(primary),
                    repeated_reference_context_cells=len(generated)-len(primary), total_visible_cells=len(all_cells),
                    seconds=time.monotonic()-start)
    output = destination / "visual_manifest_v1.json"
    output.write_text(json.dumps(manifest,indent=2)+"\n", encoding="utf-8", newline="\n")
    (destination / "visual_manifest_v1.json.sha256").write_text(f"{file_sha(output)}  {output.name}\n", encoding="ascii", newline="\n")
    print(json.dumps({"complete": True, "sheets": len(pages), "unique_generated": len(primary),
                      "repeated_reference_context": len(generated)-len(primary), "manifest_sha256": file_sha(output)}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("outputs/cctv-codeformer-u96-decoder-readout-v1-results.tar.gz"))
    parser.add_argument("--stage", type=Path, default=Path("outputs") / STAGE)
    parser.add_argument("--output", type=Path, default=Path("outputs/codeformer_u96_decoder_readout_v1_return_visuals"))
    args = parser.parse_args()
    render(args.archive, args.stage, args.output)


if __name__ == "__main__":
    main()
