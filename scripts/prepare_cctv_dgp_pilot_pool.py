"""Audit a deterministic inherited-split candidate pool; no model or training."""
import collections
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_cctv_native_comparison import sha, read, write
from run_cctv_paired_regression import SOURCES, SPLIT, SPLIT_PIN, normalize, selection
from cctv_camera_stress import reference_canvas

OUT = ROOT/"outputs/cctv_dgp_pilot_pool_v1"
SEED = "cctv-dgp-pilot-pool-v1-2026-10-03"


def main():
    if OUT.exists():
        raise ValueError("Preserve completed/partial pool; create a new version")
    if sha(ROOT/SPLIT) != SPLIT_PIN:
        raise ValueError("Inherited split fingerprint differs")
    split = read(ROOT/SPLIT)
    inherited = {role: [normalize(p) for p in split[key]] for role, key in (("train", "train"), ("validation", "validation"))}
    # Enforce the full path partition and its exact source counts before sampling.
    reviewed = selection(split)
    picked = []
    for role, paths in inherited.items():
        for source in SOURCES:
            candidates = sorted((p for p in paths if p.startswith(source+"/")),
                                key=lambda p: hashlib.sha256((SEED+"\0"+role+"\0"+p).encode()).hexdigest())
            fixed = [p for s, p in reviewed if s == source] if role == "validation" else []
            chosen = fixed+[p for p in candidates if p not in fixed]
            quota = 512 if role == "train" else 64
            if len(chosen) < quota:
                raise ValueError("Insufficient inherited source/role candidates")
            picked.extend({"role": role, "source": source, "source_file": p} for p in chosen[:quota])
    OUT.mkdir()
    protocol = {"format": "dgp-cctv-pilot-candidate-pool-v1", "date": "2026-10-03",
                "selection_seed": SEED, "historical_split": SPLIT, "historical_split_sha256": SPLIT_PIN,
                "sample_before_new_training": True, "candidates": picked,
                "sampling": "512 training and64 inherited-validation references per source; validation explicitly includes the eight preceding paired development references. Source-balanced experiment, not ethnicity labels or an established optimal ratio.",
                "budget": {"candidate_references": 1152, "seconds": 180, "model_forwards": 0, "optimizer_updates": 0},
                "script_sha256": sha(Path(__file__)), "training_ready": False,
                "next_gate": "Input-only reference quality/pose and landmark coverage before freezing a VM recipe; a readable photo header is not a clean face-quality label."}
    write(OUT/"selection_protocol.json", protocol)
    rows, artifacts = [], {}
    start = time.monotonic()
    for index, selected in enumerate(picked):
        path = ROOT/selected["source_file"]
        with Image.open(path) as img:
            size, mode, fmt = list(img.size), img.mode, img.format
            img.load()  # Complete decode, not a header-only availability claim.
        rows.append({**selected, "sha256": sha(path), "native_size": size,
                     "decoded_mode": mode, "actual_format": fmt, "bytes": path.stat().st_size})
        if time.monotonic()-start > 180:
            raise TimeoutError("Finite pool audit cap exceeded; partial selection protocol retained")
        if (index+1)%256 == 0:
            print(f"Pilot candidate pool {index+1}/1152; elapsed={time.monotonic()-start:.1f}s", flush=True)
    memberships = collections.defaultdict(list)
    for row in rows:
        memberships[row["sha256"]].append(row)
    cross = [{"sha256": pin, "rows": items} for pin, items in memberships.items() if len({r["role"] for r in items}) > 1]
    duplicates = [{"sha256": pin, "rows": items} for pin, items in memberships.items() if len(items)>1]
    groups, preview_rows = {}, []
    for role in ("train", "validation"):
        for source in SOURCES:
            items = [r for r in rows if r["role"] == role and r["source"] == source]
            dimensions = np.asarray([r["native_size"] for r in items])
            groups[role+"/"+source] = {"references": len(items), "actual_formats": dict(collections.Counter(r["actual_format"] for r in items)),
                 "modes": dict(collections.Counter(r["decoded_mode"] for r in items)),
                 "native_width_min_median_max": [int(dimensions[:,0].min()), float(np.median(dimensions[:,0])), int(dimensions[:,0].max())],
                 "native_height_min_median_max": [int(dimensions[:,1].min()), float(np.median(dimensions[:,1])), int(dimensions[:,1].max())],
                 "both_dimensions_ge256": int((dimensions.min(1)>=256).sum()),
                 "either_dimension_below32": int((dimensions.min(1)<32).sum())}
            # Eight predetermined source/role examples; no output-based replacement.
            eight = items[:8]
            sheet = Image.new("RGB", (4*164, 2*196), (238, 238, 238))
            draw = ImageDraw.Draw(sheet)
            for j, row in enumerate(eight):
                target, _, _ = reference_canvas(ROOT/row["source_file"])
                x, y = (j%4)*164, (j//4)*196
                draw.text((x+2, y+2), Path(row["source_file"]).name[:24], fill="black")
                draw.text((x+2, y+15), str(row["native_size"]), fill="black")
                sheet.paste(Image.fromarray(target).resize((160,160), Image.Resampling.BILINEAR), (x,y+32))
                preview_rows.append(row)
            name = f"input_{role}_{Path(source).name}.png"
            sheet.save(OUT/name)
            artifacts[name] = sha(OUT/name)
    seconds = time.monotonic()-start
    if seconds > 180:
        raise TimeoutError("Finite pool audit cap exceeded")
    report = {"complete": True, "selection_protocol_sha256": sha(OUT/"selection_protocol.json"),
              "candidate_references": rows, "groups": groups, "cross_role_byte_duplicates": cross,
              "selected_byte_duplicate_groups": duplicates, "preview_rows": preview_rows,
              "artifacts_sha256": artifacts, "seconds": seconds, "model_forwards": 0, "optimizer_updates": 0,
              "training_ready": False, "reserved_native_cctv_used": False,
              "limits": "Audits1152 selected source decodes/bytes, not all80000 sources or same-person identity overlap. Historical validation is development evidence. No new high-resolution target detail, clean/uncovered labels or pose labels are invented. Quality/coverage gate and VM implementation remain pending."}
    write(OUT/"data_audit.json", report)
    print(json.dumps({"complete": True, "references": len(rows), "cross_role_byte_duplicate_groups": len(cross),
                      "seconds": seconds, "training_ready": False, "audit_sha256": sha(OUT/"data_audit.json")}))


if __name__ == "__main__":
    main()
