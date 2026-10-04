"""Bounded public-data acquisition and counterpart audit; no model or training imports."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import ssl
import sys
import time
from html.parser import HTMLParser
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
import uuid

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
META_SPEC = {
    "file_url": "https://drive.google.com/uc?id=16N0RV4fHI6joBuKbQAoG34V_cQk7vxSA&export=download",
    "file_size": 267793842,
    "file_md5": "425ae20f06a4da1d4dc0f46d40ba5fd6",
}
SOURCE = "https://github.com/NVlabs/ffhq-dataset"
SPEC_SOURCE = "https://raw.githubusercontent.com/NVlabs/ffhq-dataset/master/download_ffhq.py"
HOSTS = {"drive.google.com", "drive.usercontent.google.com"}
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_TRANSFER_BYTES = 320 * 1024 * 1024
CAP_SECONDS = 600


def digest(path, algorithm="sha256"):
    result = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def allowed_url(url):
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in HOSTS or parsed.username or parsed.password or parsed.port not in (None, 443):
        raise ValueError("Not an approved public FFHQ download host")
    return url


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action = None
        self.fields = {}
        self.active = False

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        if tag == "form" and attributes.get("id") == "download-form":
            if attributes.get("method", "get").lower() != "get":
                raise ValueError("Unexpected download confirmation method")
            self.action = allowed_url(attributes["action"])
            self.active = True
        if tag == "input" and self.active and attributes.get("name"):
            key = attributes["name"]
            if key not in {"id", "export", "confirm", "uuid"} or key in self.fields:
                raise ValueError("Unexpected public download confirmation field")
            self.fields[key] = attributes.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form":
            self.active = False


def open_download(url, context):
    """Follow at most one ordinary public Drive download-confirmation form."""
    response = urlopen(Request(allowed_url(url), headers={"User-Agent": "DGP-restoration-data-audit/1.0"}), context=context, timeout=30)
    allowed_url(response.url)
    if "text/html" not in response.headers.get("Content-Type", "").lower():
        return response, False
    with response:
        body = response.read(16385)
    if len(body) > 16384:
        raise ValueError("Unexpected HTML; do not bypass access or quota restrictions")
    form = DownloadForm()
    form.feed(body.decode("utf-8"))
    if not form.action or not {"id", "export", "confirm"}.issubset(form.fields):
        raise ValueError("Public download unavailable; stop rather than bypass access restrictions")
    response = urlopen(Request(form.action + "?" + urlencode(form.fields)), context=context, timeout=30)
    allowed_url(response.url)
    if "text/html" in response.headers.get("Content-Type", "").lower():
        response.close()
        raise ValueError("Confirmation did not return the public data file")
    return response, True


def verify_blob(path, spec):
    path = Path(path)
    if path.stat().st_size != spec["file_size"] or digest(path, "md5") != spec["file_md5"]:
        raise ValueError("File size/MD5 differs from official metadata: " + path.name)
    if "pixel_size" in spec:
        with Image.open(path) as image:
            if list(image.size) != spec["pixel_size"] or digest_pixels(image) != spec["pixel_md5"]:
                raise ValueError("Decoded image differs from official metadata")


def digest_pixels(image):
    return hashlib.md5(np.asarray(image).tobytes()).hexdigest()


def fetch(path, spec, context, deadline, budget):
    path = Path(path)
    if path.exists():
        verify_blob(path, spec)
        return {"reused_verified_file": True, "bytes": path.stat().st_size}
    if time.monotonic() >= deadline:
        raise TimeoutError("Bounded acquisition time expired")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part." + uuid.uuid4().hex)
    response, confirmation = open_download(spec["file_url"], context)
    size = 0
    with response, temporary.open("xb") as stream:
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("Bounded acquisition time expired; partial file preserved")
            block = response.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            budget[0] += len(block)
            if size > spec["file_size"] or budget[0] > MAX_TRANSFER_BYTES:
                raise ValueError("Declared download byte budget exceeded")
            stream.write(block)
            if size // (16 * 1024 * 1024) != (size - len(block)) // (16 * 1024 * 1024):
                print(f"{path.name}: {size}/{spec['file_size']} bytes", flush=True)
    verify_blob(temporary, spec)
    # No overwrite, even if another process creates the destination meanwhile.
    os.link(temporary, path)
    temporary.unlink()
    return {"reused_verified_file": False, "bytes": size, "ordinary_public_confirmation": confirmation}


def choose_references(protocol):
    chosen = []
    for role, count in (("train", 10), ("validation", 6)):
        pool = sorted((r for r in protocol["references"] if r["source"] == "dataset/thumbnails128x128" and r["role"] == role), key=lambda r: int(Path(r["source_file"]).stem))
        if len(pool) < count:
            raise ValueError("Insufficient fixed references for declared audit sample")
        chosen.extend(pool[round(i * (len(pool) - 1) / (count - 1))] for i in range(count))
    if len({r["id"] for r in chosen}) != 16:
        raise ValueError("Duplicate counterpart audit reference")
    return chosen


def bind_thumbnail(reference, item, root):
    identifier = Path(reference["source_file"]).stem
    if any(Path(item[k]["file_path"]).stem != identifier for k in ("thumbnail", "image")):
        raise ValueError("Official image IDs do not match the frozen reference")
    if item["thumbnail"]["pixel_size"] != [128, 128] or item["image"]["pixel_size"] != [1024, 1024]:
        raise ValueError("Unexpected official counterpart dimensions")
    with Image.open(Path(root) / reference["native"]) as image:
        actual = digest_pixels(image)
    if actual != item["thumbnail"]["pixel_md5"]:
        raise ValueError("Current thumbnail pixels do not match the official image ID")
    allowed_url(item["image"]["file_url"])
    if item["image"]["file_size"] > MAX_IMAGE_BYTES:
        raise ValueError("Counterpart exceeds the fixed per-image byte budget")
    return actual


def run(root, output, cache, prepare_only, ca_file):
    root, output, cache = Path(root), Path(output), Path(cache)
    protocol = json.loads((root / "protocol.json").read_text())
    if digest(root / "protocol.json") != (root / "protocol.sha256").read_text().strip():
        raise ValueError("Original reference protocol changed")
    refs = choose_references(protocol)
    for ref in refs:
        if digest(root / ref["native"]) != protocol["assets_sha256"][ref["native"]]:
            raise ValueError("Pinned thumbnail changed")
    manifest = {"version": 1, "date": "2026-10-04", "purpose": "Target-quality acquisition diagnostic, not a training or evaluation protocol", "reference_protocol_sha256": digest(root / "protocol.json"), "selection": "Sorted image-ID quantiles within each original role, before counterpart inspection", "training_role_references": 10, "validation_role_references": 6, "metadata_spec": META_SPEC, "spec_source": SPEC_SOURCE, "dataset_source": SOURCE, "max_transfer_bytes": MAX_TRANSFER_BYTES, "runtime_cap_seconds": CAP_SECONDS, "native_reserved_used": False, "optimizer_updates": 0, "model_forwards": 0, "references": refs}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Existing acquisition manifest differs; use a new version")
    else:
        write(manifest_path, manifest)
    if prepare_only:
        print(json.dumps({"prepared": True, "references": 16, "manifest_sha256": digest(manifest_path), "optimizer_updates": 0}))
        return
    if (output / "results.json").exists():
        raise FileExistsError("Preserve completed acquisition/audit results")
    started = time.monotonic()
    deadline, budget = started + CAP_SECONDS, [0]
    context = ssl.create_default_context(cafile=str(ca_file)) if ca_file else ssl.create_default_context()
    metadata_path = cache / "ffhq-dataset-v2.json"
    acquisition = fetch(metadata_path, META_SPEC, context, deadline, budget)
    print("Official metadata verified; parsing 70,000 image records", flush=True)
    with metadata_path.open(encoding="utf-8") as stream:
        all_metadata = json.load(stream)
    selected = {r["id"]: all_metadata[str(int(Path(r["source_file"]).stem))] for r in refs}
    del all_metadata
    for ref in refs:
        bind_thumbnail(ref, selected[ref["id"]], root)
    rows = []
    for ref in refs:
        if time.monotonic() >= deadline:
            raise TimeoutError("Bounded acquisition time expired")
        item = selected[ref["id"]]
        path = output / "images1024" / (Path(ref["source_file"]).stem + ".png")
        transfer = fetch(path, item["image"], context, deadline, budget)
        with Image.open(path) as image:
            target = image.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
        target_path = output / "targets256" / (ref["id"] + ".png")
        if target_path.exists():
            raise FileExistsError("Preserve existing target derivation")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target.save(target_path)
        rows.append({"reference_id": ref["id"], "original_role": ref["role"], "official_category": item["category"], "native_thumbnail_pixel_md5": item["thumbnail"]["pixel_md5"], "native_thumbnail_pixel_match": True, "image": str(path.relative_to(output)).replace("\\", "/"), "image_sha256": digest(path), "image_spec": item["image"], "thumbnail_spec": item["thumbnail"], "original_photo_spec": item["in_the_wild"], "metadata": item["metadata"], "target": str(target_path.relative_to(output)).replace("\\", "/"), "target_sha256": digest(target_path), "target_derivation": "PIL RGB, LANCZOS downsample 1024 to 256; no learned sharpening", "transfer": transfer})
        print(f"Counterpart verified {len(rows)}/16: {ref['id']}", flush=True)
    # Four pages of four rows: old target, true-source 256 target, and difference.
    previews = []
    for start in range(0, 16, 4):
        sheet = Image.new("RGB", (768, 4 * 286), "#eeeeee")
        draw = ImageDraw.Draw(sheet)
        for i, (ref, row) in enumerate(zip(refs[start:start+4], rows[start:start+4])):
            with Image.open(root / ref["target"]) as image: old = image.convert("RGB")
            with Image.open(output / row["target"]) as image: new = image.convert("RGB")
            difference = Image.fromarray(np.abs(np.asarray(old).astype(np.int16) - np.asarray(new).astype(np.int16)).astype(np.uint8))
            for col, (image, label) in enumerate(((old, "old upscaled target"), (new, "1024-source target256"), (difference, "absolute RGB difference"))):
                draw.text((col * 256 + 3, i * 286 + 2), ref["id"] + " / " + label, fill="black")
                sheet.paste(image, (col * 256, i * 286 + 27))
        name = f"preview_{start//4+1}.png"
        sheet.save(output / name)
        previews.append({"path": name, "sha256": digest(output / name), "rows": 4})
    result = {"complete": True, "manifest_sha256": digest(manifest_path), "metadata_sha256": digest(metadata_path), "metadata_md5": META_SPEC["file_md5"], "metadata_acquisition": acquisition, "elapsed_seconds": time.monotonic() - started, "new_download_bytes": budget[0], "counterparts": rows, "previews": previews, "native_reserved_used": False, "model_forwards": 0, "backward_calls": 0, "optimizer_updates": 0, "visual_target_review_pending": True, "training_ready": False, "limitations": ["1024 aligned dimensions do not prove the original captured face had 1024 pixels of detail", "Fixed sample is a source audit, not proof of a useful trained restorer", "No Asian-source counterpart was acquired; original source stays separately reported", "Original roles retained even where FFHQ canonical categories differ"]}
    write(output / "results.json", result)
    print(json.dumps({"complete": True, "references": 16, "new_download_bytes": budget[0], "elapsed_seconds": result["elapsed_seconds"], "training_ready": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/cctv_dgp_vm_bundle_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/cctv_dgp_hq_counterparts_v1")
    parser.add_argument("--cache", type=Path, default=ROOT / "outputs/cctv_dgp_hq_metadata_v1")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--ca-file", type=Path)
    args = parser.parse_args()
    try:
        run(args.root, args.output, args.cache, args.prepare_only, args.ca_file)
    except Exception as error:
        write(args.output / ("failure_" + uuid.uuid4().hex + ".json"), {"complete": False, "error_type": type(error).__name__, "error": str(error), "model_forwards": 0, "optimizer_updates": 0, "training_ready": False})
        raise
