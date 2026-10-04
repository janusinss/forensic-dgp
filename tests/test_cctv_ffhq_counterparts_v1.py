"""Integrity checks for public counterpart binding; never runs network or models."""
import hashlib
import json
from pathlib import Path
import shutil
import unittest
import uuid

from PIL import Image
from scripts.acquire_cctv_ffhq_counterparts_v1 import (
    DownloadForm, allowed_url, bind_thumbnail, choose_references,
    digest_pixels, verify_blob,
)


class CounterpartTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path.cwd() / "scratch" / ("hq_counterpart_test_" + uuid.uuid4().hex)
        self.directory.mkdir(parents=True)

    def tearDown(self):
        assert self.directory.resolve().is_relative_to((Path.cwd() / "scratch").resolve())
        shutil.rmtree(self.directory)

    def fixture(self):
        path = self.directory / "00001.png"
        image = Image.new("RGB", (128, 128), (127, 60, 30))
        image.save(path)
        ref = {"source_file": "dataset/thumbnails128x128/00001.png", "native": path.name}
        item = {"thumbnail": {"file_path": "thumbnails128x128/00000/00001.png", "pixel_size": [128, 128], "pixel_md5": digest_pixels(image)}, "image": {"file_path": "images1024x1024/00000/00001.png", "pixel_size": [1024, 1024], "file_size": 2000, "file_url": "https://drive.google.com/uc?id=public"}}
        return path, ref, item

    def test_selection_keeps_original_roles_and_unique_fixed_sample(self):
        protocol = json.loads(Path("outputs/cctv_dgp_vm_bundle_v1/protocol.json").read_text())
        refs = choose_references(protocol)
        self.assertEqual(len({r["id"] for r in refs}), 16)
        self.assertEqual(sum(r["role"] == "train" for r in refs), 10)
        self.assertEqual(sum(r["role"] == "validation" for r in refs), 6)
        self.assertTrue(all(r in protocol["references"] for r in refs))

    def test_filename_alone_cannot_bind_wrong_pixels(self):
        _, ref, item = self.fixture()
        bind_thumbnail(ref, item, self.directory)
        item["thumbnail"]["pixel_md5"] = "0" * 32
        with self.assertRaisesRegex(ValueError, "pixels"):
            bind_thumbnail(ref, item, self.directory)

    def test_matching_pixels_cannot_bind_wrong_image_id(self):
        _, ref, item = self.fixture()
        item["image"]["file_path"] = "images1024x1024/00000/00002.png"
        with self.assertRaisesRegex(ValueError, "IDs"):
            bind_thumbnail(ref, item, self.directory)

    def test_checksum_rejects_same_size_corruption(self):
        path = self.directory / "data"
        path.write_bytes(b"good")
        spec = {"file_size": 4, "file_md5": hashlib.md5(b"good").hexdigest()}
        verify_blob(path, spec)
        path.write_bytes(b"evil")
        with self.assertRaisesRegex(ValueError, "MD5"):
            verify_blob(path, spec)

    def test_public_host_allowlist_rejects_lookalike_and_credentials(self):
        for url in ("https://drive.google.com.evil.test/uc", "http://drive.google.com/uc", "https://user:password@drive.google.com/uc"):
            with self.assertRaises(ValueError):
                allowed_url(url)

    def test_confirmation_is_limited_to_ordinary_public_download_fields(self):
        form = DownloadForm()
        form.feed('<form id="download-form" action="https://drive.usercontent.google.com/download" method="get"><input name="id" value="public"><input name="confirm" value="t"><input name="export" value="download"></form>')
        self.assertEqual(form.fields["id"], "public")
        for body in ('<form id="download-form" action="https://accounts.google.com/login">', '<form id="download-form" action="https://drive.usercontent.google.com/download"><input name="password">'):
            with self.assertRaises(ValueError):
                DownloadForm().feed(body)


if __name__ == "__main__":
    unittest.main()
