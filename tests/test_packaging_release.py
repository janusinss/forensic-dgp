"""Release metadata must identify exact source and never include untracked files."""
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("dgp_release_builder", Path(__file__).resolve().parents[1] / "packaging/build_release.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class ReleaseManifestTests(unittest.TestCase):
    def test_manifest_binds_payload_and_tree(self):
        sha = "a" * 40
        value = builder.source_manifest(sha, b"zip", b"100644 blob " + b"b" * 40 + b"\tapp.py\0", (sha + "\n").encode())
        self.assertEqual(value["files"], {"app.py": "b" * 40})
        self.assertEqual(value["source_zip_size"], 3)
        self.assertEqual(len(value["source_zip_sha256"]), 64)

    def test_links_and_traversal_refused(self):
        sha = "a" * 40
        for mode, name in ((b"120000", b"link"), (b"100644", b"../outside"), (b"100644", b"bad\\path")):
            with self.subTest(name=name), self.assertRaises(ValueError):
                builder.source_manifest(sha, b"zip", mode + b" blob " + b"b" * 40 + b"\t" + name + b"\0", (sha + "\n").encode())

    def test_version_and_history_validation(self):
        for value in ("1.2", "1.2.3;evil", "01.2.3", "10000.0.0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                builder.checked_version(value)
        self.assertEqual(builder.checked_version("0.1.0"), "0.1.0")
        with self.assertRaises(ValueError):
            builder.source_manifest("a" * 40, b"zip", b"100644 blob " + b"b" * 40 + b"\tapp.py\0", b"bad\n")


if __name__ == "__main__":
    unittest.main()
