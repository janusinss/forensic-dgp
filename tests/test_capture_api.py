"""Offline frozen-frame capture contract tests; no camera, network or inference."""
import asyncio
import base64
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from unittest.mock import Mock

import numpy as np
from fastapi import FastAPI
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from implementation import capture_api


async def asgi_request(app, method, path, body=b"", headers=None):
    headers = headers or {}
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method,
             "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"",
             "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
             "client": ("127.0.0.1", 1234), "server": ("127.0.0.1", 8000), "root_path": ""}
    sent = []
    delivered = False

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.Event().wait()
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)
    start = next(message for message in sent if message["type"] == "http.response.start")
    content = b"".join(message.get("body", b"") for message in sent if message["type"] == "http.response.body")
    return start["status"], content


def png(array):
    out = io.BytesIO()
    Image.fromarray(array, "RGB").save(out, format="PNG")
    return out.getvalue()


def multipart(contents, source_kind="video", source_time="1.25"):
    boundary = "capture-test-boundary"
    parts = []
    for name, value in (("source_kind", source_kind), ("source_time", source_time)):
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="frame.png"\r\nContent-Type: image/png\r\n\r\n'.encode() + contents + b"\r\n")
    return b"".join(parts) + f"--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


class CaptureAPITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = patch.object(capture_api, "STORE", Path(self.temp.name))
        self.store.start()
        self.addCleanup(self.store.stop)
        detector_patch = patch.object(capture_api, "detect_faces", return_value=[[4, 5, 40, 32]])
        self.detector = detector_patch.start()
        self.addCleanup(detector_patch.stop)
        self.app = FastAPI()
        capture_api.register(self.app)
        status, content = self.request("GET", "/capture/status")
        self.assertEqual(status, 200)
        self.token = json.loads(content)["token"]
        y, x = np.indices((64, 80))
        self.pixels = np.stack((x * 3 % 256, y * 4 % 256, (x + y) * 2 % 256), axis=-1).astype(np.uint8)
        self.frame = png(self.pixels)

    def request(self, method, path, body=b"", headers=None):
        return asyncio.run(asgi_request(self.app, method, path, body, headers))

    def freeze(self, contents=None, **fields):
        data, content_type = multipart(self.frame if contents is None else contents, **fields)
        status, content = self.request("POST", "/capture/frames", data,
                                       {"content-type": content_type, "x-capture-token": self.token})
        return status, json.loads(content)

    def crop(self, frame_id, rectangle):
        return self.request("POST", f"/capture/frames/{frame_id}/crop", json.dumps(rectangle).encode(),
                            {"content-type": "application/json", "x-capture-token": self.token})

    def test_lossless_rectangle_and_bound_evidence(self):
        status, frame = self.freeze()
        self.assertEqual(status, 200)
        self.assertEqual((frame["width"], frame["height"]), (80, 64))
        self.assertEqual(frame["faces"], [[4, 5, 40, 32]])
        status, original = self.request("GET", frame["frame_url"], headers={"x-capture-token": self.token})
        self.assertEqual((status, original), (200, self.frame))
        rect = {"x": 7, "y": 8, "width": 30, "height": 25}
        status, content = self.crop(frame["frame_id"], rect)
        self.assertEqual(status, 200)
        result = json.loads(content)
        self.assertTrue(result["image"].startswith("data:image/png;base64,"))
        crop_bytes = base64.b64decode(result["image"].split(",", 1)[1], validate=True)
        self.assertTrue(np.array_equal(np.asarray(Image.open(io.BytesIO(crop_bytes))), self.pixels[8:33, 7:37]))
        metadata = result["metadata"]
        self.assertEqual(metadata["rectangle"], [7, 8, 30, 25])
        self.assertEqual(metadata["frame_sha256"], hashlib.sha256(self.frame).hexdigest())
        self.assertEqual(metadata["crop_sha256"], hashlib.sha256(crop_bytes).hexdigest())
        evidence = capture_api.evidence_for_crop(crop_bytes, result["capture_id"])
        self.assertEqual(evidence["capture/frame.png"], self.frame)
        self.assertEqual(evidence["capture/crop.png"], crop_bytes)
        with self.assertRaises(Exception):
            capture_api.evidence_for_crop(b"altered", result["capture_id"])

    def test_missing_id_and_invalid_rectangles_fail(self):
        status, content = self.crop("missing", {"x": 0, "y": 0, "width": 16, "height": 16})
        self.assertGreaterEqual(status, 400)
        status, frame = self.freeze()
        self.assertEqual(status, 200)
        for rect in ({"x": -1, "y": 0, "width": 16, "height": 16},
                     {"x": 70, "y": 0, "width": 16, "height": 16},
                     {"x": 0, "y": 0, "width": 15, "height": 16},
                     {"x": 0, "y": 0, "width": 1601, "height": 16},
                     {"x": 0.5, "y": 0, "width": 16, "height": 16}):
            with self.subTest(rect=rect):
                status, _ = self.crop(frame["frame_id"], rect)
                self.assertGreaterEqual(status, 400)

    def test_invalid_images_and_frame_limits_fail_before_detection(self):
        for contents in (b"not a png", b"x" * (24 * 1024 * 1024 + 1),
                         png(np.zeros((1, 4097, 3), dtype=np.uint8)),
                         png(np.zeros((2049, 4096, 3), dtype=np.uint8))):
            with self.subTest(size=len(contents)):
                self.detector.reset_mock()
                status, _ = self.freeze(contents)
                self.assertGreaterEqual(status, 400)
                self.detector.assert_not_called()

    def test_token_and_foreign_origin_are_rejected(self):
        body, ctype = multipart(self.frame)
        for headers in ({"content-type": ctype},
                        {"content-type": ctype, "x-capture-token": "wrong"},
                        {"content-type": ctype, "x-capture-token": self.token, "origin": "https://example.org"}):
            with self.subTest(headers=headers):
                status, _ = self.request("POST", "/capture/frames", body, headers)
                self.assertGreaterEqual(status, 400)

    def test_source_metadata_cannot_carry_stream_credentials(self):
        status, _ = self.freeze(source_kind="rtsp://user:secret@example.org/live")
        self.assertGreaterEqual(status, 400)

    def test_opaque_canvas_frame_and_detector_failure_allow_manual_selection(self):
        rgba = np.concatenate((self.pixels, np.full((64, 80, 1), 255, dtype=np.uint8)), axis=2)
        out = io.BytesIO()
        Image.fromarray(rgba, "RGBA").save(out, format="PNG")
        self.detector.side_effect = RuntimeError("detector unavailable")
        status, frame = self.freeze(out.getvalue())
        self.assertEqual(status, 200)
        self.assertEqual(frame["faces"], [])
        self.assertTrue(frame["detection_warning"])
        status, crop = self.crop(frame["frame_id"], {"x": 1, "y": 2, "width": 20, "height": 20})
        self.assertEqual(status, 200)
        raw = base64.b64decode(json.loads(crop)["image"].split(",", 1)[1])
        self.assertTrue(np.array_equal(np.asarray(Image.open(io.BytesIO(raw))), self.pixels[2:22, 1:21]))

    def test_transparent_canvas_frame_is_rejected(self):
        rgba = np.concatenate((self.pixels, np.full((64, 80, 1), 255, dtype=np.uint8)), axis=2)
        rgba[0, 0, 3] = 0
        out = io.BytesIO()
        Image.fromarray(rgba, "RGBA").save(out, format="PNG")
        status, _ = self.freeze(out.getvalue())
        self.assertGreaterEqual(status, 400)

    def test_rtsp_resolution_rejects_nonprivate_and_pins_private_ip(self):
        request = capture_api.StreamRequest(url="rtsp://camera.local/live?channel=1", username="user@name", password="p:/? #")
        with patch.object(capture_api.subprocess, "run") as resolver:
            for addresses in (["127.0.0.1"], ["8.8.8.8"], ["192.168.1.7", "8.8.8.8"], []):
                resolver.return_value = SimpleNamespace(stdout=json.dumps(addresses).encode())
                with self.subTest(addresses=addresses), self.assertRaises(ValueError):
                    capture_api._stream_url(request)
            resolver.return_value = SimpleNamespace(stdout=b'["192.168.1.7"]')
            self.assertEqual(capture_api._stream_url(request),
                             "rtsp://user%40name:p%3A%2F%3F%20%23@192.168.1.7:554/live?channel=1")
            resolver.reset_mock()
            with self.assertRaises(ValueError):
                capture_api._stream_url(capture_api.StreamRequest(url="rtsp://user:secret@192.168.1.7/live"))
            resolver.assert_not_called()

    def test_stream_worker_credentials_use_stdin_and_owned_shutdown(self):
        process = Mock()
        process.stdin = Mock()
        process.poll.return_value = None
        fake_threads = Mock()
        capture_api.STREAMS.clear()
        self.addCleanup(capture_api.STREAMS.clear)
        with patch.object(capture_api, "_stream_url", return_value="rtsp://u:p@192.168.1.7:554/live"), \
             patch.object(capture_api, "worker_python", return_value="python.exe"), \
             patch.object(capture_api.subprocess, "Popen", return_value=process) as spawn, \
             patch.object(capture_api.threading, "Thread", return_value=fake_threads):
            result = capture_api._start_stream(capture_api.StreamRequest(url="rtsp://camera/live"), self.token)
        command = spawn.call_args.args[0]
        self.assertNotIn("u:p", " ".join(command))
        self.assertEqual(json.loads(process.stdin.write.call_args.args[0]), {"url": "rtsp://u:p@192.168.1.7:554/live"})
        self.assertEqual(fake_threads.start.call_count, 2)
        stream_id = result["stream_id"]
        other = json.loads(self.request("GET", "/capture/status")[1])["token"]
        headers = {"x-capture-token": other}
        status, _ = self.request("GET", f"/capture/streams/{stream_id}/frame", headers=headers)
        self.assertGreaterEqual(status, 400)
        self.request("DELETE", f"/capture/streams/{stream_id}", headers=headers)
        process.terminate.assert_not_called()
        self.assertIn(stream_id, capture_api.STREAMS)
        status, _ = self.request("DELETE", f"/capture/streams/{stream_id}", headers={"x-capture-token": self.token})
        self.assertEqual(status, 200)
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=3)
        self.assertNotIn(stream_id, capture_api.STREAMS)

    def test_other_page_cannot_read_or_crop_frozen_frame(self):
        status, frame = self.freeze()
        self.assertEqual(status, 200)
        other = json.loads(self.request("GET", "/capture/status")[1])["token"]
        status, _ = self.request("GET", frame["frame_url"], headers={"x-capture-token": other})
        self.assertGreaterEqual(status, 400)
        rect = json.dumps({"x": 0, "y": 0, "width": 16, "height": 16}).encode()
        status, _ = self.request("POST", f'/capture/frames/{frame["frame_id"]}/crop', rect,
                                 {"x-capture-token": other, "content-type": "application/json"})
        self.assertGreaterEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
