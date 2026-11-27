"""Unit and integration tests for P3-R3 Video/RTSP Lucky-Frame + 5-Frame ECC Huber Burst Fusion."""
from __future__ import annotations

import asyncio
import base64
from collections import deque
import hashlib
import io
import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi import FastAPI
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from implementation import capture_api, fiqa_scorer, video_burst


def _png(array: np.ndarray) -> bytes:
    out = io.BytesIO()
    Image.fromarray(array, "RGB").save(out, format="PNG")
    return out.getvalue()


def _make_textured_face(size: int = 160) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx, cy = size * 0.5, size * 0.5
    r = np.hypot((xx - cx) / (size * 0.36), (yy - cy) / (size * 0.42))
    bg = np.stack([
        215.0 + 18.0 * np.sin(xx / 11.0) + 12.0 * np.cos(yy / 13.0),
        220.0 + 14.0 * np.cos(xx / 15.0) + 14.0 * np.sin(yy / 9.0),
        225.0 + 12.0 * np.sin((xx + yy) / 14.0),
    ], axis=-1)
    skin = np.stack([
        205.0 - 38.0 * r + 14.0 * np.sin(xx / 7.0),
        158.0 - 30.0 * r + 10.0 * np.cos(yy / 8.0),
        132.0 - 24.0 * r + 8.0 * np.sin((xx - yy) / 9.0),
    ], axis=-1)
    img = np.where((r <= 1.0)[..., None], skin, bg)
    # Realistic dark hair crown + dark shoulders for natural portrait variance (~55 std)
    hair = (yy < size * 0.30) & (np.hypot((xx - cx) / (size * 0.44), (yy - cy * 0.85) / (size * 0.40)) <= 1.0)
    shoulders = (yy > size * 0.80) & (np.abs(xx - cx) < size * 0.46)
    img[hair] = [38, 32, 30]
    img[shoulders] = [44, 52, 68]
    # Crisp facial landmarks (eyes, brows, nose bridge, mouth) for strong ECC texture
    img[int(size * 0.36):int(size * 0.45), int(size * 0.28):int(size * 0.42)] = [32, 30, 35]
    img[int(size * 0.36):int(size * 0.45), int(size * 0.58):int(size * 0.72)] = [32, 30, 35]
    img[int(size * 0.31):int(size * 0.34), int(size * 0.25):int(size * 0.44)] = [50, 42, 38]
    img[int(size * 0.31):int(size * 0.34), int(size * 0.56):int(size * 0.75)] = [50, 42, 38]
    img[int(size * 0.46):int(size * 0.62), int(size * 0.47):int(size * 0.53)] = [142, 104, 88]
    img[int(size * 0.68):int(size * 0.75), int(size * 0.36):int(size * 0.64)] = [138, 62, 65]
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    h, w = a.shape[:2]
    sl = np.s_[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]
    diff = a[sl].astype(np.float32) - b[sl].astype(np.float32)
    mse = float(np.mean(diff * diff))
    return 99.0 if mse < 1e-8 else float(10.0 * math.log10((255.0 * 255.0) / mse))


async def _asgi_request(app, method, path, body=b"", headers=None):
    headers = headers or {}
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": ("127.0.0.1", 1234),
        "server": ("127.0.0.1", 8000),
        "root_path": "",
    }
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
    start = next(m for m in sent if m["type"] == "http.response.start")
    content = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return start["status"], content


def _multipart_burst(primary_bytes: bytes, *, burst_list=(), stream_id=None, source_kind="video"):
    boundary = "burst-test-boundary"
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="source_kind"\r\n\r\n{source_kind}\r\n'.encode(),
    ]
    if stream_id is not None:
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="stream_id"\r\n\r\n{stream_id}\r\n'.encode()
        )
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="frame.png"\r\nContent-Type: image/png\r\n\r\n'.encode()
        + primary_bytes
        + b"\r\n"
    )
    for idx, bbytes in enumerate(burst_list):
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="burst_files"; filename="b{idx}.png"\r\nContent-Type: image/png\r\n\r\n'.encode()
            + bbytes
            + b"\r\n"
        )
    return b"".join(parts) + f"--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


class VideoBurstCoreTests(unittest.TestCase):
    def setUp(self):
        self.clean = _make_textured_face(160)

    def test_ecc_align_affine_translation_fallback_and_bounded_rejections(self):
        h, w = self.clean.shape[:2]
        mat = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), 1.4, 1.0)
        mat[0, 2] += 2.2
        mat[1, 2] -= 1.6
        shifted = cv2.warpAffine(self.clean, mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)

        # 1) Standard affine ECC aligns shifted+rotated frame cleanly with high NCC.
        aligned, ncc, ok = video_burst.ecc_align_to_anchor(self.clean, shifted, ncc_min=0.88)
        self.assertTrue(ok)
        self.assertIsNotNone(aligned)
        self.assertGreaterEqual(ncc, 0.95)
        self.assertGreater(_psnr(aligned, self.clean), _psnr(shifted, self.clean) + 5.0)

        # 2) Translation fallback activates when affine ECC raises cv2.error.
        orig_ecc = cv2.findTransformECC
        calls = []

        def _fail_affine_only(tmpl, inp, warp, mode, criteria, mask=None, gauss=5):
            calls.append(mode)
            if mode == cv2.MOTION_AFFINE:
                raise cv2.error("simulated affine divergence")
            return orig_ecc(tmpl, inp, warp, mode, criteria, mask, gauss)

        mat_t = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.5]], dtype=np.float32)
        trans_only = cv2.warpAffine(self.clean, mat_t, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
        with patch.object(cv2, "findTransformECC", side_effect=_fail_affine_only):
            al_t, ncc_t, ok_t = video_burst.ecc_align_to_anchor(self.clean, trans_only, ncc_min=0.88)
        self.assertEqual(calls, [cv2.MOTION_AFFINE, cv2.MOTION_TRANSLATION])
        self.assertTrue(ok_t)
        self.assertIsNotNone(al_t)
        self.assertGreaterEqual(ncc_t, 0.95)

        # 3) Bounded rejection when both ECC modes fail or return NaN/Inf warp.
        for exc in (cv2.error("ecc fail"), RuntimeError("boom"), ValueError("bad")):
            with self.subTest(exc=type(exc).__name__), patch.object(cv2, "findTransformECC", side_effect=exc):
                self.assertEqual(video_burst.ecc_align_to_anchor(self.clean, shifted), (None, 0.0, False))

        nan_warp = np.array([[1.0, 0.0, np.nan], [0.0, 1.0, 0.0]], dtype=np.float32)
        with patch.object(cv2, "findTransformECC", return_value=(0.9, nan_warp)):
            self.assertEqual(video_burst.ecc_align_to_anchor(self.clean, shifted), (None, 0.0, False))

        # 4) Unrelated frame fails NCC gate cleanly.
        rng = np.random.default_rng(999)
        unrelated = rng.integers(0, 256, self.clean.shape, dtype=np.uint8)
        al_bad, ncc_bad, ok_bad = video_burst.ecc_align_to_anchor(self.clean, unrelated, ncc_min=0.88)
        self.assertFalse(ok_bad)
        self.assertIsNone(al_bad)
        self.assertLess(ncc_bad, 0.88)

        # 5) Invalid / tiny arrays degrade cleanly.
        self.assertEqual(video_burst.ecc_align_to_anchor(np.zeros((8, 8, 3), dtype=np.uint8), self.clean), (None, 0.0, False))

    def test_huber_temporal_burst_fuse_psnr_gain_and_blink_outlier_rejection(self):
        rng = np.random.default_rng(2026)
        h, w = self.clean.shape[:2]
        anchor = np.clip(self.clean.astype(np.float32) + rng.normal(0.0, 22.0, self.clean.shape), 0, 255).astype(np.uint8)

        neighbors = []
        for idx in range(4):
            mat = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), float(rng.uniform(-1.0, 1.0)), 1.0)
            mat[0, 2] += float(rng.uniform(-1.5, 1.5))
            mat[1, 2] += float(rng.uniform(-1.5, 1.5))
            warped = cv2.warpAffine(self.clean, mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
            if idx == 3:
                # Inject a realistic closed-eyelid blink outlier on left eye (~150 LSB local jump > 2.2*tau)
                # so the frame passes global ECC NCC>=0.88 and exercises per-pixel Huber trimming.
                warped[58:72, 46:66] = [190, 150, 125]
            noisy = np.clip(warped.astype(np.float32) + rng.normal(0.0, 20.0, warped.shape), 0, 255).astype(np.uint8)
            neighbors.append(noisy)

        # Add one completely unrelated frame to verify NCC rejection.
        unrelated = rng.integers(0, 256, self.clean.shape, dtype=np.uint8)

        fused, meta = video_burst.huber_temporal_burst_fuse(
            anchor, neighbors + [unrelated], ncc_min=0.88, huber_tau=12.0
        )
        self.assertEqual(meta["neighbors_accepted"], 4)
        self.assertEqual(meta["neighbors_rejected"], 1)
        self.assertEqual(meta["frames_total_with_anchor"], 5)

        # Significant PSNR gain over single noisy anchor.
        psnr_anchor = _psnr(anchor, self.clean)
        psnr_fused = _psnr(fused, self.clean)
        self.assertGreater(psnr_fused, psnr_anchor + 2.5)

        # Eye outlier must be Huber-rejected (fused eye mean stays near dark clean eye, not pulled toward 250).
        eye_clean_mean = float(self.clean[58:70, 46:66].mean())
        eye_fused_mean = float(fused[58:70, 46:66].mean())
        self.assertLess(abs(eye_fused_mean - eye_clean_mean), 18.0)

    def test_select_and_fuse_burst_passthrough_lucky_consensus_and_fiqa_fallback(self):
        # 1) Single-frame passthrough returns a copy with burst_used=False.
        out1, meta1 = video_burst.select_and_fuse_burst([self.clean])
        self.assertFalse(meta1["burst_used"])
        self.assertEqual(meta1["scorer"], "passthrough")
        self.assertEqual(meta1["frames_total_with_anchor"], 1)
        np.testing.assert_array_equal(out1, self.clean)
        self.assertIsNot(out1, self.clean)

        # 2) Lucky-Frame selection + consensus-anchor fallback when #1 FIQA frame disagrees with all neighbors.
        rng = np.random.default_rng(707)
        good_frames = [
            np.clip(self.clean.astype(np.float32) + rng.normal(0.0, 12.0, self.clean.shape), 0, 255).astype(np.uint8)
            for _ in range(3)
        ]
        # Outlier frame that gets mocked as #1 FIQA score but fails NCC against all real frames.
        bad_top1 = rng.integers(0, 256, self.clean.shape, dtype=np.uint8)
        frames = [bad_top1, *good_frames]
        with patch.object(video_burst, "score_face_crops_fiqa", return_value=[0.99, 0.82, 0.78, 0.74]):
            fused_cons, meta_cons = video_burst.select_and_fuse_burst(frames, ncc_min=0.88)
        self.assertEqual(meta_cons["scorer"], "ediffiqa_tiny")
        self.assertTrue(meta_cons["consensus_anchor_fallback"])
        self.assertEqual(meta_cons["anchor_index"], 1)
        self.assertTrue(meta_cons["burst_used"])
        self.assertGreaterEqual(meta_cons["neighbors_accepted"], 2)
        self.assertGreater(_psnr(fused_cons, self.clean), _psnr(good_frames[0], self.clean) + 1.0)

        # 3) FIQA unavailable (None scores) falls back to Laplacian sharpness and picks sharpest anchor.
        blurry = cv2.GaussianBlur(self.clean, (9, 9), 2.5)
        with patch.object(video_burst, "score_face_crops_fiqa", return_value=[None, None, None]):
            fused_lap, meta_lap = video_burst.select_and_fuse_burst([blurry, self.clean, blurry], ncc_min=0.88)
        self.assertEqual(meta_lap["scorer"], "laplacian_fallback")
        self.assertEqual(meta_lap["anchor_index"], 1)
        self.assertTrue(meta_lap["burst_used"])
        self.assertEqual(fused_lap.shape, self.clean.shape)


class CaptureApiBurstWiringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = patch.object(capture_api, "STORE", Path(self.temp.name))
        self.store.start()
        self.addCleanup(self.store.stop)
        self._det = patch.object(capture_api, "detect_faces", return_value=[[16, 16, 80, 80]])
        self._det.start()
        self.addCleanup(self._det.stop)
        capture_api.FRAMES.clear()
        capture_api.CROPS.clear()
        capture_api.STREAMS.clear()
        self.addCleanup(capture_api.FRAMES.clear)
        self.addCleanup(capture_api.CROPS.clear)
        self.addCleanup(capture_api.STREAMS.clear)
        self.app = FastAPI()
        capture_api.register(self.app)
        status, content = self._req("GET", "/capture/status")
        self.assertEqual(status, 200)
        self.token = json.loads(content)["token"]
        self.clean = _make_textured_face(128)

    def _req(self, method, path, body=b"", headers=None):
        return asyncio.run(_asgi_request(self.app, method, path, body, headers))

    def test_single_frame_backward_compat_and_burst_files_crop_fusion(self):
        rng = np.random.default_rng(1234)
        noisy_frames = [
            np.clip(self.clean.astype(np.float32) + rng.normal(0.0, 20.0, self.clean.shape), 0, 255).astype(np.uint8)
            for _ in range(5)
        ]
        pngs = [_png(f) for f in noisy_frames]

        # 1) Single-frame path stays 100% backward compatible (no burst_fusion key, exact native pixels).
        body1, ctype1 = _multipart_burst(pngs[0])
        st1, res1_b = self._req("POST", "/capture/frames", body1, {"content-type": ctype1, "x-capture-token": self.token})
        self.assertEqual(st1, 200)
        res1 = json.loads(res1_b)
        self.assertNotIn("burst_frame_count", res1)
        rect = {"x": 8, "y": 8, "width": 112, "height": 112}
        stc1, crop1_b = self._req(
            "POST", f"/capture/frames/{res1['frame_id']}/crop", json.dumps(rect).encode(),
            {"content-type": "application/json", "x-capture-token": self.token},
        )
        self.assertEqual(stc1, 200)
        crop1 = json.loads(crop1_b)
        self.assertNotIn("burst_fusion", crop1["metadata"])
        crop1_png = base64.b64decode(crop1["image"].split(",", 1)[1])
        crop1_rgb = np.asarray(Image.open(io.BytesIO(crop1_png)))
        np.testing.assert_array_equal(crop1_rgb, noisy_frames[0][8:120, 8:120])

        # 2) Multi-frame burst_files runs ECC Huber burst fusion inside _crop and improves PSNR.
        body5, ctype5 = _multipart_burst(pngs[0], burst_list=pngs[1:])
        st5, res5_b = self._req("POST", "/capture/frames", body5, {"content-type": ctype5, "x-capture-token": self.token})
        self.assertEqual(st5, 200)
        res5 = json.loads(res5_b)
        self.assertEqual(res5["burst_frame_count"], 5)
        stc5, crop5_b = self._req(
            "POST", f"/capture/frames/{res5['frame_id']}/crop", json.dumps(rect).encode(),
            {"content-type": "application/json", "x-capture-token": self.token},
        )
        self.assertEqual(stc5, 200)
        crop5 = json.loads(crop5_b)
        bf = crop5["metadata"]["burst_fusion"]
        self.assertTrue(bf["burst_used"])
        self.assertEqual(bf["frames_total_with_anchor"], 5)
        crop5_png = base64.b64decode(crop5["image"].split(",", 1)[1])
        crop5_rgb = np.asarray(Image.open(io.BytesIO(crop5_png)))
        gt_crop = self.clean[8:120, 8:120]
        self.assertGreater(_psnr(crop5_rgb, gt_crop), _psnr(crop1_rgb, gt_crop) + 2.5)
        ev = capture_api.evidence_for_crop(crop5_png, crop5["capture_id"], owner=self.token)
        self.assertEqual(ev["capture/crop.png"], crop5_png)

        # 3) >4 extra burst frames or mismatched dimensions rejected with 422.
        body_too_many, ctype_tm = _multipart_burst(pngs[0], burst_list=pngs)
        st_tm, _ = self._req("POST", "/capture/frames", body_too_many, {"content-type": ctype_tm, "x-capture-token": self.token})
        self.assertEqual(st_tm, 422)

        wrong_sz = _png(np.zeros((64, 64, 3), dtype=np.uint8))
        body_mm, ctype_mm = _multipart_burst(pngs[0], burst_list=[wrong_sz])
        st_mm, _ = self._req("POST", "/capture/frames", body_mm, {"content-type": ctype_mm, "x-capture-token": self.token})
        self.assertEqual(st_mm, 422)

    def test_rtsp_recent_frames_ring_buffer_and_owner_scoped_stream_id_burst(self):
        rng = np.random.default_rng(5678)
        frames = [
            _png(np.clip(self.clean.astype(np.float32) + rng.normal(0.0, 16.0, self.clean.shape), 0, 255).astype(np.uint8))
            for _ in range(6)
        ]

        # 1) _read_stream populates bounded deque(maxlen=5) in record['recent_frames'].
        stream_bytes = b"".join(struct.pack("!I", len(f)) + f for f in frames)

        class _FakeProc:
            def __init__(self, payload):
                self.stdout = io.BytesIO(payload)

        rec = {"process": _FakeProc(stream_bytes), "latest": None, "recent_frames": deque(maxlen=5), "failed": False, "owner": self.token}
        capture_api._read_stream(rec)
        self.assertEqual(len(rec["recent_frames"]), 5)
        self.assertEqual(rec["latest"], frames[-1])
        self.assertEqual(list(rec["recent_frames"]), frames[1:])

        # 2) Matching owner stream_id attaches up to 4 recent frames; foreign owner stream_id is ignored.
        capture_api.STREAMS["stream-owned"] = {
            "latest": frames[-1],
            "recent_frames": deque(frames[1:], maxlen=5),
            "owner": self.token,
            "failed": False,
        }
        other_token = json.loads(self._req("GET", "/capture/status")[1])["token"]
        capture_api.STREAMS["stream-foreign"] = {
            "latest": frames[-1],
            "recent_frames": deque(frames[1:], maxlen=5),
            "owner": other_token,
            "failed": False,
        }

        body_ok, ctype_ok = _multipart_burst(frames[-1], stream_id="stream-owned", source_kind="rtsp")
        st_ok, res_ok_b = self._req("POST", "/capture/frames", body_ok, {"content-type": ctype_ok, "x-capture-token": self.token})
        self.assertEqual(st_ok, 200)
        res_ok = json.loads(res_ok_b)
        self.assertEqual(res_ok["burst_frame_count"], 5)

        body_for, ctype_for = _multipart_burst(frames[-1], stream_id="stream-foreign", source_kind="rtsp")
        st_for, res_for_b = self._req("POST", "/capture/frames", body_for, {"content-type": ctype_for, "x-capture-token": self.token})
        self.assertEqual(st_for, 200)
        res_for = json.loads(res_for_b)
        self.assertNotIn("burst_frame_count", res_for)

    def test_audit_hardening_fallbacks_anisotropic_ecc_and_missing_burst_file(self):
        # 1) FIQA exception and invalid-frame FIQA score cleanly fall back to Laplacian scorer.
        blurry = cv2.GaussianBlur(self.clean, (9, 9), 2.5)
        with patch.object(video_burst, "score_face_crops_fiqa", side_effect=RuntimeError("ort crashed")):
            fused_err, meta_err = video_burst.select_and_fuse_burst([blurry, self.clean])
        self.assertEqual(meta_err["scorer"], "laplacian_fallback")
        self.assertEqual(meta_err["anchor_index"], 1)
        self.assertEqual(fused_err.shape, self.clean.shape)

        tiny_invalid = np.zeros((8, 8, 3), dtype=np.uint8)
        with patch.object(video_burst, "score_face_crops_fiqa", return_value=[None, 0.99]):
            _, meta_inv = video_burst.select_and_fuse_burst([self.clean, tiny_invalid])
        self.assertEqual(meta_inv["scorer"], "laplacian_fallback")
        self.assertEqual(meta_inv["anchor_index"], 0)

        # 2) High-aspect anisotropic crop (640x32) rescales ECC affine translations accurately.
        yy, xx = np.mgrid[0:32, 0:640].astype(np.float32)
        strip = np.stack([
            128 + 60 * np.sin(xx / 9.0) * np.cos(yy / 4.0),
            128 + 50 * np.cos(xx / 13.0) * np.sin(yy / 5.0),
            128 + 40 * np.sin((xx + 3 * yy) / 11.0),
        ], axis=-1).clip(0, 255).astype(np.uint8)
        mat_s = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.5]], dtype=np.float32)
        shifted_strip = cv2.warpAffine(strip, mat_s, (640, 32), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
        al_s, ncc_s, ok_s = video_burst.ecc_align_to_anchor(strip, shifted_strip, ncc_min=0.88)
        self.assertTrue(ok_s)
        self.assertIsNotNone(al_s)
        self.assertGreaterEqual(ncc_s, 0.95)

        # 3) RuntimeError inside select_and_fuse_burst falls back to primary crop instead of HTTP 500.
        p1 = _png(self.clean)
        p2 = _png(self.clean)
        body, ctype = _multipart_burst(p1, burst_list=[p2])
        st, res_b = self._req("POST", "/capture/frames", body, {"content-type": ctype, "x-capture-token": self.token})
        self.assertEqual(st, 200)
        fid = json.loads(res_b)["frame_id"]
        rect = {"x": 8, "y": 8, "width": 112, "height": 112}
        with patch.object(video_burst, "select_and_fuse_burst", side_effect=RuntimeError("boom")):
            stc, crop_b = self._req(
                "POST", f"/capture/frames/{fid}/crop", json.dumps(rect).encode(),
                {"content-type": "application/json", "x-capture-token": self.token},
            )
        self.assertEqual(stc, 200)
        crop = json.loads(crop_b)
        self.assertFalse(crop["metadata"]["burst_fusion"]["burst_used"])
        self.assertEqual(crop["metadata"]["burst_fusion"]["fallback_reason"], "burst_fusion_error")

        # 4) Missing burst file on disk returns HTTP 409 (not unhandled FileNotFoundError 500).
        burst_file = capture_api.FRAMES[fid]["path"] / capture_api.FRAMES[fid]["burst_frames"][0]["name"]
        burst_file.unlink()
        stc_missing, _ = self._req(
            "POST", f"/capture/frames/{fid}/crop", json.dumps(rect).encode(),
            {"content-type": "application/json", "x-capture-token": self.token},
        )
        self.assertEqual(stc_missing, 409)


if __name__ == "__main__":
    unittest.main()
