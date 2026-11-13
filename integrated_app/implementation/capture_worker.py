"""Disposable RTSP reader or frozen-frame face detector."""
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time

YUNET_MODEL = Path(__file__).with_name('assets') / 'face_detection_yunet_2023mar.onnx'
YUNET_SHA256 = '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'
MAX_SCAN_SIDE = 1600

def main():
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|protocol_whitelist;rtsp,rtp,tcp|timeout;5000000'
    import cv2
    cfg = json.loads(sys.stdin.buffer.readline(8192))
    cap = cv2.VideoCapture(cfg['url'], cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
    try:
        if not cap.isOpened():
            return 1
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            ok, frame = cap.read()
            if not ok or frame is None:
                return 1
            h, w = frame.shape[:2]
            if w < 1 or h < 1 or max(w, h) > 4096 or w * h > 8388608:
                return 1
            ok, png = cv2.imencode('.png', frame)
            if not ok or len(png) > 24 * 1024 * 1024:
                return 1
            sys.stdout.buffer.write(struct.pack('!I', len(png)))
            sys.stdout.buffer.write(png.tobytes())
            sys.stdout.buffer.flush()
            time.sleep(0.25)
    finally:
        cap.release()
    return 0

def native_face_boxes(rows, original_size, detector_size):
    if rows is None:
        return []
    width, height = original_size
    detector_width, detector_height = detector_size
    scale_x, scale_y = width / detector_width, height / detector_height
    boxes = []
    for row in rows:
        x, y, box_width, box_height = map(float, row[:4])
        right, bottom = x + box_width, y + box_height
        if not all(map(math.isfinite, (x, y, right, bottom))) or min(box_width, box_height) <= 0:
            continue
        left = max(0, math.floor(x * scale_x))
        top = max(0, math.floor(y * scale_y))
        right = min(width, math.ceil(right * scale_x))
        bottom = min(height, math.ceil(bottom * scale_y))
        if min(right - left, bottom - top) < 16:
            continue
        boxes.append([left, top, right - left, bottom - top])
        if len(boxes) == 100:
            break
    return boxes


def checked_model():
    with YUNET_MODEL.open('rb') as source:
        model = source.read(232590)
    if hashlib.sha256(model).hexdigest() != YUNET_SHA256:
        raise RuntimeError('The YuNet face detector file does not match the bundled model.')
    return model


def _valid_face_landmarks(row_scaled, bw, bh):
    """Validate 5-point YuNet landmarks against face box proportions."""
    if row_scaled is None or len(row_scaled) < 14:
        return False
    bx, by = float(row_scaled[0]), float(row_scaled[1])
    bw, bh = float(bw), float(bh)
    if not all(map(math.isfinite, (bx, by, bw, bh))) or min(bw, bh) <= 0:
        return False
    pts = []
    for idx in range(5):
        px = float(row_scaled[4 + 2 * idx])
        py = float(row_scaled[5 + 2 * idx])
        if not (math.isfinite(px) and math.isfinite(py)):
            return False
        nx, ny = (px - bx) / bw, (py - by) / bh
        if not (-0.20 <= nx <= 1.20 and -0.20 <= ny <= 1.20):
            return False
        pts.append((px, py))
    re, le, nose, rm, lm = pts
    d_eye = math.hypot(le[0] - re[0], le[1] - re[1])
    emx, emy = 0.5 * (re[0] + le[0]), 0.5 * (re[1] + le[1])
    mmx, mmy = 0.5 * (rm[0] + lm[0]), 0.5 * (rm[1] + lm[1])
    d_em = math.hypot(mmx - emx, mmy - emy)
    min_b, max_b = min(bw, bh), max(bw, bh)
    if d_eye <= 0 or d_em <= 0:
        return False
    if not (0.14 * min_b <= d_eye <= 0.78 * max_b):
        return False
    if not (0.18 * min_b <= d_em <= 0.75 * max_b):
        return False
    if d_em < 0.32 * d_eye or d_eye < 0.25 * d_em:
        return False
    band_lo = min(emy, mmy) - 0.15 * d_em
    band_hi = max(emy, mmy) + 0.15 * d_em
    return band_lo <= nose[1] <= band_hi


def _padded_face_crop(box, img_w, img_h, max_side=1600):
    """Compute the +20% margin face crop rectangle bounded inside the image."""
    x, y, w, h = map(int, box)
    img_w, img_h = int(img_w), int(img_h)
    margin = int(max(w, h) * 0.20 + 0.5)
    left = max(0, x - margin)
    top = max(0, y - margin)
    right = min(img_w, x + w + margin)
    bottom = min(img_h, y + h + margin)
    return [left, top, min(max_side, right - left), min(max_side, bottom - top)]


def _box_iou(a, b):
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[0] + a[2], b[0] + b[2])
    iy2 = min(a[1] + a[3], b[1] + b[3])
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _collect_records(detector, frame, original_size, threshold, pass_bias):
    det_height, det_width = frame.shape[:2]
    detector.setInputSize((int(det_width), int(det_height)))
    detector.setScoreThreshold(float(threshold))
    _, rows = detector.detect(frame)
    if rows is None:
        return []
    width, height = original_size
    scale_x, scale_y = width / det_width, height / det_height
    records = []
    for row in rows:
        if len(row) < 15:
            continue
        x, y, box_w, box_h = map(float, row[:4])
        raw_score = float(row[14])
        right, bottom = x + box_w, y + box_h
        if not all(map(math.isfinite, (x, y, right, bottom, raw_score))) or min(box_w, box_h) <= 0:
            continue
        left = max(0, math.floor(x * scale_x))
        top = max(0, math.floor(y * scale_y))
        right_i = min(width, math.ceil(right * scale_x))
        bottom_i = min(height, math.ceil(bottom * scale_y))
        bw_i, bh_i = right_i - left, bottom_i - top
        if min(bw_i, bh_i) < 16:
            continue
        row_scaled = [x * scale_x, y * scale_y, box_w * scale_x, box_h * scale_y]
        landmarks = []
        for idx in range(5):
            lx = float(row[4 + 2 * idx]) * scale_x
            ly = float(row[5 + 2 * idx]) * scale_y
            row_scaled.extend((lx, ly))
            landmarks.append((lx, ly))
        if not _valid_face_landmarks(row_scaled, row_scaled[2], row_scaled[3]):
            continue
        box = [left, top, bw_i, bh_i]
        crop = _padded_face_crop(box, width, height)
        records.append({
            'box': box,
            'crop': crop,
            'score': raw_score,
            '_priority': raw_score + pass_bias,
        })
    return records


def scan_face_records(image):
    """Multi-pass YuNet face detection returning validated boxes, +20% padded crops, and scores."""
    import cv2

    if image is None or getattr(image, 'ndim', 0) != 3 or image.shape[2] != 3:
        return []
    height, width = image.shape[:2]
    if min(width, height) < 16:
        return []

    if max(width, height) > MAX_SCAN_SIDE:
        scale = MAX_SCAN_SIDE / max(width, height)
        scan_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        scan_image = cv2.resize(image, scan_size, interpolation=cv2.INTER_AREA)
    else:
        scan_image = image
    scan_height, scan_width = scan_image.shape[:2]
    max_scan = max(scan_width, scan_height)
    original_size = (width, height)

    cv2.setNumThreads(2)
    detector = cv2.FaceDetectorYN.create('onnx', checked_model(), b'', (int(scan_width), int(scan_height)), 0.50, 0.3)

    # Pass 0: full bounded scan at 0.50 threshold.
    candidates = _collect_records(detector, scan_image, original_size, 0.50, pass_bias=0.02)

    max_raw_score = max((c['score'] for c in candidates), default=0.0)
    max_box_side = max((max(c['box'][2], c['box'][3]) for c in candidates), default=0)

    # Pass 1: multi-scale pyramid + mild Gaussian pre-smoothed pyramid for coarse mosaics/large faces.
    if not candidates or max_raw_score < 0.85 or max_box_side < 0.25 * min(width, height):
        for side in (480, 360, 240, 160):
            if side < max_scan:
                sc = side / max_scan
                tw, th = max(1, round(scan_width * sc)), max(1, round(scan_height * sc))
                resized = cv2.resize(scan_image, (tw, th), interpolation=cv2.INTER_AREA)
                candidates.extend(_collect_records(detector, resized, original_size, 0.45, pass_bias=0.01))

        sigma = max(1.2, 0.007 * max_scan)
        smoothed = cv2.GaussianBlur(scan_image, (0, 0), sigmaX=sigma, sigmaY=sigma)
        seen_sm = set()
        for side in (480, 360, 240, 160):
            sc = min(1.0, side / max_scan)
            tw, th = max(1, round(scan_width * sc)), max(1, round(scan_height * sc))
            if (tw, th) in seen_sm:
                continue
            seen_sm.add((tw, th))
            resized_sm = cv2.resize(smoothed, (tw, th), interpolation=cv2.INTER_AREA) if sc < 1.0 else smoothed
            candidates.extend(_collect_records(detector, resized_sm, original_size, 0.42, pass_bias=0.0))

    max_raw_score = max((c['score'] for c in candidates), default=0.0)

    # Pass 2: downscales + LAB-L CLAHE + dual unsharp filters for heavy defocus/motion/low-contrast blur.
    if not candidates or max_raw_score < 0.72:
        clahe = cv2.createCLAHE(clipLimit=2.3, tileGridSize=(4, 4))
        seen_p2 = set()
        for side in (320, 200, 128, 96):
            sc = min(1.0, side / max_scan)
            tw, th = max(1, round(scan_width * sc)), max(1, round(scan_height * sc))
            if (tw, th) in seen_p2:
                continue
            seen_p2.add((tw, th))
            down = cv2.resize(scan_image, (tw, th), interpolation=cv2.INTER_AREA) if sc < 1.0 else scan_image
            lab = cv2.cvtColor(down, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            clahe_bgr = cv2.cvtColor(cv2.merge((clahe.apply(l_ch), a_ch, b_ch)), cv2.COLOR_LAB2BGR)
            for sigma_u, amount_u, thresh_u in ((1.8, 1.5, 0.44), (2.6, 1.8, 0.40)):
                blurred = cv2.GaussianBlur(clahe_bgr, (0, 0), sigmaX=sigma_u, sigmaY=sigma_u)
                sharpened = cv2.addWeighted(clahe_bgr, 1.0 + amount_u, blurred, -amount_u, 0)
                candidates.extend(_collect_records(detector, sharpened, original_size, thresh_u, pass_bias=-0.04))

    candidates.sort(key=lambda c: (c['_priority'], c['score'], c['box'][2] * c['box'][3]), reverse=True)
    kept = []
    for cand in candidates:
        if any(_box_iou(cand['box'], prev['box']) > 0.30 for prev in kept):
            continue
        kept.append(cand)

    filtered = []
    for cand in kept:
        cx_box = cand['box'][0] + cand['box'][2] / 2.0
        cy_box = cand['box'][1] + cand['box'][3] / 2.0
        cand_area = cand['box'][2] * cand['box'][3]
        suppressed = False
        if cand['score'] < 0.68:
            for other in kept:
                if other is cand or other['score'] < 0.82:
                    continue
                other_area = other['box'][2] * other['box'][3]
                if cand_area < 0.12 * other_area:
                    ox, oy, ow, oh = other['crop']
                    if ox <= cx_box <= ox + ow and oy <= cy_box <= oy + oh:
                        suppressed = True
                        break
        if not suppressed:
            filtered.append({'box': cand['box'], 'crop': cand['crop'], 'score': float(cand['score'])})
            if len(filtered) == 100:
                break
    return filtered


def scan_faces(image):
    return [record['box'] for record in scan_face_records(image)]


def detect(details=False):
    import cv2
    import numpy as np
    raw = sys.stdin.buffer.read(24 * 1024 * 1024 + 1)
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return 1
    h, w = image.shape[:2]
    if max(w, h) > 4096 or w * h > 8388608:
        return 1
    payload = scan_face_records(image) if details else scan_faces(image)
    sys.stdout.write(json.dumps(payload))
    return 0

if __name__ == '__main__':
    try:
        arg = sys.argv[1:]
        if arg == ['detect']:
            sys.exit(detect(details=False))
        if arg == ['detect-details']:
            sys.exit(detect(details=True))
        sys.exit(main())
    except Exception:
        sys.exit(1)
