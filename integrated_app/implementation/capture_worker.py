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


def scan_faces(image):
    import cv2

    height, width = image.shape[:2]
    if max(width, height) > MAX_SCAN_SIDE:
        scale = MAX_SCAN_SIDE / max(width, height)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    scan_height, scan_width = image.shape[:2]
    size = (scan_width, scan_height)
    cv2.setNumThreads(2)
    detector = cv2.FaceDetectorYN.create('onnx', checked_model(), b'', size, 0.75, 0.3)
    _, faces = detector.detect(image)
    return native_face_boxes(faces, (width, height), size)


def detect():
    import cv2
    import numpy as np
    raw = sys.stdin.buffer.read(24 * 1024 * 1024 + 1)
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return 1
    h, w = image.shape[:2]
    if max(w, h) > 4096 or w * h > 8388608:
        return 1
    sys.stdout.write(json.dumps(scan_faces(image)))
    return 0

if __name__ == '__main__':
    try:
        sys.exit(detect() if sys.argv[1:] == ['detect'] else main())
    except Exception:
        sys.exit(1)
