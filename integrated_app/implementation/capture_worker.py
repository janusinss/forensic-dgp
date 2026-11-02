"""Disposable RTSP reader. Configuration arrives only on stdin; stdout is framed PNG."""
import json
import os
import struct
import sys
import time

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|protocol_whitelist;rtsp,rtp,tcp|timeout;5000000'

def main():
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
    scale = min(1., 1280 / max(w, h))
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    detector = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))
    if detector.empty():
        return 1
    boxes = detector.detectMultiScale(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    result = []
    for x, y, bw, bh in boxes[:100]:
        left, top = int(x / scale), int(y / scale)
        right, bottom = min(w, int((x + bw) / scale)), min(h, int((y + bh) / scale))
        result.append([left, top, right - left, bottom - top])
    sys.stdout.write(json.dumps(result))
    return 0

if __name__ == '__main__':
    try:
        sys.exit(detect() if sys.argv[1:] == ['detect'] else main())
    except Exception:
        sys.exit(1)
