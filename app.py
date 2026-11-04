"""
Flask Web Dashboard for Live YOLO Enclosure Detection
=====================================================
Target Classes:
  - 'bottom': Individual bottom base
  - 'top': Individual top cover
  - 'enclosure_assembled': Fully assembled enclosure

Hardware & Environment:
  - Windows Laptop (Development)
  - Raspberry Pi 4/5 (Deployment Ready)
"""

import os
import sys
import time
import threading
from datetime import datetime
from collections import deque

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
MODEL_PATH = "weights.pt"
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", 0))  # 0 for default webcam, /dev/video0 on Linux
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", 0.50))
FRAME_WIDTH = int(os.environ.get("FRAME_WIDTH", 1280))
FRAME_HEIGHT = int(os.environ.get("FRAME_HEIGHT", 720))
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 5000))

CLASS_ASSEMBLED = "enclosure_assembled"
CLASS_TOP = "top"
CLASS_BOTTOM = "bottom"

# Color definitions (BGR for OpenCV)
CLASS_COLORS = {
    CLASS_ASSEMBLED: (46, 204, 113),  # Emerald Green
    CLASS_TOP: (52, 152, 219),        # Cyan / Blue
    CLASS_BOTTOM: (230, 126, 34)      # Amber / Orange
}
DEFAULT_COLOR = (189, 195, 199)

def is_overlapping_assembled(part_box, assembled_boxes, overlap_threshold=0.5):
    """Prevents double-counting an assembled unit as individual top/bottom pieces."""
    px1, py1, px2, py2 = part_box
    part_area = (px2 - px1) * (py2 - py1)
    if part_area <= 0:
        return False
    for ax1, ay1, ax2, ay2 in assembled_boxes:
        ix1 = max(px1, ax1)
        iy1 = max(py1, ay1)
        ix2 = min(px2, ax2)
        iy2 = min(py2, ay2)
        if ix2 > ix1 and iy2 > iy1:
            intersection_area = (ix2 - ix1) * (iy2 - iy1)
            if (intersection_area / part_area) > overlap_threshold:
                return True
    return False

# -------------------------------------------------------------
# Video Streaming & Inference Engine (Thread-Safe)
# -------------------------------------------------------------
class VideoCamera:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.cap = None
        self.is_running = False
        self.lock = threading.Lock()
        self.thread = None

        self.current_frame_jpeg = None
        self.fps = 0.0
        self.camera_status = "Stopped"  # "Online", "Stopped", "Error"
        self.error_message = ""

        # Live counters for current frame
        self.current_counts = {
            "assembled": 0,
            "top": 0,
            "bottom": 0,
            "total": 0
        }

        # Rolling history log (last 100 detections)
        self.detection_history = deque(maxlen=100)
        self.history_counter = 0

        # Load YOLO model once
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            self.error_message = f"Model file '{self.model_path}' not found."
            self.camera_status = "Error"
            print(f"[ERROR] {self.error_message}")
            return
        try:
            print(f"[INFO] Loading YOLO model from '{self.model_path}'...")
            self.model = YOLO(self.model_path)
            print("[INFO] Model loaded successfully.")
        except Exception as e:
            self.error_message = f"Failed to load YOLO model: {e}"
            self.camera_status = "Error"
            print(f"[ERROR] {self.error_message}")

    def start(self):
        with self.lock:
            if self.is_running:
                return True
            print(f"[INFO] Opening camera index {CAMERA_INDEX}...")
            # On Windows, try cv2.CAP_DSHOW for fast startup if available, fallback to default
            if sys.platform.startswith("win"):
                self.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(CAMERA_INDEX)
            else:
                self.cap = cv2.VideoCapture(CAMERA_INDEX)

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

            if not self.cap.isOpened():
                self.camera_status = "Error"
                self.error_message = f"Could not access camera at index {CAMERA_INDEX}."
                print(f"[ERROR] {self.error_message}")
                return False

            self.is_running = True
            self.camera_status = "Online"
            self.error_message = ""

            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            print("[INFO] Camera capture thread started.")
            return True

    def stop(self):
        with self.lock:
            if not self.is_running:
                return True
            self.is_running = False
            self.camera_status = "Stopped"
            self.fps = 0.0
            self.current_counts = {"assembled": 0, "top": 0, "bottom": 0, "total": 0}

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)

        with self.lock:
            if self.cap:
                self.cap.release()
                self.cap = None
            self.current_frame_jpeg = None
        print("[INFO] Camera stopped and device released.")
        return True

    def _capture_loop(self):
        prev_time = time.time()

        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("[WARNING] Frame capture dropped.")
                time.sleep(0.05)
                continue

            # Calculate FPS
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps > 0 else (1.0 / dt)

            # YOLO inference
            counts = {"assembled": 0, "top": 0, "bottom": 0, "total": 0}
            frame_detections = []

            if self.model:
                try:
                    results = self.model.predict(source=frame, conf=CONF_THRESHOLD, verbose=False)
                    result = results[0]

                    raw_boxes = []
                    assembled_boxes = []

                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        class_name = self.model.names.get(cls_id, f"Class {cls_id}")

                        raw_boxes.append((class_name, conf, (x1, y1, x2, y2)))
                        if class_name == CLASS_ASSEMBLED:
                            assembled_boxes.append((x1, y1, x2, y2))

                    # Filter and draw
                    for class_name, conf, (x1, y1, x2, y2) in raw_boxes:
                        # Rule: suppress sub-parts inside an assembled unit
                        if class_name in (CLASS_TOP, CLASS_BOTTOM):
                            if is_overlapping_assembled((x1, y1, x2, y2), assembled_boxes):
                                continue

                        if class_name == CLASS_ASSEMBLED:
                            counts["assembled"] += 1
                        elif class_name == CLASS_TOP:
                            counts["top"] += 1
                        elif class_name == CLASS_BOTTOM:
                            counts["bottom"] += 1

                        counts["total"] += 1
                        color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)

                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        # Draw header tag
                        label = f"{class_name.upper()} {conf * 100:.1f}%"
                        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                        tag_y = max(y1 - th - 8, 0)
                        cv2.rectangle(frame, (x1, tag_y), (x1 + tw + 8, tag_y + th + bl + 4), color, -1)
                        cv2.putText(frame, label, (x1 + 4, tag_y + th + 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

                        frame_detections.append({
                            "class": class_name,
                            "confidence": round(conf * 100, 1),
                            "box": [x1, y1, x2, y2]
                        })

                except Exception as e:
                    print(f"[ERROR] Inference failed: {e}")

            # Update live state
            with self.lock:
                self.current_counts = counts

                # Add new detections to history
                ts = datetime.now().strftime("%H:%M:%S")
                for det in frame_detections:
                    self.history_counter += 1
                    self.detection_history.appendleft({
                        "id": self.history_counter,
                        "timestamp": ts,
                        "class": det["class"],
                        "confidence": det["confidence"]
                    })

            # Overlay subtle on-screen watermark
            cv2.putText(frame, f"FPS: {self.fps:.1f} | INDUSTRIAL INSPECTION LIVE",
                        (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            # Encode as JPEG
            encode_ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if encode_ok:
                with self.lock:
                    self.current_frame_jpeg = buffer.tobytes()

            # Small sleep to yield CPU (crucial for Raspberry Pi efficiency)
            time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            if not self.is_running or self.current_frame_jpeg is None:
                return self._create_placeholder_frame()
            return self.current_frame_jpeg

    def _create_placeholder_frame(self):
        """Generates a stylish dark industrial placeholder when camera is stopped."""
        h, w = 480, 720
        placeholder = np.zeros((h, w, 3), dtype=np.uint8)
        placeholder[:] = (26, 32, 44)  # Slate dark background

        # Grid lines
        for y in range(0, h, 40):
            cv2.line(placeholder, (0, y), (w, y), (36, 44, 58), 1)
        for x in range(0, w, 40):
            cv2.line(placeholder, (x, 0), (x, h), (36, 44, 58), 1)

        msg1 = "CAMERA STANDBY"
        msg2 = "Click 'Start Camera' to initiate live YOLO inspection stream"
        if self.camera_status == "Error":
            msg1 = "CAMERA ERROR"
            msg2 = self.error_message or "Unable to open webcam index."

        col = (231, 76, 60) if self.camera_status == "Error" else (52, 152, 219)
        cv2.putText(placeholder, msg1, (w // 2 - 120, h // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
        cv2.putText(placeholder, msg2, (w // 2 - 270, h // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 174, 192), 1, cv2.LINE_AA)

        _, buf = cv2.imencode(".jpg", placeholder)
        return buf.tobytes()

    def get_telemetry(self):
        with self.lock:
            return {
                "status": self.camera_status,
                "fps": round(self.fps, 1),
                "counts": dict(self.current_counts),
                "error": self.error_message,
                "history": list(self.detection_history)[:30]
            }

    def clear_history(self):
        with self.lock:
            self.detection_history.clear()

# Initialize global camera manager
camera = VideoCamera(model_path=MODEL_PATH)

# -------------------------------------------------------------
# Flask Application & Routes
# -------------------------------------------------------------
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

def generate_stream():
    """Generator for MJPEG video stream."""
    while True:
        frame = camera.get_frame()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.04)  # ~25 FPS stream cap to conserve bandwidth & Pi CPU

@app.route("/video_feed")
def video_feed():
    return Response(generate_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/telemetry")
def api_telemetry():
    return jsonify(camera.get_telemetry())

@app.route("/api/start", methods=["POST"])
def api_start():
    ok = camera.start()
    return jsonify({"success": ok, "status": camera.camera_status, "error": camera.error_message})

@app.route("/api/stop", methods=["POST"])
def api_stop():
    ok = camera.stop()
    return jsonify({"success": ok, "status": camera.camera_status})

@app.route("/api/clear_history", methods=["POST"])
def api_clear_history():
    camera.clear_history()
    return jsonify({"success": True})

if __name__ == "__main__":
    print(f"============================================================")
    print(f"  YOLO Enclosure Detection Web Dashboard")
    print(f"  Running on: http://localhost:{PORT}")
    print(f"  Network access: http://<DEVICE_IP>:{PORT}")
    print(f"============================================================")
    app.run(host=HOST, port=PORT, debug=False, threaded=True)

