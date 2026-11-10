"""
Camera Calibration & Automated Edge Measurement Module
======================================================
Automates edge isolation and physical pixel-to-millimetre calibration
for the 32 x 32 mm enclosure inspection project.
"""

import os
import sys
import time
import json
import cv2
import numpy as np

CALIBRATION_FILE = "calibration.json"
FALLBACK_CALIBRATION_FILE = "calibration_config.json"
NOMINAL_ENCLOSURE_MM = 32.0  # Known physical dimensions: 32 mm x 32 mm

class CameraCalibration:
    def __init__(self, config_path=None):
        if config_path is not None:
            self.config_path = config_path
        elif os.path.exists(CALIBRATION_FILE):
            self.config_path = CALIBRATION_FILE
        elif os.path.exists(FALLBACK_CALIBRATION_FILE):
            self.config_path = FALLBACK_CALIBRATION_FILE
        else:
            self.config_path = CALIBRATION_FILE

        self.is_calibrated = False
        self.px_per_mm_x = None
        self.px_per_mm_y = None
        self.calibrated_resolution = None
        self.calibration_timestamp = None
        self.validation_status = None
        self.load()

    def load(self):
        """Loads calibration parameters from JSON config if available."""
        if not os.path.exists(self.config_path):
            self.is_calibrated = False
            return False

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.px_per_mm_x = float(data.get("px_per_mm_x", 0))
            self.px_per_mm_y = float(data.get("px_per_mm_y", 0))
            self.calibrated_resolution = tuple(data.get("camera_resolution", data.get("resolution", (1280, 720))))
            self.calibration_timestamp = data.get("timestamp", "Unknown")
            val_info = data.get("validation", {})
            self.validation_status = val_info.get("status", "UNVALIDATED") if isinstance(val_info, dict) else str(val_info)

            if self.px_per_mm_x > 0 and self.px_per_mm_y > 0:
                self.is_calibrated = True
                return True
            else:
                self.is_calibrated = False
                return False
        except Exception as e:
            print(f"[WARNING] Could not load calibration: {e}")
            self.is_calibrated = False
            return False

    def save_auto_calibration(self, px_per_mm_x, px_per_mm_y, resolution, h_px, v_px, source_class="top"):
        """Saves automated calibration values to calibration.json without manual typing."""
        data = {
            "px_per_mm_x": round(float(px_per_mm_x), 4),
            "px_per_mm_y": round(float(px_per_mm_y), 4),
            "calibration_source": f"auto_edge_detection ({source_class})",
            "nominal_target_mm": NOMINAL_ENCLOSURE_MM,
            "measured_edge_x_px": round(float(h_px), 2),
            "measured_edge_y_px": round(float(v_px), 2),
            "pixel_aspect_ratio": round(float(px_per_mm_x / px_per_mm_y), 4),
            "camera_resolution": list(resolution),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "validation": {
                "status": "AUTO_CALIBRATED_PENDING_INDEPENDENT_VALIDATION",
                "note": "Automated calibration complete using known 32 mm enclosure. Validate against physical calipers."
            }
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        self.load()
        print(f"\n[INFO] Automated calibration saved to '{self.config_path}'.")
        return data

    def pixels_to_mm(self, width_px, height_px):
        """Converts pixel dimensions to millimetres using calibration."""
        if not self.is_calibrated:
            return None, None
        w_mm = width_px / self.px_per_mm_x
        h_mm = height_px / self.px_per_mm_y
        return round(w_mm, 2), round(h_mm, 2)


# -------------------------------------------------------------
# Automated OpenCV Edge Extraction Algorithm
# -------------------------------------------------------------
def extract_object_outer_edges(frame, yolo_box, margin=10):
    """
    Automatically isolates the visible outer edges of an enclosure part within
    the YOLO bounding box using OpenCV contour and rotated rectangle analysis.

    Returns:
        success (bool): True if reliable, clean outer edges were detected.
        dimensions (tuple): (width_px, height_px) of the physical edges.
        box_points (np.ndarray): 4 corners of the oriented rectangle (for drawing).
        diagnostic_msg (str): Reason if edge extraction was unreliable.
    """
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = yolo_box

    # Expand slightly around the YOLO box to capture entire outer perimeter
    rx1 = max(0, x1 - margin)
    ry1 = max(0, y1 - margin)
    rx2 = min(W, x2 + margin)
    ry2 = min(H, y2 + margin)

    roi = frame[ry1:ry2, rx1:rx2]
    if roi.size == 0 or roi.shape[0] < 25 or roi.shape[1] < 25:
        return False, None, None, "ROI too small"

    # 1. Grayscale & Bilateral filtering (preserves edges while removing noise)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 7, 50, 50)

    # 2. Multi-threshold edge identification
    # Candidate A: Otsu automatic thresholding
    _, thresh_otsu = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Candidate B: Canny edges + morphological closing
    edges = cv2.Canny(filtered, 40, 130)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Gather candidate contours from both edge representations
    candidate_contours = []
    for mask in [thresh_otsu, cv2.bitwise_not(thresh_otsu), closed_edges]:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidate_contours.extend(cnts)

    yolo_box_area = (x2 - x1) * (y2 - y1)
    min_area = 0.35 * yolo_box_area
    max_area = 1.40 * yolo_box_area

    valid_contours = [c for c in candidate_contours if min_area <= cv2.contourArea(c) <= max_area]
    if not valid_contours:
        return False, None, None, "No distinct outer perimeter found"

    # Select the contour with the largest area that matches enclosure geometry
    best_contour = max(valid_contours, key=cv2.contourArea)

    # Fit a minimum area rotated rectangle (accounts for part orientation on table)
    (cx_roi, cy_roi), (rw, rh), angle = cv2.minAreaRect(best_contour)

    if rw <= 10 or rh <= 10:
        return False, None, None, "Edge dimensions too small"

    # Aspect ratio check: The enclosure is physically square (32x32 mm).
    # Its pixel aspect ratio must be close to 1.0 (between 0.75 and 1.33).
    aspect_ratio = max(rw, rh) / float(min(rw, rh))
    if aspect_ratio > 1.35:
        return False, None, None, f"Contour skewed (aspect ratio {aspect_ratio:.2f})"

    # Fill density check: Solidity of the contour inside its oriented rectangle
    rect_area = rw * rh
    contour_area = cv2.contourArea(best_contour)
    if rect_area > 0 and (contour_area / rect_area) < 0.60:
        return False, None, None, "Edge contour not sufficiently solid"

    # Translate local ROI coordinates back to global camera frame coordinates
    global_center = (cx_roi + rx1, cy_roi + ry1)
    global_rot_rect = (global_center, (rw, rh), angle)
    box_points = cv2.boxPoints(global_rot_rect)
    box_points = np.int32(box_points)

    # Determine horizontal vs vertical span based on rotation
    if abs(angle) < 45 or abs(angle) > 135:
        w_px, h_px = rw, rh
    else:
        w_px, h_px = rh, rw

    return True, (w_px, h_px), box_points, "Reliable"
