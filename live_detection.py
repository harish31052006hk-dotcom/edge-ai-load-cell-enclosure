import os
import sys
import time
import csv
import cv2
import numpy as np
from ultralytics import YOLO

# Import automated edge extraction and calibration engine
from calibration import CameraCalibration, extract_object_outer_edges, NOMINAL_ENCLOSURE_MM

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
MODEL_PATH = "weights.pt"
CAMERA_INDEX = 0          # Default laptop webcam (change to 1 if using an external USB camera)
CONF_THRESHOLD = 0.50     # Minimum confidence threshold (50%)

# Target dimensions for quality inspection
TARGET_DIMENSION_MM = NOMINAL_ENCLOSURE_MM   # 32.0 mm x 32.0 mm
TOLERANCE_MM = 0.5                           # Configurable tolerance (±0.5 mm)

# File for logging inspection records
CSV_LOG_FILE = "measurements.csv"

# Target classes
CLASS_ASSEMBLED = "enclosure_assembled"
CLASS_TOP = "top"
CLASS_BOTTOM = "bottom"

# Color definitions (BGR format for OpenCV)
CLASS_COLORS = {
    CLASS_BOTTOM: (255, 128, 0),         # Sky Blue / Cyan-Orange
    CLASS_TOP: (0, 200, 255),            # Yellow-Orange
    CLASS_ASSEMBLED: (0, 255, 0)        # Green
}
STATUS_COLORS = {
    "PASS": (0, 255, 0),                 # Bright Green
    "FAIL": (0, 0, 255),                 # Red
    "UNCALIBRATED": (0, 165, 255)        # Amber / Orange
}
DEFAULT_COLOR = (200, 200, 200)

def is_overlapping_assembled(part_box, assembled_boxes, overlap_threshold=0.5):
    """
    Check if a candidate top/bottom part box substantially overlaps with an
    assembled enclosure box. Prevents double-counting an assembled unit as individual parts.
    """
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

def log_measurement(records):
    """Appends measurement records with timestamps to CSV file."""
    if not records:
        return False

    file_exists = os.path.exists(CSV_LOG_FILE)
    try:
        with open(CSV_LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "Timestamp", "Class", "Edge_Width_px", "Edge_Height_px",
                    "Width_mm", "Height_mm", "Target_mm", "Tolerance_mm",
                    "Status", "Confidence"
                ])
            for rec in records:
                writer.writerow([
                    rec["timestamp"],
                    rec["class_name"],
                    f"{rec['edge_w_px']:.1f}" if rec['edge_w_px'] else "N/A",
                    f"{rec['edge_h_px']:.1f}" if rec['edge_h_px'] else "N/A",
                    f"{rec['width_mm']:.2f}" if rec["width_mm"] is not None else "UNCALIBRATED",
                    f"{rec['height_mm']:.2f}" if rec["height_mm"] is not None else "UNCALIBRATED",
                    rec["target_mm"],
                    rec["tolerance_mm"],
                    rec["status"],
                    f"{rec['confidence']:.2f}"
                ])
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save log: {e}")
        return False

def main():
    # 1. Verify model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file '{MODEL_PATH}' was not found in the current directory.")
        print("Please verify that 'weights.pt' is located in the project folder.")
        sys.exit(1)

    # 2. Load the trained YOLO model
    print(f"[INFO] Loading YOLO model from '{MODEL_PATH}'...")
    try:
        model = YOLO(MODEL_PATH)
        print("[INFO] Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO model: {e}")
        sys.exit(1)

    # 3. Load Camera Calibration
    calibration = CameraCalibration()
    if calibration.is_calibrated:
        print(f"[INFO] Camera Calibration Active:")
        print(f"       Horizontal: {calibration.px_per_mm_x:.2f} px/mm | Vertical: {calibration.px_per_mm_y:.2f} px/mm")
        print(f"       Calibrated at: {calibration.calibration_timestamp}")
    else:
        print("[WARNING] SYSTEM UNCALIBRATED.")
        print("          Place a 32x32 mm enclosure part under the camera and press 'A' to AUTO-CALIBRATE.")

    # 4. Open the webcam using OpenCV
    print(f"[INFO] Connecting to camera (Index: {CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    FRAME_W, FRAME_H = 1280, 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        print(f"\n[ERROR] Unable to access camera at index {CAMERA_INDEX}.")
        print("Troubleshooting tips:")
        print(" 1. Ensure no other application (Zoom, Teams, Camera app) is using the webcam.")
        print(" 2. Check Windows Camera Privacy Settings (Settings > Privacy & Security > Camera).")
        print(" 3. If using an external USB camera, try changing CAMERA_INDEX = 1 in the script.\n")
        sys.exit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera connected ({actual_w}x{actual_h}).")
    print("[INFO] Keyboard Shortcuts:")
    print("       'A' - Automatically calibrate camera using detected 32x32 mm enclosure edges")
    print("       'S' - Save/Log current measurements with timestamp to CSV")
    print("       'Q' - Quit")

    prev_time = time.time()
    fps = 0.0
    notification_msg = ""
    notification_timer = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARNING] Failed to grab frame from camera. Exiting...")
            break

        # Calculate live FPS
        curr_time = time.time()
        time_diff = curr_time - prev_time
        prev_time = curr_time
        if time_diff > 0:
            fps = 1.0 / time_diff

        # Run YOLO inference
        results = model.predict(source=frame, conf=CONF_THRESHOLD, verbose=False)
        result = results[0]

        # First pass: collect raw detections
        raw_detections = []
        assembled_boxes = []

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = model.names.get(cls_id, f"Class {cls_id}")

            raw_detections.append((class_name, conf, (x1, y1, x2, y2)))
            if class_name == CLASS_ASSEMBLED:
                assembled_boxes.append((x1, y1, x2, y2))

        # Counters
        assembled_count = 0
        top_count = 0
        bottom_count = 0

        # Store detected valid edges for auto-calibration and CSV logging
        current_frame_records = []
        auto_calib_candidates = []

        # Draw optical alignment zone in center of screen
        guide_w, guide_h = 360, 360
        gx1 = (actual_w - guide_w) // 2
        gy1 = (actual_h - guide_h) // 2
        gx2 = gx1 + guide_w
        gy2 = gy1 + guide_h
        cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (70, 70, 70), 1)
        cv2.putText(frame, "OPTICAL ALIGNMENT ZONE (Normal to Camera)", (gx1 + 10, gy1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 130, 130), 1, cv2.LINE_AA)

        # Second pass: Process detections with automated edge extraction
        for class_name, conf, (x1, y1, x2, y2) in raw_detections:
            # Rule: Suppress top/bottom parts if they are part of an assembled unit
            if class_name in (CLASS_TOP, CLASS_BOTTOM):
                if is_overlapping_assembled((x1, y1, x2, y2), assembled_boxes):
                    continue

            # Update counters exclusively
            if class_name == CLASS_ASSEMBLED:
                assembled_count += 1
            elif class_name == CLASS_TOP:
                top_count += 1
            elif class_name == CLASS_BOTTOM:
                bottom_count += 1

            # -------------------------------------------------------------
            # Requirement 3: Automatically identify visible outer edges
            # -------------------------------------------------------------
            edge_ok, edge_dims, edge_box_pts, diag_msg = extract_object_outer_edges(
                frame, (x1, y1, x2, y2), margin=12
            )

            width_mm, height_mm = None, None
            edge_w_px, edge_h_px = None, None

            if edge_ok:
                edge_w_px, edge_h_px = edge_dims
                # Save candidate for auto-calibration hotkey 'A'
                auto_calib_candidates.append((class_name, edge_w_px, edge_h_px))

                # Draw high-precision oriented outer edge rectangle
                cv2.drawContours(frame, [edge_box_pts], 0, (0, 255, 255), 2)

                # Requirement 7, 8, 9, 10: Dimensional measurement evaluation
                if calibration.is_calibrated:
                    width_mm, height_mm = calibration.pixels_to_mm(edge_w_px, edge_h_px)
                    w_error = abs(width_mm - TARGET_DIMENSION_MM)
                    h_error = abs(height_mm - TARGET_DIMENSION_MM)

                    if (w_error <= TOLERANCE_MM) and (h_error <= TOLERANCE_MM):
                        status = "PASS"
                    else:
                        status = "FAIL"
                else:
                    status = "UNCALIBRATED"
            else:
                # If edges could not be reliably isolated
                status = "UNCALIBRATED" if not calibration.is_calibrated else "UNRELIABLE EDGES"

            # Record for logging
            current_frame_records.append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "class_name": class_name,
                "edge_w_px": edge_w_px,
                "edge_h_px": edge_h_px,
                "width_mm": width_mm,
                "height_mm": height_mm,
                "target_mm": TARGET_DIMENSION_MM,
                "tolerance_mm": TOLERANCE_MM,
                "status": status,
                "confidence": conf
            })

            # Bounding box color based on status
            box_color = STATUS_COLORS.get(status, (0, 165, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Build informational label tags
            line1 = f"{class_name} {conf * 100:.1f}%"
            if status in ("PASS", "FAIL") and width_mm is not None:
                line2 = f"W:{width_mm:.1f}mm H:{height_mm:.1f}mm [{status}]"
            elif status == "UNRELIABLE EDGES":
                line2 = f"Edges: {diag_msg} [FAIL]"
            else:
                line2 = f"Dim: N/A [UNCALIBRATED]"

            font = cv2.FONT_HERSHEY_SIMPLEX
            (t1_w, t1_h), _ = cv2.getTextSize(line1, font, 0.52, 2)
            (t2_w, t2_h), _ = cv2.getTextSize(line2, font, 0.52, 2)
            tag_w = max(t1_w, t2_w) + 8
            tag_h = t1_h + t2_h + 12

            label_y = max(y1 - tag_h - 4, 0)
            cv2.rectangle(frame, (x1, label_y), (x1 + tag_w, label_y + tag_h), (20, 20, 20), -1)
            cv2.rectangle(frame, (x1, label_y), (x1 + tag_w, label_y + tag_h), box_color, 1)

            cv2.putText(frame, line1, (x1 + 4, label_y + t1_h + 2), font, 0.52, CLASS_COLORS.get(class_name, DEFAULT_COLOR), 1, cv2.LINE_AA)
            cv2.putText(frame, line2, (x1 + 4, label_y + tag_h - 4), font, 0.52, box_color, 2, cv2.LINE_AA)

        # ---------------------------------------------------------
        # HUD Panel: FPS, Counters, Status, and Automated Calibration
        # ---------------------------------------------------------
        assembled_status = "DETECTED" if assembled_count > 0 else "NOT DETECTED"
        top_status = "DETECTED" if top_count > 0 else "NOT DETECTED"
        bottom_status = "DETECTED" if bottom_count > 0 else "NOT DETECTED"

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (490, 205), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

        # 1. Header: FPS & Calibration Status
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        if calibration.is_calibrated:
            calib_str = f"CALIBRATED ({calibration.px_per_mm_x:.2f}x{calibration.px_per_mm_y:.2f} px/mm)"
            calib_col = (0, 255, 0)
        else:
            calib_str = "UNCALIBRATED (Press 'A' to Auto-Calibrate)"
            calib_col = (0, 165, 255)
        cv2.putText(frame, calib_str, (150, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, calib_col, 2, cv2.LINE_AA)

        # 2. Target Spec
        cv2.putText(frame, f"Inspection Target: {TARGET_DIMENSION_MM:.1f} x {TARGET_DIMENSION_MM:.1f} mm  (Tol: +/-{TOLERANCE_MM:.1f} mm)",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

        # 3. Counters & Status
        cv2.putText(frame, f"Assembled: {assembled_count}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"[{assembled_status}]", (300, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0) if assembled_count > 0 else (130, 130, 130), 2, cv2.LINE_AA)

        cv2.putText(frame, f"Top Parts: {top_count}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"[{top_status}]", (300, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 200, 255) if top_count > 0 else (130, 130, 130), 2, cv2.LINE_AA)

        cv2.putText(frame, f"Bottom Parts: {bottom_count}", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 128, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"[{bottom_status}]", (300, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 128, 0) if bottom_count > 0 else (130, 130, 130), 2, cv2.LINE_AA)

        # 4. Shortcut instructions
        cv2.putText(frame, "Hotkeys: 'A' Auto-Calibrate (32mm) | 'S' Log CSV | 'Q' Quit", (20, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        # Notification Toast
        if time.time() < notification_timer:
            cv2.putText(frame, notification_msg, (20, 235),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        # Show stream
        cv2.imshow("Automated Enclosure Detection & Edge Inspection (Press Q to Quit)", frame)

        # Handle Keyboard Input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[INFO] 'Q' pressed. Exiting...")
            break
        elif key == ord('s'):
            if current_frame_records:
                success = log_measurement(current_frame_records)
                if success:
                    notification_msg = f"[SAVED {len(current_frame_records)} RECORD(S) TO {CSV_LOG_FILE}]"
                    notification_timer = time.time() + 2.5
                    print(f"[INFO] Logged {len(current_frame_records)} measurement(s) to '{CSV_LOG_FILE}'.")
            else:
                print("[WARNING] No detected parts in frame to log.")
        elif key == ord('a'):
            # Requirement 4 & 5: Automate calibration from detected 32 mm outer edges
            if auto_calib_candidates:
                source_cls, cand_w, cand_h = auto_calib_candidates[0]
                px_per_mm_x = cand_w / TARGET_DIMENSION_MM
                px_per_mm_y = cand_h / TARGET_DIMENSION_MM

                calibration.save_auto_calibration(
                    px_per_mm_x=px_per_mm_x,
                    px_per_mm_y=px_per_mm_y,
                    resolution=(actual_w, actual_h),
                    h_px=cand_w,
                    v_px=cand_h,
                    source_class=source_cls
                )
                notification_msg = f"[AUTO-CALIBRATED: {px_per_mm_x:.2f} x {px_per_mm_y:.2f} px/mm]"
                notification_timer = time.time() + 3.0
                print(f"[SUCCESS] Auto-calibration locked from '{source_cls}' ({cand_w:.1f}x{cand_h:.1f} px). Saved to calibration.json.")
            else:
                notification_msg = "[AUTO-CALIBRATE FAILED: No reliable outer edges in view]"
                notification_timer = time.time() + 2.5
                print("[WARNING] Auto-calibrate failed. Ensure a 32x32 mm enclosure part is centered with clear outer edges.")

    # Clean release
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera released and windows closed successfully.")

if __name__ == "__main__":
    main()
