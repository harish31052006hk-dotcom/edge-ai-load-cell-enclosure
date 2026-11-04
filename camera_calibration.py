"""
Camera Calibration & Independent Validation Tool
================================================
Project: Enclosure Dimensional Inspection
Hardware: Windows Laptop Webcam mounted on Fixed Overhead Stand
Target Specifications:
  - Calibration Target: Top Cover (known 32.0 mm x 32.0 mm)
  - Validation Target:  Bottom Base (known 32.0 mm x 32.0 mm)

Metrological Principle:
  Calibrating on the top cover and measuring that same top cover is not an
  independent accuracy test. Calibration must be verified on a separate physical
  sample (the bottom base component) before measurements are considered validated.
"""

import os
import sys
import time
import json
import cv2
import numpy as np

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
CALIBRATION_FILE = "calibration.json"
CAMERA_INDEX = 0             # 0 for default webcam, 1 for external camera
REQUESTED_WIDTH = 1280
REQUESTED_HEIGHT = 720
NOMINAL_DIMENSION_MM = 32.0  # Physical dimension of both Top and Bottom parts
VALIDATION_TOLERANCE_MM = 0.5 # Allowable validation tolerance (±0.5 mm)

def save_calibration_file(px_per_mm_x, px_per_mm_y, resolution, h_px, v_px, validation_data=None):
    """Saves calibration parameters and independent validation results to calibration.json."""
    data = {
        "px_per_mm_x": round(float(px_per_mm_x), 4),
        "px_per_mm_y": round(float(px_per_mm_y), 4),
        "calibration_target": "top_cover",
        "calibration_target_nominal_mm": NOMINAL_DIMENSION_MM,
        "measured_top_horizontal_px": round(float(h_px), 2),
        "measured_top_vertical_px": round(float(v_px), 2),
        "pixel_aspect_ratio_x_to_y": round(float(px_per_mm_x / px_per_mm_y), 4),
        "camera_resolution": list(resolution),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "validation": validation_data if validation_data else {
            "status": "PENDING_INDEPENDENT_VALIDATION",
            "message": "Top cover calibrated; bottom component validation pending."
        }
    }
    with open(CALIBRATION_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"\n[INFO] Configuration successfully saved to '{CALIBRATION_FILE}'.")
    return data

def capture_calibration_image():
    """
    Opens the webcam at 1280x720, reports the actual resolution,
    and allows the user to capture a calibration photograph.
    """
    print("\n[INFO] Connecting to webcam on fixed overhead stand...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUESTED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUESTED_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Unable to open camera (Index {CAMERA_INDEX}).")
        print("Please verify the camera connection and Windows privacy permissions.")
        return None, None

    # Query the actual resolution accepted by hardware/driver
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera stream opened.")
    print(f"       Requested Resolution: {REQUESTED_WIDTH} x {REQUESTED_HEIGHT}")
    print(f"       Actual Resolution   : {actual_w} x {actual_h}")

    print("\n" + "=" * 65)
    print("                OVERHEAD CAPTURE INSTRUCTIONS")
    print("=" * 65)
    print(" 1. Ensure laptop is mounted securely on the fixed overhead stand.")
    print(" 2. Place both components flat on the working surface:")
    print("     - TOP COVER (Calibration sample: known 32 x 32 mm)")
    print("     - BOTTOM BASE (Validation sample: known 32 x 32 mm)")
    print(" 3. Center both parts under the camera in the alignment box.")
    print(" 4. Press [SPACEBAR] to capture the photograph.")
    print(" 5. Press [Q] to quit.\n")

    captured_frame = None

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[ERROR] Frame grab failed.")
            break

        preview = frame.copy()
        h, w = preview.shape[:2]

        # Draw alignment reticle and working field
        cv2.line(preview, (w // 2, 0), (w // 2, h), (80, 80, 80), 1)
        cv2.line(preview, (0, h // 2), (w, h // 2), (80, 80, 80), 1)
        cv2.rectangle(preview, (w // 2 - 240, h // 2 - 240), (w // 2 + 240, h // 2 + 240), (0, 255, 255), 1)

        # Header HUD
        cv2.rectangle(preview, (10, 10), (660, 80), (20, 20, 20), -1)
        cv2.putText(preview, f"CAMERA PREVIEW - {actual_w}x{actual_h}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(preview, "Press [SPACE] to capture calibration photo | [Q] to quit", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("Overhead Camera Preview", preview)
        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # Spacebar
            captured_frame = frame.copy()
            print("[INFO] Photograph captured successfully.")
            break
        elif key == ord('q'):
            print("[INFO] Capture cancelled by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured_frame, (actual_w, actual_h)

def select_edge_endpoints(image, step_title, instruction, axis="horizontal"):
    """
    Interactive window allowing the user to click the two endpoints of a 32 mm edge.
    Supports resetting with 'R' and confirming with ENTER.
    """
    points = []
    window_name = f"Calibration: {step_title}"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    def mouse_callback(event, x, y, flags, param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 2:
                points.append((x, y))

    cv2.setMouseCallback(window_name, mouse_callback)

    print(f"\n>>> {step_title.upper()} <<<")
    print(f"Instruction: {instruction}")
    print(" - Left-click Endpoint 1 (corner of the 32 mm edge)")
    print(" - Left-click Endpoint 2 (opposite corner of the 32 mm edge)")
    print(" - Press [R] to Reset points if you make an error")
    print(" - Press [ENTER] to Confirm once two points are selected")
    print(" - Press [Q] to Cancel\n")

    while True:
        display = image.copy()

        # Instruction banner
        cv2.rectangle(display, (10, 10), (840, 80), (20, 20, 20), -1)
        cv2.putText(display, step_title, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)

        if len(points) == 0:
            cv2.putText(display, f"Click 1st corner of 32 mm {axis} edge | [R] Reset | [Q] Quit",
                        (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
        elif len(points) == 1:
            p1 = points[0]
            cv2.circle(display, p1, 4, (0, 0, 255), -1)
            cv2.putText(display, "P1", (p1[0] + 8, p1[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(display, f"Click 2nd corner of 32 mm {axis} edge | [R] Reset | [Q] Quit",
                        (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 200, 255), 1, cv2.LINE_AA)
        elif len(points) == 2:
            p1, p2 = points[0], points[1]
            cv2.circle(display, p1, 4, (0, 0, 255), -1)
            cv2.circle(display, p2, 4, (0, 0, 255), -1)
            cv2.putText(display, "P1", (p1[0] + 8, p1[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(display, "P2", (p2[0] + 8, p2[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(display, p1, p2, (0, 255, 0), 2, cv2.LINE_AA)

            dx = abs(p2[0] - p1[0])
            dy = abs(p2[1] - p1[1])
            dist_px = dx if axis == "horizontal" else dy
            euclidean = np.sqrt(dx**2 + dy**2)

            msg = f"Span: {dist_px:.1f} px along {axis} ({euclidean:.1f} px diagonal). Press [ENTER] to confirm."
            cv2.putText(display, msg, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('r'):
            points = []
            print("[INFO] Selection reset. Click again.")
        elif key in (13, 10):  # Enter
            if len(points) == 2:
                break
        elif key == ord('q'):
            cv2.destroyWindow(window_name)
            return None

    cv2.destroyWindow(window_name)
    p1, p2 = points[0], points[1]
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])
    return dx if axis == "horizontal" else dy

def validate_bottom_component(image, px_per_mm_x, px_per_mm_y):
    """
    Independent Validation Step:
    Allows user to select the separate BOTTOM component (known 32x32 mm)
    to independently verify calibration accuracy before declaring measurements valid.
    """
    print("\n" + "=" * 65)
    print("     INDEPENDENT METROLOGICAL VALIDATION (BOTTOM COMPONENT)")
    print("=" * 65)
    print("Principle:")
    print("  Measuring the calibrated top cover cannot serve as an independent test.")
    print("  We now inspect the separate physical BOTTOM component (32 x 32 mm).")
    print("Instructions:")
    print("  1. In the popup window, drag a tight bounding box around the BOTTOM BASE.")
    print("  2. Press [ENTER] or [SPACE] to compute dimensions.")
    print("  3. Press [C] or cancel selection if you need to exit.\n")

    window_name = "Independent Validation: Select Bottom Base (Drag box, then ENTER)"
    roi = cv2.selectROI(window_name, image, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)

    x, y, w_px, h_px = roi
    if w_px == 0 or h_px == 0:
        print("[WARNING] Validation selection cancelled. Status remains UNVALIDATED.")
        return None

    # Calculate physical dimensions of bottom component using top cover's calibration
    measured_w_mm = w_px / px_per_mm_x
    measured_h_mm = h_px / px_per_mm_y
    error_w = measured_w_mm - NOMINAL_DIMENSION_MM
    error_h = measured_h_mm - NOMINAL_DIMENSION_MM

    pass_w = abs(error_w) <= VALIDATION_TOLERANCE_MM
    pass_h = abs(error_h) <= VALIDATION_TOLERANCE_MM
    is_validated = pass_w and pass_h
    validation_status = "VALIDATED_PASS" if is_validated else "VALIDATION_FAILED"

    print("-" * 65)
    print("               INDEPENDENT VALIDATION REPORT")
    print("-" * 65)
    print(f"  Sample Under Test     : Bottom Component (Separate Sample)")
    print(f"  Nominal Dimensions    : {NOMINAL_DIMENSION_MM:.2f} mm x {NOMINAL_DIMENSION_MM:.2f} mm")
    print(f"  Tolerance Allowed     : +/- {VALIDATION_TOLERANCE_MM:.2f} mm ({NOMINAL_DIMENSION_MM - VALIDATION_TOLERANCE_MM:.1f} to {NOMINAL_DIMENSION_MM + VALIDATION_TOLERANCE_MM:.1f} mm)")
    print(f"  Measured Width        : {measured_w_mm:.2f} mm (Pixel: {w_px} px, Error: {error_w:+.2f} mm) -> {'OK' if pass_w else 'OUT OF TOLERANCE'}")
    print(f"  Measured Height       : {measured_h_mm:.2f} mm (Pixel: {h_px} px, Error: {error_h:+.2f} mm) -> {'OK' if pass_h else 'OUT OF TOLERANCE'}")
    print(f"  Validation Result     : {validation_status}")
    print("-" * 65)

    # Show visual result window
    display = image.copy()
    box_color = (0, 255, 0) if is_validated else (0, 0, 255)
    cv2.rectangle(display, (x, y), (x + w_px, y + h_px), box_color, 2)

    status_tag = f"BOTTOM VALIDATION: {validation_status}"
    dim_tag = f"W: {measured_w_mm:.2f}mm  H: {measured_h_mm:.2f}mm"
    err_tag = f"Err: W({error_w:+.2f}mm) H({error_h:+.2f}mm)"

    tag_y = max(y - 70, 10)
    cv2.rectangle(display, (x, tag_y), (x + 300, tag_y + 65), (20, 20, 20), -1)
    cv2.rectangle(display, (x, tag_y), (x + 300, tag_y + 65), box_color, 1)

    cv2.putText(display, status_tag, (x + 6, tag_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)
    cv2.putText(display, dim_tag, (x + 6, tag_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(display, err_tag, (x + 6, tag_y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.imshow("Independent Validation Result (Press any key to close)", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return {
        "status": validation_status,
        "validation_sample": "bottom_base",
        "nominal_dimension_mm": NOMINAL_DIMENSION_MM,
        "tolerance_mm": VALIDATION_TOLERANCE_MM,
        "measured_width_mm": round(measured_w_mm, 2),
        "measured_height_mm": round(measured_h_mm, 2),
        "error_width_mm": round(error_w, 2),
        "error_height_mm": round(error_h, 2),
        "pass_width": pass_w,
        "pass_height": pass_h,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

def main():
    print("=" * 65)
    print("   OVERHEAD WEBCAM CALIBRATION & INDEPENDENT VALIDATION")
    print("=" * 65)
    print("Project: YOLO Enclosure Dimensional Inspection")
    print("Target Size: 32.0 mm x 32.0 mm (Square Profile)")
    print("Calibration Component: TOP COVER (known 32 mm)")
    print("Independent Validation: BOTTOM BASE (known 32 mm)\n")

    # Step 1: Live preview and photograph capture
    frame, resolution = capture_calibration_image()
    if frame is None:
        print("[ABORTED] Calibration photograph was not captured.")
        sys.exit(0)

    # Step 2: Calibrate horizontal edge on Top Cover (32 mm)
    h_dist_px = select_edge_endpoints(
        frame,
        step_title="Step 1: Top Cover Known 32 mm HORIZONTAL Edge",
        instruction="Click the left and right corners of the TOP COVER horizontal edge.",
        axis="horizontal"
    )
    if h_dist_px is None or h_dist_px <= 0:
        print("[ABORTED] Horizontal edge calibration cancelled.")
        sys.exit(0)

    px_per_mm_x = h_dist_px / NOMINAL_DIMENSION_MM
    print(f"\n[CALCULATED] Horizontal: {h_dist_px:.2f} pixels for 32.0 mm = {px_per_mm_x:.4f} px/mm")

    # Step 3: Calibrate vertical edge on Top Cover (32 mm)
    v_dist_px = select_edge_endpoints(
        frame,
        step_title="Step 2: Top Cover Known 32 mm VERTICAL Edge",
        instruction="Click the top and bottom corners of the TOP COVER vertical edge.",
        axis="vertical"
    )
    if v_dist_px is None or v_dist_px <= 0:
        print("[ABORTED] Vertical edge calibration cancelled.")
        sys.exit(0)

    px_per_mm_y = v_dist_px / NOMINAL_DIMENSION_MM
    print(f"[CALCULATED] Vertical  : {v_dist_px:.2f} pixels for 32.0 mm = {px_per_mm_y:.4f} px/mm")

    # Step 4: Display Calibration Statistics
    print("\n" + "=" * 65)
    print("                 CALIBRATION VALUES (RAW)")
    print("=" * 65)
    print(f"  Horizontal Edge (32 mm) : {h_dist_px:.2f} px")
    print(f"  Vertical Edge (32 mm)   : {v_dist_px:.2f} px")
    print(f"  Horizontal Pixels / mm  : {px_per_mm_x:.4f} px/mm")
    print(f"  Vertical Pixels / mm    : {px_per_mm_y:.4f} px/mm")
    print(f"  Aspect Ratio (X/Y)      : {px_per_mm_x / px_per_mm_y:.4f}")
    print(f"  Active Resolution       : {resolution[0]} x {resolution[1]}")
    print("=" * 65)

    # Step 5: Independent Metrological Validation on the Bottom Base
    validation_report = validate_bottom_component(frame, px_per_mm_x, px_per_mm_y)

    # Step 6: Save parameters to calibration.json
    save_calibration_file(
        px_per_mm_x=px_per_mm_x,
        px_per_mm_y=px_per_mm_y,
        resolution=resolution,
        h_px=h_dist_px,
        v_px=v_dist_px,
        validation_data=validation_report
    )

    if validation_report and validation_report["status"] == "VALIDATED_PASS":
        print("\n[SUCCESS] Calibration validated on independent bottom component.")
        print("          Measurements are verified and ready for live inspection.")
    else:
        print("\n[CAUTION] System is calibrated but independent validation did not pass or was skipped.")
        print("          Check camera height, perpendicularity, or lighting before production use.")

if __name__ == "__main__":
    main()
