# Source Code Module Documentation

The `src/` directory contains all application code, computer vision pipelines, calibration algorithms, and web dashboard logic.

## Module Directory

### 1. Web Application (`app.py`)
- **Path:** [`app.py`](./app.py)
- **Role:** Industrial Flask web dashboard and threaded streaming server.
- **Key Classes / Functions:**
  - `VideoCamera`: Thread-safe capture and inference controller managing `cv2.VideoCapture` and `YOLO`.
  - `/video_feed`: High-efficiency multipart MJPEG video streaming route.
  - `/api/telemetry`: Real-time JSON telemetry delivering live FPS, current part counts, and rolling event logs.
  - `/api/start` & `/api/stop`: Safe hardware start/release controls preventing camera lockups.
  - `is_overlapping_assembled()`: Geometric suppression filter preventing assembled units from double-counting as loose parts.

### 2. Standalone Desktop Detection (`live_detection.py`)
- **Path:** [`live_detection.py`](./live_detection.py)
- **Role:** High-performance native OpenCV inspection application.
- **Features:**
  - Real-time HUD displaying instantaneous and rolling FPS.
  - Automated edge boundary detection with `extract_object_outer_edges()`.
  - Key bindings: <kbd>A</kbd> (Auto-calibrate on $32\text{ mm}$ part), <kbd>S</kbd> (Save CSV snapshot), <kbd>Q</kbd> (Exit).

### 3. Edge Extraction & Calibration Math (`calibration.py`)
- **Path:** [`calibration.py`](./calibration.py)
- **Role:** Algorithmic computer vision engine for dimensional inspection.
- **Features:**
  - `CameraCalibration`: Loads and saves $X$ and $Y$ pixel-to-millimeter ratios to `config/calibration.json`.
  - `extract_object_outer_edges()`: Bilateral filtering + Otsu adaptive thresholding + Canny edge maps + `cv2.minAreaRect` to isolate oriented physical component boundaries.

### 4. Interactive Overhead Calibration Tool (`camera_calibration.py`)
- **Path:** [`camera_calibration.py`](./camera_calibration.py)
- **Role:** Interactive two-point reference calibration and independent sample validation tool.
- **Metrological Principle:** Uses the known Top Cover ($32\text{ mm}$) to calibrate the camera and the independent Bottom Base ($32\text{ mm}$) to validate measurement accuracy.

### 5. Web Frontend Templates & Assets
- **HTML Template:** [`templates/index.html`](./templates/index.html) (Industrial dark dashboard UI with KPI counters and event log table).
- **CSS Stylesheet:** [`static/style.css`](./static/style.css) (Dark theme, responsive layout, pulse indicators, and typography).

