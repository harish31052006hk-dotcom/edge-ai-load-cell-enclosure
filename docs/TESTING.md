# Testing Protocol & Verification Guide

This document defines the physical and software testing procedures for the Edge AI Load Cell Enclosure detection system.

---

## 1. Test Equipment & Environment

- **Host Machine:** Windows 64-bit Laptop
- **Video Input:** Built-in webcam / USB UVC Camera (Index `0`)
- **Python Environment:** Virtual environment (`venv/`)
- **Model Checkpoint:** [`models/weights.pt`](../models/weights.pt)

---

## 2. Test Cases & Verification Results

### Test Case 1: Model Loading & Class Registry
- **Objective:** Verify checkpoint parses without corrupt weights and registers the 3 classes.
- **Command:**
  ```powershell
  python -c "from ultralytics import YOLO; m = YOLO('models/weights.pt'); print(m.names)"
  ```
- **Expected:** `{0: 'bottom', 1: 'enclosure_assembled', 2: 'top'}`
- **Observed:** Correct mapping verified.
- **Status:** **PASS**

### Test Case 2: Assembled Enclosure Detection
- **Sample:** [`datasets/test_image.jpg`](../datasets/test_image.jpg)
- **Observed:** `enclosure_assembled` detected at **85.8%** confidence.
- **Status:** **PASS**

### Test Case 3: Individual Bottom Base Detection
- **Sample:** [`datasets/bottom_test.jpg`](../datasets/bottom_test.jpg)
- **Observed:** `bottom` detected at **78.9%** confidence.
- **Status:** **PASS**

### Test Case 4: Individual Top Cover Detection
- **Sample:** [`datasets/top_test.jpg`](../datasets/top_test.jpg)
- **Observed:** `top` detected at **83.1%** confidence.
- **Status:** **PASS**

### Test Case 5: Simultaneous Multi-Part Scene
- **Sample:** [`datasets/top_bottom_separate_test.jpg`](../datasets/top_bottom_separate_test.jpg)
- **Observed:**
  - `bottom`: **85.6%**
  - `top`: **72.4%**
  - No false assembled enclosure classification triggered.
- **Status:** **PASS**

### Test Case 6: Sub-Part Anti-Double-Counting Logic
- **Objective:** Verify that when an assembled enclosure is present, its internal visible features are not double-counted as individual loose parts.
- **Mechanism:** `is_overlapping_assembled()` in `src/app.py` and `src/live_detection.py` computes intersection-over-part-area. If overlap exceeds $50\%$, loose part proposals are suppressed.
- **Status:** **PASS**

---

## 3. How to Execute Tests

### Running Native Desktop Inspection
```powershell
cd Edge_AI_Enclosure
.\venv\Scripts\Activate.ps1
python src/live_detection.py
```
- Press <kbd>S</kbd> to record timestamped measurements to `measurements.csv`.
- Press <kbd>A</kbd> to trigger auto-edge calibration on a visible $32\text{ mm}$ component.
- Press <kbd>Q</kbd> to exit.

### Running Web Inspection Dashboard
```powershell
python src/app.py
```
- Open browser at `http://localhost:5000`.
- Click **Start Camera** to initiate live MJPEG video streaming.
- Click **Stop Camera** to release webcam hardware.

