# Evidence Index & Artifact Mapping

This document provides a verified index of all physical evidence, screenshots, test files, and configuration artifacts present in the repository.

---

## 1. Milestone 1 Evidence Directory (`evidence/M1/`)

### Camera Setup & Framing (`evidence/M1/camera/`)
- Optical alignment specifications: $1280 \times 720$ resolution target, normal viewing angle.
- Overhead stand alignment reticle implemented in [`src/camera_calibration.py`](../src/camera_calibration.py).

### Vision Pipeline (`evidence/M1/pipeline/`)
- Preprocessing and bounding box rendering pipeline implemented in [`src/live_detection.py`](../src/live_detection.py) and [`src/app.py`](../src/app.py).

### Baseline Test Images (`evidence/M1/testing/`)
1. [`evidence/M1/testing/test_image.jpg`](../evidence/M1/testing/test_image.jpg) - Assembled enclosure photograph.
2. [`evidence/M1/testing/bottom.png`](../evidence/M1/testing/bottom.png) - Bottom component photograph.
3. [`evidence/M1/testing/top.png`](../evidence/M1/testing/top.png) - Top cover photograph.
4. [`evidence/M1/testing/top and bottom.png`](../evidence/M1/testing/top and bottom.png) - Simultaneous dual-part photograph.

### Baseline Interface Screenshots (`evidence/M1/screenshots/`)
1. [`evidence/M1/screenshots/Screenshot 2026-11-10 173520.png`](../evidence/M1/screenshots/Screenshot%202026-11-10%20173520.png) - Web inspection dashboard overview.
2. [`evidence/M1/screenshots/Screenshot 2026-11-10 173549.png`](../evidence/M1/screenshots/Screenshot%202026-11-10%20173549.png) - Live stream interface and controls.
3. [`evidence/M1/screenshots/Screenshot 2026-11-10 173636.png`](../evidence/M1/screenshots/Screenshot%202026-11-10%20173636.png) - KPI metric cards and telemetry status.

---

## 2. Milestone 2 Evidence Directory (`evidence/M2/`)

### Detection Verification (`evidence/M2/detection/`)
Verified bounding box coordinates and confidence ratings:
- `test_image.jpg`: `enclosure_assembled` ($85.8\%$, box $[107, 323, 742, 698]$)
- `bottom.png`: `bottom` ($78.9\%$, box $[68, 556, 739, 929]$)
- `top.png`: `top` ($83.1\%$, box $[310, 244, 928, 814]$)
- `top and bottom.png`: `bottom` ($85.6\%$) & `top` ($72.4\%$)

### Results & Calibration (`evidence/M2/results/`)
- [`evidence/M2/results/calibration.json`](../evidence/M2/results/calibration.json) - Recorded calibration run and independent validation report.

### Benchmark Data (`evidence/M2/benchmark/`)
- Documentation on measured host latency and Raspberry Pi 5 trade-offs in [`benchmark/M2_evaluation.md`](../benchmark/M2_evaluation.md).

### Evaluation Screenshots (`evidence/M2/screenshots/`)
1. [`evidence/M2/screenshots/Screenshot 2026-11-10 173646.png`](../evidence/M2/screenshots/Screenshot%202026-11-10%20173646.png) - Real-time detection event log with timestamps.
2. [`evidence/M2/screenshots/Screenshot 2026-11-10 173736.png`](../evidence/M2/screenshots/Screenshot%202026-11-10%20173736.png) - Multi-part live detection monitoring.
3. [`evidence/M2/screenshots/Screenshot 2026-11-10 173746.png`](../evidence/M2/screenshots/Screenshot%202026-11-10%20173746.png) - Quality status and confidence bar ratings.

