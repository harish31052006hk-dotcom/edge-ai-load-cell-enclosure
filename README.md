# Edge AI 3D Model Detection / Load Cell Enclosure Detection

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-YOLO26%20Small-green.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0-orange.svg)](https://opencv.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1.3-lightgrey.svg)](https://palletsprojects.com/p/flask/)

A real-time Edge AI computer vision and quality inspection system designed for detecting, tracking, and evaluating custom 3D mechanical load cell enclosure components.

---

## Project Overview

This repository contains the complete implementation of a computer vision inspection station built with **Ultralytics YOLO** and **OpenCV**, accompanied by a live industrial **Flask** web dashboard. The system detects individual components (`bottom`, `top`) and fully mated assemblies (`enclosure_assembled`), monitors live inspection telemetry, logs timestamped detection records, and provides camera calibration and sub-pixel edge detection capabilities.

---

## Objective

1. **Autonomous Part Identification**: Detect whether an enclosure on the inspection surface is an individual top cover, an individual bottom base, or a completed assembly.
2. **Real-Time Edge Telemetry**: Serve a low-latency live video stream with overlay tags, live FPS, and per-class part counts.
3. **Anti-Double-Counting**: Ensure that an assembled enclosure is never counted as an individual top or bottom component.
4. **Independent Metrological Calibration**: Provide physical pixel-to-millimeter calibration tools to verify dimensional limits ($32.0\text{ mm} \times 32.0\text{ mm}$).
5. **Cross-Platform Readiness**: Prepare the system for deployment on an overhead laptop workstation and future migration to **Raspberry Pi 5**.

---

## Target 3D Model

The physical inspection target is a modular, two-piece mechanical enclosure for load cell sensor protection:

| Component | Target Class | Nominal Dimensions | Description |
| :--- | :--- | :--- | :--- |
| **Bottom Base** | `bottom` | $32.0\text{ mm} \times 32.0\text{ mm}$ | Houses the sensor element and mounting points. |
| **Top Cover** | `top` | $32.0\text{ mm} \times 32.0\text{ mm}$ | Upper protective shield with sensor wire cutouts. |
| **Assembled Enclosure** | `enclosure_assembled` | $32.0\text{ mm} \times 32.0\text{ mm}$ | Mated assembly of top cover and bottom base. |

---

## Hardware

- **Development & Testing Host**: Windows 64-bit laptop (x86_64 CPU).
- **Vision Sensor**: Laptop webcam / USB UVC video camera ($1280 \times 720$ resolution).
- **Mounting Fixture**: Overhead laptop / camera inspection stand.
- **Planned Target Hardware**: **Raspberry Pi 5** (ARM64, headless edge deployment).
- *Hardware Accelerators*: None present; execution benchmarked purely on CPU.

---

## Software

- **Programming Language**: Python 3.14 (Virtual environment in `venv/`).
- **Deep Learning Framework**: `ultralytics` (v8.4.202) backed by PyTorch (`torch` v2.15.0+cpu, `torchvision` v0.29.2).
- **Computer Vision**: `opencv-python` (v5.0.0.93 `cv2`), `numpy` (v2.5.4).
- **Web Application & Telemetry**: `Flask` (v3.1.3), `Werkzeug` (v3.1.9), `Jinja2` (v3.1.6).

---

## Detection Pipeline

```
[ Physical Work Surface ]
          │
          ▼
[ Camera Acquisition: 1280x720 ]
          │
          ▼
[ Preprocessing & Noise Filtering ]
          │
          ▼
[ YOLO26 Small Inference (weights.pt) ]
          │
          ▼
[ Geometric Sub-Part Suppression Filter (No Double-Counting) ]
          │
          ▼
[ Telemetry & HUD Generator (FPS, Part Counts, Confidence) ]
          │
    ┌─────┴────────────────────────┐
    ▼                              ▼
[ Native Desktop GUI ]     [ Flask MJPEG Web Stream ]
```

---

## M1 - Baseline Implementation

Milestone 1 establishes the baseline edge computer vision pipeline, local hardware camera configuration, initial test datasets, and baseline object detection setup.

| Task | Status | Actual Evidence & Link |
| :--- | :--- | :--- |
| **Camera and Repository** | **Completed** | [src/app.py](./src/app.py), [src/live_detection.py](./src/live_detection.py), [evidence/M1/camera/](./evidence/M1/camera/) |
| **OpenCV Pipeline** | **Completed** | [src/calibration.py](./src/calibration.py) (Bilateral filter, Canny, Otsu, `minAreaRect`) |
| **Baseline Testing** | **Completed** | [benchmark/M1_baseline.md](./benchmark/M1_baseline.md), [docs/TESTING.md](./docs/TESTING.md) |
| **Dataset / Test Images** | **Completed** | [datasets/](./datasets/) (4 verified physical test photographs) |
| **Correct Dataset Split** | **Not completed / Not available** | Dataset was curated and split remotely on Roboflow; split manifests are not stored locally in this repository. |
| **Train and Run Detector**| **Completed** | [models/weights.pt](./models/weights.pt) (YOLO26 Small trained via Roboflow and verified) |

For comprehensive M1 details, see the [Milestone 1 Documentation](docs/M1.md).

---

## M2 - Detection and Evaluation

Milestone 2 evaluates detection performance, multi-class discrimination, anti-double-counting logic, real-time edge streaming, and metrological calibration checks.

| Task | Status | Actual Evidence & Link |
| :--- | :--- | :--- |
| **Model Export / Preparation** | **Not completed / Pending** | Checkpoint remains in PyTorch `.pt` format; export to ONNX / NCNN is documented but not executed. |
| **Convert and Run Live** | **Completed** | Full live execution in [src/live_detection.py](./src/live_detection.py) and [src/app.py](./src/app.py) |
| **Accuracy Check** | **Completed** | Confidences verified ($72.4\%\text{--}85.8\%$) across all physical test scenes in [docs/M2.md](./docs/M2.md) |
| **Fair Benchmark** | **Partially Completed** | Host laptop CPU latency evaluated; automated multi-batch loop not completed. |
| **CPU / Accelerator Eval** | **Not completed / Not measured** | Running on host CPU; Raspberry Pi 5 and hardware accelerators not measured yet. |
| **Target and Trade-off** | **Completed** | Geometric suppression and edge detection trade-offs evaluated in [benchmark/M2_evaluation.md](./benchmark/M2_evaluation.md) |

For comprehensive M2 details, see the [Milestone 2 Documentation](docs/M2.md).

---

## Testing

Comprehensive testing protocols and empirical results are detailed in [docs/TESTING.md](docs/TESTING.md).

Summary of verified physical test samples:
- `test_image.jpg`: `enclosure_assembled` detected at **85.8%** confidence.
- `bottom_test.jpg`: `bottom` detected at **78.9%** confidence.
- `top_test.jpg`: `top` detected at **83.1%** confidence.
- `top_bottom_separate_test.jpg`: `bottom` (**85.6%**) and `top` (**72.4%**) detected simultaneously.

---

## Results

1. **Multi-Class Accuracy**: Successfully detects and discriminates all three physical states with $>70\%$ confidence.
2. **Sub-Part Suppression**: Nested `top` or `bottom` proposals inside an assembled enclosure bounding box are automatically filtered out when intersection exceeds $50\%$.
3. **Host Inference Rate**: Steady-state CPU inference averages $100\text{--}140\text{ ms}$ ($\approx 7.1\text{--}10.0\text{ FPS}$) on the development laptop.
4. **Web Streaming**: Serves smooth MJPEG video stream with dynamic telemetry polling every $600\text{ ms}$.

---

## Evidence

All verified evidence files and screenshots are cataloged in [docs/EVIDENCE.md](docs/EVIDENCE.md):
- **M1 Evidence**: [evidence/M1/camera/](./evidence/M1/camera/), [evidence/M1/pipeline/](./evidence/M1/pipeline/), [evidence/M1/testing/](./evidence/M1/testing/), [evidence/M1/screenshots/](./evidence/M1/screenshots/)
- **M2 Evidence**: [evidence/M2/detection/](./evidence/M2/detection/), [evidence/M2/results/](./evidence/M2/results/), [evidence/M2/benchmark/](./evidence/M2/benchmark/), [evidence/M2/screenshots/](./evidence/M2/screenshots/)

---

## Limitations

1. **CPU Execution on Host**: Currently running PyTorch CPU inference without dedicated tensor accelerators.
2. **Perspective Foreshortening**: Non-perpendicular camera angles create perspective distortion, affecting physical millimeter measurement accuracy as shown in `config/calibration.json`.
3. **Dataset Lineage**: Full training sets and split manifests were managed via Roboflow and are not stored in local files.

---

## Conclusion

The project successfully demonstrates an operational Edge AI inspection pipeline for physical load cell enclosures. The trained YOLO26 Small model delivers robust multi-class detection, and the software stack provides both a standalone desktop application and a production-grade web dashboard with full hardware start/stop controls.

---

## Installation

1. Clone or download this repository:
   ```bash
   cd Edge_AI_Enclosure
   ```

2. Create and activate a Python virtual environment:
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - **Linux / Raspberry Pi**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Web Inspection Dashboard (Recommended)
```bash
python src/app.py
```
Open a browser and navigate to `http://localhost:5000`. Click **Start Camera** to begin live inspection.

### 2. Native Desktop Inspection Window
```bash
python src/live_detection.py
```
- Press <kbd>S</kbd> to save a timestamped detection snapshot to `measurements.csv`.
- Press <kbd>A</kbd> to perform automated edge calibration.
- Press <kbd>Q</kbd> to quit.

### 3. Interactive Camera Calibration
```bash
python src/camera_calibration.py
```

---

## Project Structure

```
Edge_AI_Enclosure/
├── benchmark/
│   ├── M1_baseline.md              # Baseline FPS, latency, and CPU benchmarks
│   └── M2_evaluation.md            # M2 detection evaluation and trade-offs
├── config/
│   └── calibration.json            # Camera calibration parameters
├── datasets/
│   ├── README.md                   # Dataset details and sample descriptions
│   ├── test_image.jpg              # Assembled enclosure test photograph
│   ├── bottom_test.jpg             # Bottom component test photograph
│   ├── top_test.jpg                # Top cover test photograph
│   └── top_bottom_separate_test.jpg# Dual-component test photograph
├── docs/
│   ├── M1.md                       # Milestone 1 documentation
│   ├── M2.md                       # Milestone 2 documentation
│   ├── TESTING.md                  # Comprehensive test procedures
│   └── EVIDENCE.md                 # Complete index mapping evidence files
├── evidence/
│   ├── M1/
│   │   ├── camera/                 # Camera framing specifications
│   │   ├── pipeline/               # Vision pipeline documentation
│   │   ├── testing/                # Raw test photographs
│   │   └── screenshots/            # Baseline UI screenshots
│   └── M2/
│       ├── detection/              # Verified coordinates and confidences
│       ├── results/                # Recorded calibration validation output
│       ├── benchmark/              # Benchmark latency notes
│       └── screenshots/            # Dashboard detection events & KPI captures
├── models/
│   ├── README.md                   # Checkpoint information and architecture
│   └── weights.pt                  # Trained YOLO26 Small PyTorch model (19.4 MB)
├── src/
│   ├── README.md                   # Source code guide
│   ├── app.py                      # Flask web dashboard application
│   ├── live_detection.py           # Native OpenCV detection script
│   ├── calibration.py              # OpenCV edge extraction & calibration engine
│   ├── camera_calibration.py       # Interactive camera calibration tool
│   ├── templates/
│   │   └── index.html              # Dashboard frontend template
│   └── static/
│       └── style.css               # Dashboard stylesheet
├── .gitignore                      # Git ignore rules for Python / Edge AI
├── README.md                       # Master assignment repository documentation
└── requirements.txt                # Exact Python dependencies
```

