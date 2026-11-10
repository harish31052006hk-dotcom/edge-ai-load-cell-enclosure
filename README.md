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
## Milestone Overview

| Milestone | Focus | Status | Evidence |
|---|---|---|---|
| M1 | Edge AI model, camera pipeline and dataset | Complete | [evidence/M1](docs/M1.md) |
| M2 | Detection evaluation and CPU baseline | Complete | [evidence/M2](docs/M2.md) |
| M3 | Real-time decision, debounce and actuation | Evidence imported | [evidence/M3](evidence/M3/README.md) |
| M4 | Sensor monitoring, anomaly detection and industrial communication | Evidence imported | [evidence/M4](evidence/M4/README.md) |
| M5 | Integrated inspection system and dashboard | Evidence imported | [evidence/M5](evidence/M5/README.md) |

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
| **Correct Dataset Split** | **Complete & Verified — Split Distribution** | [evidence/M1/05_correct_split/M1_Dataset_Split_Report.md](./evidence/M1/05_correct_split/M1_Dataset_Split_Report.md) |
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
# Milestone 3 — Real-Time Safety / Actuation System

Milestone 3 focuses on implementing real-time decision-making, dual-stream processing, temporal debounce filtering, and physical actuator triggering based on edge detection results.

### Purpose of M3
The goal of M3 is to translate raw visual detections into reliable physical control signals without false-triggering or chatter caused by transient detections or momentary occlusions.

### System Architecture & Implemented Components
- **Decision Rule / Logic**: Evaluates component presence and assembly conditions against defined pass/fail safety rules. In the group safety setup, it verifies worker compliance with safety items before activating operational indicators. Evidence is documented in the [dashboard helmet detection HUD](evidence/M3/group/decision-rule-sheet/dashboard%28helemt%29.jpeg).
- **Two-Stream Pipeline**: Captures and processes multi-source streams concurrently on edge hardware. Implemented using live camera ingestion on a Raspberry Pi 5 node, verified via [Arducam live frame capture](evidence/M3/group/two-stream-pipeline/from_raspberrypi_camera.jpeg) and the [Roboflow annotation workbench](evidence/M3/group/two-stream-pipeline/roboflow_annotation.jpeg).
- **Debounce Mechanism**: A temporal consistency filter that requires consecutive matching detection frames across a time window before committing a state change, preventing false alerts. Verified in multi-class compliance state ([compliant worker detection](evidence/M3/group/debounce/both%20glove%20and%20helmet.jpeg)).
- **Actuator / LED Firing**: Drives physical GPIO outputs upon validated rule transitions. Demonstrated via hardware LED actuation on the Raspberry Pi GPIO pins, captured in [3-LED actuation video](evidence/M3/group/actuator-fires/3_led_blinking_video.mp4).
- **Docker Status**: Docker containerization was not demonstrated with visual artifacts in the shared repository. No visual evidence is available in the imported teammate repository.
- **End-to-End Latency**: Qualitative timing is documented through the [dashboard screen recording video](evidence/M3/group/end-to-end-latency/dashboard_screen_record%28m3%29.mp4); no standalone numerical benchmark or stopwatch latency figure is independently claimed.

### Milestone 3 Evidence Index
All imported M3 evidence files are organized in [evidence/M3](evidence/M3/README.md).  
*Attribution note: Visual media in M3 are group evidence (group run on the shared Pi rig — my part: filming the group run — filmed by me).*

---

# Milestone 4 — Sensor Monitoring & Anomaly Detection

Milestone 4 extends the edge node with multi-sensor physical telemetry ingestion, statistical and machine learning anomaly detection, secure industrial IoT communication, and reliability hardening.

### Purpose of M4
The objective of M4 is condition monitoring of electromechanical components (such as servo motors driving the inspection rig) using multi-modal sensor streams to detect degradation, stalls, or abnormal operating conditions before catastrophic failure.

### Implemented Components & Evaluation
- **Sensor Data Acquisition**: Multi-channel 10 Hz / 100 Hz serial telemetry stream from an Arduino Uno acquiring MPU6050 vibration, ACS712 current, DC bus voltage, and MAX9814 acoustic data. Evidence includes [sensor waveforms plot](evidence/M4/group/sensor-data/sensor_waveforms_plot.png), [live telemetry dashboard](evidence/M4/group/sensor-data/dashboard_live_telemetry.jpeg), and [serial monitor stream](evidence/M4/group/sensor-data/serial_monitor_streaming_frame.png).
- **Anomaly Detection / Model**: Employs sliding-window multivariate feature extraction coupled with regularized statistical Z-Score / Isolation Forest scoring. Verified via [anomaly score trajectory chart](evidence/M4/group/anomaly-model/anomaly_score_trajectory.png) and [live anomaly detection event](evidence/M4/group/anomaly-model/dashboard_anomaly_detected.jpeg).
- **False-Alarm Evaluation**: Tuned noise floor regularization (FEATURE_STD_FLOORS) and calibrated current thresholds reflect a group-reported 0.00% false-alarm rate across normal baseline cycles in the group evaluation, supported by the [confusion matrix heatmap](evidence/M4/group/false-alarms/confusion_matrix_heatmap.png) and [anomaly score distribution](evidence/M4/group/false-alarms/anomaly_score_distribution.png).
- **Secure MQTT**: Hardened MQTTS on port 8883 utilizing TLS encryption and X.509 certificate authentication. Verified in [Mosquitto TLS configuration](evidence/M4/group/secure-mqtt/mosquitto_secure.conf).
- **MQTT Topic Hierarchy**: Structured 7-topic taxonomy (motor/servo01/...) adhering to ISO 8601 timestamps, QoS policies, and schema specifications, illustrated in [topic hierarchy tree](evidence/M4/group/topics/topic_hierarchy_tree.png).
- **PLC / Modbus Connection**: Industrial Modbus TCP slave bridge mapping 16-bit registers (Input, Holding, Coils) for supervisory control and E-Stop interlock, documented in [Modbus register map table](evidence/M4/group/plc-link/modbus_register_map_table.png).
- **Threat Model (STRIDE / DREAD)**: Comprehensive security threat analysis evaluating 8 attack vectors across serial, network, and host boundaries, detailed in [STRIDE matrix](evidence/M4/group/threat-model/threat_model_stride_matrix.png).
- **Reliability & Service Monitoring**: System resilience verified through serial auto-reconnection (<5s recovery), SQLite atomic batching, and systemd watchdog supervision shown in [systemd service status](evidence/M4/group/reliability/service_status_systemd.png).

### Milestone 4 Evidence Index
All imported M4 verification artifacts are indexed in [evidence/M4](evidence/M4/README.md).  
*Attribution note: Results and metrics in M4 are group-reported from the shared teammate instrumentation bench.*

---

# Milestone 5 — Integrated Edge AI Inspection System

Milestone 5 connects the vision inspection subsystem (M1/M2/M3) and the condition monitoring subsystem (M4) as an integrated inspection-system demonstration.

### Purpose of M5
To demonstrate an integrated edge inspection setup combining computer vision inspection, physical pass/fail sorting, electromechanical condition monitoring, and edge telemetry.

### Integrated Subsystems & Performance
- **Inspection Rig**: Physical testing bench combining the camera imaging assembly, workpiece staging surface, and edge compute hardware. Documented in [physical camera feed](evidence/M5/group/inspection-rig/01_physical_camera_feed.jpeg) and [Raspberry Pi 5 edge node photo](evidence/M5/group/inspection-rig/03_raspberry_pi_5_edge_node.jpeg).
- **Pass/Fail Station**: Automated quality assessment station evaluating parts against nominal specifications and logging inspection decisions. Backed by [assessment benchmark screenshot](evidence/M5/group/pass-fail-station/06_assessment_benchmarks.png) and [20-sample pass/fail results CSV](evidence/M5/group/pass-fail-station/pass_fail_20_results.csv).
- **Full Device Processing Chain**: Continuous 10-minute end-to-end execution of the combined vision and sensor pipeline running directly on the edge device is group-reported, verified in [10-minute run log](evidence/M5/group/full-chain-on-the-device/10_minute_chain_run_log.txt) and [chain run report](evidence/M5/group/full-chain-on-the-device/10_minute_chain_report.json).
- **Live Dashboard**: Unified industrial web dashboard displaying synchronized camera feeds, detection HUD, motor condition telemetry oscilloscope, and anomaly alarm centers. Documented via [dashboard overview](evidence/M5/group/live-dashboard/01_dashboard_overview.png), [vision inspection rig panel](evidence/M5/group/live-dashboard/02_vision_inspection_rig.png), and [motor anomaly alert display](evidence/M5/group/live-dashboard/04_live_motor_anomaly_alert.png).
- **Product Readiness & System Health**: Real-time diagnostic monitoring tracking CPU load, memory utilization, disk usage, and temperature to verify operational readiness, recorded in [system health status](evidence/M5/group/product-readiness/07_system_health.png).
- **Headline Metric**: Statistical sorting results across 50 consecutive cycles are group-reported and documented in the imported [headline metric 50 results CSV](evidence/M5/group/headline-metric/headline_metric_50_results.csv).
- **Documents & Live Demo Status**: Architectural datasheets and operating workflows are preserved in teammate documentation; no dedicated standalone visual recordings exist for live demo Q&A in the imported repository.

### Milestone 5 Evidence Index
All imported M5 artifacts are indexed in [evidence/M5](evidence/M5/README.md).  
*Attribution note: Hardware setup media represents group evidence on the shared rig; operational logs and performance metrics are group-reported.*

---


## Testing

Comprehensive testing protocols and empirical results are detailed in [docs/TESTING.md](docs/TESTING.md).

Summary of verified physical test samples:
- `test_image.jpg`: `enclosure_assembled` detected at **85.8%** confidence.
- `bottom.png`: `bottom` detected at **78.9%** confidence.
- `top.png`: `top` detected at **83.1%** confidence.
- `top and bottom.png`: `bottom` (**85.6%**) and `top` (**72.4%**) detected simultaneously.

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
│   ├── bottom.png             # Bottom component test photograph
│   ├── top.png                # Top cover test photograph
│   └── top and bottom.png# Dual-component test photograph
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

