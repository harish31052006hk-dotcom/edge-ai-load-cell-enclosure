# Milestone 1: Baseline Benchmark & Latency Notes

This document records the baseline execution performance of the YOLO26 Small model and the vision capture pipeline on the development host.

---

## 1. Benchmark Environment

| Specification | Details |
| :--- | :--- |
| **Host Architecture** | x86_64 Windows Laptop CPU |
| **Execution Framework** | PyTorch v2.15.0+cpu / Ultralytics v8.4.202 |
| **Model Evaluated** | [`models/weights.pt`](../models/weights.pt) (YOLO26 Small, 19.39 MB) |
| **Frame Dimensions** | $1280 \times 720$ (native acquisition) |

---

## 2. Baseline Measurements

| Metric | Measured Value | Notes |
| :--- | :--- | :--- |
| **Cold Start / Initialization** | $\approx 7.9\text{ s}$ | Includes initial Torch weight unpacking and memory allocation |
| **Steady-State CPU Inference** | $\approx 100\text{--}140\text{ ms}$ | Measured across single frame predictions |
| **Effective Inference Rate** | $\approx 7.1\text{--}10.0\text{ FPS}$ | Running purely on host CPU without GPU acceleration |
| **Web Streaming Target Rate** | $\approx 25\text{ FPS}$ | MJPEG generator sleep cap ($0.04\text{ s}$) to prevent network saturation |

---

## 3. Unavailable / Pending Benchmarks

* **Automated Multi-Batch Benchmark:** *Not completed / Not measured* (automated latency benchmarking across hundreds of batches has not been run yet).
* **GPU Acceleration (CUDA / TensorRT):** *Not available* on current test environment.
* **Power & Thermal Consumption:** *Not measured* on host machine.

