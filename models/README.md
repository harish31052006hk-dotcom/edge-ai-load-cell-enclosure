# Model Zoo & Weights Repository

This directory contains the trained neural network checkpoint for the Load Cell Enclosure detection project.

## Checkpoint Information

- **Model File:** [`weights.pt`](./weights.pt)
- **File Size:** **19.39 MB** (20,337,157 bytes)
- **Model Architecture:** **YOLO26 Small** (`yolo26s.yaml` DetectionModel)
- **Training Environment:** Trained via **Roboflow** cloud training pipeline (`roboflow_train` metadata tag)
- **Framework Compatibility:** Ultralytics YOLO (`ultralytics>=8.0.0`) and PyTorch (`torch>=2.0.0`)

## Target Classes

| Class ID | Class Name | Description | Color Code (BGR) |
| :--- | :--- | :--- | :--- |
| `0` | `bottom` | Lower base piece of the enclosure | `(230, 126, 34)` (Orange/Amber) |
| `1` | `enclosure_assembled` | Fully assembled enclosure unit | `(46, 204, 113)` (Emerald Green) |
| `2` | `top` | Upper cover piece of the enclosure | `(52, 152, 219)` (Cyan/Blue) |

## Model Export & Optimization Status

| Format | Target Platform | Status |
| :--- | :--- | :--- |
| **PyTorch (`.pt`)** | Windows Host / General Python | **Available** (`weights.pt`) |
| **ONNX (`.onnx`)** | Cross-platform acceleration | *Not completed / Not measured* |
| **NCNN / OpenVINO** | Raspberry Pi 5 CPU accelerator | *Not completed / Not measured* |
| **TFLite / INT8** | Low-power Edge TPU | *Not completed / Not measured* |

To export the model for edge deployment in the future:
```bash
yolo export model=models/weights.pt format=onnx imgsz=640
```

