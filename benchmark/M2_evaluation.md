# Milestone 2: Detection Evaluation & Trade-offs

This document details the trade-offs, target considerations, detection quality, and hardware evaluation for the Load Cell Enclosure detector.

---

## 1. Detection Quality & Class Discrimination

### Confusion Mitigation
When detecting a multi-part mechanical enclosure, the assembled unit presents exterior features identical to the top cover and bottom base.
- **Problem:** Standard YOLO non-maximum suppression (NMS) evaluates classes independently. A high-confidence `enclosure_assembled` detection often co-occurs with overlapping `top` and `bottom` detections.
- **Solution:** A geometric containment check (`is_overlapping_assembled()`) was implemented:
  $$\text{Intersection Ratio} = \frac{\text{Area}(\text{Box}_{\text{part}} \cap \text{Box}_{\text{assembled}})}{\text{Area}(\text{Box}_{\text{part}})}$$
  If $\text{Intersection Ratio} > 0.50$, the nested sub-part is suppressed, eliminating duplicate counts on assembled units.

---

## 2. Hardware & Architecture Trade-off Analysis

| Target Platform | Runtime Format | Expected Latency | Optimization Required | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Windows Laptop CPU** | PyTorch `.pt` | $\approx 100\text{--}140\text{ ms}$ | None (Baseline) | **Tested** |
| **Raspberry Pi 5 (CPU)** | PyTorch `.pt` | $\approx 250\text{--}400\text{ ms}$ | High CPU usage | *Pending Deployment* |
| **Raspberry Pi 5 (ONNX)** | ONNX Runtime | $\approx 90\text{--}150\text{ ms}$ | `yolo export format=onnx` | *Not completed / Not measured* |
| **Raspberry Pi 5 (NCNN)** | NCNN (Vulkan/ARM) | $\approx 40\text{--}70\text{ ms}$ | `yolo export format=ncnn` | *Not completed / Not measured* |
| **Hailo-8L / AI Kit** | HEF format | $\approx 10\text{--}20\text{ ms}$ | Hailo Dataflow Compiler | *Not completed / Hardware absent* |

---

## 3. Metrological Calibration Trade-offs

Observations from [`config/calibration.json`](../config/calibration.json):
1. **Camera Angle:** Non-perpendicular viewing creates an asymmetric aspect ratio ($X/Y = 3.85$), skewing physical dimension conversion.
2. **Lighting & Part Reflections:** Glare on plastic/metal enclosures creates edge noise during Canny segmentation. Diffuse overhead lighting is recommended.
3. **Verification Policy:** Measurements are kept marked as unvalidated until certified by a physical Vernier caliper.

