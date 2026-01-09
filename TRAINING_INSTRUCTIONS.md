# UNINA-DLA: Model Training & Deployment Instructions

This guide provides step-by-step instructions for training the UNINA-DLA detector, from initial teacher training to DLA-optimized ONNX export.

## Prerequisites

- **Environment**: Ensure you have installed the requirements from `requirements.txt`.
- **Hardware**: Training is recommended on a machine with a CUDA-enabled GPU (≥ RTX 2070). Deployment targets the **NVIDIA Jetson Orin AGX DLA**.
- **Dataset**: Your dataset should be organized in YOLO format and described in `unina_dla/config/unina_dla_data.yaml`.

---

## Phase 0: Teacher Training

We use a high-capacity teacher (**YOLOv10-X**) to guide the training of our lightweight, DLA-compatible student model.

```bash
python scripts/train_teacher.py --data unina_dla/config/unina_dla_data.yaml --epochs 300 --batch 16
```

- **Output**: The best weights will be saved in `runs/unina_dla_teacher/yolov10x_fsg/weights/best.pt`.
- **Note**: This teacher uses advanced augmentations (Mosaic, Mixup) and SiLU activations, which are not DLA-compatible but provide a strong performance baseline.

---

## Phase 1: Tri-Vector Student Distillation

In this phase, we train the **UNINA-DLA** model (student). It uses structural re-parameterization (RepVGG), ReLU activations, and a DLA-optimized neck/head. We use knowledge distillation (SDF, Logit, and DFL losses) to transfer knowledge from the teacher.

```bash
python scripts/train_distillation.py \
    --teacher runs/unina_dla_teacher/yolov10x_fsg/weights/best.pt \
    --data unina_dla/config/unina_dla_data.yaml \
    --epochs 100 \
    --batch 16 \
    --exp_name unina_dla
```

- **Objective**: Match the feature maps and prediction distributions of the teacher while maintaining a DLA-native architecture.
- **Checkpoints**: Saved in `checkpoints/best.pth`.

---

## Phase 2: Quantization-Aware Training (QAT)

To run efficiently in INT8 precision on the DLA without losing accuracy, we perform QAT. This simulates quantization noise during fine-tuning.

```bash
python scripts/train_qat.py \
    --checkpoint checkpoints/best.pth \
    --data unina_dla/config/unina_dla_data.yaml \
    --epochs 10
```

- **Process**:
    1. **Fuse**: Merges RepVGG branches into a single path.
    2. **Calibrate**: Uses a subset of data to determine optimal quantization scales (Entropy Calibration).
    3. **Fine-tune**: Adapts weights to quantization errors using the original FP32 student as a reference (Self-Distillation).

---

## Phase 3: DLA-Optimized ONNX Export

The final step is to export the model to ONNX. We apply strict **Static Shape** constraints and use a **DLA Wrapper** to ensure the output format is optimized for zero-copy DLA inference.

### Option A: Export FP32 (No QAT)
```bash
python scripts/export_onnx.py --checkpoint checkpoints/best.pth --output unina_dla_fp32.onnx
```

### Option B: Export QAT (INT8 Optimized)
```bash
python scripts/export_onnx.py --checkpoint checkpoints/unina_dla_qat_epoch_10.pth --output unina_dla_qat.onnx --qat
```

---

## Phase 4: Verification

### 1. DLA Compatibility Check
Check if there are any unsupported layers that might cause GPU fallback:
```bash
python scripts/check_model_properties.py --onnx unina_dla_qat.onnx
```

### 2. Validation
Evaluate the performance on your validation set:
```bash
python scripts/val.py --checkpoint checkpoints/best.pth --data unina_dla/config/unina_dla_data.yaml
```

---

## Summary of Operations
| Operation | Goal | Target Hardware |
| :--- | :--- | :--- |
| **Teacher Training** | Establish performance ceiling | GPU (Cloud/Desktop) |
| **Distillation** | Compress knowledge into student | GPU (Cloud/Desktop) |
| **QAT** | Prepare for INT8 precision | GPU (Cloud/Desktop) |
| **Export** | Generate deployment artifact | Jetson Orin (DLA) |
