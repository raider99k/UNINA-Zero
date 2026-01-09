# UNINA-DLA: Training & Deployment Guide

A complete guide to training the UNINA-DLA cone detector for Formula Student Driverless.

## Prerequisites

| Requirement | Details |
|:---|:---|
| **OS** | Linux (Ubuntu 22.04) recommended. WSL2 works on Windows. |
| **GPU** | CUDA-enabled (≥ RTX 2070, 8GB VRAM) |
| **Python** | 3.10+ |
| **Dependencies** | `pip install -r requirements.txt` |
| **Dataset** | YOLO format, configured in `unina_dla/config/unina_dla_data.yaml` |

---

## Training Workflow

The training consists of **two main steps**:

```
┌──────────────────┐      ┌─────────────────────────────────────────┐
│ 1. Train Teacher │ ───▶ │ 2. Train Student (Distill → QAT → ONNX) │
└──────────────────┘      └─────────────────────────────────────────┘
```

---

## Step 1: Train the Teacher

Train a high-capacity **YOLOv10-X** model. This is a one-time cost per dataset.

```bash
python scripts/train_teacher.py \
    --data unina_dla/config/unina_dla_data.yaml \
    --epochs 300 \
    --batch 32
```

**Output**: `runs/unina_dla_teacher/yolov10x_fsg/weights/best.pt`

> [!TIP]
> **RTX 4090 User**: You can comfortably increase the batch size to **32** or **48** for teacher training. The 24GB VRAM also allows for high-resolution validation concurrently.
> The teacher uses SiLU activations and advanced augmentations (Mosaic/Mixup) to maximize accuracy. These are intentionally not DLA-compatible.

---

## Step 2: Train the Student (Unified Pipeline)

A single command automates the complete student pipeline:

| Phase | Description |
|:---|:---|
| **1. Distillation** | Tri-Vector knowledge transfer (SDF, Logit, DFL) |
| **2. QAT** | Quantization-Aware Training for INT8 precision |
| **3. Export** | DLA-optimized ONNX with static shapes |

```bash
python scripts/train_student.py \
    --teacher runs/unina_dla_teacher/yolov10x_fsg/weights/best.pt \
    --data unina_dla/config/unina_dla_data.yaml \
    --epochs_distill 100 \
    --epochs_qat 10 \
    --batch 32 \
    --onnx_output unina_dla.onnx
```

### Outputs
The training script generates both model checkpoints and visual validation artifacts in the `--output_dir` (default: `checkpoints/`).

#### 1. Models
| File | Description |
|:---|:---|
| `checkpoints/best_distill.pth` | Best distilled student (FP32) |
| `checkpoints/qat_final.pth` | QAT-trained student (INT8-ready) |
| `unina_dla.onnx` | Final deployment artifact |

#### 2. Visualizations
Automatic "YOLO-style" images are generated for inspection:
| File | Description |
|:---|:---|
| `train_batch0.jpg` | First training batch with Ground Truth labels |
| `results_distill.png` | Loss and mAP curves for Phase 1 |
| `results_qat.png` | Loss and mAP curves for Phase 2 |
| `confusion_matrix_*.png` | Matrix showing class-wise precision and recall |

---

## Advanced Options

### Skip Phases
```bash
# Skip distillation (use existing checkpoint)
python scripts/train_student.py \
    --skip_distillation \
    --resume checkpoints/best_distill.pth \
    --epochs_qat 10

# Skip QAT (export distilled model as FP32)
python scripts/train_student.py \
    --teacher <path> \
    --skip_qat
```

### All Arguments
| Argument | Default | Description |
|:---|:---|:---|
| `--teacher` | *Required* | Path to teacher weights |
| `--data` | `unina_dla/config/unina_dla_data.yaml` | Dataset config |
| `--epochs_distill` | `100` | Distillation epochs |
| `--epochs_qat` | `10` | QAT fine-tuning epochs |
| `--batch` | `16` | Batch size |
| `--skip_distillation` | `False` | Skip Phase 1 |
| `--skip_qat` | `False` | Skip Phase 2 |
| `--resume` | `None` | Checkpoint to resume from |
| `--output_dir` | `checkpoints` | Directory for checkpoints |
| `--onnx_output` | `unina_dla.onnx` | ONNX output path |

---

## Verification

### 1. DLA Compatibility Check
```bash
python scripts/check_model_properties.py --onnx unina_dla.onnx
```

### 2. Validation (mAP Evaluation)
```bash
python scripts/val.py \
    --checkpoint checkpoints/best_distill.pth \
    --data unina_dla/config/unina_dla_data.yaml
```

---

## Deployment to Jetson Orin

After training, convert the ONNX to a TensorRT engine on the Jetson:

```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=unina_dla.onnx \
    --saveEngine=unina_dla.engine \
    --useDLACore=0 \
    --int8 \
    --allowGPUFallback
```

> [!IMPORTANT]
> Use `--allowGPUFallback` only for debugging. For production, verify 100% DLA residency with `nsys profile`.

---

## Quick Reference

| Task | Command |
|:---|:---|
| Train Teacher | `python scripts/train_teacher.py --data <yaml> --epochs 300` |
| Train Student | `python scripts/train_student.py --teacher <pt> --data <yaml>` |
| Validate | `python scripts/val.py --checkpoint <pth> --data <yaml>` |
| Export Only | `python scripts/export_onnx.py --checkpoint <pth> --output <onnx>` |
