"""
UNINA-DLA Unified Student Training Pipeline.

This script automates:
    Phase 1: Knowledge Distillation (Teacher -> Student)
    Phase 2: Quantization-Aware Training (QAT)
    Phase 3: DLA-Optimized ONNX Export

Usage:
    python scripts/train_student.py --teacher runs/teacher/best.pt --data unina_dla/config/unina_dla_data.yaml
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from unina_dla.model.unina_dla import UNINA_DLA
from unina_dla.model.teacher_model import get_teacher_model
from unina_dla.model.losses.distillation_losses import SDFDistillationLoss, LogitDistillationLoss, DFLDistillationLoss
from unina_dla.utils.validator import validate
from unina_dla.utils.qat_utils import prepare_qat_model, enable_calibration, enable_quantization
from unina_dla.model.losses.v10_loss import v10DetectionLoss
try:
    from pytorch_quantization import nn as quant_nn
except ImportError:
    quant_nn = None

from ultralytics.data import build_yolo_dataset, build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG, IterableSimpleNamespace
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.plotting import Annotator, colors


# ============================================================================
# UTILITIES: VISUALIZATION
# ============================================================================
def save_batch_image(batch, filename, names):
    """Save a batch of images with ground truth boxes for inspection."""
    imgs = batch['img'].numpy()  # [B, 3, H, W]
    targets = batch['cls']  # [N, 6] (batch_idx, cls, x, y, w, h)
    
    # Find first image in batch that has targets, or default to 0
    target_idx = 0
    unique_indices = torch.unique(targets[:, 0])
    if len(unique_indices) > 0:
        target_idx = int(unique_indices[0].item())
    
    # Take selected image
    img = imgs[target_idx].transpose(1, 2, 0)
    img = (img * 1.0).astype(np.uint8) # Ultralytics dataloader might give 0-255 uint8 or float
    # Ensure it's 0-255 uint8 for PIL
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    
    h, w, _ = img.shape
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    
    # Filter targets for the selected image
    img_targets = targets[targets[:, 0] == target_idx]
    
    for t in img_targets:
        cls_id = int(t[1])
        # x, y, w, h are normalized
        cx, cy, bw, bh = t[2], t[3], t[4], t[5]
        x1 = (cx - bw/2) * w
        y1 = (cy - bh/2) * h
        x2 = (cx + bw/2) * w
        y2 = (cy + bh/2) * h
        
        color = (255, 0, 0) # Default red
        label = names.get(cls_id, f"class_{cls_id}")
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, y1 - 10), label, fill=color)
    
    img_pil.save(filename)
    print(f"Batch inspection image saved to {filename}")


def plot_results(metrics, filename, title="Training Metrics"):
    """Plot evolution of Loss and mAP."""
    epochs = range(1, len(metrics['loss']) + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, metrics['loss'], color=color, marker='o', label='Total Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('mAP50', color=color)
    ax2.plot(epochs, metrics['map'], color=color, marker='s', label='mAP@50')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(title)
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Results plot saved to {filename}")


# ============================================================================
# PHASE 1: DISTILLATION
# ============================================================================
class YOLOv8LossAdapter:
    """
    Adapter to use YOLOv8/v10 loss logic with the custom UNINA_DLA model.
    Mocks the expected Ultralytics model structure.
    """
    def __init__(self, real_model, cfg):
        self.device = next(real_model.parameters()).device
        
        class MockHead:
            def __init__(self, nc, reg_max, stride):
                self.nc = nc
                self.reg_max = reg_max
                self.stride = torch.tensor(stride).to(real_model.parameters().__next__().device)
                self.no = nc + reg_max * 4

        class MockModel:
            def __init__(self, args, head_attrs):
                self.args = args
                self.model = [MockHead(**head_attrs)]
            def parameters(self):
                return real_model.parameters()

        head = real_model.head
        head_attrs = {
            'nc': head.num_classes, 
            'reg_max': head.reg_max, 
            'stride': [8., 16., 32.]
        }
        
        self.mock_model = MockModel(cfg, head_attrs)
        # v10DetectionLoss initializes self.proj using head.reg_max from MockModel
        # CRITICAL: Force student_only=True (O2O) to resolve Dual Assignment conflict on Single Head
        self.loss_fn = v10DetectionLoss(self.mock_model, student_only=True)

    def __call__(self, preds, batch):
        reg_outs, cls_outs = preds
        # Merge reg and cls outputs for each scale to match loss function expectation
        feats = [torch.cat([r, c], dim=1) for r, c in zip(reg_outs, cls_outs)]
        return self.loss_fn(feats, batch)


def run_distillation(args, device, data_cfg, train_loader, val_loader, writer):
    print("\n" + "="*60)
    print("PHASE 1: TRI-VECTOR KNOWLEDGE DISTILLATION")
    print("="*60)
    
    # Load Teacher
    print(f"Loading Teacher from {args.teacher}...")
    teacher_wrapper = get_teacher_model(args.teacher)
    if teacher_wrapper is None:
        raise RuntimeError(f"Failed to load teacher model from {args.teacher}")
    teacher_model = teacher_wrapper.model.to(device)
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()
    
    # Teacher Hooks
    t_outputs = {'feats': [], 'cls': [], 'box': []}
    
    def get_activation(storage_list):
        def hook(model, input, output):
            storage_list.append(output)
        return hook

    head = teacher_model.model[-1]
    head_from = head.f
    
    # INDAGINE: Verify Teacher Head Structure
    print(f"Teacher Head Type: {type(head)}")
    print(f"Teacher Head Attributes: {[a for a in dir(head) if not a.startswith('_')]}")
    
    hooks = []
    for i, idx in enumerate(head_from):
        layer = teacher_model.model[idx]
        hooks.append(layer.register_forward_hook(get_activation(t_outputs['feats'])))
        if hasattr(head, 'cv2') and len(head.cv2) > i:
            hooks.append(head.cv2[i].register_forward_hook(get_activation(t_outputs['box'])))
        if hasattr(head, 'cv3') and len(head.cv3) > i:
            hooks.append(head.cv3[i].register_forward_hook(get_activation(t_outputs['cls'])))
    
    # SAFETY CHECK: Class Count
    # Teacher might be trained on 5 classes (Background + 4 cones) but Student expected 4.
    # We force Student to match Data Config to prevent shape mismatch in distillation.
    nc_data = data_cfg.get('nc', 4)
    print(f"Dataset Class Count: {nc_data}")
    
    if hasattr(teacher_model, 'nc'):
        print(f"Teacher Class Count: {teacher_model.nc}")
        if teacher_model.nc != nc_data:
             print(f"WARNING: Teacher nc ({teacher_model.nc}) != Dataset nc ({nc_data}).")
             print("This is acceptable if Teacher has BACKGROUND class and Student does not, but usually indicates config drift.")

    # Initialize Student with Dataset's Class Count
    student = UNINA_DLA(num_classes=nc_data, deploy=False).to(device)
    print(f"Student initialized with {nc_data} classes.")
    student.train()
    
    # Init Adapters
    with torch.no_grad():
        t_outputs['feats'].clear()
        teacher_model(torch.zeros(1, 3, 640, 640).to(device))
        t_channels = [f.shape[1] for f in t_outputs['feats']]
        t_outputs['feats'].clear(); t_outputs['cls'].clear(); t_outputs['box'].clear()
        
        s_dummy = torch.zeros(1, 3, 640, 640).to(device)
        s_feats = student.backbone(s_dummy)
        s_neck = student.neck(s_feats)
        s_channels = [f.shape[1] for f in s_neck]
        print(f"Teacher Channels: {t_channels}, Student Channels: {s_channels}")

    adapters = nn.ModuleList([
        nn.Conv2d(s, t, 1).to(device) for s, t in zip(s_channels, t_channels)
    ])
    
    # Optimizer with smart parameter grouping
    # Group 0: Weights (with decay)
    # Group 1: Biases (no decay)
    # Group 2: BN/Normalization (no decay)
    g_weights, g_biases, g_bns = [], [], []
    for m in list(student.modules()) + list(adapters.modules()):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            g_weights.append(m.weight)
            if m.bias is not None:
                g_biases.append(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            g_bns.append(m.weight)
            g_bns.append(m.bias)
            
    optimizer = optim.AdamW([
        {'params': g_weights, 'weight_decay': 0.0005},
        {'params': g_biases, 'weight_decay': 0.0},
        {'params': g_bns, 'weight_decay': 0.0}
    ], lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_distill)
    
    # Losses
    loss_sdf = SDFDistillationLoss().to(device)
    loss_logit = LogitDistillationLoss(temperature=4.0).to(device)
    loss_dfl = DFLDistillationLoss().to(device)
    
    # Task Loss (Ground Truth)
    dataset_cfg = IterableSimpleNamespace(**DEFAULT_CFG.__dict__)
    task_loss_fn = YOLOv8LossAdapter(student, dataset_cfg)
    
    best_map = 0.0
    best_ckpt_path = os.path.join(args.output_dir, 'best_distill.pth')
    
    history = {'loss': [], 'map': []}
    
    for epoch in range(args.epochs_distill):
        student.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Distill Epoch {epoch+1}/{args.epochs_distill}")
        
        for i, batch in enumerate(pbar):
            # Save inspection image on first batch of first epoch
            if epoch == 0 and i == 0:
                save_batch_image(batch, os.path.join(args.output_dir, 'train_batch0.jpg'), data_cfg.get('names', {}))

            imgs = batch['img'].to(device).float() / 255.0
            optimizer.zero_grad()
            
            t_outputs['feats'].clear()
            t_outputs['cls'].clear()
            t_outputs['box'].clear()
            
            with torch.no_grad():
                teacher_model(imgs)
                
            s_feats = student.backbone(imgs)
            s_neck = student.neck(s_feats)
            s_box, s_cls = student.head(s_neck)
                
            l_sdf = loss_sdf(s_neck, t_outputs['feats'], adapters)
            l_logit = loss_logit(s_cls, t_outputs['cls'])
            l_dfl = loss_dfl(s_box, t_outputs['box'])
            
            # Compute Task Loss (GT)
            # student.head returns (reg, cls) - YOLOv8LossAdapter handles it
            l_task, _ = task_loss_fn((s_box, s_cls), batch)
            
            total_loss = args.lambda_sdf * l_sdf + args.lambda_logit * l_logit + args.lambda_dfl * l_dfl + args.lambda_task * l_task
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            pbar.set_postfix({'loss': total_loss.item()})
            
            step = epoch * len(train_loader) + i
            writer.add_scalar('Distill/Loss', total_loss.item(), step)

        scheduler.step()
        
        avg_loss = epoch_loss / len(train_loader)
        history['loss'].append(avg_loss)
        
        # Validation
        val_metrics, metrics_obj = validate(student, val_loader, device, num_classes=data_cfg.get('nc', 4), names=data_cfg.get('names'), save_dir=args.output_dir)
        print(f"Epoch {epoch+1} mAP50: {val_metrics['mAP50']:.4f}")
        writer.add_scalar('Distill/mAP50', val_metrics['mAP50'], epoch)
        history['map'].append(val_metrics['mAP50'])
        
        # Plot periodically
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs_distill:
            plot_results(history, os.path.join(args.output_dir, 'results_distill.png'), "Phase 1: Distillation Metrics")
        
        # Checkpoint
        torch.save(student.state_dict(), os.path.join(args.output_dir, 'last_distill.pth'))
        if val_metrics['mAP50'] > best_map:
            best_map = val_metrics['mAP50']
            torch.save(student.state_dict(), best_ckpt_path)
            print(f"New Best Model (mAP50: {best_map:.4f})")
            
    for h in hooks:
        h.remove()
    
    # Final confusion matrix for distillation phase
    metrics_obj.plot_confusion_matrix(os.path.join(args.output_dir, 'confusion_matrix_distill.png'))
    
    print(f"Phase 1 Complete. Best mAP50: {best_map:.4f}")
    return best_ckpt_path



# ============================================================================
# PHASE 2: QAT
# ============================================================================



def run_qat(args, device, data_cfg, train_loader, distill_ckpt, writer):
    print("\n" + "="*60)
    print("PHASE 2: QUANTIZATION-AWARE TRAINING (QAT)")
    print("="*60)
    
    # Load Distilled Student
    model = UNINA_DLA(num_classes=data_cfg.get('nc', 4), deploy=False)
    model.load_state_dict(torch.load(distill_ckpt, map_location=device))
    
    # Prepare for QAT
    model = prepare_qat_model(model)
    model.to(device)
    
    # Calibration
    print("Running Entropy Calibration...")
    enable_calibration(model)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(train_loader, desc="Calibrating")):
            if i >= 20:
                break
            imgs = batch['img'].to(device).float() / 255.0
            model(imgs)
    
    # Fine-tuning
    print("Starting QAT Fine-Tuning...")
    enable_quantization(model)
    model.train()
    
    dataset_cfg = IterableSimpleNamespace(**DEFAULT_CFG.__dict__)
    loss_adapter = YOLOv8LossAdapter(model, dataset_cfg)
    # QAT Optimizer with smart grouping
    g_weights, g_biases, g_bns = [], [], []
    
    # Define valid weight-bearing modules
    weight_modules = (nn.Conv2d, nn.Linear)
    if quant_nn is not None:
        weight_modules += (quant_nn.QuantConv2d, quant_nn.QuantLinear)
        
    for m in model.modules():
        if isinstance(m, weight_modules):
            if hasattr(m, 'weight') and m.weight is not None:
                g_weights.append(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                g_biases.append(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            if m.weight is not None: g_bns.append(m.weight)
            if m.bias is not None: g_bns.append(m.bias)

    optimizer = optim.AdamW([
        {'params': g_weights, 'weight_decay': 0.0005},
        {'params': g_biases, 'weight_decay': 0.0},
        {'params': g_bns, 'weight_decay': 0.0}
    ], lr=1e-5)
    
    qat_ckpt_path = os.path.join(args.output_dir, 'qat_final.pth')
    history = {'loss': [], 'map': []}
    
    for epoch in range(args.epochs_qat):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"QAT Epoch {epoch+1}/{args.epochs_qat}")
        for batch in pbar:
            imgs = batch['img'].to(device).float() / 255.0
            optimizer.zero_grad()
            preds = model(imgs)
            loss, loss_items = loss_adapter(preds, batch)
            l_sum = loss.sum()
            l_sum.backward()
            optimizer.step()
            
            epoch_loss += l_sum.item()
            pbar.set_postfix({'loss': l_sum.item()})
        
        avg_loss = epoch_loss / len(train_loader)
        history['loss'].append(avg_loss)
        writer.add_scalar('QAT/Loss', avg_loss, epoch)
        
        # QAT validation (optional but good for plots)
        val_metrics, metrics_obj = validate(model, val_loader, device, num_classes=data_cfg.get('nc', 4), names=data_cfg.get('names'), save_dir=args.output_dir)
        history['map'].append(val_metrics['mAP50'])
        writer.add_scalar('QAT/mAP50', val_metrics['mAP50'], epoch)
        
        plot_results(history, os.path.join(args.output_dir, 'results_qat.png'), "Phase 2: QAT Metrics")
    
    # Final confusion matrix for QAT phase
    metrics_obj.plot_confusion_matrix(os.path.join(args.output_dir, 'confusion_matrix_qat.png'))
        
    torch.save(model.state_dict(), qat_ckpt_path)
    print(f"Phase 2 Complete. Saved to {qat_ckpt_path}")
    return qat_ckpt_path



# ============================================================================
# PHASE 3: EXPORT
# ============================================================================
def run_export(args, data_cfg, qat_ckpt, is_qat):
    print("\n" + "="*60)
    print("PHASE 3: DLA-OPTIMIZED ONNX EXPORT")
    print("="*60)
    
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.export_onnx import export_onnx
    num_classes = data_cfg.get('nc', 4)
    export_onnx(qat_ckpt, args.onnx_output, qat=is_qat, num_classes=num_classes)
    print(f"Phase 3 Complete. ONNX saved to {args.onnx_output}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='UNINA-DLA Unified Student Training')
    parser.add_argument('--teacher', type=str, default=None, help='Path to teacher weights (required unless --skip_distillation)')
    parser.add_argument('--data', type=str, default='unina_dla/config/unina_dla_data.yaml')
    parser.add_argument('--epochs_distill', type=int, default=100)
    parser.add_argument('--epochs_qat', type=int, default=10)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lambda_sdf', type=float, default=1.0, help='Weight for SDF Loss')
    parser.add_argument('--lambda_logit', type=float, default=1.0, help='Weight for Logit Loss')
    parser.add_argument('--lambda_dfl', type=float, default=0.5, help='Weight for DFL Loss')
    parser.add_argument('--lambda_task', type=float, default=1.0, help='Weight for Task Loss (Ground Truth)')
    parser.add_argument('--skip_distillation', action='store_true')
    parser.add_argument('--skip_qat', action='store_true')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--onnx_output', type=str, default='unina_dla.onnx')
    args = parser.parse_args()
    
    # Validate arguments
    if not args.skip_distillation and args.teacher is None:
        parser.error("--teacher is required unless --skip_distillation is set")
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join('runs', 'unina_dla_student'))
    
    # Data
    data_cfg = check_det_dataset(args.data)
    dataset_cfg = IterableSimpleNamespace(**DEFAULT_CFG.__dict__)
    dataset_cfg.imgsz = 640
    
    # CRITICAL: Overwrite defaults with YAML values (Augmentations)
    for k, v in data_cfg.items():
        if k in dataset_cfg.__dict__:
            setattr(dataset_cfg, k, v)
    
    train_set = build_yolo_dataset(dataset_cfg, data_cfg['train'], args.batch, data_cfg, mode='train', rect=False)
    train_loader = build_dataloader(train_set, batch=args.batch, workers=4, shuffle=True)
    val_set = build_yolo_dataset(dataset_cfg, data_cfg['val'], args.batch, data_cfg, mode='val', rect=False)
    val_loader = build_dataloader(val_set, batch=args.batch, workers=4, shuffle=False)
    
    # --- PHASE 1 ---
    if not args.skip_distillation:
        distill_ckpt = run_distillation(args, device, data_cfg, train_loader, val_loader, writer)
    else:
        if args.resume is None:
            raise ValueError("--resume must be provided when using --skip_distillation")
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        distill_ckpt = args.resume
        print(f"Skipping Phase 1. Using resume checkpoint: {distill_ckpt}")
    
    # --- PHASE 2 ---
    if not args.skip_qat:
        qat_ckpt = run_qat(args, device, data_cfg, train_loader, distill_ckpt, writer)
    else:
        qat_ckpt = distill_ckpt
        print(f"Skipping Phase 2. Using distillation checkpoint for export.")
    
    # --- PHASE 3 ---
    is_qat = not args.skip_qat
    run_export(args, data_cfg, qat_ckpt, is_qat)
    
    writer.close()
    print("\n" + "="*60)
    print("ALL PHASES COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
