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

from unina_dla.model.unina_dla import UNINA_DLA
from unina_dla.model.teacher_model import get_teacher_model
from unina_dla.model.losses.distillation_losses import SDFDistillationLoss, LogitDistillationLoss, DFLDistillationLoss
from unina_dla.utils.validator import validate
from unina_dla.utils.qat_utils import prepare_qat_model, enable_calibration, enable_quantization

from ultralytics.data import build_yolo_dataset, build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG, IterableSimpleNamespace
from ultralytics.utils.loss import v8DetectionLoss


# ============================================================================
# PHASE 1: DISTILLATION
# ============================================================================
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
    hooks = []
    for i, idx in enumerate(head_from):
        layer = teacher_model.model[idx]
        hooks.append(layer.register_forward_hook(get_activation(t_outputs['feats'])))
        if hasattr(head, 'cv2') and len(head.cv2) > i:
            hooks.append(head.cv2[i].register_forward_hook(get_activation(t_outputs['cls'])))
        if hasattr(head, 'cv3') and len(head.cv3) > i:
            hooks.append(head.cv3[i].register_forward_hook(get_activation(t_outputs['box'])))
    
    # Initialize Student
    student = UNINA_DLA(num_classes=data_cfg.get('nc', 4), deploy=False).to(device)
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
    
    # Optimizer
    optimizer = optim.AdamW(
        list(student.parameters()) + list(adapters.parameters()),
        lr=0.001, weight_decay=0.0005
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_distill)
    
    # Losses
    loss_sdf = SDFDistillationLoss().to(device)
    loss_logit = LogitDistillationLoss(temperature=4.0).to(device)
    loss_dfl = DFLDistillationLoss().to(device)
    
    best_map = 0.0
    best_ckpt_path = os.path.join(args.output_dir, 'best_distill.pth')
    
    for epoch in range(args.epochs_distill):
        student.train()
        pbar = tqdm(train_loader, desc=f"Distill Epoch {epoch+1}/{args.epochs_distill}")
        
        for i, batch in enumerate(pbar):
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
            
            total_loss = 1.0 * l_sdf + 1.0 * l_logit + 0.5 * l_dfl
            
            total_loss.backward()
            optimizer.step()
            
            pbar.set_postfix({'loss': total_loss.item()})
            
            step = epoch * len(train_loader) + i
            writer.add_scalar('Distill/Loss', total_loss.item(), step)

        scheduler.step()
        
        # Validation
        val_metrics = validate(student, val_loader, device, num_classes=data_cfg.get('nc', 4), names=data_cfg.get('names'))
        print(f"Epoch {epoch+1} mAP50: {val_metrics['mAP50']:.4f}")
        writer.add_scalar('Distill/mAP50', val_metrics['mAP50'], epoch)
        
        # Checkpoint
        torch.save(student.state_dict(), os.path.join(args.output_dir, 'last_distill.pth'))
        if val_metrics['mAP50'] > best_map:
            best_map = val_metrics['mAP50']
            torch.save(student.state_dict(), best_ckpt_path)
            print(f"New Best Model (mAP50: {best_map:.4f})")
            
    for h in hooks:
        h.remove()
    
    print(f"Phase 1 Complete. Best mAP50: {best_map:.4f}")
    return best_ckpt_path


# ============================================================================
# PHASE 2: QAT
# ============================================================================
class YOLOv8LossAdapter:
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
        head_attrs = {'nc': head.num_classes, 'reg_max': head.reg_max, 'stride': [8., 16., 32.]}
        self.mock_model = MockModel(cfg, head_attrs)
        self.loss_fn = v8DetectionLoss(self.mock_model)

    def __call__(self, preds, batch):
        reg_outs, cls_outs = preds
        feats = [torch.cat([r, c], dim=1) for r, c in zip(reg_outs, cls_outs)]
        return self.loss_fn(feats, batch)


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
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    
    qat_ckpt_path = os.path.join(args.output_dir, 'qat_final.pth')
    
    for epoch in range(args.epochs_qat):
        model.train()
        pbar = tqdm(train_loader, desc=f"QAT Epoch {epoch+1}/{args.epochs_qat}")
        for batch in pbar:
            imgs = batch['img'].to(device).float() / 255.0
            optimizer.zero_grad()
            preds = model(imgs)
            loss, loss_items = loss_adapter(preds, batch)
            loss.sum().backward()
            optimizer.step()
            pbar.set_postfix({'loss': loss.sum().item()})
        
        writer.add_scalar('QAT/Loss', loss.sum().item(), epoch)
        
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
    parser.add_argument('--batch', type=int, default=16)
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
