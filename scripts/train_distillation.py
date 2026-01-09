import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from unina_dla.model.unina_dla import UNINA_DLA
from unina_dla.model.teacher_model import get_teacher_model
from unina_dla.model.losses.distillation_losses import SDFDistillationLoss, LogitDistillationLoss, DFLDistillationLoss
from unina_dla.utils.validator import validate
from tqdm import tqdm
from ultralytics.data import build_yolo_dataset, build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG, IterableSimpleNamespace
import argparse
import yaml
import os

def train_distillation(epochs=100, batch_size=16, data_yaml='unina_dla/config/unina_dla_data.yaml', teacher_path=None, exp_name='unina_dla'):
    print(f"Initializing Distillation Pipeline (Tri-Vector)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # TensorBoard
    log_dir = os.path.join('runs', exp_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"Logging to {log_dir}")

    # 1. Load Data
    data_cfg = check_det_dataset(data_yaml)
    
    # Create a config namespace for the dataset
    dataset_cfg = IterableSimpleNamespace(**DEFAULT_CFG.__dict__)
    dataset_cfg.imgsz = 640
    
    print(f"Building Train Dataset from {data_cfg['train']}...")
    train_set = build_yolo_dataset(dataset_cfg, data_cfg['train'], batch_size, data_cfg, mode='train', rect=False)
    train_loader = build_dataloader(train_set, batch=batch_size, workers=4, shuffle=True)
    
    print(f"Building Val Dataset from {data_cfg['val']}...")
    val_set = build_yolo_dataset(dataset_cfg, data_cfg['val'], batch_size, data_cfg, mode='val', rect=False)
    val_loader = build_dataloader(val_set, batch=batch_size, workers=4, shuffle=False)
    
    # 2. Load Teacher (Frozen)
    print(f"Loading Teacher from {teacher_path if teacher_path else 'Pretrained'}...")
    teacher_wrapper = get_teacher_model(teacher_path)
    teacher_model = teacher_wrapper.model.to(device)
    for param in teacher_model.parameters(): param.requires_grad = False
    teacher_model.eval()
    
    # Hooks setup
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
    
    # 3. Initialize Student
    student = UNINA_DLA(num_classes=data_cfg.get('nc', 4), deploy=False).to(device)
    student.train()
    
    # Init Adapters
    with torch.no_grad():
        # Teacher Channels
        t_outputs['feats'].clear()
        teacher_model(torch.zeros(1, 3, 640, 640).to(device))
        t_channels = [f.shape[1] for f in t_outputs['feats']]
        t_outputs['feats'].clear(); t_outputs['cls'].clear(); t_outputs['box'].clear()
        
        # Student Channels (Dynamic)
        # We need to run a dummy pass to see what RepPAN outputs (it might be unified 256)
        s_dummy = torch.zeros(1, 3, 640, 640).to(device)
        s_feats = student.backbone(s_dummy)
        s_neck = student.neck(s_feats)
        s_channels = [f.shape[1] for f in s_neck]
        print(f"Teacher Channels: {t_channels}")
        print(f"Student Channels: {s_channels}")

    adapters = nn.ModuleList([
        nn.Conv2d(s, t, 1).to(device) for s, t in zip(s_channels, t_channels)
    ])
    
    # 4. Optimization
    optimizer = optim.AdamW(
        list(student.parameters()) + list(adapters.parameters()),
        lr=0.001, weight_decay=0.0005
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 5. Losses
    loss_sdf = SDFDistillationLoss().to(device)
    loss_logit = LogitDistillationLoss(temperature=4.0).to(device)
    loss_dfl = DFLDistillationLoss().to(device)
    
    best_map = 0.0
    
    print("Starting Training...")
    
    for epoch in range(epochs):
        student.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        total_loss_avg = 0
        
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
            
            total_loss_avg += total_loss.item()
            
            pbar.set_postfix({
                'loss': total_loss.item(),
                'sdf': l_sdf.item(),
            })
            
            # Step-wise Logging
            step = epoch * len(train_loader) + i
            writer.add_scalar('Train/Loss', total_loss.item(), step)
            writer.add_scalar('Train/SDF_Loss', l_sdf.item(), step)
            writer.add_scalar('Train/Logit_Loss', l_logit.item(), step)
            writer.add_scalar('Train/DFL_Loss', l_dfl.item(), step)
            writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], step)

        scheduler.step()
        
        # Validation
        print("\nValidating...")
        val_metrics = validate(student, val_loader, device, num_classes=data_cfg.get('nc', 4), names=data_cfg.get('names'))
        print(f"Epoch {epoch+1} mAP50: {val_metrics['mAP50']:.4f}")
        
        writer.add_scalar('Val/mAP50', val_metrics['mAP50'], epoch)
        writer.add_scalar('Val/mAP50-95', val_metrics['mAP50-95'], epoch)
        
        # Checkpointing
        os.makedirs('checkpoints', exist_ok=True)
        current_map = val_metrics['mAP50']
        
        # Save Last
        torch.save(student.state_dict(), f"checkpoints/last.pth")
        
        # Save Best
        if current_map > best_map:
            best_map = current_map
            torch.save(student.state_dict(), f"checkpoints/best.pth")
            print(f"New Best Model Saved (mAP50: {best_map:.4f})")
            
        if (epoch+1) % 10 == 0:
             torch.save(student.state_dict(), f"checkpoints/epoch_{epoch+1}.pth")
            
    # Cleanup
    for h in hooks: h.remove()
    writer.close()
    print("Training Finished.")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--data', type=str, default='unina_dla/config/unina_dla_data.yaml')
    parser.add_argument('--teacher', type=str, default=None, help='Path to trained teacher weights')
    parser.add_argument('--exp_name', type=str, default='distill_run')
    args = parser.parse_args()
    
    train_distillation(args.epochs, args.batch, args.data, args.teacher, args.exp_name)
