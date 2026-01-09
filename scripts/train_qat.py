import torch
import torch.nn as nn
import torch.optim as optim
from unina_dla.model.unina_dla import UNINA_DLA
from unina_dla.utils.qat_utils import prepare_qat_model, enable_calibration, enable_quantization
from ultralytics.data import build_yolo_dataset, build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG, IterableSimpleNamespace
import argparse
import yaml
import os
from tqdm import tqdm

def train_qat(checkpoint_path, data_yaml, epochs=30, batch_size=16):
    print("Starting QAT Pipeline...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Data
    data_cfg = check_det_dataset(data_yaml)
    
    # Create a config namespace for the dataset
    dataset_cfg = IterableSimpleNamespace(**DEFAULT_CFG.__dict__)
    dataset_cfg.imgsz = 640
    
    print(f"Building Dataset from {data_cfg['train']}...")
    dataset = build_yolo_dataset(dataset_cfg, data_cfg['train'], batch_size, data_cfg, mode='train', rect=False)
    dataloader = build_dataloader(dataset, batch=batch_size, workers=4, shuffle=True)
    
    # 2. Load Student (Pre-trained FP32)
    # Ideally load from checkpoint
    print(f"Loading Student from {checkpoint_path}...")
    model = UNINA_DLA(num_classes=data_cfg.get('nc', 4), deploy=False)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        print("Warning: Checkpoint not found, using random weights for QAT test.")
    
    # 3. Prepare for QAT
    # This fuses RepVGG and replaces layers with QuantModules
    model = prepare_qat_model(model)
    model.to(device)
    
    # 4. Calibration
    print("Running Entropy Calibration...")
    enable_calibration(model)
    
    # Run calibration on a subset of data (e.g. 256 images)
    model.eval()
    with torch.no_grad():
        calib_batches = 20
        count = 0
        for batch in tqdm(dataloader, desc="Calibrating"):
            imgs = batch['img'].to(device).float() / 255.0
            model(imgs)
            count += 1
            if count >= calib_batches:
                break
    
    print("Calibration done. Switching to QAT Fine-Tuning...")
    enable_quantization(model)
    model.train()
    
    # 5. Fine-tuning
    # Since we lack the complex YOLO Task Loss in this standalone script,
    # we will use Self-Distillation (Feature Matching) from the static FP32 model (Teacher)
    # just to ensure weights adapt to quantization noise without drifting.
    # Ideally, use the original Teacher or Task Loss.
    
    # For robust "No Dummy Code", we just minimize MSE on output features vs FP32 model
    # Or just run the loop (assuming user might plug in loss). 
    # Let's implement Output MSE Loss (simplest QAT recovery).
    
    # We need a reference "Teacher" (the original FP32 model)
    # But we modified 'model' in-place. We should have kept a copy if we wanted self-distill.
    # We will assume simple fine-tuning is desired. 
    # Lacking task loss is critical.
    # I will perform "Dummy Fine-Tuning" - i.e. just running the loop with a dummy loss 
    # to demonstrate mechanics, but warn user.
    # User said "No Dummy Code".
    # I must implement a loss.
    # Feature Matching with itself (Distillation from pre-quantized state) is a valid QAT strategy.
    
    # Reload FP32 reference
    ref_model = UNINA_DLA(num_classes=data_cfg.get('nc', 4), deploy=True) # Fused
    if os.path.exists(checkpoint_path):
        ref_model.load_state_dict(torch.load(checkpoint_path))
        # We need to fuse it to match the QAT model structure
        ref_model.switch_to_deploy()
    ref_model.to(device)
    ref_model.eval()
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-5) # Low learning rate
    mse_loss = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"QAT Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            imgs = batch['img'].to(device).float() / 255.0
            
            optimizer.zero_grad()
            
            # Forward QAT Model
            # UNINA_DLA now returns (reg_outs, cls_outs)
            q_reg, q_cls = model(imgs) 
            
            # Forward Reference FP32 Model
            with torch.no_grad():
                f_reg, f_cls = ref_model(imgs)
                
            # Compute Loss: MSE between FP32 outputs and QAT outputs
            loss = 0.0
            for i in range(len(q_reg)):
                loss += mse_loss(q_reg[i], f_reg[i])
                loss += mse_loss(q_cls[i], f_cls[i])
                
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({'qat_loss': loss.item()})
            
        # Save QAT Model
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), f"checkpoints/unina_dla_qat_epoch_{epoch+1}.pth")

    print("QAT Fine-Tuning Finished.")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='unina_dla.pth')
    parser.add_argument('--data', type=str, default='unina_dla/config/unina_dla_data.yaml')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    
    os.makedirs('checkpoints', exist_ok=True)
    train_qat(args.checkpoint, args.data, args.epochs)
