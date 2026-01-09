import torch
import torch.nn as nn
import torch.optim as optim
from unina_dla.model.unina_dla import UNINA_DLA
from unina_dla.utils.qat_utils import prepare_qat_model, enable_calibration, enable_quantization
from ultralytics.data import build_yolo_dataset, build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG, IterableSimpleNamespace
from ultralytics.utils.loss import v8DetectionLoss
import argparse
import yaml
import os
from tqdm import tqdm

class YOLOv8LossAdapter:
    """
    Adapter to use ultralytics v8DetectionLoss with UNINA_DLA model.
    """
    def __init__(self, real_model, cfg):
        self.device = next(real_model.parameters()).device

        # Create a MockModel to satisfy v8DetectionLoss requirements
        class MockHead:
            def __init__(self, nc, reg_max, stride):
                self.nc = nc
                self.reg_max = reg_max
                # Ensure stride is a tensor on the correct device
                self.stride = torch.tensor(stride).to(real_model.parameters().__next__().device) if isinstance(stride, list) else stride
                self.no = nc + reg_max * 4

        class MockModel:
            def __init__(self, args, head_attrs):
                self.args = args
                self.model = [MockHead(**head_attrs)]

            def parameters(self):
                return real_model.parameters()

        # Extract head attributes from UNINA_DLA
        # UNINA_DLA has self.head which is YOLOv10Head
        # YOLOv10Head has num_classes and reg_max
        # Stride is fixed [8, 16, 32] for this architecture
        head = real_model.head
        head_attrs = {
            'nc': head.num_classes,
            'reg_max': head.reg_max,
            'stride': [8., 16., 32.]
        }

        # We need to make sure cfg contains box, cls, dfl gains
        # DEFAULT_CFG should have them.
        self.mock_model = MockModel(cfg, head_attrs)
        self.loss_fn = v8DetectionLoss(self.mock_model)

    def __call__(self, preds, batch):
        """
        Args:
            preds: (reg_outputs, cls_outputs) from UNINA_DLA
            batch: dict containing 'batch_idx', 'cls', 'bboxes', etc.
        """
        reg_outs, cls_outs = preds

        # Concatenate reg and cls outputs for each scale to match v8DetectionLoss expectation
        # v8DetectionLoss expects a list of tensors [B, NO, H, W] where NO = 4*reg_max + nc
        # And specifically, it expects concatenated (reg, cls) order because it splits them:
        # pred_distri, pred_scores = ...split((self.reg_max * 4, self.nc), 1)

        feats = []
        for r, c in zip(reg_outs, cls_outs):
            # r: [B, 4*reg_max, H, W]
            # c: [B, nc, H, W]
            feats.append(torch.cat([r, c], dim=1))

        return self.loss_fn(feats, batch)

def train_qat(checkpoint_path, data_yaml, epochs=30, batch_size=16):
    print("Starting QAT Pipeline...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Data
    data_cfg = check_det_dataset(data_yaml)
    
    # Create a config namespace for the dataset and loss
    dataset_cfg = IterableSimpleNamespace(**DEFAULT_CFG.__dict__)
    dataset_cfg.imgsz = 640
    # Ensure dataset_cfg has loss hyperparameters
    # box: 7.5, cls: 0.5, dfl: 1.5 are defaults in DEFAULT_CFG
    
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
    # Use full YOLO Task Loss (IoU, DFL, Cls)
    # Initialize Loss Adapter
    loss_adapter = YOLOv8LossAdapter(model, dataset_cfg)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-5) # Low learning rate
    
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"QAT Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            imgs = batch['img'].to(device).float() / 255.0
            
            optimizer.zero_grad()
            
            # Forward QAT Model
            # UNINA_DLA returns (reg_outs, cls_outs)
            preds = model(imgs)
            
            # Compute Loss using YOLO Task Loss
            loss, loss_items = loss_adapter(preds, batch)

            loss.sum().backward()
            optimizer.step()
            
            # loss_items is a tensor of 3 values: box, cls, dfl
            box_loss, cls_loss, dfl_loss = loss_items[0].item(), loss_items[1].item(), loss_items[2].item()

            pbar.set_postfix({
                'loss': loss.sum().item(),
                'box': box_loss,
                'cls': cls_loss,
                'dfl': dfl_loss
            })
            
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
