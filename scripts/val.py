
import torch
from unina_dla.model.unina_dla import UNINA_DLA
from unina_dla.utils.validator import validate
from ultralytics.data import build_dataloader
import argparse
import yaml
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data', type=str, default='unina_dla/config/unina_dla_data.yaml')
    parser.add_argument('--batch', type=int, default=16)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(args.data) as f:
        data_cfg = yaml.safe_load(f)
        
    model = UNINA_DLA(num_classes=data_cfg.get('nc', 4), deploy=False).to(device)
    
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint {args.checkpoint}...")
        model.load_state_dict(torch.load(args.checkpoint))
    else:
        print("Checkpoint not found!")
        exit(1)
        
    dataloader = build_dataloader(
        data_cfg, batch=args.batch, imgsz=640, mode='val', rect=False, augment=False 
    )
    
    results = validate(model, dataloader, device)
    print(f"Validation Results: mAP50: {results['mAP50']:.4f}, mAP50-95: {results['mAP50-95']:.4f}")
