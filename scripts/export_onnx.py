import torch
from unina_dla.model.unina_dla import UNINA_DLA
try:
    from pytorch_quantization import nn as quant_nn
    from unina_dla.utils.qat_utils import prepare_qat_model
except ImportError:
    quant_nn = None
    prepare_qat_model = None
    print("Warning: pytorch_quantization not found. QAT export will strictly fail.")
import argparse
import os

def export_onnx(checkpoint, output, qat=False, num_classes=5):
    print(f"Exporting model to {output} (QAT={qat}, num_classes={num_classes})...")
    
    # 1. Load Model
    model = UNINA_DLA(num_classes=num_classes, deploy=False)
    
    # 2. Prepare for Deploy
    if qat:
        if quant_nn is None:
            raise ImportError("Cannot export QAT model: pytorch_quantization not installed.")
        # CRITICAL: For QAT export, we must:
        # 1. Prepare the model for QAT (inserts QuantConv layers)
        # 2. THEN load the QAT-trained weights (which have QuantConv keys)
        quant_nn.TensorQuantizer.use_fb_fake_quant = True
        model = prepare_qat_model(model)
        
        # Now load QAT weights
        if checkpoint and os.path.exists(checkpoint):
            print(f"Loading QAT weights from {checkpoint}")
            model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
        else:
            raise FileNotFoundError(f"QAT checkpoint required but not found: {checkpoint}")
    else:
        # FP32 Export: Load weights first, then fuse RepVGG
        if checkpoint and os.path.exists(checkpoint):
            print(f"Loading weights from {checkpoint}")
            model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
        else:
            print("No checkpoint found or provided. Exporting random weights.")
        model.switch_to_deploy()
    
    # Enable Export Mode for Flattened Output
    # User modified head to return tuple. We use a Wrapper to enforce DLA-friendly flattened output.
    
    class DLAWrapper(torch.nn.Module):
        def __init__(self, model, num_classes):
            super().__init__()
            self.model = model
            self.num_classes = num_classes
            
        def forward(self, x):
            # Get split outputs from model head
            reg_outs, cls_outs = self.model(x)
            
            # Flatten and Concat for DLA Zero-Copy
            preds = []
            for reg, cls in zip(reg_outs, cls_outs):
                # reg: [B, 64, H, W], cls: [B, NC, H, W]
                # Permute to [B, H, W, C]
                reg = reg.permute(0, 2, 3, 1)
                cls = cls.permute(0, 2, 3, 1)
                
                b, h, w, _ = reg.shape
                reg = reg.reshape(b, -1, 64)
                cls = cls.reshape(b, -1, self.num_classes).sigmoid() # Apply sigmoid
                
                # Re-concat: [Box, Cls]
                preds.append(torch.cat([reg, cls], dim=-1))
                
            # [B, 8400, 64+NC]
            return torch.cat(preds, dim=1)
            
    model = DLAWrapper(model, num_classes)
        
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 3. Dummy Input (Static)
    dummy_input = torch.randn(1, 3, 640, 640).to(device)
    
    # 4. Export
    # The forward pass now returns a single tensor [B, 8400, 64+NC]
    torch.onnx.export(
        model,
        dummy_input,
        output,
        opset_version=13,
        input_names=['images'],
        output_names=['detections'], # Single output tensor
        dynamic_axes=None # STRICT STATIC SHAPES FOR DLA
    )
    
    print("Export Success.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='unina-dla.pth')
    parser.add_argument('--output', type=str, default='unina-dla.onnx')
    parser.add_argument('--qat', action='store_true', help="Export QAT model")
    parser.add_argument('--num_classes', type=int, default=5, help="Number of classes")
    
    args = parser.parse_args()
    
    export_onnx(args.checkpoint, args.output, args.qat, args.num_classes)
