import torch
import torch.nn as nn
from unina_dla.model.backbone.repvgg_backbone import RepVGGBackbone
from unina_dla.model.neck.rep_pan import RepPAN
from unina_dla.model.head.yolov10_head import YOLOv10Head

class UNINA_DLA(nn.Module):
    """
    UNINA-DLA-v1 Model: RepVGG-B0 + Rep-PAN + YOLOv10Head.
    """
    def __init__(self, num_classes=4, deploy=False):
        super().__init__()
        self.deploy = deploy
        
        # Backbone
        self.backbone = RepVGGBackbone(deploy=deploy)
        
        # Neck
        # RepVGG-B0 outs: [128, 256, 512]
        self.neck = RepPAN(
            in_channels=[128, 256, 512],
            out_channels=256, # Unified output channel width
            deploy=deploy
        )
        
        # Head
        self.head = YOLOv10Head(
            num_classes=num_classes,
            in_channels=[256, 256, 256],
            deploy=deploy
        )
        
    def forward(self, x):
        # Backbone
        features = self.backbone(x) # [P3, P4, P5]
        
        # Neck
        neck_features = self.neck(features)
        
        # Head
        return self.head(neck_features)

    def switch_to_deploy(self):
        """
        Switch entire model to inference mode (fuse RepVGG blocks).
        """
        self.deploy = True
        self.backbone.switch_to_deploy()
        self.neck.switch_to_deploy()
        self.head.switch_to_deploy()
        
if __name__ == "__main__":
    # Smoke test
    model = UNINA_DLA(num_classes=4, deploy=False)
    dummy_input = torch.randn(1, 3, 640, 640)
    output = model(dummy_input)
    print("Training Output shapes:")
    for i, (cls, box) in enumerate(zip(output[0], output[1])):
        print(f"Scale {i}: Cls {cls.shape}, Box {box.shape}")
        
    print("\nSwitching to deploy...")
    model.switch_to_deploy()
    output_deploy = model(dummy_input)
    print("Deploy successful.")
