import torch
import torch.nn as nn
from unina_dla.model.backbone.repvgg_backbone import RepVGGBackbone as RepVGG_B0
from unina_dla.model.neck.rep_pan import RepPAN
from unina_dla.model.head.yolov10_head import YOLOv10Head

class UNINA_DLA(nn.Module):
    """
    Complete UNINA-DLA Detector.
    
    Architecture: RepVGG-B0 + Rep-PAN + YOLOv10-DLA
    
    Features:
    - 100% DLA-compatible operations
    - Structural re-parameterization for inference
    - Pseudo-One-to-One (Requires NMS if trained with standard loss)
    - 4 Cone Classes: Blue, Yellow, Small Orange, Big Orange
    
    Args:
        num_classes (int): Number of classes (default: 4 for cones)
        input_size (tuple): Fixed input resolution (default: (640, 640))
        deploy (bool): Enable inference mode
    """
    
    def __init__(self, num_classes=4, input_size=(640, 640), deploy=False):
        super(UNINA_DLA, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.deploy = deploy
        
        # Backbone (P3, P4, P5)
        self.backbone = RepVGG_B0(out_indices=(2, 3, 4), deploy=deploy)
        
        # Determine backbone output channels
        if hasattr(self.backbone, 'out_channels'):
            bb_channels = self.backbone.out_channels
        else:
            # Strict check: Backbones MUST expose their output channels
            raise ValueError(
                f"Backbone {type(self.backbone).__name__} does not have 'out_channels' attribute. "
                "Please ensure the backbone is correctly initialized and exposes its output steps."
            )
        
        # Neck
        self.neck = RepPAN(
            in_channels=bb_channels,
            out_channels=256,
            deploy=deploy
        )
        
        # Head (Neck output is always 256ch per scale)
        self.head = YOLOv10Head(
            num_classes=num_classes,
            in_channels=[256, 256, 256],
            reg_max=16,
            deploy=deploy
        )
    
    def switch_to_deploy(self):
        """Convert all RepVGG blocks to fused inference mode."""
        self.backbone.switch_to_deploy()
        self.neck.switch_to_deploy()
        self.head.switch_to_deploy()
        self.deploy = True
    
    def forward(self, x):
        # Backbone: extract multi-scale features
        features = self.backbone(x)  # [P3, P4, P5]
        
        # Neck: feature pyramid + path aggregation
        pyramid = self.neck(features) # [P3, P4, P5] (all 256ch)
        
        # Head: detection predictions
        outputs = self.head(pyramid)
        
        return outputs
