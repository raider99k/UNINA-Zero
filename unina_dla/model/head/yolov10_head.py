import torch
import torch.nn as nn
from unina_dla.model.backbone.repvgg_block import RepVGGBlock

class YOLOv10Head(nn.Module):
    """
    YOLOv10 Detection Head with One-to-One assignment.
    
    Features:
    - NMS-free inference via Hungarian matching during training (implied)
    - Distribution Focal Loss (DFL) for box regression
    - Decoupled classification and regression branches
    - RepVGG blocks for stems to maximize DLA efficiency
    
    Args:
        num_classes (int): Number of object classes (default: 4 for cones)
        in_channels (list): Input channels from neck [P3, P4, P5]
        reg_max (int): DFL distribution bins (default: 16)
        deploy (bool): Enable inference mode for RepVGG blocks
    """
    
    def __init__(self, num_classes=4, in_channels=[256, 256, 256], reg_max=16, deploy=False):
        super(YOLOv10Head, self).__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.deploy = deploy
        
        # Per-scale detection modules
        self.stems = nn.ModuleList()      # Initial convolution
        self.cls_convs = nn.ModuleList()  # Classification branch
        self.reg_convs = nn.ModuleList()  # Regression branch
        self.cls_preds = nn.ModuleList()  # Class prediction layer
        self.reg_preds = nn.ModuleList()  # Box prediction layer
        
        for ch in in_channels:
            # Stem: reduce channels. 
            self.stems.append(RepVGGBlock(ch, ch, deploy=deploy))
            
            # Classification branch (2 convs)
            self.cls_convs.append(nn.Sequential(
                RepVGGBlock(ch, ch, deploy=deploy),
                RepVGGBlock(ch, ch, deploy=deploy)
            ))
            
            # Regression branch (2 convs)
            self.reg_convs.append(nn.Sequential(
                RepVGGBlock(ch, ch, deploy=deploy),
                RepVGGBlock(ch, ch, deploy=deploy)
            ))
            
            # Prediction heads (1x1 convs)
            # These are NOT RepVGG, just standard linear projections.
            
            # Class prediction: num_classes
            self.cls_preds.append(nn.Conv2d(ch, num_classes, 1))
            
            # Box prediction: 4 * reg_max (DFL distribution)
            self.reg_preds.append(nn.Conv2d(ch, 4 * reg_max, 1))

    def forward(self, feats):
        """
        Args:
            feats (tuple): Features [P3, P4, P5]
            
        Returns:
            tuple: (reg_outputs, cls_outputs)
        """
        reg_outputs = []
        cls_outputs = []
        for i, x in enumerate(feats):
            x = self.stems[i](x)
            
            # Branching
            cls_feat = self.cls_convs[i](x)
            reg_feat = self.reg_convs[i](x)
            
            # Predictions
            cls_out = self.cls_preds[i](cls_feat)
            reg_out = self.reg_preds[i](reg_feat)
            
            reg_outputs.append(reg_out)
            cls_outputs.append(cls_out)
            
        return tuple(reg_outputs), tuple(cls_outputs)

    def switch_to_deploy(self):
        """Convert all RepVGG blocks to fused inference mode."""
        if self.deploy:
            return
            
        for module in self.modules():
            if module is not self and hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
        self.deploy = True
