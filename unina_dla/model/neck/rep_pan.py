import torch
import torch.nn as nn
from unina_dla.model.backbone.repvgg_block import RepVGGBlock

class RepPAN(nn.Module):
    """
    Re-parameterizable Path Aggregation Network (Rep-PAN).
    
    DLA Optimizations:
    - Uses concatenation instead of element-wise addition (more efficient on DLA)
    - Nearest-neighbor upsampling (DLA native, avoids bilinear complexity)
    - RepVGG blocks for fusion at inference
    
    Args:
        in_channels (list): [P3, P4, P5] input channels from backbone
        out_channels (int): Unified output channel count
        deploy (bool): Enable inference mode
    """
    
    def __init__(self, in_channels=[256, 512, 512], out_channels=256, deploy=False):
        super(RepPAN, self).__init__()
        self.deploy = deploy
        
        # P3, P4, P5
        c3, c4, c5 = in_channels
        
        # Top-down pathway
        # Reduce P5 to out_channels
        self.reduce_p5 = RepVGGBlock(c5, out_channels, deploy=deploy)
        # Reduce P4 to out_channels
        self.reduce_p4 = RepVGGBlock(c4, out_channels, deploy=deploy)
        
        # Upsample P5 + Concat with reduced P4 -> Fuse
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # After concat: out_channels (from P5 up) + out_channels (from P4) = 2*out_channels
        self.p4_fusion = RepVGGBlock(out_channels * 2, out_channels, deploy=deploy)
        
        # Upsample P4_fused + Concat with P3 -> Fuse
        # Note: P3 is c3 (256), already matches out_channels usually. 
        # But if c3 != out_channels, we might need a reduction?
        # Assuming c3 == out_channels for now (256), or we treat it as generic.
        # Let's add a reduction for P3 just in case, or handle mismatch in fusion.
        # Usually YOLO FPN reduces lateral connections first.
        self.reduce_p3 = RepVGGBlock(c3, out_channels, deploy=deploy)
        
        self.p3_fusion = RepVGGBlock(out_channels * 2, out_channels, deploy=deploy)
        
        # Bottom-up pathway
        # P3_out (out_channels) -> Downsample -> Concat P4_fused -> Fuse
        self.downsample_p3 = RepVGGBlock(out_channels, out_channels, stride=2, deploy=deploy)
        
        # Concat: P3_down (out) + P4_fusion (out) = 2*out
        self.n3_fusion = RepVGGBlock(out_channels * 2, out_channels, deploy=deploy)
        
        # P4_out -> Downsample -> Concat P5_reduced -> Fuse
        self.downsample_p4 = RepVGGBlock(out_channels, out_channels, stride=2, deploy=deploy)
        
        # Concat: P4_down (out) + P5_reduced (out) = 2*out
        self.n4_fusion = RepVGGBlock(out_channels * 2, out_channels, deploy=deploy) # P5 output
        
    def forward(self, inputs):
        # inputs: [P3, P4, P5]
        p3, p4, p5 = inputs
        
        # --- Top-Down ---
        p5_red = self.reduce_p5(p5)
        p5_up = self.upsample(p5_red)
        p4_red = self.reduce_p4(p4)
        
        # Concat P4 and P5_up
        p4_cat = torch.cat([p4_red, p5_up], dim=1)
        p4_fused = self.p4_fusion(p4_cat)
        
        p4_up = self.upsample(p4_fused)
        p3_red = self.reduce_p3(p3)
        
        # Concat P3 and P4_up
        p3_cat = torch.cat([p3_red, p4_up], dim=1)
        p3_out = self.p3_fusion(p3_cat) # This is the final P3 output
        
        # --- Bottom-Up ---
        p3_down = self.downsample_p3(p3_out)
        
        # Concat P3_down and P4_fused (from top-down phase)
        # Note: Standard PANet uses the "laterals" from top-down.
        n3_cat = torch.cat([p3_down, p4_fused], dim=1)
        n3_out = self.n3_fusion(n3_cat) # This is the final P4 output
        
        n3_down = self.downsample_p4(n3_out)
        
        # Concat n3_down and p5_red
        n4_cat = torch.cat([n3_down, p5_red], dim=1)
        n4_out = self.n4_fusion(n4_cat) # This is the final P5 output
        
        # Return [P3, P4, P5]
        return (p3_out, n3_out, n4_out)

    def switch_to_deploy(self):
        """Convert all RepVGG blocks to fused inference mode."""
        if self.deploy:
            return
            
        for module in self.modules():
            if module is not self and hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
        self.deploy = True
