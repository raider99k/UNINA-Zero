import torch
import torch.nn as nn
from unina_dla.model.backbone.repvgg_block import RepVGGBlock

class RepVGG_B0(nn.Module):
    """
    RepVGG-B0 backbone optimized for DLA.
    
    Features:
    - Strict ReLU activations (no SiLU)
    - Channel depths capped at 512 for CBUF residency
    - Multi-scale outputs at P3, P4, P5
    
    Args:
        out_indices (tuple): Stages to output (default: (2, 3, 4))
        deploy (bool): Enable inference mode
    """
    
    def __init__(self, out_indices=(2, 3, 4), deploy=False):
        super(RepVGG_B0, self).__init__()
        self.out_indices = out_indices
        self.deploy = deploy
        
        # RepVGG-B0 configuration
        # num_blocks: [1, 4, 6, 16, 1]
        # width_multiplier: [1, 1, 1, 2.5, 2.5] -> [64, 64, 128, 256, 512]?
        # Standard B0:
        # Stage 0: 1 layer, 64 ch
        # Stage 1: 2 layers, 64 ch
        # Stage 2: 4 layers, 128 ch
        # Stage 3: 14 layers, 256 ch
        # Stage 4: 1 layer, 512 ch
        
        # Wait, the config in IMPLEMENTATION.md said:
        # B0: [1, 4, 6, 16, 1] layers
        # Channels: [64, 128, 256, 512, 512] (capped at 512)
        # Let's follow the implementation plan strictly.
        
        self.stages = nn.ModuleList()
        self.in_channels = 3  # RGB image
        
        # Stage 0: 1 layer, 64 channels, stride 2 (resolution 320x320)
        self.stage0 = self._make_stage(64, 1, stride=2)
        
        # Stage 1: 4 layers, 128 channels, stride 2 (resolution 160x160) - P2
        self.stage1 = self._make_stage(128, 4, stride=2)
        
        # Stage 2: 6 layers, 256 channels, stride 2 (resolution 80x80) - P3
        self.stage2 = self._make_stage(256, 6, stride=2)
        
        # Stage 3: 16 layers, 512 channels, stride 2 (resolution 40x40) - P4
        # Original RepVGG-B0 has 256->512 here? No, let's use the plan's widths.
        # Plan says: S3: 512ch, S4: 512ch.
        self.stage3 = self._make_stage(512, 16, stride=2)
        
        # Stage 4: 1 layer, 512 channels, stride 2 (resolution 20x20) - P5
        self.stage4 = self._make_stage(512, 1, stride=2)

    def _make_stage(self, width, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(RepVGGBlock(
                in_channels=self.in_channels,
                out_channels=width,
                stride=s,
                deploy=self.deploy
            ))
            self.in_channels = width
        return nn.Sequential(*blocks)

    def forward(self, x):
        outs = []
        
        x = self.stage0(x)
        # S0 output is usually not needed
        
        x = self.stage1(x)
        if 2 in self.out_indices: # Note: strictly following index convention P2=2? 
            # In typical detection, indices usually map to P_levels. 
            # Stage 1 is P2 (stride 4).
            pass 
            # But wait, usually we want P3, P4, P5.
        
        # Actually my stages are:
        # S0: stride 2 -> x2 downsample
        # S1: stride 2 -> x4 downsample (P2)
        # S2: stride 2 -> x8 downsample (P3)
        # S3: stride 2 -> x16 downsample (P4)
        # S4: stride 2 -> x32 downsample (P5)
        
        # So P2 is output of stage1
        # P3 is output of stage2
        # P4 is output of stage3
        # P5 is output of stage4
        
        x = self.stage2(x) # P3
        if 2 in self.out_indices:
            outs.append(x)
            
        x = self.stage3(x) # P4
        if 3 in self.out_indices:
            outs.append(x)
            
        x = self.stage4(x) # P5
        if 4 in self.out_indices:
            outs.append(x)
            
        return tuple(outs)

    def switch_to_deploy(self):
        """Convert all RepVGG blocks to fused inference mode."""
        if self.deploy:
            return
        
        for module in self.modules():
            if module is not self and hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
        self.deploy = True
