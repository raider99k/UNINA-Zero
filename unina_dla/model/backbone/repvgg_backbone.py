import torch
import torch.nn as nn
from .repvgg_block import RepVGGBlock

class RepVGGBackbone(nn.Module):
    """
    RepVGG-B0 Backbone optimized for DLA (ReLU only).
    """
    def __init__(self, deploy=False):
        super().__init__()
        self.deploy = deploy
        
        # RepVGG-B0 config: [1, 2, 4, 14, 1] blocks
        # Widths: [64, 64, 128, 256, 512]
        self.stages = nn.ModuleList()
        self.in_planes = 64
        
        # Stage 0: Stem
        self.stem = RepVGGBlock(3, 64, stride=2, deploy=deploy)
        
        # Stage 1: 2 blocks, 64 channels
        layer1 = self._make_layer(64, 2, stride=2, deploy=deploy)
        self.stages.append(layer1)
        
        # Stage 2: 4 blocks, 128 channels (P3)
        layer2 = self._make_layer(128, 4, stride=2, deploy=deploy)
        self.stages.append(layer2)
        
        # Stage 3: 14 blocks, 256 channels (P4)
        layer3 = self._make_layer(256, 14, stride=2, deploy=deploy)
        self.stages.append(layer3)
        
        # Stage 4: 1 block, 512 channels (P5)
        layer4 = self._make_layer(512, 1, stride=2, deploy=deploy)
        self.stages.append(layer4)
        
        # Define output channels for P3, P4, P5
        self.out_channels = [128, 256, 512]

    def _make_layer(self, planes, num_blocks, stride, deploy):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(RepVGGBlock(self.in_planes, planes, stride=stride, deploy=deploy))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            # self.stages contains [Stage1(P2), Stage2(P3), Stage3(P4), Stage4(P5)]
            # We want P3, P4, P5 -> indices 1, 2, 3
            if i in [1, 2, 3]: 
                outs.append(x)
        return outs # [P3, P4, P5]

    def switch_to_deploy(self):
        self.deploy = True
        self.stem.switch_to_deploy()
        for stage in self.stages:
            for block in stage:
                block.switch_to_deploy()
