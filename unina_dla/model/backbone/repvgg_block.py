import torch
import torch.nn as nn
from unina_dla.utils.repvgg_fusion import merge_repvgg_branches

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class RepVGGBlock(nn.Module):
    """
    RepVGG Block with DLA-compatible ReLU activation.
    
    Training: Multi-branch (3x3 + 1x1 + identity) for better gradients
    Inference: Single fused 3x3 conv for DLA efficiency
    
    Attributes:
        deploy (bool): If True, use fused inference mode
        groups (int): Groups for depthwise separable (default=1)
    """
    
    def __init__(self, in_channels, out_channels, stride=1, 
                 groups=1, deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        # STRICTLY ReLU for DLA compatibility
        self.nonlinearity = nn.ReLU(inplace=True)
        
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                         kernel_size=3, stride=stride, padding=1, groups=groups, bias=True)
        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, groups=groups)
            
    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))
        
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
            
        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)
    
    def switch_to_deploy(self):
        """Fuse branches into single 3x3 conv for inference."""
        if hasattr(self, 'rbr_reparam'):
            return
        
        kernel, bias = merge_repvgg_branches(
            self.rbr_dense.conv, self.rbr_dense.bn,
            self.rbr_1x1.conv, self.rbr_1x1.bn,
            self.rbr_identity
        )
        
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, 
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=3, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding,
                                     dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups, 
                                     bias=True)
        
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        
        # Remove training branches to save memory
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        
        self.deploy = True
