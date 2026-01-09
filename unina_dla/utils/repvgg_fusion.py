import torch
import torch.nn as nn
import numpy as np

def fuse_bn_into_conv(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple:
    """
    Fold BatchNorm parameters into convolution.
    
    Mathematical formulation:
    W' = (γ/σ) * W
    b' = β - (μ*γ/σ)
    
    Args:
        conv (nn.Conv2d): Convolution layer
        bn (nn.BatchNorm2d): BatchNorm layer
        
    Returns:
        tuple: (fused_weight, fused_bias)
    """
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)
    
    beta = bn.bias
    gamma = bn.weight
    
    if beta is None:
        beta = torch.zeros_like(mean)
    if gamma is None:
        gamma = torch.ones_like(mean)
        
    # Calculate fusion factor (γ/σ)
    factor = gamma / var_sqrt
    
    # Reshape factor for broadcasting: [C_out, 1, 1, 1]
    fused_weight = w * factor.view(-1, 1, 1, 1)
    
    # Calculate fused bias: β - (μ*γ/σ)
    # If conv has bias, add it scaled by factor: (b - μ) * (γ/σ) + β  => b*(γ/σ) + β - μ*(γ/σ)
    if conv.bias is not None:
        fused_bias = (conv.bias - mean) * factor + beta
    else:
        fused_bias = beta - mean * factor
        
    return fused_weight, fused_bias

def pad_1x1_to_3x3(kernel_1x1: torch.Tensor) -> torch.Tensor:
    """
    Zero-pad 1x1 kernel to 3x3 (value at center).
    
    Args:
        kernel_1x1: Tensor of shape [C_out, C_in, 1, 1]
        
    Returns:
        Tensor of shape [C_out, C_in, 3, 3]
    """
    if kernel_1x1 is None:
        return 0
    
    # Pad 1 pixel on all sides
    return torch.nn.functional.pad(kernel_1x1, [1, 1, 1, 1])

def create_identity_kernel(channels: int, groups: int) -> torch.Tensor:
    """
    Create 3x3 identity kernel for residual connection.
    
    The kernel has 1 at the center for i==j (respecting groups) and 0 elsewhere.
    
    Args:
        channels (int): Number of input/output channels
        groups (int): Number of groups
        
    Returns:
        Tensor of shape [channels, channels/groups, 3, 3]
    """
    kernel_value = np.zeros((channels, channels // groups, 3, 3), dtype=np.float32)
    
    for i in range(channels):
        # Determine the input channel index corresponding to the output channel i
        # For standard conv (groups=1), input channel is i % input_channels, but here weight shape is [C_out, C_in/groups, k, k]
        # In grouped conv, the ith output channel connects to the (i % (C_in/groups)) input channel of the group.
        # Identity mapping implies output i comes from input i.
        # The weight tensor element w[i, input_idx, k_h, k_w] connects input (group_idx * (C_in/groups) + input_idx) to output i.
        # where group_idx = i // (C_out/groups).
        
        # We want input i to map to output i. 
        # so input_idx should be i % (channels // groups).
        input_idx = i % (channels // groups)
        kernel_value[i, input_idx, 1, 1] = 1
        
    return torch.from_numpy(kernel_value)

def merge_repvgg_branches(conv3x3, bn3x3, conv1x1, bn1x1, identity_bn=None):
    """
    Merge all RepVGG branches into single 3x3 convolution.
    
    Steps:
    1. Fold BN into each branch to get (W, b) for each
    2. Pad 1x1 and identity kernels to 3x3
    3. Sum all kernels and biases
    
    Args:
        conv3x3, bn3x3: Dense 3x3 branch
        conv1x1, bn1x1: 1x1 branch
        identity_bn: BatchNorm from identity branch (optional)
        
    Returns:
        tuple: (merged_weight, merged_bias)
    """
    # 1. Get fused weights/bias for 3x3 branch
    w3, b3 = fuse_bn_into_conv(conv3x3, bn3x3)
    
    # 2. Get fused weights/bias for 1x1 branch
    w1, b1 = fuse_bn_into_conv(conv1x1, bn1x1)
    
    # Pad 1x1 weight to 3x3
    w1 = pad_1x1_to_3x3(w1)
    
    # Initialize total weight/bias
    w_final = w3 + w1
    b_final = b3 + b1
    
    # 3. Handle Identity branch
    if identity_bn is not None:
        # Create a pseudo 1x1 identity conv to fuse with the BN
        # We can simulate fusing by calculations manually, but creating a dummy conv is safer for consistency
        groups = conv3x3.groups
        channels = conv3x3.in_channels
        
        # Create identity kernel
        id_kernel = create_identity_kernel(channels, groups).to(conv3x3.weight.device)
        
        # Calculate fused parameters for identity BN
        # W_id_fused = (γ/σ) * I
        # b_id_fused = β - (μ*γ/σ)
        
        mean = identity_bn.running_mean
        var_sqrt = torch.sqrt(identity_bn.running_var + identity_bn.eps)
        beta = identity_bn.bias
        gamma = identity_bn.weight
        
        if beta is None:
            beta = torch.zeros_like(mean)
        if gamma is None:
            gamma = torch.ones_like(mean)
            
        factor = gamma / var_sqrt
        
        # Broadcast factor: [C, 1, 1, 1]
        b_factor = factor.view(-1, 1, 1, 1)
        
        w_id_fused = id_kernel * b_factor
        b_id_fused = beta - mean * factor
        
        # Add to total
        w_final += w_id_fused
        b_final += b_id_fused
        
    return w_final, b_final
