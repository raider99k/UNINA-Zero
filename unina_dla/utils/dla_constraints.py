import torch
import torch.nn as nn

DLA_SUPPORTED_ACTIVATIONS = (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.ReLU6)
# Hardswish, Hardsigmoid are sometimes supported but less efficient than ReLU
DLA_FORBIDDEN_ACTIVATIONS = (nn.SiLU, nn.GELU, nn.LeakyReLU, nn.Mish, nn.ELU)

DLA_MAX_POOL_SIZE = 8
DLA_CBUF_SAFE_CHANNELS = 512 # Heuristic for standard 3x3 conv layers

def validate_dla_compatibility(model: nn.Module):
    """
    Check model for DLA compatibility issues.
    
    Returns:
        dict: {
            'compatible': bool,
            'warnings': list[str],
            'errors': list[str]
        }
    """
    errors = []
    warnings = []
    
    for name, module in model.named_modules():
        # Check Activations
        if isinstance(module, DLA_FORBIDDEN_ACTIVATIONS):
            errors.append(f"Forbidden activation {type(module).__name__} at {name}. Use ReLU.")
            
        # Check Pooling
        if isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
            k = module.kernel_size
            if isinstance(k, tuple):
                k = max(k)
            if k > DLA_MAX_POOL_SIZE:
                errors.append(f"Pooling kernel {k} > {DLA_MAX_POOL_SIZE} at {name}. DLA HW constraint.")
                
        # Check CBUF (Heuristic)
        if isinstance(module, nn.Conv2d):
            # Check input/output channels
            c_in = module.in_channels
            c_out = module.out_channels
            k = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
            
            # Very rough heuristic: if 3x3 and channels > 512, might thrash
            if k >= 3 and (c_in > DLA_CBUF_SAFE_CHANNELS or c_out > DLA_CBUF_SAFE_CHANNELS):
                warnings.append(f"High channel count ({c_in}->{c_out}) at {name}. May exceed CBUF (1MB) and thrash.")

    return {
        'compatible': len(errors) == 0,
        'warnings': warnings,
        'errors': errors
    }
