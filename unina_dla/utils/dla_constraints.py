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
                
        # Check weight volume against CBUF (1MB)
        if isinstance(module, nn.Conv2d):
            c_in = module.in_channels
            c_out = module.out_channels
            groups = module.groups
            k = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
            
            # Calculate weight size in bytes (INT8 assumed for DLA)
            # Size = (C_in / groups) * C_out * k_h * k_w
            weight_size = (c_in // groups) * c_out * k[0] * k[1]
            
            if weight_size > 1_000_000:
                errors.append(f"Weight volume {weight_size/1e6:.2f}MB > 1MB at {name}. "
                             f"This WILL cause memory thrashing on DLA. Use higher groups or fewer channels.")
            elif weight_size > 800_000:
                warnings.append(f"Weight volume {weight_size/1e6:.2f}MB is close to 1MB CBUF limit at {name}. "
                                "High risk of performance degradation.")

    return {
        'compatible': len(errors) == 0,
        'warnings': warnings,
        'errors': errors
    }
