import torch
import torch.nn as nn
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor

def prepare_qat_model(model, per_channel_weights=True):
    """
    Prepare model for QAT:
    1. Fuse RepVGG blocks (MUST be done before QAT)
    2. Replace standard convs/relus with QuantConv/QuantReLU
    3. Exclude sensitive layers (Head)
    """
    # 1. Fuse RepVGG branches
    if hasattr(model, 'switch_to_deploy'):
        print("Fusing RepVGG blocks for QAT...")
        model.switch_to_deploy()
        
    # 2. Initialize Quant Descriptors
    # DLA supports INT8. Use symmetric quantization for weights usually?
    # TensorRT DLA usually prefers symmetric for weights.
    
    quant_desc_input = QuantDescriptor(calib_method='histogram')
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
    
    # 3. Replace modules
    # iterating and replacing...
    # NOTE: pytorch_quantization has a simpler API for this usually, but strict control is better.
    # We will simply monkey-patch submodules or traverse.
    
    replace_modules_with_quant(model)
    
    return model

def replace_modules_with_quant(module, parent_name=''):
    """
    Recursively replace Conv2d and ReLU with QuantConv2d and QuantReLU.
    Skipping sensitive layers.
    """
    for name, child in module.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        
        # SENSITIVE LAYER EXEMPTION
        # Exclude the final detection head layers to keep them in FP16/FP32
        if "head" in full_name:
            # print(f"Keeping sensitive layer in FP16: {full_name}")
            continue
            
        if isinstance(child, nn.Conv2d):
            new_module = quant_nn.QuantConv2d(
                child.in_channels, child.out_channels, child.kernel_size,
                child.stride, child.padding, child.dilation, child.groups,
                child.bias is not None
            )
            # Copy weights
            new_module.weight.data = child.weight.data
            if child.bias is not None:
                new_module.bias.data = child.bias.data
            
            # Replace
            setattr(module, name, new_module)
            
        elif isinstance(child, nn.ReLU):
            new_module = quant_nn.QuantReLU(inplace=child.inplace)
            setattr(module, name, new_module)
            
        else:
            replace_modules_with_quant(child, full_name)

def enable_calibration(model):
    """Enable calibration mode for all quantized modules."""
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.enable_calib()
            module.disable_quant()

def enable_quantization(model):
    """Enable proper quantization (disable calibration)."""
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.disable_calib()
            module.enable_quant()
