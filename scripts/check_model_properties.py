import onnx
import sys
import argparse

def check_model(model_path):
    print(f"Checking model: {model_path}")
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    
    # 1. Check Opset
    opset = model.opset_import[0].version
    print(f"Opset Version: {opset}")
    
    # 2. Check Inputs
    print("Inputs:")
    for input in model.graph.input:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in input.type.tensor_type.shape.dim]
        print(f"  {input.name}: {shape}")
        
        # Check for dynamic axes (strings usually indicate dynamic)
        if any(isinstance(d, str) for d in shape):
            print(f"  [WARNING] Dynamic axis detected in input {input.name}!")
            
    # 3. Check Outputs
    print("Outputs:")
    for output in model.graph.output:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in output.type.tensor_type.shape.dim]
        print(f"  {output.name}: {shape}")
        
    print("Check complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to ONNX model")
    args = parser.parse_args()
    check_model(args.model)
