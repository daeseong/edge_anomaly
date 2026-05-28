# Approximate FP32 weights from the quantized model.
import torch
import torch.nn as nn

def dequantize(quantized_model: nn.Module, target_layers: list[str]) -> dict:
    """
    {layer_name: tensor of dequantized FP32 weights}
    
    Approximates weights from a quantized model (int8 -> fp32). 
    """

    dq = {}

    for name, module in quantized_model.named_modules():

        if name not in target_layers:
            continue
 
        if not hasattr(module, "weight"):
            print(f"No weight attr: {name} - skip.")
            continue
 
        w = module.weight()
 
        if not w.is_quantized:
            print(f"Not quantized: {name} - skip")
            continue
 
        # INT8 → approximate FP32
        dq[name] = w.dequantize()
 
    missing = set(target_layers) - set(dq.keys())
    
    if missing:
        print(f"No DQ weights: {missing}")
 
    return dq

