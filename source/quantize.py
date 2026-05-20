from pathlib import Path
 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
 
from cnn import ConvAutoencoder


 
def quantize(checkpoint_path: Path,
             calibration_loader: DataLoader,
             target_layers: list[str],
             backend: str = "fbgemm",
             n_calibration: int = 50
             ) -> tuple[nn.Module, dict]:
    """
    Applies static PTQ.
 
    Args:
        checkpoint_path: path to best_model.pth
        calibration_loader: DataLoader for calibration data (train/good)
        target_layers: list of modules to quantize
        backend: quantization backend
        n_calibration: number of images to use for calibration
 
    Returns:
        quantized_model: model with target layers converted to INT8
        layer_stats: per-layer activation + weight quantization params
    """
 
    # Load checkpt. to CPU (GPU not supported)
    model = ConvAutoencoder()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    # Set qconfig
    model.qconfig = torch.quantization.get_default_qconfig(backend)

    for _, module in model.named_modules():

        if isinstance(module, nn.ConvTranspose2d):
            module.qconfig = torch.quantization.default_qconfig

    # Observers
    torch.quantization.prepare(model, inplace=True)
 
    # Build calibration dataset
    full_dataset = calibration_loader.dataset
    n = min(n_calibration, len(full_dataset))
    calibration_subset = Subset(full_dataset, range(n))
    calibration_subset_loader = DataLoader(calibration_subset,
                                           batch_size=calibration_loader.batch_size,
                                           shuffle=False
                                           )
 
    print(f"Calibrating on {n} images across {len(calibration_subset_loader)} batches...")
 
    # Activation ranges
    with torch.no_grad():

        for imgs, _ in calibration_subset_loader:
            model(imgs)
 
    # Per-layer stats (before conversion)
    layer_stats = {}

    for name, module in model.named_modules():
        
        if name in target_layers and hasattr(module, "activation_post_process"):
            observer = module.activation_post_process
            scale, zero_point = observer.calculate_qparams()
            layer_stats[name] = {"activation_scale": scale.item(),
                                 "activation_zero_point": zero_point.item(),
                                 "activation_min": observer.min_val.item(),
                                 "activation_max": observer.max_val.item()
                                 }

    # Convert to int8
    torch.quantization.convert(model, inplace=True)
 
    # Post-conversion weights
    for name, module in model.named_modules():

        if name in target_layers and name in layer_stats:

            if hasattr(module, "weight"):
                w = module.weight()
                
                if w.qscheme() == torch.per_channel_affine:
                    layer_stats[name]["weight_scale"]      = w.q_per_channel_scales().mean().item()
                    layer_stats[name]["weight_zero_point"] = w.q_per_channel_zero_points().float().mean().item()
                else:
                    layer_stats[name]["weight_scale"]      = w.q_scale()
                    layer_stats[name]["weight_zero_point"] = w.q_zero_point()
    
    """
    Verify the FP to ensure quantized model produces valid result. Helps
    to ensure errors are not from the visualization script.
    """
    model.eval()

    with torch.no_grad():

        sample, _ = next(iter(calibration_subset_loader))
        output = model(sample[:1])
        assert output.shape == sample[:1].shape, (f"Shape mismatch after quantization: "f"expected {sample[:1].shape}, got {output.shape}")
 
    # Summary
    print(f"Quantization complete. Output shape: {output.shape}")
    for name, stats in layer_stats.items():
        for k, v in stats.items():
            print(f"Layer: {name:<30} Key: {k:<25} Value: {v:.5f}\n")
 
    return model, layer_stats