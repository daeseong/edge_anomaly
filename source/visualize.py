import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# Helper functions

def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """tensor [0,1] → Hnumpy array"""
    return tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def _reconstruct(model: nn.Module, image: torch.Tensor) -> torch.Tensor:
    """Run a single image through the model on CPU. Returns CHW tensor."""
    model.eval()
    model = model.cpu()
    with torch.no_grad():
        output = model(image.unsqueeze(0).cpu())
    return output.squeeze(0)


def _residual_map(original: torch.Tensor, reconstruction: torch.Tensor) -> np.ndarray:
    """Per-pixel MSE averaged over channels → (H, W) numpy array."""
    error = (original.cpu() - reconstruction.cpu()) ** 2
    return error.mean(dim=0).numpy()


# Reconstruction

def plot_reconstruction(fp32_model: nn.Module,
                        quantized_model: nn.Module,
                        normal_image: torch.Tensor,
                        anomaly_image: torch.Tensor,
                        anomaly_type: str
                        ) -> None:
    """
    Side-by-side reconstruction comparison for one normal and one anomalous image.
    Args:
        fp32_model: trained model
        quantized_model: quantized model (int8)
        normal_image: tensor from train/good
        anomaly_image: tensor from test/<anomaly_type>
        anomaly_type: string label for the anomaly row title
    """

    rows = [("Normal", normal_image), (anomaly_type, anomaly_image)]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    col_titles = ["Original", "FP32 Reconstruction", "INT8 Reconstruction"]

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold")

    for row_idx, (label, image) in enumerate(rows):
        fp32_recon = _reconstruct(fp32_model, image)
        int8_recon = _reconstruct(quantized_model, image)

        axes[row_idx, 0].imshow(_to_numpy(image))
        axes[row_idx, 1].imshow(_to_numpy(fp32_recon))
        axes[row_idx, 2].imshow(_to_numpy(int8_recon))

        for ax in axes[row_idx]:
            ax.axis("off")

        axes[row_idx, 0].set_ylabel(label, fontsize=11, rotation=90, labelpad=10)
        axes[row_idx, 0].axis("on")
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])

    plt.suptitle("FP32 vs INT8 Reconstruction", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


# Heatmaps for residuals

def plot_residual_heatmap(fp32_model: nn.Module,
                          quantized_model: nn.Module,
                          normal_image: torch.Tensor,
                          anomaly_image: torch.Tensor,
                          anomaly_type: str
                          ) -> None:
    """
    Brighter pixels = higher reconstruction error = likely anomaly.
    """

    rows = [("Normal",normal_image), (anomaly_type,  anomaly_image)]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    col_titles = ["Original", "FP32 Residual", "INT8 Residual"]

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold")

    for row_idx, (label, image) in enumerate(rows):
        fp32_recon = _reconstruct(fp32_model, image)
        int8_recon = _reconstruct(quantized_model, image)

        fp32_map = _residual_map(image, fp32_recon)
        int8_map = _residual_map(image, int8_recon)

        #color scale across both heatmaps per row
        vmax = max(fp32_map.max(), int8_map.max())

        axes[row_idx, 0].imshow(_to_numpy(image))
        axes[row_idx, 1].imshow(fp32_map, cmap="hot", vmin=0, vmax=vmax)
        axes[row_idx, 2].imshow(int8_map, cmap="hot", vmin=0, vmax=vmax)

        for ax in axes[row_idx]:
            ax.axis("off")

        axes[row_idx, 0].set_ylabel(label, fontsize=11, rotation=90, labelpad=10)
        axes[row_idx, 0].axis("on")
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])

    plt.suptitle("FP32 vs INT8 Residual Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


# Quantization error

def plot_weight_error(fp32_weights: dict,
                      dq_weights: dict
                      ) -> None:
    """
    Distribution of weight quantization error per target layer.

    Error = fp32 weight − int8 weight.
    Wider distribution -> more precision lost in that layer.

    Args:
        fp32_weights: from checkpoint
        dq_weights: from dequantize()
    """

    layers = list(fp32_weights.keys())
    n = len(layers)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers):
        if layer not in dq_weights:
            ax.set_title(f"{layer}\n|no dq weights")
            continue

        fp32_w = fp32_weights[layer].float().detach().cpu().flatten().numpy()
        deq_w = dq_weights[layer].float().detach().cpu().flatten().numpy()
        error = fp32_w - deq_w

        ax.hist(error, bins=100, color="steelblue", edgecolor="none", alpha=0.85)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="Zero error")
        ax.set_title(layer, fontsize=11, fontweight="bold")
        ax.set_xlabel("Quantization Error (FP32 − INT8)")
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)

    plt.suptitle("Per-Layer Weight Quantization Error", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


# Anomaly score comparison

def plot_anomaly_scores(
    fp32_model: nn.Module,
    quantized_model: nn.Module,
    test_loader: DataLoader,
) -> None:
    """
    Mean reconstruction MSE per anomaly type for FP32 vs INT8.
    Side-by-side bar chart to determine whether PTQ degrades detection.

    Args:
        fp32_model: trained model
        quantized_model: quantized model 
        test_loader: DataLoader for test split
    """

    fp32_model.eval()
    fp32_model = fp32_model.cpu()
    quantized_model.eval()

    scores: dict[str, dict[str, list]] = {}

    with torch.no_grad():
        for images, anomaly_types in test_loader:
            images = images.cpu()

            fp32_recons = fp32_model(images)
            int8_recons = quantized_model(images)

            fp32_errors = ((images - fp32_recons) ** 2).mean(dim=[1, 2, 3])
            int8_errors = ((images - int8_recons) ** 2).mean(dim=[1, 2, 3])

            for atype, fp32_err, int8_err in zip(anomaly_types, fp32_errors, int8_errors):

                if atype not in scores:
                    scores[atype] = {"fp32": [], "int8": []}

                scores[atype]["fp32"].append(fp32_err.item())
                scores[atype]["int8"].append(int8_err.item())

    # Aggregate mean per anomaly type
    anomaly_types = sorted(scores.keys())
    fp32_means = [np.mean(scores[t]["fp32"]) for t in anomaly_types]
    int8_means = [np.mean(scores[t]["int8"]) for t in anomaly_types]

    x = np.arange(len(anomaly_types))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, fp32_means, width, label="FP32", color="steelblue",  alpha=0.85)
    ax.bar(x + width / 2, int8_means, width, label="INT8", color="darkorange", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xlabel("Anomaly Type", fontsize=11)
    ax.set_ylabel("Mean MSE", fontsize=11)
    ax.set_title("Anomaly Score: FP32 vs INT8 by Anomaly Type", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.show()

    # ROC

from sklearn.metrics import roc_auc_score, roc_curve

def plot_roc_curve(
    fp32_model: nn.Module,
    quantized_model: nn.Module,
    test_loader: DataLoader,
) -> None:
    """
    ROC curve and AUROC for FP32 vs INT8. 
    0 = good, 1 = anomaly.
    A score of 1.0 = perfect separation, 0.5 = random.
    """

    fp32_model.eval()
    fp32_model = fp32_model.cpu()
    quantized_model.eval()

    fp32_scores, int8_scores, labels = [], [], []

    with torch.no_grad():

        for images, anomaly_types in test_loader:
            images = images.cpu()

            fp32_recons = fp32_model(images)
            int8_recons = quantized_model(images)

            fp32_errors = ((images - fp32_recons) ** 2).mean(dim=[1, 2, 3])
            int8_errors = ((images - int8_recons) ** 2).mean(dim=[1, 2, 3])

            for atype, fp32_err, int8_err in zip(anomaly_types, fp32_errors, int8_errors):
                label = 0 if atype == "good" else 1
                labels.append(label)
                fp32_scores.append(fp32_err.item())
                int8_scores.append(int8_err.item())

    labels = np.array(labels)
    fp32_scores = np.array(fp32_scores)
    int8_scores = np.array(int8_scores)

    fp32_auc = roc_auc_score(labels, fp32_scores)
    int8_auc = roc_auc_score(labels, int8_scores)

    fp32_fpr, fp32_tpr, _ = roc_curve(labels, fp32_scores)
    int8_fpr, int8_tpr, _ = roc_curve(labels, int8_scores)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fp32_fpr, fp32_tpr, label=f"FP32 (AUROC = {fp32_auc:.3f})")
    ax.plot(int8_fpr, int8_tpr, label=f"INT8 (AUROC = {int8_auc:.3f})")
    ax.plot([0, 1], [0, 1], linewidth=1, linestyle="--", label="Random (AUROC = 0.5)")

    ax.set_xlabel("FPR", fontsize=11)
    ax.set_ylabel("TPR", fontsize=11)
    ax.set_title("ROC Curve: FP32 vs INT8", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.show()

    print(f"FP32 AUROC : {fp32_auc:.4f}")
    print(f"INT8 AUROC : {int8_auc:.4f}")
    print(f"Difference : {abs(fp32_auc - int8_auc):.4f}")