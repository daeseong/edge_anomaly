# edge_anomaly

A Post-Training Quantization (PTQ) pipeline applied to a Convolutional Autoencoder for unsupervised anomaly detection on industrial inspection images.

---

## Goal

This project investigates the if **compressing a model from FP32 to INT8 using Post-Training Quantization degrade its ability to detect anomalies.**

The model is a Convolutional Autoencoder trained exclusively on normal images from the [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad). It learns to reconstruct normal images well and flags anomalies by measuring how poorly it reconstructs unseen defects. No anomaly labels are used during training. PTQ is applied after training with no retraining or fine-tuning.

The short answer: at convergence, fp32 and int8 performed similarly, where fp32 performed slightly better than the quantized int8. **PTQ preserves discriminative ability.**

---

## Project Structure

```
edge_anomaly/
│
├── data/
│   └── data_processing.py
│
├── source/
│   ├── encoder.py
│   ├── decoder.py
│   ├── cnn.py
│   ├── train.py
│   ├── quantize.py
│   ├── dequantize.py
│   └── visualize.py
│
├── notebooks/
│   └── quantization_CNN.ipynb
│
├── checkpoints/
│   └── best_model.pth
│
├── config/
│   └── cnn.yaml
│
├── results/
├── requirements.txt
└── README.md
```

---

## Challenges

**PyTorch PTQ limitations**

PyTorch's eager mode static PTQ has two constraints that shaped the implementation:

- The quantized model runs on **CPU only**.
- **Per-channel weight quantization is not supported for `ConvTranspose2d`**. The first decoder layer was forced onto per-tensor quantization instead, a coarser scheme that assigns one scale factor to the entire weight tensor rather than one per channel. This layer consistently showed the widest weight error distribution of the three analysed layers and is the weakest link in the quantization pipeline.

**Small dataset**

MVTec's bottle category provides only 209 training images which were all normal. This constrained batch diversity per epoch (7 batches at batch size 32) and required careful learning rate tuning to avoid poor optimization.

---

## Stack

| | |
|---|---|
| **Framework** | PyTorch 2.6.0 |
| **Dataset** | MVTec AD |
| **Quantization** | Post-Training Quantization (PTQ), FP32 → INT8 |
| **Evaluation** | AUROC, per-layer weight error, residual heatmaps |
