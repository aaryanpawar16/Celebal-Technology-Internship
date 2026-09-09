# MNIST Image Denoising with a Convolutional Autoencoder

## Overview

This notebook builds a convolutional autoencoder that learns to remove Gaussian noise from MNIST handwritten digit images, reconstructing clean digits from noisy inputs.

## Dataset

- **Source:** `awsaf49/mnist-dataset` via `kagglehub`
- **Images:** loaded from disk, converted to grayscale, resized to 28×28, normalized to [0, 1]
- **Shapes:** `x_train` (60000, 28, 28, 1), `x_test` (10000, 28, 28, 1)

### Noise Generation

Gaussian noise is added to create noisy versions of the training and test images:

- `noise_factor = 0.4`
- Noise sampled from a standard normal distribution, scaled and added to pixel values
- Result clipped back to the valid [0, 1] range

## Model Architecture

A symmetric convolutional autoencoder:

**Encoder:** `Conv2D(32) → MaxPooling2D → Conv2D(32) → MaxPooling2D`
**Decoder:** `Conv2D(32) → UpSampling2D → Conv2D(32) → UpSampling2D → Conv2D(1, sigmoid)`

- Input/output shape: 28×28×1
- Bottleneck: 7×7×32
- **Total params:** 28,353 (all trainable)
- **Loss:** binary cross-entropy (appropriate since pixel values are in [0, 1])
- **Optimizer:** Adam

## Training

- Input: noisy images (`x_train_noisy`) → Target: clean images (`x_train`)
- 10 epochs, batch size 128, shuffled
- Validation on noisy/clean test pairs

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 0.1487 | 0.1038 |
| 5 | 0.0917 | 0.0906 |
| 10 | 0.0879 | 0.0870 |

Training and validation loss track closely throughout, with no signs of overfitting.

## Results

**PSNR (Peak Signal-to-Noise Ratio, higher = better):**

| Comparison | PSNR |
|---|---|
| Noisy vs. Clean | 10.98 dB |
| Denoised vs. Clean | 20.85 dB |

The autoencoder nearly doubles the PSNR relative to the raw noisy input, quantitatively confirming the visual denoising quality.

## Key Observations

- The bottleneck architecture forces the model to encode only the structure needed to reconstruct a plausible digit, effectively discarding the noise while preserving overall stroke shape.
- The PSNR improvement (10.98 → 20.85 dB) gives a quantitative confirmation of the visible denoising effect.
- At higher noise factors (> 0.5), thin-stroke digits (e.g. "1", "7") start to blur or lose detail due to the spatial compression from pooling/upsampling.
- Close tracking between train and validation loss indicates the model generalizes well at this dataset size and epoch count.
- **Possible improvements:** add skip connections (U-Net-style autoencoder), train for more epochs, or test robustness against other noise types (salt-and-pepper, speckle).

## Requirements

```
tensorflow
numpy
pillow
matplotlib
kagglehub
```

## How to Run

1. Ensure a Kaggle API token is configured for `kagglehub` to download the dataset (or provide the MNIST images locally in the expected folder structure).
2. Install dependencies: `pip install tensorflow numpy pillow matplotlib kagglehub`
3. Run the notebook cells sequentially from top to bottom.
