# CIFAR-10 Image Classification Learning Project — ANN vs CNN

## Overview

A learning-focused deep learning notebook that builds and compares an Artificial Neural Network (ANN) and a Convolutional Neural Network (CNN) on CIFAR-10, then explores training strategies (dropout, batch norm, data augmentation) to show how architecture and technique affect image classification performance.

**Learning goal:** understand the full deep learning pipeline — from raw pixels to a trained, evaluated model — and see concretely why CNNs outperform ANNs on image data.

## Dataset

- **CIFAR-10:** 60,000 32×32 color images across 10 classes
  - 50,000 training images / 10,000 test images
- **Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Loaded directly via `tf.keras.datasets.cifar10`

## Environment

- TensorFlow 2.21.0
- Trained on CPU (native Windows — GPU not available for TensorFlow ≥ 2.11 without WSL2 or DirectML)

## Pipeline Steps

1. **Load Data** — Load CIFAR-10 train/test splits via Keras.
2. **Visualize Samples** — Plot a grid of sample images with their class labels.
3. **Preprocessing** — Normalize pixel values to [0, 1]; flatten images to 3,072-length vectors for the ANN.
4. **ANN Model** — Fully connected network (512 → 256 → 128 → 10) with dropout, trained on flattened pixels.
5. **CNN Model** — 3-block convolutional network (32 → 64 → 128 filters) with batch normalization, max pooling, and dropout, trained on raw image tensors.
6. **Learning Curve Comparison** — Plot ANN vs CNN validation accuracy over epochs.
7. **Data Augmentation Upgrade** — Add random flip/rotation/zoom layers and retrain a CNN to test generalization improvements.
8. **Final Comparison Table** — Side-by-side test accuracy for all three models.
9. **Beginner Task Solutions** — Worked examples for extending the ANN, inspecting the CNN's filter progression, and confirming early stopping/epoch/augmentation settings.

## Results

| Model | Test Accuracy |
|---|---|
| ANN | 0.4441 |
| CNN | 0.6860 |
| Augmented CNN | 0.4926 |

- **ANN:** trained ~19 epochs (early stopping), plateaus around 44% test accuracy — treats each image as a flat vector, so it can't exploit spatial structure.
- **CNN:** trained 7 epochs (early stopping triggered), reaches 68.6% test accuracy — convolution + pooling extract spatial/hierarchical features far more effectively.
- **Augmented CNN:** trained only 3 epochs before early stopping and underperformed the plain CNN (49.3%) in this run — augmentation typically helps generalization, but here it needed more training epochs (unaugmented data trains faster/converges sooner) to show its benefit; the early-stopping patience cut training short relative to the harder, augmented training signal.
- **Deeper ANN (bonus task):** adding an extra dense layer did not help — it stopped early at 3 epochs with 34.5% test accuracy, underperforming the original ANN.

## Beginner Tasks Covered

- ✅ Increase ANN depth and observe performance (deeper ANN underperformed — more layers ≠ better without the right architecture)
- ✅ Progressive CNN filters: 32 → 64 → 128
- ✅ Train for up to 20 epochs
- ✅ Add `EarlyStopping` (`monitor='val_loss', patience=3, restore_best_weights=True`)
- ✅ Add data augmentation (random flip, rotation, zoom) and retrain

## Key Takeaways

- ANNs ignore spatial structure in images, capping their performance well below CNNs on the same task.
- CNNs' convolution and pooling layers let them learn spatial and hierarchical features, giving a large accuracy jump (+24 points) over the ANN.
- Adding more layers to an ANN doesn't fix its fundamental limitation for image data.
- Data augmentation is a generalization technique, not a guaranteed short-term accuracy boost — its benefit depends on training long enough for the model to learn from the added variation.
- This notebook is a solid foundation for computer vision interview prep and further deep learning projects.

## Requirements

```
tensorflow
matplotlib
numpy
pandas
```

## How to Run

1. Ensure TensorFlow can download CIFAR-10 (internet access on first run, or a cached copy).
2. Install dependencies: `pip install tensorflow matplotlib numpy pandas`
3. Run the notebook cells sequentially from top to bottom. Training the CNN models is the most time-consuming step; a GPU-enabled environment (e.g. Colab, WSL2, or Linux) is recommended for faster runs.
