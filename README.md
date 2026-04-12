# DeepLense Evaluation Tests — GSoC 2026

**Project:** Unsupervised Super-Resolution and Analysis of Real Lensing Images  
**Organization:** [ML4Sci](https://ml4sci.org/) / [DeepLense](https://github.com/ML4SCI/DeepLense) | Google Summer of Code 2026

---

## Task Mapping

| Folder | Test | Description |
|--------|------|-------------|
| `Classifier/` | **Common Test I** | Multi-class classification (no substructure / subhalo / vortex) |
| `SR_1/` | **Specific Test VI.A** | Super-resolution on simulated data (75×75 → 150×150) |
| `SR_2/` | **Specific Test VI.B** | Super-resolution on real telescope data — 300 pairs (64×64 → 128×128) |

---

## Approach

### Classifier — Custom CNN

Classifies strong lensing images into 3 dark matter substructure types using a custom 4-block CNN trained from scratch on single-channel 150×150 images. Data is normalized using dataset statistics and augmented with flips and rotations (exploiting rotational symmetry of lensing systems). Class weights handle any imbalance. Evaluated with ROC curves and AUC scores.

### SR_1 — SRGAN on Simulated Pairs (75→150)

Upscales simulated low-resolution lensing images using paired HR ground truths. The generator is SRResNet — 16 residual blocks with a single pixel-shuffle 2× upsample block. Training is two-phase: L1 pretrain first for a stable generator baseline, then GAN training with pixel + adversarial loss. Data normalized to [0,1] globally. 90:10 split. Evaluated with MSE, PSNR, SSIM.

### SR_2 — Transfer Learning to Real Data (64→128, 300 pairs)

Enhances real telescope images with only 300 paired samples. The same SRResNet architecture is rebuilt for 64×64 input / 128×128 output, and convolutional weights are transferred layer-by-layer from the SR_1 model — this works because Conv2D kernels are spatial-size agnostic. Transfer learning is essential here: 270 training images (after 90:10 split) cannot train any meaningful SR model from scratch, but the SR_1 generator already understands lensing-specific super-resolution from simulations that share the same underlying physics. Fine-tuned with L1 + VGG perceptual loss at low learning rate (5e-5) with heavy augmentation. Evaluated with MSE, PSNR, SSIM against both bicubic baseline and pre-fine-tuned model.

---

## Results

### Classifier

| Metric | Score |
|--------|-------|
| Accuracy | 90.83% |
| AUC (no substructure) | 0.9866 |
| AUC (sub-halo) | 0.9708 |
| AUC (vortex) | 0.9846 |

### SR_1 — Simulated Data (75×75 → 150×150)

| Method | MSE | PSNR (dB) | SSIM |
|--------|-----|-----------|------|
| Bicubic | 0.000059 | 42.31 | 0.9736 |
| SRGAN (ours) | 0.000326 | 35.03 | 0.9510 |

### SR_2 — Real Telescope Data (64×64 → 128×128)

| Method | MSE | PSNR (dB) | SSIM |
|--------|-----|-----------|------|
| Bicubic interpolation | 0.006901 | 24.72 | 0.6933 |
| Before fine-tuning | 0.008657 | 23.72 | 0.7435 |
| After fine-tuning | 0.000379 | 36.85 | 0.9494 |

Improvement over bicubic: **PSNR +12.13 dB | SSIM +0.2561**

---

## Data & Model Placement

```
Tasks/
├── Classifier/dataset/dataset/
│   ├── train/
│   │   ├── no/          ← .npy lensing images (no substructure)
│   │   ├── sphere/      ← .npy lensing images (sub-halo)
│   │   └── vort/        ← .npy lensing images (vortex)
│   └── val/
│       ├── no/
│       ├── sphere/
│       └── vort/
├── SR_1/Dataset/
│   ├── LR/              ← 75×75 low-resolution .npy images
│   └── HR/              ← 150×150 high-resolution .npy images
└── SR_2/Dataset/
    ├── LR/              ← 64×64 low-resolution .npy images
    └── HR/              ← 128×128 high-resolution .npy images

generator_simulated_final.keras   ← Saved SR_1 generator (loaded by SR_2 for transfer learning)
```

---

## How to Run

Run in **Google Colab** (GPU runtime) in order: **Classifier → SR_1 → SR_2** (SR_2 depends on SR_1's saved generator).

File paths in notebooks point to:
```
/content/dataset/Tasks/Classifier/dataset/dataset
/content/dataset/Tasks/SR_1/Dataset
/content/drive/MyDrive/Tasks/SR_2/Dataset
```

Mount Google Drive and place datasets accordingly.  
To run locally, replace the above paths with your cloned repo folder path.

**Framework:** Keras / TensorFlow  
**Split:** 90:10 train-test for all tasks
