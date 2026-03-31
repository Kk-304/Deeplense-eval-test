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

### Classifier — ResNet50 + ImageNet Transfer Learning

Classifies strong lensing images into 3 dark matter substructure types. We use ResNet50 pretrained on ImageNet because the dataset is too small to train a deep CNN from scratch — ImageNet provides universal low-level features (edges, textures) while residual connections capture the fine-grained structural differences between classes. Training is two-phase: first the classification head trains alone with the backbone frozen, then the last 30 ResNet layers are unfrozen and fine-tuned at 20× lower learning rate to adapt to lensing-specific patterns without catastrophic forgetting. Rotational augmentation is applied since lensing images are rotationally invariant. Class weights handle any imbalance. All data is tested from Train and Val files in given dataset folder. Evaluated with ROC curves and AUC scores.

### SR_1 — SRGAN on Simulated Pairs (75→150)

Upscales simulated low-resolution lensing images using paired HR ground truths. The generator is SRResNet — 16 residual blocks with a single pixel-shuffle 2× upsample block. The discriminator is PatchGAN-style, classifying overlapping patches as real/fake for locally consistent texture generation. L1 loss alone produces blurry outputs that average over plausible high-frequency details, losing the sharp arc structures critical for lensing analysis. SRGAN's combined loss — L1 pixel + VGG19 perceptual (block5_conv4 features) + adversarial — balances pixel accuracy, structural fidelity, and sharpness. Training is two-phase: L1 pretrain first for a stable generator baseline, then full GAN training. 90:10 split. Evaluated with MSE, PSNR, SSIM.

### SR_2 — Transfer Learning to Real Data (64→128, 300 pairs)

Enhances real telescope images (HSC→HST) with only 300 paired samples. The same SRResNet architecture is rebuilt for 64×64 input / 128×128 output, and convolutional weights are transferred layer-by-layer from the SR_1 model — this works because Conv2D kernels are spatial-size agnostic. Transfer learning is essential here: 270 training images (after 90:10 split) cannot train any meaningful SR model from scratch, but the SR_1 generator already understands lensing-specific super-resolution from simulations that share the same underlying physics. Adversarial loss is intentionally dropped — the discriminator memorizes 270 images within ~5 epochs and provides only noise gradients. Instead we fine-tune with L1 + VGG perceptual loss at very low learning rate (5e-5) to make small domain-specific adjustments. Heavy augmentation is applied. Evaluated with MSE, PSNR, SSIM and the pre-fine-tuned model.

---

## Data & Model Placement

```
Classifier/
├── dataset/           ← Place extracted classification dataset here
└── classifier.ipynb

SR_1/
├── Dataset/           ← Place extracted simulated HR/LR pairs here
├── sr1.ipynb
└── best_generator_simulated.keras   ← Saved generator weights (after training)

SR_2/
├── Dataset/           ← Place extracted real HR/LR pairs here
├── sr2.ipynb
└── best_generator_real.keras        ← Saved generator weights (after training)
```

Trained generator weights are saved inside their respective task folders.  
SR_2 loads the pretrained generator from `SR_1/best_generator_simulated.keras` for transfer learning.

---

## How to Run

Run in **Google Colab** (GPU runtime) in order: **Classifier → SR_1 → SR_2** (SR_2 depends on SR_1's saved generator).

File paths in notebooks point to:
```
/content/drive/MyDrive/Tasks/Classifier/dataset/dataset
/content/drive/MyDrive/Tasks/SR_1/Dataset
/content/drive/MyDrive/Tasks/SR_2/Dataset
```

Mount Google Drive and place datasets accordingly.

**Framework:** Keras / TensorFlow  
**Split:** 90:10 train-test for all tasks
