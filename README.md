# License Plate State Classification

This repository contains a comprehensive evaluation of multiple approaches for classifying U.S. license plates by state using both vision-only and OCR-based techniques. The project explores how state identification can be reframed as a visual recognition task rather than relying solely on text extraction through OCR.

---

## 🚀 Project Overview
Automatic License Plate Recognition (ALPR) systems typically rely on OCR to extract the state name from license plates. However, real-world conditions such as motion blur, stylized fonts, and complex backgrounds often make OCR unreliable.

This project evaluates four distinct approaches:

1. **ResNet-18 Transfer Learning (Full Fine-Tuning)**
2. **OCR with Fuzzy Text Matching**
3. **CLIP Zero-Shot Classification**
4. **CLIP Few-Shot Classification**

This uses the dataset found here:

https://www.kaggle.com/datasets/gpiosenka/us-license-plates-image-classification

---

## 📊 Key Findings
- **ResNet-18 (Full Fine-Tune)** achieved the highest performance with **92.13% accuracy**, with no states dropping below the 60% reliability threshold.
- **CLIP Zero-Shot** performed surprisingly well with **88.31% accuracy**, despite no domain-specific training.
- **CLIP Few-Shot** improved robustness for hard states but slightly reduced overall accuracy.
- **OCR Baseline** reached **85.94%** but showed large per-state variance, making it unsuitable for production.

Overall, treating state identification as a **visual classification** problem significantly outperforms traditional OCR-based approaches.

---

## 📁 Dataset
The project uses the **US License Plates - Image Classification** dataset from Kaggle.
- ~8,000 labeled images
- Size: **128 × 224 × 3** JPEGs
- 56 classes (50 states + Washington DC + 5 territories)
- Custom **70/15/15 stratified split**
- Data augmentation applied:
  - Color jitter
  - Random rotation
  - Random horizontal flip

---

## 🧠 Methods
### 1. **ResNet-18 Transfer Learning**
Four configurations were tested:
- Full fine-tuning
- Frozen backbone + linear classifier
- Frozen backbone + 3-layer classifier
- Partially frozen (unfreezing last residual block)

Full fine-tuning significantly outperformed all others.

### 2. **OCR Baseline**
- Implemented with **EasyOCR**
- Used fuzzy matching via Levenshtein distance
- High variability between states

### 3. **CLIP Zero-Shot**
- Used text prompts such as "A picture of a Virginia license plate"
- Classified via cosine similarity in shared embedding space

### 4. **CLIP Few-Shot**
- Misclassified training examples used as additional embedding references
- Improved the worst-case states

---

## 🧪 Experimental Setup
- Frameworks: **PyTorch**, **Transformers**, **EasyOCR**
- Optimizer: **Adam**, LR = 0.001 (full FT), 0.0001 for partially unfrozen blocks
- ~1,200 test images

Metrics reported:
- Overall accuracy
- Per-state accuracy analysis
- Overfitting diagnostics via loss/accuracy curves

---

## 📈 Results Summary
| Method | Accuracy |
|--------|-----------|
| **ResNet-18 (Full Fine-Tune)** | **92.13%** |
| CLIP Zero-Shot | 88.31% |
| CLIP Few-Shot | 86.33% |
| OCR Baseline | 85.94% |
| ResNet-18 (Partial, Layer4) | 79.70% |
| ResNet-18 (Frozen, 1 Layer) | <50% |
| ResNet-18 (Frozen, 3 Layers) | <50% |

## 📄 License
This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## 🔗 Repository
https://github.com/andreasparas1/plate-classification

---
Feel free to open issues or PRs if you'd like to extend the project or improve model performance!

