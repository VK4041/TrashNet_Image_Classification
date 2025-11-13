# TrashNet Image Classification

An image classification project using transfer learning with ResNet50 to classify waste materials into 6 categories for improved recycling and waste management.

## 📋 Overview

Waste contamination in recycling is a growing problem due to lack of awareness about which items are recyclable. This project builds a deep learning classifier that takes RGB images of waste materials and categorizes them into 6 classes, helping to reduce contamination and improve recycling efficiency.

## 🎯 Problem Statement

Improper sorting of recyclables leads to entire batches being discarded, exacerbating pollution and resource waste. This classifier aims to automate waste categorization using computer vision and deep learning.

## 📊 Dataset

**TrashNet Dataset** by Gary Thung and Mindy Yang
- **Total Images**: ~2,500 RGB images
- **Classes**: 6 categories
  - Cardboard
  - Glass
  - Metal
  - Paper
  - Plastic
  - Trash

**Key Characteristics**:
- Images organized in class-labeled folders
- Class imbalance present (Trash: 137 samples, Paper: 594 samples)
- Real-world waste material images with varying backgrounds

**Dataset Split**: 70% Train / 15% Validation / 15% Test (Stratified)

## 🏗️ Architecture

### Transfer Learning with ResNet50

**Base Model**: ResNet50 pretrained on ImageNet
- **Total Parameters**: ~25M
- **Architecture**: 50-layer deep residual network
- **Custom Classifier Head**:
  ```
  Linear(2048 → 512) → ReLU → Dropout(0.5) → Linear(512 → 6)
  ```

### Three Model Variants Explored

1. **Model 1 (Frozen Features)**: Only classifier trained
   - Test Accuracy: **83.95%**
   
2. **Model 2 (Full Fine-Tuning)**: All layers trainable
   - Test Accuracy: **75.26%**
   
3. **Model 3 (Selective Fine-Tuning)**: Last conv block (layer4) + classifier
   - Test Accuracy: **91.05%** ✅ **Best Baseline**

## 🔧 Data Preprocessing

### Image Transformations
- **Resize**: 224×224 pixels
- **Normalization**: ImageNet mean/std
- **Augmentation Pipeline**:
  - Random horizontal flip
  - Random rotation (±5-10°)
  - Color jitter (brightness, contrast, saturation)
  - Random affine transformations
  - Random erasing (Aug2)
  - Gaussian blur (Aug3)

### Class Balancing Strategy
- Oversampling minority classes to match maximum class count
- Augmentation applied only to oversampled images
- **Result**: 416 samples per class in balanced training set

## 📈 Results

### Best Model Performance

| Configuration | Test Accuracy | Notes |
|--------------|---------------|-------|
| Baseline (No Aug, No Balance) | 91.05% | Model 3 architecture |
| **Aug2 + Balanced** | **92.11%** | ✅ Best overall |
| Aug1 + Balanced | 89.74% | Lighter augmentation |
| Aug3 + Balanced | 90.00% | Aggressive transforms |

### Augmentation Comparison

**Aug1** (Geometric + Color):
- Random flip, rotation, color jitter, affine, perspective

**Aug2** (Spatial + Erasing):
- Random crop, rotation, affine, random erasing
- **Best performer at 92.11%**

**Aug3** (Filter-based):
- Gaussian blur, invert, posterize, grayscale

### Key Findings

✅ **Balancing + Augmentation**: +1.06% improvement over baseline  
✅ **Selective Fine-Tuning**: Better than full fine-tuning (91% vs 75%)  
✅ **Data Efficiency**: 2,500 images sufficient with proper augmentation  
✅ **Cross-Domain Test**: 81% accuracy on completely new waste images  

## 💡 Data Analysis Insights

### Correlation Heatmap (ResNet18 Features)
- **Glass ↔ Plastic**: Highest similarity (0.95+)
- **Metal ↔ Glass**: Strong correlation due to shine/reflectivity
- **Paper ↔ Cardboard**: Similar texture patterns
- **Trash**: Dispersed across all categories

### t-SNE Clustering
- Moderately distinct clusters for each material
- Overlap between visually similar pairs (Glass-Plastic, Paper-Cardboard)
- Trash class dispersed throughout feature space

### Common Misclassifications
Based on test set errors:
- Paper ↔ Cardboard (texture similarity)
- Glass ↔ Plastic (transparency/reflectivity)
- Metal ↔ Glass (surface properties)

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision numpy matplotlib scikit-learn tqdm seaborn
```

### Installation

```bash
git clone https://github.com/VK4041/TrashNet_Image_Classification.git
cd TrashNet_Image_Classification
```

### Dataset Setup

1. Download TrashNet dataset from [GitHub](https://github.com/garythung/trashnet)
2. Place in `Data/TrashNet/trashnet/` directory
3. Structure should be:
   ```
   trashnet/
   ├── cardboard/
   ├── glass/
   ├── metal/
   ├── paper/
   ├── plastic/
   └── trash/
   ```

### Training

```python
# Load best model configuration
model = ResNet50WasteClassifier(num_classes=6, freeze_features=True)
for param in model.base_model.layer4.parameters():
    param.requires_grad = True

# Train with balanced augmented data
train_model(model, balanced_train_loader, val_loader, test_loader, epochs=20)
```

## 🔬 Profiling & Optimization

### Performance Bottleneck Analysis

**Before Optimization**:
- DataLoader CPU time: ~730ms
- Main bottleneck: Data loading

**After Optimization** (num_workers=4, pin_memory=True):
- DataLoader CPU time: ~72ms
- **90% reduction in data loading time**

### Training Efficiency
- Epochs: 20 with early stopping (patience=3)
- Batch size: 128
- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss
- Device: CUDA (T4 GPU)

## 📊 Visualization Features

- Class distribution plots
- Random sample display with denormalization
- Training/validation loss curves
- Accuracy progression graphs
- Misclassified image analysis
- Feature correlation heatmaps
- t-SNE cluster visualization

## 🎓 Academic Context

**Course**: SIT744 Deep Learning - Deakin University  
**Author**: Varun Kumar  
**Research Paper**: [Classification of Trash for Recyclability Status](https://cs229.stanford.edu/proj2016/report/ThungYang-ClassificationOfTrashForRecyclabilityStatus-report.pdf)

## 💡 Key Techniques Demonstrated

- ✅ Transfer learning with pretrained CNNs
- ✅ Class imbalance handling through oversampling
- ✅ Strategic data augmentation
- ✅ Selective layer fine-tuning
- ✅ Early stopping to prevent overfitting
- ✅ Cross-domain generalization testing
- ✅ Feature extraction and visualization
- ✅ Performance profiling and optimization

## 📝 Project Highlights

1. **Smart Fine-Tuning**: Unfreezing only the last convolutional block achieved best results
2. **Balanced Learning**: Oversampling + augmentation improved accuracy by 1%
3. **Real-World Testing**: 81% accuracy on completely new waste images
4. **Efficient Pipeline**: 90% reduction in data loading time through optimization
5. **Interpretability**: Correlation analysis explains misclassification patterns

## 🔗 References

- [TrashNet Dataset](https://github.com/garythung/trashnet)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [Australian Recycling Facts](https://www.cleanup.org.au/recycle)

## ⚠️ Notes

- Designed for Google Colab with GPU support
- Requires ~2GB storage for dataset
- Training time: ~10-15 minutes on T4 GPU
- Best results with balanced + Aug2 configuration

## 🤝 Contributing

Contributions welcome! Feel free to:
- Add new augmentation strategies
- Test on different architectures
- Expand to more waste categories
- Improve cross-domain performance

## 📄 License

This project is open source and available under the MIT License.

## 📧 Contact

For questions or collaborations, please open an issue on GitHub.

---

**Impact**: This classifier can help reduce recycling contamination, saving resources and reducing environmental pollution through automated waste sorting.

Find the resources here: https://drive.google.com/drive/folders/14TOWegOtAg8Tfdjy9NRyVN0jLoLlrlA7?usp=sharing

---

**Note**: This project was developed as part of academic coursework. The techniques demonstrated are applicable to various NLP tasks requiring domain adaptation with limited computational resources.
