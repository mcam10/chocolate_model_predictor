# StankNet

StankNet: Deep Learning-Based Stool Classification using Transfer Learning
StankNet is a deep learning model designed to classify stool samples according to the Purina Fecal Score Chart, powered by ResNet50 and transfer learning.

## Overview
This tool serves pet owners in monitoring animal digestive health through automated, consistent scoring of stool samples. Our latest implementation shows significant improvements in classification accuracy through transfer learning.

### Technical Highlights
- Built on ResNet50 architecture pre-trained on ImageNet
- Fine-tuned for specific stool classification tasks
- Demonstrated improved accuracy with strong diagonal pattern in confusion matrix
- Specialized in distinguishing between subtle differences in stool consistency

### The Purina Fecal Score Chart (1-7)

- Score 1: Very hard and dry
- Score 2: Firm but not hard
- Score 3: Log-shaped, moist
- Score 4: Very moist but has shape
- Score 5: Very moist and barely has shape
- Score 6: Has texture but no shape
- Score 7: Watery, no texture

## Model Performance
- Successfully differentiates between 7 different stool consistency classes
- Strongest performance in distinguishing middle-range consistencies (classes 2-5)
- Validated through confusion matrix analysis showing clear diagonal pattern
- Uses transfer learning to leverage ImageNet features while specializing in stool characteristics

## Data Structure
Data collection starts inside of Google Drive then preprocessed and cleaned inside of an AWS S3 Bucket. 

```bash
data/
    1/  # Very hard and dry samples
        image1.jpg
        image2.jpg
        ...
    2/  # Firm but not hard samples
        image1.jpg
        image2.jpg
        ...
    ...
```

## Implementation Details
- ResNet50 backbone with custom classification head
- Transfer learning approach with frozen feature extraction layers
- Fine-tuned final layers for stool-specific feature detection
- Data augmentation and normalization for robust performance
