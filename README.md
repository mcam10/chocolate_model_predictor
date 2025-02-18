# StankNet

StankNet: Deep Learning-Based Stool Classification
StankNet is a deep learning model designed to classify stool samples according to the Purina Fecal Score Chart.

This tool will start as research and hopefully serve pet owners in monitoring animal digestive health through automated, consistent scoring of stool samples.

Overview
The Purina Fecal Score Chart ranges from 1 to 7:

Score 1: Very hard and dry

Score 2: Firm but not hard

Score 3: Log-shaped, moist

Score 4: Very moist but has shape

Score 5: Very moist and barely has shape

Score 6: Has texture but no shape

Score 7: Watery, no texture

## Data Structure
Data collection starts inside of Google Drive then prepocessed and cleaned inside of an AWS S3 Bucket. 

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
