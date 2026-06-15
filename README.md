# CNN_image_classifier - Base Description
This project implements a CNN architecture (3 convolutional blocks + fully  connected classifier) built on PyTorch trained on the CIFAR-10 dataset (50,000 training images,  10,000 test images, 32x32 RGB)

# CIFAR-10 CNN Image Classifier

A convolutional neural network built from scratch in PyTorch to classify images 
into 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

## Live Demo
(https://cnnimageclassifier-hd96eprefa99y2u8phfkjw.streamlit.app/)

## Overview
This project implements a CNN architecture (3 convolutional blocks + fully 
connected classifier) trained on the CIFAR-10 dataset (50,000 training images, 
10,000 test images, 32x32 RGB).

## Architecture
- Conv Block 1: 32 filters, 3x3 kernel → ReLU → MaxPool
- Conv Block 2: 64 filters, 3x3 kernel → ReLU → MaxPool
- Conv Block 3: 128 filters, 3x3 kernel → ReLU → MaxPool
- Flatten → Dense(512) → ReLU → Dense(10)

## Results
- Test accuracy: 74.78%
- Trained for 10 epochs, batch size 32, Adam optimizer (lr=0.001)
- CrossEntropyLoss decreased from 1.31 to 0.44 over training

## How to Run Locally
\`\`\`bash
pip install -r requirements.txt
streamlit run app.py
\`\`\`

## Future Improvements
- **Batch normalization**: would stabilize training and likely boost accuracy 
  by 3-5%
- **More epochs**: training loss was still decreasing at epoch 10, suggesting 
  the model hasn't converged yet
- **Deeper architecture**: additional conv blocks or residual connections 
  (ResNet-style) for higher capacity
- **Learning rate scheduling**: decaying the learning rate over time for 
  finer convergence

## Tech Stack
PyTorch, torchvision, Streamlit, PIL
