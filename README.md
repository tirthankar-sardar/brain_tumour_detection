# brain_tumour_detection
## Overview 
This project focuses on detecting brain tumors from MRI images using a deep learning approach implemented in TensorFlow. The model classifies MRI scans into tumor and non-tumor categories, assisting in early diagnosis and medical analysis. The solution is designed for academic and research purposes, demonstrating the use of convolutional neural networks (CNNs) in medical image classification.
Model Architecture

# The model is a Convolutional Neural Network (CNN) built using TensorFlow and Keras.
## Key layers include:

Convolutional Layers: Extract spatial features from MRI scans

Max Pooling Layers: Reduce spatial dimensions and prevent overfitting

Dense Layers: Learn complex patterns for classification

Softmax Output Layer: Predicts the probability of tumor presence

The model is trained using categorical cross-entropy loss and the Adam optimizer.
